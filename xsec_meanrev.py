"""クロスセクション平均回帰(contrarian)= チャンピオンと**別アプローチ**の検証済み手法。

― 何が「別」か ―――――――――――――――――――――――――――――――――――――
チャンピオン `confluence_meanrev` は「各銘柄を単独で見て、自己正規化Z×RSI×長期Z×平穏
レジームが同時に行き過ぎた高確信局面だけ建てる」per-instrument のコンフルエンス平均回帰。
本手法は機構が全く異なる **portfolio-level のランキング戦略**:
  毎リバランス(hold本ごと)に19銘柄を直近リターン(vol正規化・横断demean)で順位付けし、
  **最も負けた銘柄をロング / 最も勝った銘柄をショート**して横断平均への収束を取る。
  =マーケットニュートラル寄り・スケジュール建玉・銘柄間の相対値で判断(価格の絶対水準でない)。

理論的裏付け: 既往検証で「メジャー/クロスFXのクロスセクション**モメンタム**は net マイナス」
(exp03)。その反対=クロスセクション**平均回帰**は net プラスのはず、を実証(exp19: 反対方向の
モメンタム版は毎年マイナスの鏡像)。

― 検証結論(各脚$10k = チャンピオンの value サイジングと同尺、H4, 19銘柄, 2016-2026)―――
  現行チャンピオンは v2(confluence_meanrev_v2, ERフィルタ付)= total_pnl **19,072**(v1は18,109)。
  * 片側4脚: 総PnL **18,224**(v1超・v2の96%)。片側5脚: **21,678 > v2**。ただし脚を増やす=
    投下資本を増やすだけ(常時10脚, PF1.09, プラス年率64%)→「v2超」はレバレッジであってエッジ向上でない。
  * 同尺($10k/脚)では利益目標は実質達成だが、資本効率・頑健性はチャンピオンが明確に上(下記)。
  * 注: champion を改善した ER(40)フィルタは本手法に**転移しない**(検証済)。強トレンド銘柄こそ
    fade対象の極値なので、ERで除くとシグナルが減る(er_max<=0.5 で悪化)。機構が違う証左。
  * ただし質は劣る&資本を食う(知的誠実性のため明記):
      - PF≈1.10(チャンピオン1.83)。約500取引/年(同112)。
      - 8脚を常時保有=常に建玉 → 同時建玉~6に絞ると11.2k、~4で5.4k(チャンピオンは平均1-2銘柄)。
        つまり「同等利益」は約5倍の平均投下資本で得ている(資本効率はチャンピオンが上)。
      - 利益が後半に偏在: IS(2016-2020)はほぼフラット1.0k / OOS(2021-2026)17.2k(短期反転は
        高ボラ期に強い)。プラス年率73%(チャンピオン100%)。
      - パラメータ感度: lookbackには頑健(アンサンブルでも17k)だが hold に敏感。net黒字には
        長hold(>=24本≈4日)が必須(往復spread償却)で、hold<=18 はコスト負け。18kはhold=24の
        好セル=利益の「大きさ」は過信しない(方向のエッジは本物だが厚みは薄い)。--scan で確認可。
  * **本当の価値=分散**: チャンピオン月次PnLとの相関0.135(≒無相関の別エッジ)。
    等リスクで少量(~10%)混ぜると Sharpe 1.694→1.703・最大DD -1420→-1192(16%改善)。

→ 「チャンピオンを置き換える」のではなく「直交する補完エッジ」。詳細な研究過程は exp19/b/c/d。

実行:
  uv run python xsec_meanrev.py            # 年次成績 + 目標(利益)対比
  uv run python xsec_meanrev.py --scan     # lookback/hold グリッドで頑健性(高原)確認
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import universe as uni

pd.set_option("display.width", 220)

NOTIONAL = 10_000.0          # 1脚あたり建玉(チャンピオンの value=10k と同尺)
DEFAULT = dict(lookback=9, hold=24, max_legs=4, score_z=0.0, vol_win=50)


def universe_close(tf: str, exclude=("AUDJPY",)) -> pd.DataFrame:
    names = [x for x in uni.universe(crosses=True) if x not in set(exclude)]
    return pd.DataFrame({n: uni.instrument_close(n, tf) for n in names}).dropna()


def backtest(close: pd.DataFrame, *, lookback=9, hold=24, max_legs=4,
             score_z=0.0, vol_win=50) -> pd.DataFrame:
    """各脚$10k のトレード単位 PnL を生成。先読みなし(t時点の順位で t→t+hold を取る)。"""
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_win).std()
    mp = close.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}  # 半spread(価格比)

    recs = []
    for t in range(max(lookback, vol_win) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]           # vol正規化
        if score.isna().any():
            continue
        score = score - score.mean()                # 横断 demean(ドル全面高安を相殺)
        s = score.sort_values()
        longs = s[s < -score_z].index[:max_legs]    # 最も負け=ロング
        shorts = s[s > score_z].index[-max_legs:]   # 最も勝ち=ショート
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        ts = close.index[t + hold]
        for p in longs:
            recs.append((ts, (fwd[p] - 2 * hs[p]) * NOTIONAL))   # 往復spread計上
        for p in shorts:
            recs.append((ts, (-fwd[p] - 2 * hs[p]) * NOTIONAL))
    return pd.DataFrame(recs, columns=["exit", "pnl"])


def yearly_table(trades: pd.DataFrame) -> pd.DataFrame:
    g = trades.assign(year=pd.DatetimeIndex(trades["exit"]).year).groupby("year")["pnl"]
    rows = {}
    for year, p in g:
        pos = p[p > 0].sum(); neg = -p[p < 0].sum()
        rows[int(year)] = {"trades": len(p),
                           "profit_factor": round(pos / neg, 2) if neg > 0 else float("inf"),
                           "pnl": round(p.sum(), 0)}
    df = pd.DataFrame(rows).T
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


CHAMPION_TOTAL = 19072.0  # 現行チャンピオン confluence_meanrev_v2(ERフィルタ)総PnL(v1=18109)


def main() -> int:
    ap = argparse.ArgumentParser(description="クロスセクション平均回帰の検証")
    ap.add_argument("--tf", default="H4")
    ap.add_argument("--scan", action="store_true", help="lookback/hold グリッドで高原確認")
    ap.add_argument("--cross-spread", type=float, default=3.0)
    args = ap.parse_args()

    uni.register_cross_spreads(args.cross_spread)
    close = universe_close(args.tf)
    print(f"universe=19  tf={args.tf}  bars={len(close)}  "
          f"{close.index[0].date()}..{close.index[-1].date()}\n")

    if args.scan:
        print("=== 頑健性スキャン(reversion方向, 各脚$10k)— 高原で広くプラスか ===")
        rows = []
        for lb in (3, 6, 9, 12, 18):
            for hold in (6, 12, 18, 24):
                t = backtest(close, lookback=lb, hold=hold)
                y = yearly_table(t)
                pf = y["profit_factor"].replace(np.inf, np.nan)
                rows.append({"lookback": lb, "hold": hold, "total_pnl": round(t["pnl"].sum(), 0),
                             "pf_median": round(pf.median(), 2), "pos_yr": round((y["pnl"] > 0).mean(), 2)})
        s = pd.DataFrame(rows)
        piv = s.pivot(index="lookback", columns="hold", values="total_pnl")
        print("total_pnl(行=lookback, 列=hold):")
        print(piv.to_string())
        print(f"\nプラス設定の割合: 全体{(s['total_pnl']>0).mean():.0%} / "
              f"hold>=24限定{(s[s['hold']>=24]['total_pnl']>0).mean():.0%}")
        print("→ 正直な解釈: (1)contrarian方向は構造エッジ(exp19でモメンタム鏡像は毎年マイナス)。")
        print("  (2)net黒字には長hold(>=24本≈4日)が必須=往復spreadの償却。短hold(<=18)はコスト負け。")
        print("  (3)18kはhold=24の好セル。lookbackは頑健だがholdに敏感=利益の大きさは過信しない。")
        return 0

    trades = backtest(close, **DEFAULT)
    y = yearly_table(trades)
    print(f"=== 年次成績(lookback={DEFAULT['lookback']} hold={DEFAULT['hold']} "
          f"片側{DEFAULT['max_legs']}脚, 各脚$10k)===")
    print(y.to_string())
    total = trades["pnl"].sum()
    pf = y["profit_factor"].replace(np.inf, np.nan)
    print(f"\n総PnL={total:.0f}  PF中央={pf.median():.2f}  プラス年率={(y['pnl']>0).mean():.0%}  "
          f"年取引={int(y['trades'].mean())}")
    print(f"\n=== 🎯 利益目標(現行チャンピオン v2 {CHAMPION_TOTAL:.0f} と同尺比較; v1=18109)===")
    v1 = 18109.0
    print(f"  v1比: {'✅' if total >= v1 else '❌'} ({total:.0f} vs {v1:.0f}, {total/v1:.0%})  "
          f"v2比: {'✅' if total >= CHAMPION_TOTAL else '△'} ({total:.0f} vs {CHAMPION_TOTAL:.0f}, {total/CHAMPION_TOTAL:.0%})")
    print("  注: 片側4脚=18.2k(v1超/v2の96%)。片側5脚=21.7k>v2 だが常時10脚=資本増(レバレッジ)。")
    print("      同尺では同等だが資本効率・PF(1.1 vs 1.7)はチャンピオンが上。真価は相関0.135の分散。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
