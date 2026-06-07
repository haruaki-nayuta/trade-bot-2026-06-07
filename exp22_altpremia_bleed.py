"""イテレーション22: 代替リスクプレミア(クロスセクション・モメンタム / キャリー)の失血窓ヘッジ。

戦略フレーム(reports/09 後, bleed_lab): 補完エッジは「平均相関の低さ」でなく
**チャンピオンv2が失血しているまさにその窓で稼ぐか** で評価する。チャンピオンの失血窓は
高ER(トレンド継続レジーム)に集中(2021-2023, 最深=2022 USDラリー)。

これまでの族(tsmom/ma/adx/breakout = 単一銘柄の方向性トレンド)とは別系統の **直交プレミア** 2つ:
 (1) クロスセクション・モメンタム = 19銘柄を直近リターンで順位付けし「勝ち組ロング/負け組ショート」。
     = xsec_meanrev(contrarian, チャンピオンと相関0.135の既存補完)の **鏡像**。
     方向はトレンド追随なので失血窓(トレンド期)で稼ぐ仮説。マーケットニュートラル寄り。
 (2) キャリー = 金利差順方向「高金利ロング/低金利ショート」(内蔵近似金利 fxlab/carry.py)。
     価格と独立な収益源(金利受払)を含む別レジームのプレミア。

評価軸(優先順):
  1) mean_in_bleed_IS と mean_in_bleed_OOS が **両方プラス** = 持続ヘッジ(2022一発でない)
  2) hedge_edge = mean_in_bleed - mean_normal が大きい(失血窓で相対的に稼ぐ)
  3) mean_in_bleed > 0(窓内で絶対的にプラス)
  単体PF/total は無視(トレンド系は単体net負が既定 — reports/02)。

月次PnLストリームは自前構築(bleed_lab.strategy_monthly_pnl は strategies/ モジュール用なので、
xsec/carry はここで mask × 月次PnL を突き合わせる)。

― 検証結論(実測)―――――――――――――――――――――――――――――――――――――
 ● キャリーが圧勝(失血窓ヘッジとして)。 best=hold=42,k=5:
     in_bleed=+420.6 vs normal=+27.0 (edge +393.6), IS=+284.6 / OOS=+570.3 = 強い持続(2022一発でない)。
     失血窓貢献プラス設定92%, 持続67% = 広い高原。単体 standalone も +11,533(保険なのに黒字)。
     金利受払を除く『価格のみ』成分でも in_bleed=+305.2(IS+199/OOS+422)= 高金利通貨が失血窓(=USD
     ラリー等のトレンド期)で素直に伸びる為替方向エッジが本体。受払は薄い上乗せクッション。
 ● クロスセクション・モメンタムは弱い。 best=lb18,hold24,k3:
     in_bleed=+160.3, だが持続設定わずか3%・IS+20.8(ほぼゼロ)・standalone -13,078。
     失血窓貢献はほぼOOS(2022以降)依存=持続性が薄い。エッジとして劣後。
 ● 統合DDテスト(champion + carry overlay を1口座でDD=20%再較正; champ単独=CAGR+21.6%/Sharpe1.21/p95-28.7%):
     - 落とし穴: キャリーは常時~10脚を保有しスロットを食う。max_pos=8 だと champion を締め出し CAGR崩壊(+2.5%)。
       max_pos を広げて両者に枠を与えると回復(max_pos=16, w=0.25 で CAGR+12%/p95-24%)。
     - 最良=集中キャリー(hold=42,k=2=常時4脚で資本食い小), max_pos=12, weight=0.25:
         CAGR +21.0% / Sharpe 1.34 / 100%プラス年 / 理論DD p95 -25.0%
       → champ単独の CAGR(+21.6%)は 0.6pp 届かないが、Sharpe(1.21→1.34)と
         テール(p95 -28.7%→-25.0%, 3.7pp改善)を Pareto 改善。「同リターンでテール縮小」型の保険。

実行: uv run python exp22_altpremia_bleed.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import bleed_lab as bl
from fxlab import carry, config
from fxlab import universe as uni

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

NOTIONAL = 10_000.0  # 1脚あたり建玉(チャンピオンの value=10k と同尺)


def universe_close(tf: str, exclude=("AUDJPY",)) -> pd.DataFrame:
    names = [x for x in uni.universe(crosses=True) if x not in set(exclude)]
    return pd.DataFrame({n: uni.instrument_close(n, tf) for n in names}).dropna()


def _half_spread(names, mp) -> dict:
    return {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}


# --- (1) クロスセクション・モメンタム(トレード単位 → 月次PnL) ----------
def xsec_momentum_trades(close: pd.DataFrame, *, lookback, hold, k,
                          vol_win=50, demean=True) -> pd.DataFrame:
    """勝ち組ロング/負け組ショート(xsec平均回帰の鏡像)。各脚$10k, 往復spread計上。先読みなし。"""
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_win).std()
    mp = close.mean()
    hs = _half_spread(names, mp)
    recs = []
    for t in range(max(lookback, vol_win) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]            # vol正規化
        if score.isna().any():
            continue
        if demean:
            score = score - score.mean()             # 横断 demean(ドル全面高安を相殺)
        s = score.sort_values()
        losers = s.index[:k]                          # 最も負け → ショート
        winners = s.index[-k:]                         # 最も勝ち → ロング(モメンタム)
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        ts = close.index[t + hold]
        for p in winners:
            recs.append((ts, (fwd[p] - 2 * hs[p]) * NOTIONAL))
        for p in losers:
            recs.append((ts, (-fwd[p] - 2 * hs[p]) * NOTIONAL))
    return pd.DataFrame(recs, columns=["exit", "pnl"])


# --- (2) キャリー(トレード単位 → 月次PnL) -----------------------------
def carry_trades(close: pd.DataFrame, *, hold, k) -> pd.DataFrame:
    """高金利ロング/低金利ショート。金利差(carry_annual)で順位付け。各脚$10k。
    価格変動PnL + 保有ぶんのキャリー受払 − 往復spread。先読みなし(t→t+hold)。"""
    names = list(close.columns)
    mp = close.mean()
    hs = _half_spread(names, mp)
    recs = []
    for t in range(0, len(close) - hold, hold):
        ts_entry = close.index[t]
        year = ts_entry.year
        car = pd.Series({p: carry.carry_annual(p, year) for p in names}).sort_values()
        lows = car.index[:k]                           # 低金利 → ショート
        highs = car.index[-k:]                          # 高金利 → ロング
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        ts = close.index[t + hold]
        days = (ts - ts_entry).total_seconds() / 86400.0
        for p in highs:
            price_pnl = fwd[p] - 2 * hs[p]
            carry_pnl = (carry.carry_annual(p, year) / 100.0) * (days / 365.0)
            recs.append((ts, (price_pnl + carry_pnl) * NOTIONAL))
        for p in lows:
            price_pnl = -fwd[p] - 2 * hs[p]
            carry_pnl = -(carry.carry_annual(p, year) / 100.0) * (days / 365.0)
            recs.append((ts, (price_pnl + carry_pnl) * NOTIONAL))
    return pd.DataFrame(recs, columns=["exit", "pnl"])


def monthly_from_trades(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    m = pd.DatetimeIndex(trades["exit"]).to_period("M")
    return trades.assign(m=m).groupby("m")["pnl"].sum()


def yearly_total(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    y = pd.DatetimeIndex(trades["exit"]).year
    return trades.assign(y=y).groupby("y")["pnl"].sum()


def fmt_score(sc: dict) -> str:
    return (f"in_bleed={sc['mean_in_bleed']:+.1f}  normal={sc['mean_normal']:+.1f}  "
            f"edge={sc['hedge_edge']:+.1f}  win={sc['winrate_in_bleed']:.0%}  "
            f"IS={sc['mean_in_bleed_IS']:+.1f}  OOS={sc['mean_in_bleed_OOS']:+.1f}  "
            f"standalone_total={sc['total_all']:+.0f}")


def main():
    tf = "H4"
    uni.register_cross_spreads(3.0)
    close = universe_close(tf)
    print(f"universe={close.shape[1]}  tf={tf}  bars={len(close)}  "
          f"{close.index[0].date()}..{close.index[-1].date()}")

    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)
    print(f"失血窓: {int(mask.sum())}/{len(mask)} ヶ月\n")

    cols = ["lookback", "hold", "k", "standalone", "mean_in_bleed", "mean_normal",
            "hedge_edge", "winrate_in_bleed", "mean_in_bleed_IS", "mean_in_bleed_OOS", "persist"]

    # ========== (1) クロスセクション・モメンタム ==========
    print("=" * 100)
    print("(1) クロスセクション・モメンタム(勝ち組ロング/負け組ショート = xsec_meanrev の鏡像)")
    print("=" * 100)
    mom_grid = []
    for lb in (3, 6, 9, 12, 18):
        for hold in (6, 12, 18, 24):
            for k in (2, 3, 4):
                tr = xsec_momentum_trades(close, lookback=lb, hold=hold, k=k)
                mser = monthly_from_trades(tr)
                if mser.empty:
                    continue
                sc = bl.conditional_score(mser, mask)
                mom_grid.append({"lookback": lb, "hold": hold, "k": k,
                                 "standalone": tr["pnl"].sum(), **sc, "_m": mser})
    mdf = pd.DataFrame(mom_grid)
    mdf["persist"] = (mdf["mean_in_bleed_IS"] > 0) & (mdf["mean_in_bleed_OOS"] > 0)
    mdf_sorted = mdf.sort_values("mean_in_bleed", ascending=False)
    print("失血窓貢献 上位(mean_in_bleed 降順):")
    print(mdf_sorted[cols].head(15).round(1).to_string(index=False))
    print(f"\n失血窓貢献プラスの設定: {(mdf['mean_in_bleed']>0).mean():.0%}  "
          f"持続(IS&OOS両プラス): {mdf['persist'].mean():.0%}")

    persist_mom = mdf[mdf["persist"]].sort_values("mean_in_bleed", ascending=False)
    if len(persist_mom):
        bm = persist_mom.iloc[0]; tag = "持続"
    else:
        bm = mdf_sorted.iloc[0]; tag = "(持続なし→窓内最大)"
    print(f"\n--- 最良モメンタム {tag}: lookback={int(bm['lookback'])} "
          f"hold={int(bm['hold'])} k={int(bm['k'])} ---")
    print(f"  {fmt_score(bl.conditional_score(bm['_m'], mask))}")
    trm = xsec_momentum_trades(close, lookback=int(bm['lookback']), hold=int(bm['hold']), k=int(bm['k']))
    print("  年次PnL(standalone):")
    print("  " + yearly_total(trm).round(0).to_string().replace("\n", "\n  "))

    # ========== (2) キャリー ==========
    print("\n" + "=" * 100)
    print("(2) キャリー(高金利ロング/低金利ショート, 内蔵近似金利)")
    print("=" * 100)
    car_grid = []
    for hold in (6, 12, 18, 24, 30, 42):
        for k in (2, 3, 4, 5):
            tr = carry_trades(close, hold=hold, k=k)
            mser = monthly_from_trades(tr)
            if mser.empty:
                continue
            sc = bl.conditional_score(mser, mask)
            car_grid.append({"lookback": np.nan, "hold": hold, "k": k,
                             "standalone": tr["pnl"].sum(), **sc, "_m": mser})
    cdf = pd.DataFrame(car_grid)
    cdf["persist"] = (cdf["mean_in_bleed_IS"] > 0) & (cdf["mean_in_bleed_OOS"] > 0)
    cdf_sorted = cdf.sort_values("mean_in_bleed", ascending=False)
    ccols = ["hold", "k", "standalone", "mean_in_bleed", "mean_normal", "hedge_edge",
             "winrate_in_bleed", "mean_in_bleed_IS", "mean_in_bleed_OOS", "persist"]
    print("失血窓貢献 上位(mean_in_bleed 降順):")
    print(cdf_sorted[ccols].head(15).round(1).to_string(index=False))
    print(f"\n失血窓貢献プラスの設定: {(cdf['mean_in_bleed']>0).mean():.0%}  "
          f"持続(IS&OOS両プラス): {cdf['persist'].mean():.0%}")

    persist_car = cdf[cdf["persist"]].sort_values("mean_in_bleed", ascending=False)
    if len(persist_car):
        bc = persist_car.iloc[0]; tag = "持続"
    else:
        bc = cdf_sorted.iloc[0]; tag = "(持続なし→窓内最大)"
    print(f"\n--- 最良キャリー {tag}: hold={int(bc['hold'])} k={int(bc['k'])} ---")
    print(f"  {fmt_score(bl.conditional_score(bc['_m'], mask))}")
    trc = carry_trades(close, hold=int(bc['hold']), k=int(bc['k']))
    print("  年次PnL(standalone):")
    print("  " + yearly_total(trc).round(0).to_string().replace("\n", "\n  "))

    # ========== まとめ ==========
    print("\n" + "=" * 100)
    print("まとめ: 最良候補の失血窓貢献(チャンピオンの保険として)")
    print("=" * 100)
    sm = bl.conditional_score(bm['_m'], mask)
    sc2 = bl.conditional_score(bc['_m'], mask)
    print(f"モメンタム: lookback={int(bm['lookback'])} hold={int(bm['hold'])} k={int(bm['k'])}")
    print(f"  {fmt_score(sm)}  persist={bool(bm['persist'])}")
    print(f"キャリー  : hold={int(bc['hold'])} k={int(bc['k'])}")
    print(f"  {fmt_score(sc2)}  persist={bool(bc['persist'])}")

    return {"mom": (bm, sm), "carry": (bc, sc2), "mask": mask, "close": close}


if __name__ == "__main__":
    main()
