"""イテレーション19 実験: クロスセクション平均回帰(contrarian)= チャンピオンと別アプローチ。

仮説: メジャー/クロスFXでは「クロスセクション・モメンタムは net マイナス」(exp03で確認)。
ならばその反対=「直近の負け組をロング・勝ち組をショートし、横断平均への収束を取る」
クロスセクション平均回帰は net プラスのはず。チャンピオン(各銘柄の自己正規化コンフルエンス)
とは機構が全く異なる(銘柄間ランキング・スケジュールリバランス・マーケットニュートラル寄り)。

ドル建てで公平比較: 各脚 $10k notional(チャンピオンの value サイジングと同じ規模)。
往復スプレッドを建て/手仕舞いで計上。年次でプラス率・PF・総PnLを出す。

実行: uv run python exp19.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import universe as uni

pd.set_option("display.width", 220)

NOTIONAL = 10_000.0  # 1脚あたりの建玉(チャンピオンの value=10k に合わせる)


def build_universe_close(tf: str, instruments: list[str]) -> pd.DataFrame:
    cols = {name: uni.instrument_close(name, tf) for name in instruments}
    return pd.DataFrame(cols).dropna()


def half_spread_frac(name: str, mean_price: float) -> float:
    """半スプレッド(価格比)。クロスは register 済みの SPREADS_PIPS を使う。"""
    return config.spread_pips(name) * config.pip_size(name) / 2.0 / mean_price


def backtest_xs_meanrev(close: pd.DataFrame, lookback: int, hold: int, k: int,
                        direction: str = "reversion") -> pd.DataFrame:
    names = list(close.columns)
    mom = close.pct_change(lookback)
    mean_price = close.mean()
    hs = {p: half_spread_frac(p, mean_price[p]) for p in names}

    rows = []  # (year, pnl_usd, is_win)
    for t in range(lookback + 1, len(close) - hold, hold):
        m = mom.iloc[t]
        if m.isna().any():
            continue
        order = m.sort_values()
        losers = order.index[:k]     # 直近の負け組
        winners = order.index[-k:]   # 直近の勝ち組
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        year = close.index[t + hold].year
        if direction == "reversion":
            longs, shorts = losers, winners   # 負け組ロング・勝ち組ショート(収束狙い)
        else:
            longs, shorts = winners, losers   # モメンタム(比較用)
        for p in longs:
            r = fwd[p] - 2 * hs[p]
            rows.append((year, r * NOTIONAL))
        for p in shorts:
            r = -fwd[p] - 2 * hs[p]
            rows.append((year, r * NOTIONAL))

    df = pd.DataFrame(rows, columns=["year", "pnl"])
    out = {}
    for year, g in df.groupby("year"):
        pos = g["pnl"][g["pnl"] > 0].sum()
        neg = -g["pnl"][g["pnl"] < 0].sum()
        out[int(year)] = {
            "trades": len(g),
            "profit_factor": round(pos / neg, 2) if neg > 0 else float("inf"),
            "pnl": round(g["pnl"].sum(), 0),
        }
    res = pd.DataFrame(out).T
    for c in res.columns:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    tf = "H4"
    close = build_universe_close(tf, instruments)
    print(f"universe={len(instruments)} 銘柄  tf={tf}  bars={len(close)}  "
          f"{close.index[0].date()}..{close.index[-1].date()}\n")

    grid = [
        (6, 6), (6, 3), (12, 6), (12, 12), (3, 3),
        (24, 12), (24, 24), (12, 3), (18, 6), (6, 12),
    ]
    print("=== クロスセクション平均回帰(contrarian, 各脚$10k) ===")
    summary = []
    for lb, hold in grid:
        for k in (2, 3, 4):
            res = backtest_xs_meanrev(close, lb, hold, k, "reversion")
            if res.empty:
                continue
            pf = res["profit_factor"].replace(np.inf, np.nan)
            pos_rate = (res["pnl"] > 0).mean()
            total = res["pnl"].sum()
            summary.append({
                "lookback": lb, "hold": hold, "k": k,
                "total_pnl": round(total, 0),
                "pf_median": round(pf.median(), 2),
                "pf_min": round(pf.min(), 2),
                "pos_yr_rate": round(pos_rate, 2),
                "avg_trades": int(res["trades"].mean()),
            })
    s = pd.DataFrame(summary).sort_values("total_pnl", ascending=False)
    print(s.to_string(index=False))

    # 最良設定の年次内訳 + モメンタム版(反対方向)との対比
    best = s.iloc[0]
    lb, hold, k = int(best["lookback"]), int(best["hold"]), int(best["k"])
    print(f"\n=== 最良 reversion 設定 lookback={lb} hold={hold} k={k} の年次 ===")
    print(backtest_xs_meanrev(close, lb, hold, k, "reversion").to_string())
    print(f"\n=== 同設定の momentum(反対方向, 比較用)年次 ===")
    print(backtest_xs_meanrev(close, lb, hold, k, "momentum").to_string())


if __name__ == "__main__":
    main()
