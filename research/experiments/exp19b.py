"""イテレーション19b: クロスセクション平均回帰の精錬。

exp19 で contrarian 方向が net プラス(最良17k, ただし64%プラス年)と確認。
ここでは品質を上げて 18.1k(チャンピオン)超え+頑健性向上を狙う:
  (A) vol正規化スコア: score_i = ret_i(lookback) / vol_i  → 横断で demean
  (B) しきい値ゲート  : |score| が score_z 超の脚だけ建てる(常に top/bottom k を建てない)
  (C) 長め hold も探索(lb=6 周辺)

各脚 $10k。往復スプレッド計上。実行: uv run python exp19b.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import universe as uni

pd.set_option("display.width", 220)
NOTIONAL = 10_000.0


def build_universe_close(tf, instruments):
    return pd.DataFrame({n: uni.instrument_close(n, tf) for n in instruments}).dropna()


def half_spread_frac(name, mean_price):
    return config.spread_pips(name) * config.pip_size(name) / 2.0 / mean_price


def backtest(close, lookback, hold, score_z, vol_win=50, demean=True, max_legs=4):
    names = list(close.columns)
    rets = close.pct_change()
    mom = close.pct_change(lookback)
    vol = rets.rolling(vol_win).std()
    mean_price = close.mean()
    hs = {p: half_spread_frac(p, mean_price[p]) for p in names}

    rows = []
    for t in range(max(lookback, vol_win) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]            # vol正規化
        if score.isna().any():
            continue
        if demean:
            score = score - score.mean()             # 横断平均からの乖離
        # しきい値超の銘柄のみ。負け組(score<<0)ロング・勝ち組(score>>0)ショート
        s = score.sort_values()
        cand_long = s[s < -score_z].index[:max_legs]      # 最も負け
        cand_short = s[s > score_z].index[-max_legs:]     # 最も勝ち
        if len(cand_long) == 0 and len(cand_short) == 0:
            continue
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        year = close.index[t + hold].year
        for p in cand_long:
            rows.append((year, (fwd[p] - 2 * hs[p]) * NOTIONAL))
        for p in cand_short:
            rows.append((year, (-fwd[p] - 2 * hs[p]) * NOTIONAL))

    df = pd.DataFrame(rows, columns=["year", "pnl"])
    if df.empty:
        return pd.DataFrame()
    out = {}
    for year, g in df.groupby("year"):
        pos = g["pnl"][g["pnl"] > 0].sum(); neg = -g["pnl"][g["pnl"] < 0].sum()
        out[int(year)] = {"trades": len(g),
                          "profit_factor": round(pos / neg, 2) if neg > 0 else float("inf"),
                          "pnl": round(g["pnl"].sum(), 0)}
    res = pd.DataFrame(out).T
    for c in res.columns:
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    close = build_universe_close("H4", instruments)
    print(f"universe={len(instruments)}  bars={len(close)}\n")

    summary = []
    for lb in (3, 6, 9, 12):
        for hold in (6, 12, 18, 24):
            for score_z in (0.0, 0.5, 1.0, 1.5):
                res = backtest(close, lb, hold, score_z)
                if res.empty:
                    continue
                pf = res["profit_factor"].replace(np.inf, np.nan)
                summary.append({
                    "lb": lb, "hold": hold, "z": score_z,
                    "total_pnl": round(res["pnl"].sum(), 0),
                    "pf_med": round(pf.median(), 2),
                    "pf_min": round(pf.min(), 2),
                    "pos_yr": round((res["pnl"] > 0).mean(), 2),
                    "trd/yr": int(res["trades"].mean()),
                })
    s = pd.DataFrame(summary).sort_values("total_pnl", ascending=False)
    print("=== 精錬版スキャン(vol正規化+demean+しきい値, 各脚$10k) top20 ===")
    print(s.head(20).to_string(index=False))
    print("\n=== 毎年プラス(pos_yr=1.0)のものを利益順 ===")
    allpos = s[s["pos_yr"] >= 1.0].sort_values("total_pnl", ascending=False)
    print(allpos.head(15).to_string(index=False) if not allpos.empty else "  なし")

    # 利益最大の年次内訳
    best = s.iloc[0]
    print(f"\n=== 利益最大 lb={int(best['lb'])} hold={int(best['hold'])} z={best['z']} の年次 ===")
    print(backtest(close, int(best["lb"]), int(best["hold"]), float(best["z"])).to_string())


if __name__ == "__main__":
    main()
