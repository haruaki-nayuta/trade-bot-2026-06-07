"""イテレーション3 実験: クロスセクション・モメンタム(7銘柄の相対強弱ロングショート)。

時系列モメンタム(各銘柄を単独で順張り)とは別系統の、資産横断で頑健とされるエッジ。
毎リバランスで 7 ペアを直近モメンタムで順位付けし、上位 k をロング・下位 k をショート。
市場全体の方向に依存しにくい(ドル全面高/全面安を相殺)ので年次が安定しやすい仮説を検証。

リターンベースの簡易バックテスト(往復スプレッドをターンオーバーに計上)。
PF/年次プラス率を目標基準で評価する。  実行: uv run python exp03.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab.data import available_pairs, load

pd.set_option("display.width", 200)


def build_close(tf: str) -> pd.DataFrame:
    pairs = available_pairs()
    cols = {p: load(p, tf)["close"] for p in pairs}
    return pd.DataFrame(cols).dropna()


def half_spread_frac(pair: str) -> float:
    return config.spread_pips(pair) * config.pip_size(pair) / 2.0


def backtest_xsec(tf="D1", lookback=63, hold=5, k=2):
    close = build_close(tf)
    pairs = list(close.columns)
    rets = close.pct_change()                      # 1バー単純リターン
    mom = close.pct_change(lookback)               # 順位付け用モメンタム

    # 半スプレッド(価格比)。建て/手仕舞いで往復、ポジション転換時に発生。
    hs = pd.Series({p: half_spread_frac(p) / close[p].mean() for p in pairs})

    trade_rets = []   # 各保有(=1トレード)の損益率
    trade_year = []
    idxs = range(lookback + 1, len(close) - hold, hold)
    for t in idxs:
        m = mom.iloc[t]
        if m.isna().any():
            continue
        order = m.sort_values()
        longs = order.index[-k:]
        shorts = order.index[:k]
        # 次の hold バンドのリターン(t→t+hold)。look-ahead 無し(t時点の順位で t以降を取る)。
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        for p in longs:
            r = fwd[p] - 2 * hs[p]                  # ロング: 順方向 - 往復コスト
            trade_rets.append(r); trade_year.append(close.index[t + hold].year)
        for p in shorts:
            r = -fwd[p] - 2 * hs[p]                 # ショート: 逆方向 - 往復コスト
            trade_rets.append(r); trade_year.append(close.index[t + hold].year)

    df = pd.DataFrame({"ret": trade_rets, "year": trade_year})
    rows = {}
    for year, g in df.groupby("year"):
        pos = g["ret"][g["ret"] > 0].sum()
        neg = -g["ret"][g["ret"] < 0].sum()
        rows[int(year)] = {
            "trades": len(g),
            "profit_factor": round(pos / neg, 2) if neg > 0 else float("inf"),
            "ret_sum%": round(g["ret"].sum() * 100, 1),
            "positive": g["ret"].sum() > 0,
        }
    out = pd.DataFrame(rows).T
    return out, df


def main():
    for tf, lb, hold, k in [("D1", 63, 5, 2), ("D1", 126, 10, 2), ("D1", 21, 5, 3), ("H4", 120, 30, 2)]:
        out, df = backtest_xsec(tf, lb, hold, k)
        pf = out["profit_factor"].replace(np.inf, np.nan)
        pos = (out["ret_sum%"] > 0).mean()
        print(f"\n=== XSEC mom  tf={tf} lookback={lb} hold={hold} k={k} ===")
        print(out.to_string())
        print(f"  → プラス年率 {pos:.0%} / PF中央値 {pf.median():.2f} / PF最小 {pf.min():.2f} "
              f"/ 年平均トレード {int(out['trades'].mean())} / 通算 {df['ret'].sum()*100:.0f}%")


if __name__ == "__main__":
    main()
