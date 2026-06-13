"""Champion-conditional SETUP — build the trade pool with wick features.

Run the BASE champion (confluence_meanrev_v2 = confluence_meanrev_filtered.generate_signals
with v2 PARAMS) on EACH of the 7 majors at H4, size_mode="value", size_value=10000 so that
per-trade return_pct is comparable across the whole sample.

For EACH trade we look up its ENTRY (trigger) bar candle in load(pair,"H4") and compute CAUSAL
candle/wick features of that bar and the prior bars (all <= entry bar, NO look-ahead):
  clv1, up_w, lo_w, wick_diff1, body1, wick_diff3 (mean last 3 incl entry), clv3,
  range_rel20 (rng / rng.rolling(20).mean()),
  z = zscore(close,50) at entry, rsi(14) at entry, er = Kaufman ER(40) at entry, dir.

Direction-aligned helpers:
  wick_support = wick_diff1 for longs, -wick_diff1 for shorts (+ => wick supports trade dir).
  clv_support  = -clv1 for longs (close-near-low supports a long bounce, reports/12),
                  +clv1 for shorts.

Plus: entry year, is_oos flag (year >= 2022 = champion OOS split).

Concatenate all 7 majors -> research/outputs/wickbias_champion_pool.csv (one row per trade).

Run: uv run python -m research.experiments.wickbias_build_pool
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

from fxlab import load, run, trades
from strategies.confluence_meanrev_filtered import generate_signals
from strategies.confluence_meanrev_v2 import PARAMS

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
OUT = "research/outputs/wickbias_champion_pool.csv"

# feature params (match champion semantics for redundancy columns)
Z_WIN = 50      # zscore window (matches PARAMS["window"])
RSI_P = 14
ER_WIN = 40     # Kaufman ER window (matches PARAMS["er_win"])


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _efficiency_ratio(close: pd.Series, w: int) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar causal candle/wick features. Each row uses only that bar's OHLC (formed
    at its close) plus earlier bars (rolling)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).replace(0.0, np.nan)  # NaN if no range (synthetic close-only -> all NaN)
    clv1 = (c - l) / rng                # close location value: high => close near high
    up_w = (h - np.maximum(o, c)) / rng
    lo_w = (np.minimum(o, c) - l) / rng
    wick_diff1 = lo_w - up_w            # + => lower wick dominates, - => upper wick dominates
    body1 = (c - o) / rng

    feats = pd.DataFrame(
        {
            "clv1": clv1,
            "up_w": up_w,
            "lo_w": lo_w,
            "wick_diff1": wick_diff1,
            "body1": body1,
            "wick_diff3": wick_diff1.rolling(3).mean(),  # mean of last 3 incl current bar
            "clv3": clv1.rolling(3).mean(),
            "range_rel20": (h - l) / (h - l).rolling(20).mean(),
        },
        index=df.index,
    )
    # redundancy columns (champion's own signal ingredients)
    feats["z"] = _zscore(c, Z_WIN)
    feats["rsi"] = vbt.RSI.run(c, RSI_P).rsi
    feats["er"] = _efficiency_ratio(c, ER_WIN)
    return feats


def build_pair(pair: str) -> pd.DataFrame:
    df = load(pair, "H4")
    pf = run(pair, "H4", generate_signals, PARAMS, size_mode="value", size_value=10000)
    tt = trades.trade_table(pf, df)
    if tt.empty:
        return tt

    feats = candle_features(df)
    # entry bar = trigger candle (BASE v2: entry timestamp == z-trigger candle).
    # NOTE: reindex with the tz-aware Series (NOT .values, which drops tz -> all-NaN).
    f_at_entry = feats.reindex(tt["entry"])
    f_at_entry.index = tt.index

    out = pd.concat([tt.reset_index(drop=True), f_at_entry.reset_index(drop=True)], axis=1)
    out.insert(0, "pair", pair)

    is_long = out["dir"].str.lower().eq("long")
    # direction-aligned wick / clv support
    out["wick_support"] = np.where(is_long, out["wick_diff1"], -out["wick_diff1"])
    out["clv_support"] = np.where(is_long, -out["clv1"], out["clv1"])

    out["year"] = pd.to_datetime(out["entry"]).dt.year
    out["is_oos"] = out["year"] >= 2022
    return out


def main() -> None:
    parts = []
    print(f"Building champion-conditional wick pool over {len(MAJORS)} majors (H4)...\n")
    for pair in MAJORS:
        p = build_pair(pair)
        n = len(p)
        if n:
            wr = (p["return_pct"] > 0).mean() * 100
            ret = p["return_pct"].sum()
            nlong = p["dir"].str.lower().eq("long").sum()
            print(f"  {pair}: n={n:4d}  long={nlong:4d} short={n - nlong:4d}  "
                  f"win_rate={wr:5.1f}%  sum_return_pct={ret:+7.2f}")
            parts.append(p)
        else:
            print(f"  {pair}: n=0 (no trades)")

    pool = pd.concat(parts, ignore_index=True)
    pool.to_csv(OUT, index=False)

    n = len(pool)
    wr = (pool["return_pct"] > 0).mean() * 100
    ret = pool["return_pct"].sum()
    print("\n" + "=" * 70)
    print(f"WROTE: {OUT}")
    print(f"TOTAL n trades : {n}")
    print(f"win rate       : {wr:.2f}%  (return_pct > 0)")
    print(f"total net ret  : {ret:+.2f}  (sum of per-trade return_pct, %)")
    print(f"  long/short   : {pool['dir'].str.lower().eq('long').sum()} / "
          f"{pool['dir'].str.lower().eq('short').sum()}")
    print(f"  is_oos True  : {int(pool['is_oos'].sum())}  "
          f"(year>=2022) / False {int((~pool['is_oos']).sum())}")
    print("\nper-pair counts:")
    print(pool["pair"].value_counts().reindex(MAJORS).to_string())
    print("\ncolumn list:")
    print(list(pool.columns))


if __name__ == "__main__":
    main()
