"""flip variant の plateau & per-pair half/full 詳細、tsmomとの相関(同一エッジか)。

実行: cwd から  uv run python -m research.experiments.exp_stf_breakout3
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals as tsmom_signals

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
LOW3 = ["EURUSD", "USDJPY", "GBPUSD"]
_ORIG = copy.deepcopy(C.SPREADS_PIPS); _ORIGC = C.COMMISSION_FRACTION

def set_cost(mode):
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in _ORIG}; C.COMMISSION_FRACTION = 0.0
    elif mode == "half":
        C.SPREADS_PIPS = {k: v*0.5 for k,v in _ORIG.items()}; C.COMMISSION_FRACTION = _ORIGC
    else:
        C.SPREADS_PIPS = copy.deepcopy(_ORIG); C.COMMISSION_FRACTION = _ORIGC

def donch_flip(data, entry=36):
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift(); lower = low.rolling(entry).min().shift()
    le = close > upper; se = close < lower
    return le, se, se, le

def sh(pair, sigfn, params, data):
    pf = run(pair, "H1", sigfn, params, data=data, size_mode="value")
    return float(metrics(pf)["sharpe"].iloc[0])

if __name__ == "__main__":
    cache = {p: load(p, "H1") for p in PAIRS}
    print("=== Donchian FLIP plateau (entry 24/30/36/42, low3 focus) ===")
    for mode in ["gross", "half", "full"]:
        set_cost(mode)
        print(f"[{mode}]")
        for entry in [24, 30, 36, 42]:
            d = {p: sh(p, donch_flip, {"entry": entry}, cache[p]) for p in PAIRS}
            v = np.array([d[p] for p in PAIRS]); l3 = np.array([d[p] for p in LOW3])
            print(f"  entry={entry}: 7mean={v.mean():+.3f} pos={(v>0).sum()}/7 | "
                  f"low3={l3.mean():+.3f} pos={(l3>0).sum()}/3 | "
                  + " ".join(f"{p}={d[p]:+.2f}" for p in LOW3))

    print("\n=== Is flip the SAME edge as tsmom? per-pair half-spread side-by-side ===")
    set_cost("half")
    print("pair      flip36   tsmom24")
    for p in PAIRS:
        f = sh(p, donch_flip, {"entry": 36}, cache[p])
        t = sh(p, tsmom_signals, {"lookback": 24, "band": 0.0}, cache[p])
        print(f"  {p}: {f:+.3f}   {t:+.3f}")

    set_cost("full")
    print("\nDONE")
