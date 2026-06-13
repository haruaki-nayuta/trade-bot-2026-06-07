"""補完検証: (1) tsmom を同一ハーネスで再測定し breakout との差を確認、
(2) breakout の出口を変えても(opposite-channel only / no trailing)結論が変わらないか、
(3) NET full/half を best な breakout variant でフル7ペア表示。

実行: cwd から  uv run python -m research.experiments.exp_stf_breakout2
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
_ORIG = copy.deepcopy(C.SPREADS_PIPS)
_ORIGC = C.COMMISSION_FRACTION


def set_cost(mode):
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in _ORIG}; C.COMMISSION_FRACTION = 0.0
    elif mode == "half":
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in _ORIG.items()}; C.COMMISSION_FRACTION = _ORIGC
    else:
        C.SPREADS_PIPS = copy.deepcopy(_ORIG); C.COMMISSION_FRACTION = _ORIGC


# Donchian with no-trailing exit: only exit on opposite channel breakout (pure reversal flip)
def donch_flip(data, entry=36):
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    long_e = close > upper
    short_e = close < lower
    return long_e, short_e, short_e, long_e  # exit only when opposite channel breaks (flip)


def sh(pair, sigfn, params, data, side="both"):
    pf = run(pair, "H1", sigfn, params, data=data, size_mode="value", side=side)
    return float(metrics(pf)["sharpe"].iloc[0])


def row(sigfn, params, cache, mode):
    set_cost(mode)
    d = {p: sh(p, sigfn, params, cache[p]) for p in PAIRS}
    vals = np.array([d[p] for p in PAIRS])
    return vals.mean(), int((vals > 0).sum()), np.mean([d[p] for p in LOW3]), \
        int(sum(d[p] > 0 for p in LOW3)), d


if __name__ == "__main__":
    cache = {p: load(p, "H1") for p in PAIRS}

    print("=" * 80)
    print("tsmom (signed momentum) H1 lb24 — same harness, for contrast")
    print("=" * 80)
    for mode in ["gross", "full", "half"]:
        m, pos, l3, l3p, d = row(tsmom_signals, {"lookback": 24, "band": 0.0}, cache, mode)
        print(f"  [{mode:5}] 7pair mean={m:+.3f} pos={pos}/7 | low3={l3:+.3f} pos={l3p}/3 | "
              + " ".join(f"{p}={d[p]:+.2f}" for p in PAIRS))

    print("\n" + "=" * 80)
    print("Donchian FLIP (exit only on opposite breakout, no trailing) — entry sweep")
    print("=" * 80)
    set_cost("gross")
    best = None
    for entry in [12, 20, 24, 36, 48, 72, 96]:
        m, pos, l3, l3p, d = row(donch_flip, {"entry": entry}, cache, "gross")
        print(f"  entry={entry:3} GROSS 7pair mean={m:+.3f} pos={pos}/7 | low3={l3:+.3f} pos={l3p}/3")
        if best is None or m > best[1]:
            best = ({"entry": entry}, m)
    print("  best flip entry by mean:", best[0])
    for mode in ["gross", "full", "half"]:
        m, pos, l3, l3p, d = row(donch_flip, best[0], cache, mode)
        print(f"    [{mode:5}] mean={m:+.3f} pos={pos}/7 | low3={l3:+.3f} pos={l3p}/3")

    set_cost(_ORIGC if False else "full")
    print("\nDONE")
