"""best NET-positive 3pair cells の頑健性チェック。

exp_stf_tftradeoff の結果:
 - 3p(EUR/JPY/GBP) NET(full) で正が残るのは M30lb48 band=0.002 (+0.166, 2/3) など band版。
 - 3p NET(half) は H1lb24 band=0 (+0.290, 3/3) が最良だが band=0=turnover最大。
ここでは候補セルの per-pair NET 内訳 と 年次安定性(robust性) を見る。
plateau判定: band帯 {0, 0.002, 0.005} で 3p NET が滑らかに正を保つか。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals

LOWSPREAD3 = ["EURUSD", "USDJPY", "GBPUSD"]
ORIG_SPREADS = dict(C.SPREADS_PIPS)
ORIG_COMM = C.COMMISSION_FRACTION


def set_cost(mode):
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in ORIG_SPREADS}; C.COMMISSION_FRACTION = 0.0
    elif mode == "full":
        C.SPREADS_PIPS = dict(ORIG_SPREADS); C.COMMISSION_FRACTION = ORIG_COMM
    elif mode == "half":
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in ORIG_SPREADS.items()}; C.COMMISSION_FRACTION = ORIG_COMM


# 候補セル: (tf, lb)
CANDS = [("M30", 48), ("H1", 24)]
BANDS = [0.0, 0.001, 0.002, 0.003, 0.005]


def per_pair_net(tf, lb, band, mode):
    set_cost(mode)
    out = {}
    for pair in LOWSPREAD3:
        data = load(pair, tf)
        pf = run(pair, tf, generate_signals, {"lookback": lb, "band": band},
                 data=data, size_mode="value", side="both")
        m = metrics(pf).iloc[0]
        out[pair] = (float(m["sharpe"]), float(m["total_return"]), int(m["num_trades"]))
    return out


def yearly_sharpe(tf, lb, band, mode, pair):
    """年次のグロス/NET sharpe を測り、プラス年の割合で頑健性を見る。"""
    set_cost(mode)
    data = load(pair, tf)
    years = sorted(set(data.index.year))
    res = {}
    for y in years:
        sl = data[data.index.year == y]
        if len(sl) < lb + 50:
            continue
        pf = run(pair, tf, generate_signals, {"lookback": lb, "band": band},
                 data=sl, size_mode="value", side="both")
        res[y] = float(metrics(pf).iloc[0]["sharpe"])
    return res


def main():
    print("=" * 100)
    print("PLATEAU check: 3pair(EUR/JPY/GBP) NET across band, candidate cells")
    print("=" * 100)
    for tf, lb in CANDS:
        for mode in ("gross", "full", "half"):
            print(f"\n--- {tf}lb{lb}  [{mode}] ---")
            for band in BANDS:
                pp = per_pair_net(tf, lb, band, mode)
                shs = [v[0] for v in pp.values()]
                pos = sum(1 for s in shs if s > 0)
                mean = float(np.nanmean(shs))
                med = float(np.nanmedian(shs))
                trd = float(np.mean([v[2] for v in pp.values()]))
                detail = " ".join(f"{p[:3]}={pp[p][0]:+.2f}" for p in LOWSPREAD3)
                print(f"  band={band:.3f}: mean={mean:+.3f} med={med:+.3f} "
                      f"pos={pos}/3 trd={trd:.0f} | {detail}")

    # ---- yearly robustness for the most promising NET-full 3p cell --------
    print("\n" + "=" * 100)
    print("YEARLY NET(full) sharpe per pair @ M30lb48 band=0.002 (best NET-full 3p)")
    print("=" * 100)
    for pair in LOWSPREAD3:
        ys = yearly_sharpe("M30", 48, 0.002, "full", pair)
        posfrac = sum(1 for v in ys.values() if v > 0) / max(len(ys), 1)
        print(f"{pair}: pos_years={posfrac:.0%} ({sum(1 for v in ys.values() if v>0)}/{len(ys)}) "
              + " ".join(f"{y}:{v:+.1f}" for y, v in ys.items()))

    # equal-weight 3pair portfolio yearly (concat returns) - approx via mean sharpe
    print("\n" + "=" * 100)
    print("YEARLY NET(half) sharpe per pair @ H1lb24 band=0 (best NET-half 3p)")
    print("=" * 100)
    for pair in LOWSPREAD3:
        ys = yearly_sharpe("H1", 24, 0.0, "half", pair)
        posfrac = sum(1 for v in ys.values() if v > 0) / max(len(ys), 1)
        print(f"{pair}: pos_years={posfrac:.0%} ({sum(1 for v in ys.values() if v>0)}/{len(ys)}) "
              + " ".join(f"{y}:{v:+.1f}" for y, v in ys.items()))

    C.SPREADS_PIPS = dict(ORIG_SPREADS); C.COMMISSION_FRACTION = ORIG_COMM


if __name__ == "__main__":
    main()
