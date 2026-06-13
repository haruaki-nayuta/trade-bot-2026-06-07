"""再構築B: IS/OOS + plateau (th, lookback) スイープ。

GROSS と NET を別プロセスで測るため、CLI 引数 mode=net|gross を取る。
出力は機械可読 (RESULT_ 行) にして親で集約。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/hopeful-pike-e2e515")

from fxlab import load, run, metrics
import fxlab.config as C
from research.experiments.exp_trendB_riskadj import generate_signals

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "D1"
IS_RANGE = ("2016-01-01", "2021-12-31")
OOS_RANGE = ("2022-01-01", "2026-12-31")


def measure_avg(params, data_slice):
    sh, tr, pos = [], [], 0
    for p in PAIRS:
        d = load(p, TF)
        if data_slice:
            d = d.loc[data_slice[0]:data_slice[1]]
        pf = run(p, TF, generate_signals, params, data=d, size_mode="value", side="both")
        m = metrics(pf).iloc[0]
        sh.append(float(m["sharpe"]))
        tr.append(float(m["total_return"]))
        if float(m["total_return"]) > 0:
            pos += 1
    return float(np.nanmean(sh)), float(np.nanmean(tr)), pos


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "net"
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
        C.COMMISSION_FRACTION = 0.0

    BASE = {"lookback": 252, "th": 1.0, "rebalance": 5}

    # IS / OOS
    is_sh, is_tr, is_pos = measure_avg(BASE, IS_RANGE)
    oos_sh, oos_tr, oos_pos = measure_avg(BASE, OOS_RANGE)
    print(f"RESULT[{mode}] IS  sharpe={is_sh:.4f} tr={is_tr:.4f} pos={is_pos}")
    print(f"RESULT[{mode}] OOS sharpe={oos_sh:.4f} tr={oos_tr:.4f} pos={oos_pos}")

    # plateau: th x lookback (フル期間)
    print(f"--- plateau [{mode}] (full period, 7pair avg) ---")
    for lb in [126, 252]:
        for th in [0.5, 1.0, 1.5]:
            sh, tr, pos = measure_avg({"lookback": lb, "th": th, "rebalance": 5}, None)
            print(f"RESULT[{mode}] PLATEAU lb={lb} th={th} sharpe={sh:.4f} tr={tr:.4f} pos={pos}")
