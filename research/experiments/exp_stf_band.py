"""H1 ~1日tsmom + ノートレードバンド: 取引数削減でNET黒字化するか。

tsmom.generate_signals(data, lookback, band) を使い、band を上げて微小モメンタムを
無視する(取引数削減)。GROSS / NET(通常スプレッド) / NET(半スプレッド)を
7メジャー & 低スプレッド3ペア(EUR/JPY/GBP)で測る。plateau(band×lookback)も確認。

実行: uv run python -m research.experiments.exp_stf_band
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
LOWSPREAD3 = ["EURUSD", "USDJPY", "GBPUSD"]
TF = "H1"

# 元のスプレッドを退避
ORIG_SPREADS = copy.deepcopy(C.SPREADS_PIPS)
ORIG_COMM = C.COMMISSION_FRACTION


def set_cost(mode: str):
    """mode: gross / full / half"""
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in ORIG_SPREADS}
        C.COMMISSION_FRACTION = 0.0
    elif mode == "full":
        C.SPREADS_PIPS = copy.deepcopy(ORIG_SPREADS)
        C.COMMISSION_FRACTION = ORIG_COMM
    elif mode == "half":
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in ORIG_SPREADS.items()}
        C.COMMISSION_FRACTION = ORIG_COMM
    else:
        raise ValueError(mode)


def f(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def eval_cell(lookback: int, band: float):
    """各ペアで gross/full/half の Sharpe と取引数を測る。"""
    rows = {}
    # データは一度ロードしてキャッシュ。コストモードは run() 内のconfig参照で切替
    datacache = {p: load(p, TF) for p in PAIRS}
    for mode in ("gross", "full", "half"):
        set_cost(mode)
        for p in PAIRS:
            data = datacache[p]
            pf = run(p, TF, generate_signals,
                     {"lookback": lookback, "band": band},
                     data=data, size_mode="value", side="both")
            m = metrics(pf)
            rows.setdefault(p, {})[mode + "_sharpe"] = f(m["sharpe"])
            rows.setdefault(p, {})[mode + "_ret"] = f(m["total_return"])
            rows[p]["num_trades"] = f(m["num_trades"])
    return rows


def summarize(rows, pairs):
    out = {}
    for mode in ("gross", "full", "half"):
        vals = [rows[p][mode + "_sharpe"] for p in pairs]
        out[mode + "_mean"] = float(np.nanmean(vals))
        out[mode + "_pos"] = int(sum(1 for v in vals if v > 0))
    out["trades_mean"] = float(np.nanmean([rows[p]["num_trades"] for p in pairs]))
    return out


def main():
    lookbacks = [24]
    bands = [0.0, 0.003, 0.005, 0.01]
    # plateau確認のため lookback も広げる
    lb_grid = [12, 24, 36]
    band_grid = [0.0, 0.003, 0.005, 0.01]

    print("=" * 90)
    print(f"H1 tsmom band sweep | lookback=24 | 7-pair & lowspread3(EUR/JPY/GBP)")
    print("=" * 90)

    base_trades = None
    results = {}
    for band in bands:
        rows = eval_cell(24, band)
        s7 = summarize(rows, PAIRS)
        s3 = summarize(rows, LOWSPREAD3)
        results[band] = (s7, s3, rows)
        if band == 0.0:
            base_trades = s7["trades_mean"]
        red = (1 - s7["trades_mean"] / base_trades) * 100 if base_trades else 0.0
        print(f"\n--- band={band} | trades/pair(7avg)={s7['trades_mean']:.0f} "
              f"(reduction vs band0: {red:.1f}%) ---")
        print(f"  7-pair  GROSS {s7['gross_mean']:+.3f}(pos{s7['gross_pos']}/7) "
              f"NET-full {s7['full_mean']:+.3f}(pos{s7['full_pos']}/7) "
              f"NET-half {s7['half_mean']:+.3f}(pos{s7['half_pos']}/7)")
        print(f"  low3    GROSS {s3['gross_mean']:+.3f}(pos{s3['gross_pos']}/3) "
              f"NET-full {s3['full_mean']:+.3f}(pos{s3['full_pos']}/3) "
              f"NET-half {s3['half_mean']:+.3f}(pos{s3['half_pos']}/3)")
        # per-pair full sharpe
        per = " ".join(f"{p}:{rows[p]['full_sharpe']:+.2f}" for p in PAIRS)
        print(f"    per-pair NET-full sharpe: {per}")

    # plateau grid (NET-full 7pair mean & low3 mean)
    print("\n" + "=" * 90)
    print("PLATEAU grid: NET-full-spread mean Sharpe (7-pair / low3)")
    print("=" * 90)
    header = "lb\\band " + " ".join(f"{b:>8}" for b in band_grid)
    print(header)
    plateau_grid_7 = {}
    plateau_grid_3 = {}
    plateau_grid_half3 = {}
    for lb in lb_grid:
        cells7 = []
        cells3 = []
        cellsh3 = []
        for band in band_grid:
            if lb == 24 and band in results:
                s7, s3, _rows = results[band]
            else:
                rows = eval_cell(lb, band)
                s7 = summarize(rows, PAIRS)
                s3 = summarize(rows, LOWSPREAD3)
            cells7.append(s7["full_mean"])
            cells3.append(s3["full_mean"])
            cellsh3.append(s3["half_mean"])
        plateau_grid_7[lb] = cells7
        plateau_grid_3[lb] = cells3
        plateau_grid_half3[lb] = cellsh3
        row7 = " ".join(f"{v:>+8.3f}" for v in cells7)
        row3 = " ".join(f"{v:>+8.3f}" for v in cells3)
        print(f"lb={lb:<4} 7p {row7}")
        print(f"         3p {row3}")

    print("\nPLATEAU grid: NET-half-spread mean Sharpe (low3)")
    print(header)
    for lb in lb_grid:
        rowh3 = " ".join(f"{v:>+8.3f}" for v in plateau_grid_half3[lb])
        print(f"lb={lb:<4} h3 {rowh3}")

    # 復元
    C.SPREADS_PIPS = copy.deepcopy(ORIG_SPREADS)
    C.COMMISSION_FRACTION = ORIG_COMM


if __name__ == "__main__":
    main()
