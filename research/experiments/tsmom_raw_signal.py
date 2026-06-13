"""seed数値の出所確定: 純tsmom exposure の bar-level Sharpe (GROSS, コストなし)。

position_t = sign(close_t/close_{t-lb} - 1)  (確定バーのみ, 先読みなし)
pnl_{t+1}  = position_t * (close_{t+1}/close_t - 1)
これのbar-Sharpe(年率)を全7ペアで測る。seedの +0.51/+0.52 がこれかを確認。
さらにUTC20-23除外でも測り、ロールオーバーアーティファクト署名を点検。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
BARS_PER_YEAR = {"M15": 96 * 252, "M30": 48 * 252, "H1": 24 * 252}


def sig_sharpe(close, lookback, hour=None, exclude_hours=None):
    mom = close / close.shift(lookback) - 1.0
    pos = np.sign(mom)                      # 確定バーt
    fwd = close.shift(-1) / close - 1.0     # t->t+1 リターン
    pnl = pos * fwd
    pnl = pnl.dropna()
    if exclude_hours is not None:
        h = pnl.index.hour
        mask = ~pd.Series(h, index=pnl.index).isin(exclude_hours)
        pnl = pnl[mask.values]
    mu, sd = pnl.mean(), pnl.std()
    return (mu / sd) if sd > 0 else 0.0


def main():
    for tf, lbs in [("H1", [12, 24, 36, 48, 72]), ("M30", [24, 48, 96]), ("M15", [96])]:
        bpy = BARS_PER_YEAR[tf]
        for lb in lbs:
            raw, clean = [], []
            per = {}
            for pair in PAIRS:
                c = load(pair, tf)["close"]
                sr = sig_sharpe(c, lb) * np.sqrt(bpy)
                sc = sig_sharpe(c, lb, exclude_hours=range(20, 24)) * np.sqrt(bpy)
                raw.append(sr)
                clean.append(sc)
                per[pair] = (sr, sc)
            mean_raw = float(np.mean(raw))
            mean_clean = float(np.mean(clean))
            pos_raw = sum(1 for x in raw if x > 0)
            pos_clean = sum(1 for x in clean if x > 0)
            print(f"{tf} lb{lb:<3} RAW mean={mean_raw:+.3f} pos={pos_raw}/7 | "
                  f"CLEAN(no20-23) mean={mean_clean:+.3f} pos={pos_clean}/7")
            if (tf, lb) in [("H1", 24), ("M30", 48), ("M30", 96)]:
                print("        per-pair RAW: " +
                      ", ".join(f"{p}={per[p][0]:+.3f}" for p in PAIRS))
                print("        per-pair CLEAN:" +
                      ", ".join(f"{p}={per[p][1]:+.3f}" for p in PAIRS))


if __name__ == "__main__":
    main()
