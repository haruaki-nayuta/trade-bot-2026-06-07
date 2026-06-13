"""検証2: ~1日tsmomモメンタムのグロスエッジの広がり(breadth)とplateau。

H1で lookback in {12,24,36,48,72}、M30で {24,48,96} を全7メジャーでGROSS測定。
問い: グロス正が広いlookback帯×多ペアで高原を作るか、単一セルか。

GROSS = コスト0 (SPREADS_PIPS全0, COMMISSION_FRACTION=0)。size_mode=value で時系列条件一定。
"""

from __future__ import annotations

import numpy as np

import fxlab.config as C

# --- GROSS化: コストを退避してから0に ---
_ORIG_SPREADS = dict(C.SPREADS_PIPS)
_ORIG_COMM = C.COMMISSION_FRACTION
C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
C.COMMISSION_FRACTION = 0.0

from fxlab import load, run, metrics  # noqa: E402
from strategies.tsmom import generate_signals  # noqa: E402

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

CONFIGS = [
    ("H1", [12, 24, 36, 48, 72]),
    ("M30", [24, 48, 96]),
]


def sharpe_of(pair, tf, lookback):
    data = load(pair, tf)
    pf = run(pair, tf, generate_signals, {"lookback": lookback, "band": 0.0},
             data=data, size_mode="value", side="both")
    m = metrics(pf)
    return float(m["sharpe"]), int(m["num_trades"])


def main():
    results = {}  # (tf, lookback) -> {pair: (sharpe, ntr)}
    for tf, lbs in CONFIGS:
        for lb in lbs:
            cell = {}
            for pair in PAIRS:
                s, n = sharpe_of(pair, tf, lb)
                cell[pair] = (s, n)
            results[(tf, lb)] = cell

    # --- per-cell 表 ---
    print("=" * 90)
    print("GROSS Sharpe  (tf, lookback) x pair")
    print("=" * 90)
    header = "tf   lb   " + " ".join(f"{p:>8}" for p in PAIRS) + "   mean   pos/7"
    print(header)
    for (tf, lb), cell in results.items():
        sh = [cell[p][0] for p in PAIRS]
        mean = float(np.mean(sh))
        npos = sum(1 for x in sh if x > 0)
        row = f"{tf:<4} {lb:<4} " + " ".join(f"{cell[p][0]:>8.3f}" for p in PAIRS)
        row += f"   {mean:>6.3f}  {npos}/7"
        print(row)

    print()
    print("num_trades per cell (sanity):")
    for (tf, lb), cell in results.items():
        nt = [cell[p][1] for p in PAIRS]
        print(f"{tf:<4} {lb:<4} " + " ".join(f"{cell[p][1]:>8d}" for p in PAIRS))

    # --- H1 lb24 全7ペア平均(報告用 gross_sharpe_raw) ---
    h1_24 = results[("H1", 24)]
    h1_24_mean = float(np.mean([h1_24[p][0] for p in PAIRS]))
    h1_24_pos = sum(1 for p in PAIRS if h1_24[p][0] > 0)
    print()
    print(f">>> H1 lb24 per-pair gross Sharpe: " +
          ", ".join(f"{p}={h1_24[p][0]:.3f}" for p in PAIRS))
    print(f">>> H1 lb24 mean gross Sharpe = {h1_24_mean:.4f}, positive {h1_24_pos}/7")

    # --- plateau判定: 各セルの pos/7 と mean ---
    print()
    print(">>> Plateau summary (cells with mean>0 AND pos>=5/7):")
    plateau_cells = []
    for (tf, lb), cell in results.items():
        sh = [cell[p][0] for p in PAIRS]
        mean = float(np.mean(sh))
        npos = sum(1 for x in sh if x > 0)
        if mean > 0 and npos >= 5:
            plateau_cells.append((tf, lb, mean, npos))
            print(f"   {tf} lb{lb}: mean={mean:.3f} pos={npos}/7")
    if not plateau_cells:
        print("   (none)")

    # --- broad判定主軸: H1 lb24 が>=5/7か ---
    broad = h1_24_pos >= 5
    print()
    print(f">>> BROAD (H1 lb24 gross-positive >=5/7) = {broad}")

    # 広い帯での頑健性: H1の中央帯(24,36,48)が全て>=5/7なら強いplateau
    band_cells = [results[("H1", lb)] for lb in (24, 36, 48)]
    band_pos = [sum(1 for p in PAIRS if c[p][0] > 0) for c in band_cells]
    print(f">>> H1 mid-band (lb24/36/48) pos/7 = {band_pos}")


if __name__ == "__main__":
    main()
    # 復元
    C.SPREADS_PIPS = _ORIG_SPREADS
    C.COMMISSION_FRACTION = _ORIG_COMM
