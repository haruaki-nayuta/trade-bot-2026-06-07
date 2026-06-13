"""TF x コストのトレードオフ最適点を探す。

~1日モメンタム(tsmom.py 忠実実装)を H1(lb24) / H2相当 / H4(lb6=1日) で、
GROSS / NET(通常スプレッド) / NET(半スプレッド=ECN指値近似) を7メジャーで測り、
「グロスエッジを保ちつつコストを最小化する最適TF」を探す。

- 7ペア平均 と 低スプレッド3ペア(EURUSD/USDJPY/GBPUSD)平均 を両方出す。
- ノートレードバンド(band)で取引数を減らした版も測り、turnover削減が
  NETを正に押し上げるかを確認(コスト最小化の本命レバー)。
- 全数値は run()/metrics() の実測。捏造なし。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals

PAIRS = list(C.PAIRS)                       # 7メジャー
LOWSPREAD3 = ["EURUSD", "USDJPY", "GBPUSD"]  # 実弾はこの3本に絞る想定

# TF -> 1日相当の lookback。M30=48本, H1=24本, H4=6本 が「約1日」。
# 中間として H1 lb12(半日)/ H4 lb12(2日) も比較に入れる。
CELLS = [
    ("M30", 48, "1day"),
    ("H1", 24, "1day"),
    ("H4", 6, "1day"),
    ("H4", 12, "2day"),  # H4で取引数をさらに絞った版(比較用)
]

# ノートレードバンド(モメンタム閾値)。0以外でフリップ頻度=取引数を減らす。
BANDS = [0.0, 0.002, 0.005]

ORIG_SPREADS = dict(C.SPREADS_PIPS)
ORIG_COMM = C.COMMISSION_FRACTION


def set_cost(mode: str):
    """GROSS / FULL / HALF のコスト体系をグローバルに設定。"""
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in ORIG_SPREADS}
        C.COMMISSION_FRACTION = 0.0
    elif mode == "full":
        C.SPREADS_PIPS = dict(ORIG_SPREADS)
        C.COMMISSION_FRACTION = ORIG_COMM
    elif mode == "half":
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in ORIG_SPREADS.items()}
        C.COMMISSION_FRACTION = ORIG_COMM
    else:
        raise ValueError(mode)


def measure(pair, tf, lookback, band, mode):
    set_cost(mode)
    data = load(pair, tf)
    pf = run(pair, tf, generate_signals, {"lookback": lookback, "band": band},
             data=data, size_mode="value", side="both")
    m = metrics(pf)
    col = m.index[0]
    return {
        "sharpe": float(m.loc[col, "sharpe"]),
        "total_return": float(m.loc[col, "total_return"]),
        "num_trades": int(m.loc[col, "num_trades"]),
        "profit_factor": float(m.loc[col, "profit_factor"]),
    }


def agg(rows, pairs):
    """対象 pairs の平均 sharpe / total_return と pos数(>0のペア数)。"""
    sub = [r for r in rows if r["pair"] in pairs]
    sh = np.array([r["sharpe"] for r in sub], dtype=float)
    tr = np.array([r["total_return"] for r in sub], dtype=float)
    nt = np.array([r["num_trades"] for r in sub], dtype=float)
    return {
        "mean_sharpe": float(np.nanmean(sh)),
        "median_sharpe": float(np.nanmedian(sh)),
        "pos": int(np.sum(sh > 0)),
        "n": len(sub),
        "mean_ret": float(np.nanmean(tr)),
        "mean_trades": float(np.nanmean(nt)),
    }


def main():
    base_trades = {}  # (tf,lb): band=0 の7ペア平均取引数(turnover基準)
    print("=" * 110)
    print(f"{'cell':<14}{'band':>6}{'mode':>7} | "
          f"{'7p mSh':>8}{'7p med':>8}{'7p pos':>7}{'7p ret%':>8}{'7p trd':>7} | "
          f"{'3p mSh':>8}{'3p pos':>7}{'3p ret%':>8}{'3p trd':>7}")
    print("=" * 110)

    summary = []  # 集計行を貯める

    for tf, lb, label in CELLS:
        for band in BANDS:
            cellrows = {}
            for mode in ("gross", "full", "half"):
                rows = []
                for pair in PAIRS:
                    r = measure(pair, tf, lb, band, mode)
                    r["pair"] = pair
                    rows.append(r)
                cellrows[mode] = rows
                a7 = agg(rows, PAIRS)
                a3 = agg(rows, LOWSPREAD3)
                cellname = f"{tf}lb{lb}"
                print(f"{cellname:<14}{band:>6.3f}{mode:>7} | "
                      f"{a7['mean_sharpe']:>8.3f}{a7['median_sharpe']:>8.3f}"
                      f"{a7['pos']:>7d}{a7['mean_ret']*100:>8.1f}{a7['mean_trades']:>7.0f} | "
                      f"{a3['mean_sharpe']:>8.3f}{a3['pos']:>7d}"
                      f"{a3['mean_ret']*100:>8.1f}{a3['mean_trades']:>7.0f}")
                summary.append({
                    "cell": cellname, "tf": tf, "lb": lb, "band": band, "mode": mode,
                    "p7_mSh": a7["mean_sharpe"], "p7_medSh": a7["median_sharpe"],
                    "p7_pos": a7["pos"], "p7_trd": a7["mean_trades"],
                    "p3_mSh": a3["mean_sharpe"], "p3_pos": a3["pos"], "p3_trd": a3["mean_trades"],
                })
                if mode == "gross" and band == 0.0:
                    base_trades[(tf, lb)] = a7["mean_trades"]
            print("-" * 110)

    # ---- per-pair gross detail at the canonical cells (band=0) -------------
    print("\n=== per-pair GROSS sharpe (band=0) ===")
    set_cost("gross")
    for tf, lb, _ in CELLS:
        line = f"{tf}lb{lb:<3} "
        for pair in PAIRS:
            data = load(pair, tf)
            pf = run(pair, tf, generate_signals, {"lookback": lb, "band": 0.0},
                     data=data, size_mode="value", side="both")
            sh = float(metrics(pf).iloc[0]["sharpe"])
            line += f" {pair[:6]}={sh:+.2f}"
        print(line)

    # ---- best cell selection: turnover-reduced focus ----------------------
    # NET(full)で最良の 7p / 3p セルを探す
    df = pd.DataFrame(summary)
    print("\n=== BEST cells ===")
    for mode in ("full", "half"):
        sub = df[df["mode"] == mode]
        b7 = sub.loc[sub["p7_mSh"].idxmax()]
        b3 = sub.loc[sub["p3_mSh"].idxmax()]
        print(f"[{mode}] best 7p: {b7['cell']} band={b7['band']:.3f} "
              f"mSh={b7['p7_mSh']:.3f} pos={int(b7['p7_pos'])}/7 trd={b7['p7_trd']:.0f}")
        print(f"[{mode}] best 3p: {b3['cell']} band={b3['band']:.3f} "
              f"mSh={b3['p3_mSh']:.3f} pos={int(b3['p3_pos'])}/3 trd={b3['p3_trd']:.0f}")

    # ---- turnover reduction at H4 (the cost-minimal TF) -------------------
    print("\n=== turnover vs base (gross band=0 baseline 7p mean trades) ===")
    for tf, lb, _ in CELLS:
        base = base_trades.get((tf, lb))
        for band in BANDS:
            sub = df[(df.tf == tf) & (df.lb == lb) & (df.band == band) & (df["mode"] == "gross")]
            if len(sub):
                t = sub.iloc[0]["p7_trd"]
                print(f"{tf}lb{lb} band={band:.3f}: 7p trd={t:.0f} "
                      f"({100*t/base:.0f}% of band0)")

    # restore
    C.SPREADS_PIPS = dict(ORIG_SPREADS)
    C.COMMISSION_FRACTION = ORIG_COMM


if __name__ == "__main__":
    main()
