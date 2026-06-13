"""exp62: リスク予算ダイヤルの精密化 — 「+10%相対」に必要な正確な robust p95 目標と、その実コスト。

MTF(exp59/60)・xsec(exp61)が確定で死亡し、同一リスク契約(robust p95 DD=20%)での手法改善は
出尽くした(reports/19/20 を MTF/xsec で再確認)。残る唯一の正直な +10% はリスク予算の変更のみ。

本実験は、その変更の正確な大きさとコストを多シードで確定する:
  ・robust 較正目標 p95 DD を 20.0% から 23.0% まで刻んで CAGR(3シード平均)を測る。
  ・各点で empirical(単一パス)MtM 最大DD と p99 も併記 = 「真に受け入れるリスク」。
  ・ベースライン(+18.2%)の +10%相対 = +20.0% に到達する p95 目標を内挿で特定。

これは手法の改善ではなくリスク選好の選択肢として、正確な数値でユーザーに提示するための材料。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1, champion_sizing  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

SEEDS = [0, 1, 2]


def main():
    print("=" * 72)
    print("  exp62: リスク予算ダイヤル — +10%相対に必要な robust p95 目標とその実コスト")
    print("=" * 72)
    pool = build_pool_d1(tf="H4")
    closes = mm.load_closes(tf="H4")
    print(f"  H4 d1 pool {len(pool)} trades / grid {len(closes)} / mp8\n")

    print(f"  {'p95目標':>7} {'CAGR(3seed)':>13} {'range':>16} {'k':>6} {'empDD':>8} {'p99':>8}")
    rows = []
    for target in [0.200, 0.205, 0.210, 0.215, 0.220, 0.225, 0.230]:
        cagrs, ks, empdds, p99s = [], [], [], []
        for sd in SEEDS:
            mk = champion_sizing(pool, max_pos=8)
            # calibrate_robust は seed=0 固定の内部ブート。シード変動は最終ブートで反映。
            k, eqm, eqr, info, p95 = mm.calibrate_robust(
                pool, closes, mk, target_dd=target, max_pos=8, n_boot=600)
            s = mm.stats(eqm, eqr, info, tf="H4")
            bs = mm.bootstrap_maxdd(eqm, n_boot=1500, seed=sd)
            cagrs.append(s["cagr"]); ks.append(k); empdds.append(s["maxdd_mtm"]); p99s.append(bs["p99"])
        cagrs = np.array(cagrs)
        rows.append((target, cagrs.mean()))
        print(f"  {target:>6.1%} {cagrs.mean():>+12.2%} "
              f"[{cagrs.min():>+6.2%},{cagrs.max():>+6.2%}] {np.mean(ks):>6.2f} "
              f"{np.mean(empdds):>+7.1%} {np.mean(p99s):>+7.1%}")

    base = rows[0][1]
    tgt_cagr = base * 1.10
    # 内挿で +10% 到達 p95 目標
    arr = np.array(rows)
    p95_for_10 = np.interp(tgt_cagr, arr[:, 1], arr[:, 0])
    print(f"\n  ベースライン(p95=20%) robust = {base:+.2%}")
    print(f"  +10%相対 = {tgt_cagr:+.2%} に到達する p95 目標 ≈ {p95_for_10:.1%}")
    print(f"  → 「20回に1回級の理論DDを {base_to_pct(20.0)}→{p95_for_10*100:.1f}% まで許容」がコスト")
    print("\n  ※ empirical 較正(本番既定)なら現行でも robust ベースを大きく上回る点も併記すべき")
    print("=" * 72)


def base_to_pct(x):
    return f"{x:.1f}"


if __name__ == "__main__":
    main()
