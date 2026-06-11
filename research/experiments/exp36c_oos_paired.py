"""exp36c: exp36 最終確認 — robust-IS較正→OOSペア比較 + w0.5 較正不連続の監査。

(1) cushion mp15 g0.05 w0.5 の robust 較正 k が3シードで完全一致(10.6707)
    = DDゲート(dd_mtm<-gate)の自己フィードバックで p95(k) が不連続になり境界に張り付いた疑い。
    較正点±εの p95(seed0/1/2)を実測し、達成 p95 が 20% 以下(=制約充足)かを監査。
(2) OOS維持の最終判定: champion mp11 と cushion mp15 g0.05 {w0.5,w1.0}(+USD複合)を
    **IS期間で robust(p95=20%, seed0)較正 → OOS素検証**でペア比較(運用想定は robust 較正のため)。

実行: PYTHONPATH=. uv run python research/experiments/exp36c_oos_paired.py
出力: research/outputs/exp36c_oos_paired.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import mm_lab as mm  # noqa: E402
from tail_protocol import boot_dd, cagr_of, calibrate_robust_seeded, max_dd  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from exp36_cushion_joint import (  # noqa: E402
    OVL_PARAMS, OOS_START, build_both, make_sizing_factory, usd_factor_er, usd_gate_mult,
)

OUT = ROOT / "research" / "outputs" / "exp36c_oos_paired.json"


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    import strategies.adx_trend as adx
    ovl = mm.build_pool_for(adx, OVL_PARAMS, tf="H4", side="short",
                            tag="adx_trend_30_100_14_20_short")
    both, fbar, keysrc = build_both(pool, ovl)
    er_f = usd_factor_er(40)
    gm = usd_gate_mult(pool, er_f, 0.35, 0.5)
    res = {}

    # --- (1) w0.5 較正不連続の監査 --------------------------------------------
    mk_w05 = make_sizing_factory(fbar, keysrc, 0.5, 0.05, 15)
    fn = lambda k: mm.simulate(both, closes, mk_w05(k), max_pos=15)[0]  # noqa: E731
    k_star = 10.670732421875
    audit = {}
    for kk in (k_star * 0.98, k_star, k_star * 1.02):
        eq = fn(kk)
        ps = {f"seed{sd}": float(boot_dd(eq, n_boot=600, seed=sd)["p95"]) for sd in (0, 1, 2)}
        audit[f"k={kk:.3f}"] = {**ps, "dd": max_dd(eq), "cagr": cagr_of(eq)}
        print(f"  k={kk:6.3f} p95 s0={ps['seed0']:+.1%} s1={ps['seed1']:+.1%} "
              f"s2={ps['seed2']:+.1%} | DD={max_dd(eq):+.1%} CAGR={cagr_of(eq):+.2%}", flush=True)
    res["w05_audit"] = audit

    # --- (2) robust-IS較正 → OOSペア比較 --------------------------------------
    print("\n=== robust(p95=20%, seed0) IS較正 → OOS素検証 ===", flush=True)
    is_both = both[both["entry"] < OOS_START].reset_index(drop=True)
    oos_both = both[both["entry"] >= OOS_START].reset_index(drop=True)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]

    configs = {
        "champ mp11": (champion_sizing(pool, max_pos=11), is_pool, oos_pool, 11),
        "cushion mp15 g0.05 w0.5": (mk_w05, is_both, oos_both, 15),
        "cushion mp15 g0.05 w1.0": (
            make_sizing_factory(fbar, keysrc, 1.0, 0.05, 15), is_both, oos_both, 15),
        "cushion mp15 g0.05 w1.0 +usd th0.35 g0.5": (
            make_sizing_factory(fbar, keysrc, 1.0, 0.05, 15, gatemult=gm), is_both, oos_both, 15),
    }
    res["oos_robust_is"] = {}
    for lab, (mk, ip, op, mp) in configs.items():
        fn_is = (lambda mk=mk, ip=ip, mp=mp: (
            lambda k: mm.simulate(ip, is_cl, mk(k), max_pos=mp)[0]))()
        k_is = calibrate_robust_seeded(fn_is, target=0.20, n_boot=600, seed=0)
        eqo, _, _ = mm.simulate(op, oos_cl, mk(k_is), max_pos=mp)
        bso = boot_dd(eqo, n_boot=600, seed=0)
        row = {"k_is_rob": k_is, "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo),
               "oos_p95": float(bso["p95"])}
        res["oos_robust_is"][lab] = row
        print(f"  {lab:44s} k_is={k_is:5.2f} OOS CAGR={row['oos_cagr']:+7.2%} "
              f"DD={row['oos_dd']:+6.1%} p95={row['oos_p95']:+6.1%}", flush=True)

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
