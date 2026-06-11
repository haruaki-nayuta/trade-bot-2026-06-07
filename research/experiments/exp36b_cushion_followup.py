"""exp36b: exp36 のフォローアップ — 純クッション(USDゲート無し)の確定計測と高原性検証。

exp36 の発見: gate=0.05 クッション × 高mp が robust を底上げ(mp15 で seed0 +17.1〜17.3%)。
USDゲート複合はさらに +0.7〜0.9pp 上に出るが、th 0.30↔0.35 で 2pp 振れる(exp30 同様ノイズ含み)。
本フォローアップで判定を固める:
  A. gate 軸の高原性: mp15 × w{0.5,1.0} × gate{0.03,0.065}(0.05 が1点突出でないか)
  B. 純クッション上位の stage2(seeds 1,2 → mean3): mp15 g0.05 w0.5 / w1.0, mp13 g0.05 w1.0
  C. 純クッション mp15 g0.05 w0.5 / w1.0 の IS較正→OOS素検証(overlay OOS PnL 分離)
  D. ヘッドライン構成の検証用 boot p95(n_boot=1500, seed0)と robust k での年次リターン

実行: PYTHONPATH=. uv run python research/experiments/exp36b_cushion_followup.py
出力: research/outputs/exp36b_followup.json
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

import mm_lab as mm  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd, yearly_returns,
)

# exp36 の実装を再利用
sys.path.insert(0, str(ROOT / "research" / "experiments"))
from exp36_cushion_joint import (  # noqa: E402
    OVL_PARAMS, build_both, is_oos, make_sizing_factory, stage1, usd_factor_er, usd_gate_mult,
)

OUT = ROOT / "research" / "outputs" / "exp36b_followup.json"


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    import strategies.adx_trend as adx
    ovl = mm.build_pool_for(adx, OVL_PARAMS, tf="H4", side="short",
                            tag="adx_trend_30_100_14_20_short")
    both, fbar, keysrc = build_both(pool, ovl)
    res = {"gate_plateau": [], "stage2_pure": [], "oos_pure": [], "verify": {}}

    def fn_of(w, gate, mp):
        mk = make_sizing_factory(fbar, keysrc, w, gate, mp)
        return mk, (lambda k: mm.simulate(both, closes, mk(k), max_pos=mp)[0])

    # --- A. gate 軸の高原性 ---------------------------------------------------
    print("=== A. gate 高原性(mp15) ===", flush=True)
    for w in (0.5, 1.0):
        for gate in (0.03, 0.065):
            _, fn = fn_of(w, gate, 15)
            res["gate_plateau"].append(
                stage1(f"cushion mp15 g{gate} w{w}", fn,
                       {"mp": 15, "gate": gate, "w": w, "th": None, "g": None}))

    # --- B. 純クッション上位の stage2 -----------------------------------------
    print("\n=== B. 純クッション stage2(seeds 1,2) ===", flush=True)
    for (mp, w) in ((15, 0.5), (15, 1.0), (13, 1.0)):
        _, fn = fn_of(w, 0.05, mp)
        row = {"label": f"cushion mp{mp} g0.05 w{w}"}
        cs = []
        for sd in (0, 1, 2):
            k_r = calibrate_robust_seeded(fn, target=0.20, n_boot=600, seed=sd)
            c = cagr_of(fn(k_r))
            row[f"rob_k{sd}"] = k_r
            row[f"rob_cagr{sd}"] = c
            cs.append(c)
        row["rob_mean3"] = float(np.mean(cs))
        res["stage2_pure"].append(row)
        print(f"  {row['label']:30s} s0={cs[0]:+.2%} s1={cs[1]:+.2%} s2={cs[2]:+.2%} "
              f"mean3={row['rob_mean3']:+.2%}", flush=True)

    # --- C. 純クッションの IS→OOS ---------------------------------------------
    print("\n=== C. 純クッション IS較正→OOS素検証 ===", flush=True)
    for (mp, w) in ((15, 0.5), (15, 1.0)):
        mk, _ = fn_of(w, 0.05, mp)
        o = is_oos(both, closes, mk, mp, keysrc=keysrc)
        o["label"] = f"cushion mp{mp} g0.05 w{w}"
        res["oos_pure"].append(o)
        print(f"  {o['label']:30s} k_is={o['k_is']:5.2f} OOS CAGR={o['oos_cagr']:+7.2%} "
              f"DD={o['oos_dd']:+6.1%} ovlN={o['ovl_oos_n']} ovlPnL={o['ovl_oos_pnl_frac']:+.2%}",
              flush=True)

    # --- D. ヘッドラインの検証 boot(n=1500)+ robust k 年次 --------------------
    print("\n=== D. 検証 boot(n=1500, seed0)と robust k 年次 ===", flush=True)
    er_f = usd_factor_er(40)
    heads = {
        "cushion mp15 g0.05 w0.5": fn_of(0.5, 0.05, 15)[1],
        "cushion mp15 g0.05 w1.0": fn_of(1.0, 0.05, 15)[1],
    }
    gm = usd_gate_mult(pool, er_f, 0.35, 0.5)
    mk_usd = make_sizing_factory(fbar, keysrc, 1.0, 0.05, 15, gatemult=gm)
    heads["cushion mp15 g0.05 w1.0 +usd th0.35 g0.5"] = (
        lambda k: mm.simulate(both, closes, mk_usd(k), max_pos=15)[0])
    for lab, fn in heads.items():
        k_e = calibrate_empirical(fn, target=0.20, hi=24.0)
        eq_e = fn(k_e)
        bs = boot_dd(eq_e, n_boot=1500, seed=0)
        k_r = calibrate_robust_seeded(fn, target=0.20, n_boot=600, seed=0)
        eq_r = fn(k_r)
        yr = yearly_returns(eq_r)
        res["verify"][lab] = {
            "emp_k": k_e, "emp_cagr": cagr_of(eq_e), "emp_p95_1500": bs["p95"],
            "emp_p99_1500": bs["p99"],
            "rob_k0": k_r, "rob_cagr0": cagr_of(eq_r), "rob_dd": max_dd(eq_r),
            "rob_worst_year": float(yr.min()),
            "rob_yearly": {str(y): float(v) for y, v in yr.items()},
        }
        print(f"  {lab:44s} emp p95(1500)={bs['p95']:+.1%} | rob k={k_r:.2f} "
              f"CAGR={cagr_of(eq_r):+.2%} worst_yr={yr.min():+.1%}", flush=True)

    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(f"\nsaved -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
