"""exp57: 第3ラウンド採用候補スタックの決定的監査 — M1粒度 + 署名 + h20単年 + Pプラトー。

対象(exp56): d1+h20+P4.5+mp10 = robust 5シード +20.25%(+1.62pp)。構成要素の残関門:
  - mp9/mp10(用量再審): mp11 は H4 +0.99pp でも M1 谷比 1.161 → 掛け目 x0.861 で逆転死
    (exp44)。中間用量が d1 幾何でゲート(比率≤1.15)を通り、掛け目込み実効CAGRで
    mp8 を上回るかが採否を決める。
  - 署名: mp 系は emp較正時 p95 が0.4〜1.0pp悪化(exp56 単一シード)。ブートシード 0-2 の
    ペア測定で「2/3 以上のシードで >0.5pp 悪化」かを判定(exp47 §6 方式)。
    ※ mp 系の最終判定は M1 直接測定(p95_M1近似=-0.20×谷比)を優先する(exp44 の前例:
      mp11 の H4 証拠はクリーン扱いで、棄却根拠は専ら M1 ゲートだった)。
  - h20: 増分(対 d1)の年次分解。最良年残存率 <50% なら単年依存でスタックから外す。
  - P4.5: OOS-emp の生CAGR割れは較正の機械効果(OOS DD -18.79→-17.82%、MAR は rob/emp
    両経路で改善=過剰適合の証拠なし)。IS-argmax は d1 プールで 4.5(d0 時代 3.5)。
    残るは近傍平滑性: P∈{4.25, 4.75} が 4.5 と滑らかに繋がるか(4.75 は測定のみ・採用禁止)。

事前登録の採用規則:
  {mp10系スタック} のうち M1 谷比 ≤1.15 かつ 実効CAGR(掛け目込み, rob_m5)が
  d1mp8 実効(≈+17.7%)+0.5pp 以上、かつ h20/P4.5 が各監査を通る最大構成を採用。
  どれも通らなければ「20%未達」を正直に報告する。

実行: PYTHONPATH=. uv run python research/experiments/exp57_stack_m1audit.py
出力: research/outputs/exp57_audit.csv / exp57_result.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import mm_lab as mm  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded,
)
from exp47_entry_delay import reconstruct, delayed_pool  # noqa: E402
from exp56_round3_protocol import make_sizing  # noqa: E402
import exp52_d1_m1audit as a52  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
EXP56_JSON = OUT_DIR / "exp56_result.json"


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def eq_fn(pool, closes, mk, max_pos):
    c = {}

    def f(k):
        kk = round(float(k), 10)
        if kk not in c:
            c[kk] = mm.simulate(pool, closes, mk(kk), max_pos=max_pos)[0]
        return c[kk]
    return f


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool0 = mm.build_pool()
    closes = mm.load_closes()
    rc = reconstruct(pool0)
    mod1, kept1, ret_new1, ex1 = delayed_pool(pool0, rc, 1)
    mod_h, kept_h, ret_h, ex_h = delayed_pool(pool0, rc, 1, skip_h20=True)
    e56 = json.loads(EXP56_JSON.read_text())["results"]

    # 構成: (label, pool, p, max_pos, exp56キー)
    CFGS = [
        ("d1mp8", mod1, 4.0, 8, "base_d1"),
        ("d1mp9", mod1, 4.0, 9, "mp9"),
        ("d1mp10", mod1, 4.0, 10, "mp10"),
        ("h20mp10", mod_h, 4.0, 10, "h20+mp10"),
        ("h20P45mp10", mod_h, 4.5, 10, "h20+P4.5+mp10"),
    ]

    sec("0. 較正再計算と exp56 照合(emp + rob_s0)+ rob_m5 k")
    cals = {}
    for label, pl, p, mp, key in CFGS:
        mk = make_sizing(pl, p=p, max_pos=mp)
        f = eq_fn(pl, closes, mk, mp)
        k_emp = calibrate_empirical(f, 0.20)
        k_r0 = calibrate_robust_seeded(f, 0.20, seed=0)
        c_emp, c_r0 = cagr_of(f(k_emp)), cagr_of(f(k_r0))
        ref = e56[key]
        m = (abs(k_emp - ref["emp_k"]) < 0.02 and abs(c_emp - ref["emp_cagr"]) < 5e-4
             and abs(c_r0 - ref["rob"]["0"]["cagr"]) < 5e-4)
        k_m5 = float(np.mean([ref["rob"][str(s)]["k"] for s in range(5)]))
        cals[label] = {"mk": mk, "pool": pl, "mp": mp, "k_emp": k_emp, "c_emp": c_emp,
                       "k_r0": k_r0, "c_r0": c_r0, "k_m5": k_m5, "eq": f,
                       "rob_m5_cagr": ref["rob_cagr_mean"], "match": m}
        print(f"  {label:12s} emp k={k_emp:6.3f}/{c_emp:+.2%}  rob_s0 k={k_r0:6.3f}/{c_r0:+.2%}"
              f"  rob_m5 k={k_m5:6.3f}  exp56一致: {m}   [{time.time()-t0:.0f}s]")
    if not all(c["match"] for c in cals.values()):
        print("!! exp56 と較正不一致 — 中断")
        return 1

    sec("1. 署名ブートシード監査(emp_k 固定, n_boot=1500, seeds 0-2, 対 d1mp8)")
    base_eq = cals["d1mp8"]["eq"](cals["d1mp8"]["k_emp"])
    p95_base = {sd: boot_dd(base_eq, n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)}
    print("  d1mp8 : " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95_base.items()))
    sig_audit = {}
    for label in ("d1mp9", "d1mp10", "h20mp10", "h20P45mp10"):
        c = cals[label]
        eq = c["eq"](c["k_emp"])
        p95s = {sd: boot_dd(eq, n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)}
        emp_up = c["c_emp"] > cals["d1mp8"]["c_emp"]
        sigs = {sd: emp_up and (abs(p95s[sd]) > abs(p95_base[sd]) + 0.005) for sd in p95s}
        sig_audit[label] = {"p95": p95s, "n_sig": sum(sigs.values())}
        print(f"  {label:12s} " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95s.items()) +
              f"   署名: {sum(sigs.values())}/3 " +
              " ".join("X" if v else "-" for v in sigs.values()))

    sec("2. h20 増分(対 d1)の年次分解 — 単年依存監査")
    # kept_h ⊆ kept1。両方 kept: ret_h - ret_new1 / d1のみ: -ret_new1
    diff_tr = np.where(kept_h & kept1, ret_h - ret_new1,
                       np.where(kept1 & ~kept_h, -ret_new1, 0.0))
    yr = pd.Series(diff_tr).groupby(pool0["exit"].dt.year).sum()
    total = float(diff_tr.sum())
    best_y = int(yr.idxmax())
    keep_share = float(yr.drop(best_y).sum() / total) if total > 0 else np.nan
    n_moved = int((np.abs(diff_tr) > 0).sum())
    print(f"  影響トレード {n_moved}件  増分合計 {total:+.4f}")
    print("  年次: " + "  ".join(f"{int(y)}:{v:+.4f}" for y, v in yr.items() if abs(v) > 1e-4))
    print(f"  最良年 {best_y}({float(yr[best_y]):+.4f})  除外後 {float(yr.drop(best_y).sum()):+.4f}"
          f"  残存率 {keep_share:.0%}  -> {'PASS' if keep_share >= 0.5 else 'FAIL(単年依存)'}")
    h20_audit = {"total": total, "best_year": best_y, "keep_share": keep_share,
                 "yearly": {int(y): float(v) for y, v in yr.items()}}

    sec("3. P プラトー(4.25/4.75, rob seed0)— 4.5 の1点スパイク検査")
    p_plateau = {}
    for ctx_label, pl, mp in [("d1mp8", mod1, 8), ("h20mp10", mod_h, 10)]:
        row = {}
        for p in (4.0, 4.25, 4.5, 4.75):
            mk = make_sizing(pl, p=p, max_pos=mp)
            f = eq_fn(pl, closes, mk, mp)
            k = calibrate_robust_seeded(f, 0.20, seed=0)
            row[p] = cagr_of(f(k))
        p_plateau[ctx_label] = row
        print(f"  [{ctx_label}] " + "  ".join(f"P{p}:{v:+.2%}" for p, v in row.items()))

    sec("4. M1粒度監査(rob_m5 k, 谷比ゲート1.15 + 掛け目込み実効CAGR)")
    grid_idx = pd.DatetimeIndex(load_m1("EURUSD").index.tz_localize(None))
    print(f"M1 grid: {len(grid_idx):,} bars")
    audits = {}
    for label, pl, p, mp, key in CFGS:
        c = cals[label]
        a52.MAX_POS = mp  # m1_audit_one / simulate_with_log が参照するモジュール定数
        audits[label] = a52.m1_audit_one(label, pl, closes, c["mk"], c["k_m5"], grid_idx)
        a = audits[label]
        print(f"      谷比={a['ratio']:.3f} ({'PASS' if a['ratio'] <= 1.15 else 'FAIL'})"
              f"  掛け目 x{a['haircut']:.3f}  実効CAGR={a['cagr_adj']:+.2%}"
              f"  p95_M1近似={a['p95_m1_approx']:+.1%}  [{time.time()-t0:.0f}s]")
    a52.MAX_POS = 8

    sec("5. 判定表(実効CAGR = 掛け目込み, rob_m5 基準)")
    base_eff = audits["d1mp8"]["cagr_adj"]
    rows = []
    for label, pl, p, mp, key in CFGS:
        a = audits[label]
        c = cals[label]
        rows.append({
            "cfg": label, "rob_m5_h4": c["rob_m5_cagr"], "ratio_m1": a["ratio"],
            "gate_115": a["ratio"] <= 1.15, "haircut": a["haircut"],
            "eff_cagr": a["cagr_adj"], "eff_gain_pp": (a["cagr_adj"] - base_eff) * 100,
            "p95_m1": a["p95_m1_approx"],
            "n_sig": sig_audit.get(label, {}).get("n_sig"),
            "trough_m1": a["trough_m1"], "skipped": a["skipped"],
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "exp57_audit.csv", index=False)
    payload = {
        "cals": {k: {kk: vv for kk, vv in v.items() if kk not in ("mk", "pool", "eq")}
                 for k, v in cals.items()},
        "sig_audit": {k: {"p95": {str(s): float(x) for s, x in v["p95"].items()},
                          "n_sig": v["n_sig"]} for k, v in sig_audit.items()},
        "h20_audit": h20_audit,
        "p_plateau": {k: {str(p): float(v) for p, v in row.items()}
                      for k, row in p_plateau.items()},
        "m1_audits": {k: {kk: vv for kk, vv in v.items() if kk != "episodes"}
                      for k, v in audits.items()},
        "episodes": {k: v["episodes"] for k, v in audits.items()},
    }
    (OUT_DIR / "exp57_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp57_audit.csv'} / exp57_result.json\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
