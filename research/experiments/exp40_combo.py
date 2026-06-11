"""exp40: 合成(z-power P × DDクッション mp15)のフル測定 — 2本のレバーは積み上がるか。

背景(前フェーズの2発見):
  A(exp37): champ z-power 指数 P=2.0→4.0 で robust mean3 +14.98→+17.21%(mp11)。
            p95フラット、OOSピークは P3.5-4.0。
  B(exp36): DD連動 ADXトレンド・クッション(adx_trend 30/100 short, gate0.05)× mp15 で
            robust mean3 +17.27%(w0.5)/+17.37%(w1.0)。p95 -29.4→-24〜-25%。
  本実験: 両レバー同時の 8構成グリッド(P∈{2,3,3.5,4} × w∈{0.5,1.0}, mp15, g0.05)を測り、
  加法性(超加法/加法/劣加法)と干渉を判定する。

プロトコル = 同一テール判定(reports/11):
  stage1: 8構成 empirical較正(hi=24)+boot p95(600,seed0)+robust(p95=20%)seed0
  stage2: 上位3構成に seeds0..4(5シード)→ mean5(参考 mean3=seeds0-2)
  筆頭フル検査:
    (a) empirical較正+そのkで p95(n_boot=1500,seed0) — レバ偽装署名(emp↑+p95悪化)
    (b) robust較正k(seed0)の年次リターン全表・最悪年・プラス年率
    (c) robust-IS較正(seed0,600)→OOS素検証 — champ mp11 ペア(基準 +19.48% / DD -11.1%)
    (d) ブロック長感応 block∈{21,63,126}(robust seed0)
    (e) 較正点k*±5% の p95(seeds0-2)滑らかさ(較正の崖チェック)
  相互作用: 合成改善幅(mean3, 対 +14.98%)vs 単独和(P4.0 +2.23pp + cushion w1.0 +2.39pp)

実行: PYTHONPATH=. uv run python research/experiments/exp40_combo.py
出力: research/outputs/exp40_combo.csv / .json
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

import mm_lab as mm  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd, yearly_returns,
)

pd.set_option("display.width", 260)

OUT_CSV = ROOT / "research" / "outputs" / "exp40_combo.csv"
OUT_JSON = ROOT / "research" / "outputs" / "exp40_combo.json"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OVL_PARAMS = {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}

MP = 15
GATE = 0.05
P_GRID = (2.0, 3.0, 3.5, 4.0)
W_GRID = (0.5, 1.0)

# 物差し(前フェーズ確定値)
BASE_P95 = -0.294          # mp11 empirical較正kでの boot p95
BASE_EMP_CAGR = 0.2383     # mp11 empirical CAGR
BASELINE_MEAN3 = 0.1498    # champ mp11 robust mean3
SINGLE_P4_PP = 0.0223      # exp37: P4.0 単独改善幅(mean3, 対 +14.98%)
SINGLE_CUSH_PP = 0.0239    # exp36: cushion mp15 w1.0 単独改善幅
BASE_OOS_ROB = 0.1948      # champ mp11 robust-IS較正→OOS素 CAGR(exp36c)
BASE_OOS_DD = -0.111

Z0, CLIP_LO, CLIP_HI = 2.2, 0.3, 3.0


def fz_p(p):
    """champ z-power(P パラメータ化版)。P=2.0 が mm_production._fz と同値。"""
    def f(z):
        return float(np.clip((z / Z0) ** p, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0
    return f


# --- exp36 build_both のコピー(fbar は P 別に外で計算するため返さない) ------
def build_both(pool_c: pd.DataFrame, overlay_pool: pd.DataFrame):
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    src = both["src"].to_numpy()
    keysrc = {}
    for i in range(len(both)):
        keysrc[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = src[i]
    return both, keysrc


# --- exp36 make_sizing_factory のコピー + champ 側 fz を P でパラメータ化 ----
def make_sizing_factory(fz, fbar, keysrc, w, gate, max_pos):
    """champ=z-power(fz/fbar)、overlay=dd_mtm<-gate のとき weight w で建玉。"""
    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
            if keysrc.get(key, "champ") == "champ":
                return ctx["equity_real"] * base * (fz(ctx["z"]) / fbar)
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * w
            return 0.0
        return sizing
    return make_sizing


def champ_sizing_p(fz, fbar, max_pos):
    """チャンピオン単独 × z-power P(クッション無し診断用)。"""
    def make_sizing(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (fz(ctx["z"]) / fbar)
    return make_sizing


def cached_eq(pool_any, closes, make_sizing, mp):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pool_any, closes, make_sizing(kk), max_pos=mp)[0]
        return cache[kk]
    return eq_of_k


def stage1(label: str, eq_of_k, meta: dict) -> dict:
    t0 = time.time()
    k_emp = calibrate_empirical(eq_of_k, target=0.20, hi=24.0)
    eq_e = eq_of_k(k_emp)
    bs = boot_dd(eq_e, n_boot=600, seed=0)
    yr = yearly_returns(eq_e)
    k_r0 = calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, seed=0)
    eq_r0 = eq_of_k(k_r0)
    row = {"label": label, **meta,
           "emp_k": k_emp, "emp_cagr": cagr_of(eq_e), "emp_dd": max_dd(eq_e),
           "emp_p95": bs["p95"], "worst_year_emp": float(yr.min()),
           "rob_k0": k_r0, "rob_cagr0": cagr_of(eq_r0)}
    print(f"  {label:34s} emp k={k_emp:5.2f} CAGR={row['emp_cagr']:+7.2%} "
          f"p95={bs['p95']:+6.1%} wy={row['worst_year_emp']:+6.1%} | "
          f"rob s0 k={k_r0:5.2f} CAGR={row['rob_cagr0']:+7.2%}  ({time.time()-t0:.0f}s)",
          flush=True)
    return row


def stage2_seeds(row: dict, eq_of_k, seeds=(1, 2, 3, 4)) -> dict:
    for sd in seeds:
        k_r = calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, seed=sd)
        row[f"rob_k{sd}"] = k_r
        row[f"rob_cagr{sd}"] = cagr_of(eq_of_k(k_r))
    cs = [row[f"rob_cagr{s}"] for s in (0, 1, 2, 3, 4)]
    row["rob_mean3"] = float(np.mean(cs[:3]))
    row["rob_mean5"] = float(np.mean(cs))
    print(f"  {row['label']:34s} seeds " + "/".join(f"{c:+.2%}" for c in cs)
          + f"  mean3={row['rob_mean3']:+.2%} mean5={row['rob_mean5']:+.2%}", flush=True)
    return row


def main() -> int:
    t_all = time.time()
    pool = mm.build_pool()
    closes = mm.load_closes()
    import strategies.adx_trend as adx
    ovl = mm.build_pool_for(adx, OVL_PARAMS, tf="H4", side="short",
                            tag="adx_trend_30_100_14_20_short")
    both, keysrc = build_both(pool, ovl)
    zs = pool["z_entry"].to_numpy()
    fbar_of = {p: (float(np.mean([fz_p(p)(z) for z in zs])) or 1.0) for p in P_GRID}
    print(f"champ pool {len(pool)} / overlay pool {len(ovl)} / grid {len(closes)}本", flush=True)
    print("fbar(P): " + ", ".join(f"P{p}={fbar_of[p]:.4f}" for p in P_GRID), flush=True)

    rows = []
    eqfn = {}
    mkmap = {}   # label -> (pool_any, make_sizing)

    # --- 1) 合成グリッド: P × w(mp15, gate0.05) ------------------------------
    print("\n=== 合成グリッド(P × cushion w, mp15 g0.05) stage1 ===", flush=True)
    for p in P_GRID:
        for w in W_GRID:
            mk = make_sizing_factory(fz_p(p), fbar_of[p], keysrc, w, GATE, MP)
            fn = cached_eq(both, closes, mk, MP)
            lab = f"combo P{p} w{w}"
            rows.append(stage1(lab, fn, {"P": p, "w": w, "kind": "combo"}))
            eqfn[lab] = fn
            mkmap[lab] = (both, mk)

    # --- 2) 診断: P4 単独(クッション無し)を mp15/mp11 で --------------------
    print("\n=== 診断: P4 単独(クッション無し) ===", flush=True)
    for mp_d in (15, 11):
        mk = champ_sizing_p(fz_p(4.0), fbar_of[4.0], mp_d)
        fn = cached_eq(pool, closes, mk, mp_d)
        lab = f"champ-only P4.0 mp{mp_d}"
        rows.append(stage1(lab, fn, {"P": 4.0, "w": None, "kind": "diag"}))
        eqfn[lab] = fn
        mkmap[lab] = (pool, mk)

    # --- 3) stage2: 合成グリッド上位3 に seeds1..4 -----------------------------
    df = pd.DataFrame(rows)
    cand = df[df["kind"] == "combo"].sort_values("rob_cagr0", ascending=False).head(3)
    print(f"\n=== stage2(5シード): {', '.join(cand['label'])} ===", flush=True)
    for idx in cand.index:
        rows[idx] = stage2_seeds(rows[idx], eqfn[rows[idx]["label"]])

    # --- 4) 筆頭フル検査 --------------------------------------------------------
    df = pd.DataFrame(rows)
    lead_lab = df.sort_values("rob_mean5", ascending=False, na_position="last").iloc[0]["label"]
    lead_i = int(df.index[df["label"] == lead_lab][0])
    fn = eqfn[lead_lab]
    pool_any, mk = mkmap[lead_lab]
    full = {"lead": lead_lab}
    print(f"\n=== 筆頭フル検査: {lead_lab} ===", flush=True)

    # (a) empirical較正 + p95(n_boot=1500, seed0) — レバ偽装署名
    k_emp = rows[lead_i]["emp_k"]
    eq_e = fn(k_emp)
    bs15 = boot_dd(eq_e, n_boot=1500, seed=0)
    full["a_emp_k"] = k_emp
    full["a_emp_cagr"] = cagr_of(eq_e)
    full["a_p95_1500"] = bs15["p95"]
    full["a_p99_1500"] = bs15["p99"]
    sig = (full["a_emp_cagr"] > BASE_EMP_CAGR) and (bs15["p95"] < BASE_P95 - 0.01)
    full["a_lev_disguise"] = bool(sig)
    print(f"  (a) emp k={k_emp:.2f} CAGR={full['a_emp_cagr']:+.2%} "
          f"p95(1500)={bs15['p95']:+.1%} p99={bs15['p99']:+.1%} "
          f"(基準 emp {BASE_EMP_CAGR:+.2%} / p95 {BASE_P95:+.1%}) → 署名={'あり' if sig else 'クリーン'}",
          flush=True)

    # (b) robust較正k(seed0)の年次リターン全表
    k_r0 = rows[lead_i]["rob_k0"]
    eq_r = fn(k_r0)
    yr = yearly_returns(eq_r)
    full["b_rob_k0"] = k_r0
    full["b_rob_cagr0"] = cagr_of(eq_r)
    full["b_yearly"] = {str(y): float(v) for y, v in yr.items()}
    full["b_worst_year"] = float(yr.min())
    full["b_pos_year_rate"] = float((yr > 0).mean())
    print(f"  (b) robust k0={k_r0:.2f} CAGR={full['b_rob_cagr0']:+.2%} "
          f"最悪年={full['b_worst_year']:+.1%} プラス年率={full['b_pos_year_rate']:.0%}", flush=True)
    print("      年次: " + " ".join(f"{y}:{v:+.1%}" for y, v in yr.items()), flush=True)

    # (c) robust-IS較正(seed0)→OOS素検証(champ mp11 ペア再計算)
    print("  (c) robust-IS較正→OOS素検証", flush=True)
    from mm_production import champion_sizing  # noqa: E402  (P=2.0 基準)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    full["c_oos"] = {}
    for lab2, (pa, mk2, mp2) in {
        "champ mp11 (基準)": (pool, champion_sizing(pool, max_pos=11), 11),
        lead_lab: (pool_any, mk, MP),
    }.items():
        ip = pa[pa["entry"] < OOS_START].reset_index(drop=True)
        op = pa[pa["entry"] >= OOS_START].reset_index(drop=True)
        fn_is = cached_eq(ip, is_cl, mk2, mp2)
        k_is = calibrate_robust_seeded(fn_is, target=0.20, n_boot=600, seed=0)
        eqo, _, _ = mm.simulate(op, oos_cl, mk2(k_is), max_pos=mp2)
        bso = boot_dd(eqo, n_boot=600, seed=0)
        row2 = {"k_is_rob": k_is, "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo),
                "oos_p95": float(bso["p95"])}
        full["c_oos"][lab2] = row2
        print(f"      {lab2:28s} k_is={k_is:5.2f} OOS CAGR={row2['oos_cagr']:+7.2%} "
              f"DD={row2['oos_dd']:+6.1%} p95={row2['oos_p95']:+6.1%}", flush=True)

    # (d) ブロック長感応(robust seed0)
    print("  (d) ブロック長感応(robust seed0)", flush=True)
    full["d_block"] = {}
    for blk in (21, 63, 126):
        k_b = calibrate_robust_seeded(fn, target=0.20, n_boot=600, block=blk, seed=0)
        c_b = cagr_of(fn(k_b))
        full["d_block"][str(blk)] = {"k": k_b, "cagr": c_b}
        print(f"      block={blk:4d}  k={k_b:5.2f}  CAGR={c_b:+7.2%}", flush=True)

    # (e) 較正点 k*±5% の p95 滑らかさ(seeds0-2)
    print("  (e) k*±5% の p95 滑らかさ", flush=True)
    full["e_smooth"] = {}
    for kk in (k_r0 * 0.95, k_r0, k_r0 * 1.05):
        eqx = fn(kk)
        ps = {f"seed{sd}": float(boot_dd(eqx, n_boot=600, seed=sd)["p95"]) for sd in (0, 1, 2)}
        full["e_smooth"][f"k={kk:.3f}"] = {**ps, "dd": max_dd(eqx), "cagr": cagr_of(eqx)}
        print(f"      k={kk:6.3f} p95 s0={ps['seed0']:+.1%} s1={ps['seed1']:+.1%} "
              f"s2={ps['seed2']:+.1%} | DD={max_dd(eqx):+.1%} CAGR={cagr_of(eqx):+.2%}", flush=True)

    # --- 5) 相互作用の符号 ------------------------------------------------------
    lead_mean3 = rows[lead_i]["rob_mean3"]
    combo_pp = lead_mean3 - BASELINE_MEAN3
    sum_pp = SINGLE_P4_PP + SINGLE_CUSH_PP
    tol = 0.003
    verdict = ("超加法" if combo_pp > sum_pp + tol
               else ("劣加法" if combo_pp < sum_pp - tol else "加法"))
    full["interaction"] = {"combo_pp": combo_pp, "sum_singles_pp": sum_pp, "verdict": verdict}
    print(f"\n=== 相互作用: 合成改善幅(mean3)={combo_pp*100:+.2f}pp vs 単独和={sum_pp*100:+.2f}pp "
          f"→ {verdict} ===", flush=True)

    # --- 保存 -------------------------------------------------------------------
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps({"rows": rows, "full": full}, indent=2,
                                   default=lambda x: None if pd.isna(x) else float(x)))
    print(f"\nsaved -> {OUT_CSV}\n      -> {OUT_JSON}\n total {time.time()-t_all:.0f}s", flush=True)

    print("\n=== サマリ(robust 降順) ===")
    cols = [c for c in ["label", "emp_k", "emp_cagr", "emp_p95", "rob_cagr0", "rob_mean3",
                        "rob_mean5", "worst_year_emp"] if c in out.columns]
    print(out.sort_values("rob_cagr0", ascending=False)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
