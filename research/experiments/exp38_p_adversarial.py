"""exp38: 発見A(z-power 指数 P=4.0)の敵対検証 — 逐次探索バイアスを暴く/晴らす。

スタンス: 懐疑者。P=2.0(本番確定値)→P=4.0 の改善が「調整済み関数の再調整」という
後知恵でないかを反証する。exp37 のコードは参照せず、mm_production.champion_sizing を
コピーして _fz の P だけをパラメータ化した独立再実装で測る。

検証項目:
  1) 独立再実装: P∈{2,3,4,5}×mp11 を protocol_eval(seeds 0,1,2)。exp37 の
     mean3 曲線(2.0:14.98 / 3.0:16.36 / 4.0:17.21 / 5.0:17.64 %)と ±0.2pp 一致か。
  2) IS単独選択: IS(<2022-01-01)プールのみで P∈{2.0..5.0} robust(seed0) を測り、
     「ISだけ見た研究者の P_IS」を確定 → robust-IS較正k→OOS素シミュで持続を検証。
  3) 追加シード: P=4(と P=2 ペア)を seeds {3,4} に拡張(計5シード)。
  4) 配分集中の現実性監査: robust k(3シード平均)で 1玉最大レバ / top10利益集中 /
     同時総エクスポージャ最大 / z>2.5 件数(P=2 vs P=4)。
  5) 年次安定: P4 と P2 の年次リターン全表と差分(改善の年集中チェック)。
  6) ブロック長感応: robust 較正 block∈{21,63,126}(seed0)で P4−P2 差が保つか。
  7) mp相互作用: P=4 × mp12/mp13(robust seed0)で mp11 特異性チェック。

実行: PYTHONPATH=. uv run python -u research/experiments/exp38_p_adversarial.py
出力: research/outputs/exp38_p_adversarial.json / exp38_log.txt(リダイレクト)
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
    boot_dd, cagr_of, calibrate_robust_seeded, max_dd, protocol_eval, yearly_returns,
)

pd.set_option("display.width", 220)

OUT_JSON = ROOT / "research" / "outputs" / "exp38_p_adversarial.json"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")

# --- 独立再実装: mm_production.champion_sizing のコピー(P をパラメータ化) ----
Z0, CLIP_LO, CLIP_HI = 2.2, 0.3, 3.0


def fz(z, P):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def champion_sizing_p(pool, P, max_pos):
    """mm_production.champion_sizing と同形。_fz の指数だけ P。"""
    fbar = float(np.mean([fz(z, P) for z in pool["z_entry"].to_numpy()])) or 1.0

    def make_sizing(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (fz(ctx["z"], P) / fbar)
    return make_sizing


def eq_fn(pool, closes, P, max_pos):
    mk = champion_sizing_p(pool, P, max_pos)

    def _f(k):
        eqm, _, _ = mm.simulate(pool, closes, mk(k), max_pos=max_pos)
        return eqm
    return _f


# --- 監査用シミュレータ: mm_lab.simulate の忠実コピー + 配分記録 --------------
def simulate_audit(pool, closes, sizing, *, init=10_000.0, max_pos=6, vol_win=120):
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)

    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    bars_arr = pool["bars_held"].to_numpy()

    equity = init
    peak_mtm = init
    open_pos = []
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    mtm_ret_hist = np.empty(n)
    prev_mtm = init
    conc = []
    skipped = 0
    taken = []            # 監査: (ti, alloc, mtm_at_entry)
    gross_exp = np.zeros(n)  # 監査: バー毎の総建玉/equity_mtm

    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                else:
                    still.append(p)
            open_pos = still

        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            run_ret = p["dir"] * (px / p["eprice"] - 1.0)
            unreal += p["alloc"] * run_ret
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity
        peak_mtm = max(peak_mtm, mtm)
        mtm_ret_hist[b] = (mtm / prev_mtm - 1.0) if prev_mtm > 0 else 0.0
        prev_mtm = mtm

        if b >= vol_win:
            rv = mtm_ret_hist[b - vol_win + 1:b + 1]
            recent_vol = float(np.std(rv) * np.sqrt(mm.BARS_PER_YEAR.get("H4", 1512)))
        else:
            recent_vol = float("nan")

        dd_mtm = mtm / peak_mtm - 1.0

        if b in by_entry:
            for ti in by_entry[b]:
                if len(open_pos) >= max_pos:
                    skipped += 1
                    continue
                ctx = {
                    "equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                    "dd_mtm": dd_mtm, "n_open": len(open_pos), "max_pos": max_pos,
                    "recent_vol": recent_vol, "z": float(z_arr[ti]),
                    "instr": instr_arr[ti], "ret": float(ret_arr[ti]),
                    "bars_held": int(bars_arr[ti]),
                }
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({
                    "ti": ti, "col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                    "eprice": eprice_arr[ti], "alloc": alloc,
                    "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti]),
                })
                conc.append(len(open_pos))
                taken.append((ti, alloc, mtm))

        gross_exp[b] = sum(p["alloc"] for p in open_pos) / mtm if mtm > 0 else np.nan

    eq_mtm = pd.Series(eq_mtm, index=grid)
    eq_real = pd.Series(eq_real, index=grid)
    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0,
            "avg_conc": float(np.mean(conc)) if conc else 0.0}
    return eq_mtm, eq_real, info, taken, gross_exp


def jdump(res):
    OUT_JSON.write_text(json.dumps(res, indent=2, default=float))


def main() -> int:
    t0 = time.time()
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / grid {len(closes)} bars  ({time.time()-t0:.0f}s)", flush=True)

    res = {}

    # === 1) 独立再実装 vs exp37 ============================================
    print("\n=== [1] 独立再実装: P x mp11, protocol_eval(seeds 0,1,2) ===", flush=True)
    EXP37_MEAN3 = {2.0: 0.1498, 3.0: 0.1636, 4.0: 0.1721, 5.0: 0.1764}
    res["repro"] = {}
    for P in [2.0, 3.0, 4.0, 5.0]:
        r = protocol_eval(eq_fn(pool, closes, P, 11), label=f"P={P} mp11", seeds=(0, 1, 2))
        ref = EXP37_MEAN3[P]
        diff = r["rob_cagr_mean"] - ref
        ok = abs(diff) <= 0.002
        print(f"    -> exp37 mean3 {ref:+.2%}  diff {diff*100:+.2f}pp  {'MATCH' if ok else 'MISMATCH'}",
              flush=True)
        res["repro"][f"P{P}"] = {**{k: v for k, v in r.items() if k != "rob"},
                                 "rob": {str(s): v for s, v in r["rob"].items()},
                                 "exp37_ref": ref, "diff_pp": diff * 100, "match": ok}
        jdump(res)

    # === 2) IS単独選択テスト ================================================
    print("\n=== [2] IS単独選択: ISプール+ISグリッドのみで P を選ぶ ===", flush=True)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    print(f"  IS {len(is_pool)} trades / OOS {len(oos_pool)} trades", flush=True)
    res["is_selection"] = {}
    for P in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        mk_is = champion_sizing_p(is_pool, P, 11)   # fbar も IS のみ(因果)
        fn_is = (lambda mk=mk_is: (lambda k: mm.simulate(is_pool, is_cl, mk(k), max_pos=11)[0]))()
        k_is = calibrate_robust_seeded(fn_is, target=0.20, n_boot=600, seed=0)
        eq_is = fn_is(k_is)
        eqo, _, _ = mm.simulate(oos_pool, oos_cl, mk_is(k_is), max_pos=11)
        bso = boot_dd(eqo, n_boot=600, seed=0)
        row = {"k_is": k_is, "is_cagr": cagr_of(eq_is), "is_dd": max_dd(eq_is),
               "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo), "oos_p95": float(bso["p95"])}
        res["is_selection"][f"P{P}"] = row
        print(f"  P={P:3.1f}  k_is={k_is:5.2f}  IS CAGR={row['is_cagr']:+7.2%}  "
              f"| OOS CAGR={row['oos_cagr']:+7.2%}  DD={row['oos_dd']:+6.1%}  "
              f"p95={row['oos_p95']:+6.1%}", flush=True)
        jdump(res)
    best_is = max(res["is_selection"].items(), key=lambda kv: kv[1]["is_cagr"])
    res["is_selection"]["P_IS_argmax"] = best_is[0]
    print(f"  -> ISだけ見た研究者の選択: {best_is[0]} (IS CAGR {best_is[1]['is_cagr']:+.2%})",
          flush=True)
    jdump(res)

    # === 3) 追加シード(P=4 と P=2 ペア, seeds 3,4) ==========================
    print("\n=== [3] 追加シード seeds {3,4} (full pool, mp11) ===", flush=True)
    res["extra_seeds"] = {}
    for P in [2.0, 4.0]:
        fn = eq_fn(pool, closes, P, 11)
        for sd in [3, 4]:
            k = calibrate_robust_seeded(fn, target=0.20, n_boot=600, seed=sd)
            c = cagr_of(fn(k))
            res["extra_seeds"][f"P{P}_s{sd}"] = {"k": k, "cagr": c}
            print(f"  P={P} seed{sd}: k={k:5.2f} CAGR={c:+.2%}", flush=True)
            jdump(res)

    # === 4) 配分集中の現実性監査(robust k 3シード平均で) ====================
    print("\n=== [4] 配分集中の現実性監査 (k = robust 3シード平均) ===", flush=True)
    res["concentration"] = {}
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    audit_eq = {}
    for P in [2.0, 4.0]:
        k_mean = res["repro"][f"P{P}"]["rob_k_mean"]
        mk = champion_sizing_p(pool, P, 11)
        eqm, eqr, info, taken, gross = simulate_audit(pool, closes, mk(k_mean), max_pos=11)
        audit_eq[P] = eqm
        ti_a = np.array([t[0] for t in taken])
        alloc_a = np.array([t[1] for t in taken])
        mtm_a = np.array([t[2] for t in taken])
        lev = alloc_a / mtm_a
        pnl = alloc_a * pool["ret"].to_numpy()[ti_a]
        top10 = np.sort(pnl)[-10:]
        share_net = float(top10.sum() / pnl.sum())
        share_gross = float(top10.sum() / pnl[pnl > 0].sum())
        z_taken = pool["z_entry"].to_numpy()[ti_a]
        n_z25 = int((z_taken > 2.5).sum())
        row = {
            "k_mean": k_mean, "cagr": cagr_of(eqm), "dd": max_dd(eqm),
            "max_single_lev": float(lev.max()), "p99_single_lev": float(np.percentile(lev, 99)),
            "med_single_lev": float(np.median(lev)),
            "top10_share_net": share_net, "top10_share_gross": share_gross,
            "max_gross_exposure": float(np.nanmax(gross)),
            "n_z_gt_2.5_per_year": n_z25 / years, "n_taken": len(taken),
            "skipped": info["skipped"],
        }
        res["concentration"][f"P{P}"] = row
        print(f"  P={P}: k={k_mean:.2f} CAGR={row['cagr']:+.2%} DD={row['dd']:+.1%}\n"
              f"    1玉レバ max={row['max_single_lev']:.2f}x p99={row['p99_single_lev']:.2f}x "
              f"med={row['med_single_lev']:.2f}x\n"
              f"    top10利益シェア net={share_net:.1%} gross={share_gross:.1%}\n"
              f"    総エクスポージャ max={row['max_gross_exposure']:.2f}x  "
              f"z>2.5 採用 {n_z25}件 ({n_z25/years:.1f}/年)  taken={len(taken)} skip={info['skipped']}",
              flush=True)
        jdump(res)

    # === 5) 年次安定(P4 vs P2, robust k平均の系列) ==========================
    print("\n=== [5] 年次リターン (robust k平均) ===", flush=True)
    yr2 = yearly_returns(audit_eq[2.0])
    yr4 = yearly_returns(audit_eq[4.0])
    tab = pd.DataFrame({"P2": yr2, "P4": yr4})
    tab["diff_pp"] = (tab["P4"] - tab["P2"]) * 100
    print((tab * np.array([100, 100, 1])).round(1).to_string(), flush=True)
    res["yearly"] = {str(y): {"P2": float(yr2.loc[y]), "P4": float(yr4.loc[y]),
                              "diff_pp": float((yr4.loc[y] - yr2.loc[y]) * 100)}
                     for y in yr2.index}
    jdump(res)

    # === 6) ブロック長感応(seed0) ==========================================
    print("\n=== [6] ブロック長感応 block in {21,63,126} (seed0) ===", flush=True)
    res["block_sens"] = {}
    for P in [2.0, 4.0]:
        fn = eq_fn(pool, closes, P, 11)
        for blk in [21, 126]:
            k = calibrate_robust_seeded(fn, target=0.20, n_boot=600, block=blk, seed=0)
            c = cagr_of(fn(k))
            res["block_sens"][f"P{P}_b{blk}"] = {"k": k, "cagr": c}
            print(f"  P={P} block={blk:3d}: k={k:5.2f} CAGR={c:+.2%}", flush=True)
            jdump(res)
    for P in [2.0, 4.0]:  # block=63 は [1] の seed0 を転記
        s0 = res["repro"][f"P{P}"]["rob"]["0"]
        res["block_sens"][f"P{P}_b63"] = {"k": s0["k"], "cagr": s0["cagr"]}
    jdump(res)

    # === 7) mp相互作用(P=4 x mp12/mp13, robust seed0; P=2 ペア) =============
    print("\n=== [7] mp相互作用 robust seed0 ===", flush=True)
    res["mp_interaction"] = {}
    for mp in [12, 13]:
        for P in [2.0, 4.0]:
            fn = eq_fn(pool, closes, P, mp)
            k = calibrate_robust_seeded(fn, target=0.20, n_boot=600, seed=0)
            c = cagr_of(fn(k))
            res["mp_interaction"][f"P{P}_mp{mp}"] = {"k": k, "cagr": c}
            print(f"  mp{mp} P={P}: k={k:5.2f} CAGR={c:+.2%}", flush=True)
            jdump(res)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min  -> {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
