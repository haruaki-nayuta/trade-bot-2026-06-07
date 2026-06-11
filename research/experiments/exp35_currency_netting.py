"""exp35: 通貨ファクター・ネッティング上限 — サイジング層の構造レバー(完全未踏)。

仮説: チャンピオンのテール(2022型失血)の正体は「反USD方向の建玉が同時に積み上がり、
USD共通モードで全玉同時失血」(reports/10 §1, exp30)。既存の資金管理は建玉「数」(max_pos)
しか見ておらず、**通貨別の正味エクスポージャ集中**は無制限。新規建玉時に通貨別の正味
エクスポージャへ上限を課せば、共通モード集中だけを構造的に削り(平時の分散済み利益は
削らない)、テール p95 が下がる → robust 較正で k を上げ直して CAGR 増、が狙い。

実装(mm_lab.simulate を本ファイルにコピー拡張。mm_lab 本体は不変):
  - champion z-power サイジング内蔵: alloc = equity_real * (k/max_pos) * f(z)/f̄
  - 通貨レグ分解: 銘柄 XXXYYY の建玉 (alloc, dir) は base=XXX に +dir*alloc、
    quote=YYY に -dir*alloc のエクスポージャ。open positions の通貨別正味 net[cur] を維持。
  - 新規トレード時、対象通貨について |net+Δ| が「増える方向」かつ cap 超過なら、
    ちょうど cap に収まるよう alloc を縮小。減らす方向はフリーパス。縮小後 alloc≤0 は見送り。
  - cap = gamma * (k/max_pos) * equity_mtm。**cap ∝ k** なので較正で k を動かしても
    配分の「形」は不変。

スイープ: gamma∈{1.0,1.5,2.0,3.0} × currencies∈{USDのみ, ALL=8通貨}(mp=11固定)。
gamma=∞ が mp11 ベースラインと一致することを最初に検算(ハーネスバグ検出)。
判定: tail_protocol 2段階(empirical+boot p95 seed0 → 上位のみ seeds1,2)+ IS/OOS +
gamma 方向の高原性。合格 = robust(p95=20%) seeds(0,1,2) 平均 ≥ +16.6%(reports/11 準拠)。

実行: PYTHONPATH=. uv run python research/experiments/exp35_currency_netting.py
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
from mm_production import champion_sizing, _fz  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
)

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
CUR_ALL = ("USD", "EUR", "JPY", "GBP", "AUD", "NZD", "CAD", "CHF")
MAX_POS = 11
OUT_CSV = ROOT / "research" / "outputs" / "exp35_currency_netting.csv"
OUT_JSON = ROOT / "research" / "outputs" / "exp35_currency_netting.json"

BASE_ROB_MEAN3 = 0.1504   # mp12 robust 平均(最良ベースライン)
PASS_LINE = 0.166         # +10% 相対
STAGE2_GATE = 0.156       # robust seed0 がこれ未満なら seeds1,2 は省略
BASE_P95 = -0.294         # mp11 empirical k での boot p95(レバ偽装チェック基準)


# --- simulate_netcap(mm_lab.simulate のコピー拡張 + champion sizing 内蔵) ----
def make_runner(pool: pd.DataFrame, closes: pd.DataFrame, *, max_pos=MAX_POS,
                gamma=np.inf, cur_set=CUR_ALL, fbar=None, init=10_000.0):
    """前計算を共有し run(k)->(eq_mtm,eq_real,info) と eq_of_k(k)->eq_mtm(キャッシュ付)を返す。

    gamma=inf なら mm_lab.simulate + champion_sizing と完全一致(検算で保証)。
    fbar を渡すと f(z) 正規化定数を外部固定(IS較正→OOS素検証で IS の f̄ を使うため)。
    """
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)

    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry: dict[int, list[int]] = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    cols = np.array([col_of[x] for x in instr_arr])
    # champion z-power(f̄ は mm_production と同一: プール平均で正規化)
    fz_raw = np.array([_fz(z) for z in pool["z_entry"].to_numpy()])
    fbar_used = float(fbar) if fbar is not None else (float(fz_raw.mean()) or 1.0)
    fzn = fz_raw / fbar_used
    # 通貨レグ(制約対象通貨のみ追跡。対象外は None)
    cur_set = frozenset(cur_set)
    base_cur = [str(x)[:3] if str(x)[:3] in cur_set else None for x in instr_arr]
    quote_cur = [str(x)[3:] if str(x)[3:] in cur_set else None for x in instr_arr]
    capped = np.isfinite(gamma)

    def run(k):
        per = k / max_pos
        equity = init
        peak_mtm = init
        open_pos = []  # [col, dir, eprice, alloc, exit_pos, ret, base, quote]
        net = {c: 0.0 for c in cur_set}
        eq_mtm = np.empty(n)
        eq_real = np.empty(n)
        n_taken = 0
        max_conc = 0
        skip_room = 0
        skip_cap = 0
        n_shrunk = 0
        shrink_sum = 0.0
        for b in range(n):
            # ① 決済(エクスポージャ解放)
            if open_pos:
                still = []
                for p in open_pos:
                    if p[4] <= b:
                        equity += p[3] * p[5]
                        if p[6] is not None:
                            net[p[6]] -= p[1] * p[3]
                        if p[7] is not None:
                            net[p[7]] += p[1] * p[3]
                    else:
                        still.append(p)
                open_pos = still
            # ② MtM
            unreal = 0.0
            for p in open_pos:
                unreal += p[3] * (p[1] * (carr[b, p[0]] / p[2] - 1.0))
            mtm = equity + unreal
            eq_mtm[b] = mtm
            eq_real[b] = equity
            if mtm > peak_mtm:
                peak_mtm = mtm
            # ③ 新規エントリー
            if b in by_entry:
                cap = gamma * per * mtm if capped else np.inf
                for ti in by_entry[b]:
                    if len(open_pos) >= max_pos:
                        skip_room += 1
                        continue
                    alloc0 = equity * per * fzn[ti]
                    alloc = alloc0
                    d = dir_arr[ti]
                    bc = base_cur[ti]
                    qc = quote_cur[ti]
                    if capped:
                        for cur, s in ((bc, d), (qc, -d)):
                            if cur is None:
                                continue
                            nv = net[cur]
                            new = nv + s * alloc
                            # 減らす方向 or cap 内ならフリーパス
                            if abs(new) <= abs(nv) or abs(new) <= cap:
                                continue
                            bound = cap if new > 0 else -cap
                            allowed = (bound - nv) / s
                            if allowed < alloc:
                                alloc = allowed
                        if alloc <= 1e-12:
                            skip_cap += 1
                            continue
                        if alloc < alloc0 * (1 - 1e-12):
                            n_shrunk += 1
                            shrink_sum += alloc / alloc0
                    open_pos.append([cols[ti], d, eprice_arr[ti], alloc,
                                     int(exit_pos[ti]), float(ret_arr[ti]), bc, qc])
                    if bc is not None:
                        net[bc] += d * alloc
                    if qc is not None:
                        net[qc] -= d * alloc
                    n_taken += 1
                    if len(open_pos) > max_conc:
                        max_conc = len(open_pos)
        eqm = pd.Series(eq_mtm, index=grid)
        eqr = pd.Series(eq_real, index=grid)
        info = {"final": equity, "skipped": skip_room + skip_cap, "n_taken": n_taken,
                "max_conc": max_conc, "skip_room": skip_room, "skip_cap": skip_cap,
                "n_shrunk": n_shrunk,
                "shrink_mean": (shrink_sum / n_shrunk) if n_shrunk else float("nan")}
        return eqm, eqr, info

    cache: dict[float, pd.Series] = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = run(kk)[0]
        return cache[kk]

    return run, eq_of_k, fbar_used


def sanity_check(pool, closes):
    """gamma=∞ が mm_lab.simulate + champion_sizing(mp11) と完全一致するか検算。"""
    k0 = 11.509042685  # exp33 の mp11 empirical k
    run_inf, _, _ = make_runner(pool, closes, gamma=np.inf, cur_set=CUR_ALL)
    eq_a, eqr_a, _ = run_inf(k0)
    mk = champion_sizing(pool, max_pos=MAX_POS)
    eq_b, eqr_b, _ = mm.simulate(pool, closes, mk(k0), max_pos=MAX_POS)
    ok_m = np.allclose(eq_a.to_numpy(), eq_b.to_numpy(), rtol=1e-9)
    ok_r = np.allclose(eqr_a.to_numpy(), eqr_b.to_numpy(), rtol=1e-9)
    print(f"[sanity] gamma=inf vs mm.simulate: MtM一致={ok_m} 実現一致={ok_r} "
          f"CAGR={cagr_of(eq_a):+.2%} (期待 +23.8%)")
    if not (ok_m and ok_r):
        raise SystemExit("sanity check failed: gamma=inf がベースラインと不一致")


def eval_config(pool, closes, label, gamma, cur_set):
    """段階1評価: empirical較正 + boot p95(seed0) + robust seed0 + IS/OOS + cap統計。"""
    t0 = time.time()
    run, eq_of_k, fbar_full = make_runner(pool, closes, gamma=gamma, cur_set=cur_set)
    # empirical 20%
    k_emp = calibrate_empirical(eq_of_k, 0.20, hi=32.0, iters=24)
    eqm, eqr, info = run(k_emp)
    s = mm.stats(eqm, eqr, info)
    bs = boot_dd(eqm, n_boot=600, seed=0)
    # robust(p95=20%) seed0
    k_r0 = calibrate_robust_seeded(eq_of_k, 0.20, n_boot=600, seed=0, hi=32.0)
    rob0 = cagr_of(eq_of_k(k_r0))
    # IS較正 → OOS素検証(empirical, f̄ は IS プールで固定=因果)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    _, eq_is, fbar_is = make_runner(is_pool, is_cl, gamma=gamma, cur_set=cur_set)
    k_is = calibrate_empirical(eq_is, 0.20, hi=32.0, iters=24)
    run_oos, _, _ = make_runner(oos_pool, oos_cl, gamma=gamma, cur_set=cur_set, fbar=fbar_is)
    eqo, _, _ = run_oos(k_is)
    res = {
        "label": label, "gamma": (gamma if np.isfinite(gamma) else -1.0),
        "currencies": ("ALL" if len(cur_set) > 1 else "USD"),
        "emp_k": float(k_emp), "emp_cagr": float(s["cagr"]),
        "emp_dd": float(s["maxdd_mtm"]), "emp_p95": float(bs["p95"]),
        "emp_sharpe": float(s["sharpe"]), "worst_year": float(s["worst_year"]),
        "pos_year": float(s["pos_year_rate"]),
        "rob_k_seed0": float(k_r0), "rob_cagr_seed0": float(rob0),
        "k_is": float(k_is), "oos_emp_cagr": float(cagr_of(eqo)),
        "oos_emp_dd": float(max_dd(eqo)),
        "n_taken": int(info["n_taken"]), "skip_room": int(info["skip_room"]),
        "skip_cap": int(info["skip_cap"]), "n_shrunk": int(info["n_shrunk"]),
        "shrink_mean": float(info["shrink_mean"]),
    }
    print(f"  {label:14s} emp k={k_emp:5.2f} CAGR={s['cagr']:+7.2%} p95={bs['p95']:+6.1%} "
          f"worst年={s['worst_year']:+6.1%} | rob s0 k={k_r0:5.2f} CAGR={rob0:+7.2%} | "
          f"OOS={res['oos_emp_cagr']:+7.2%} DD={res['oos_emp_dd']:+6.1%} | "
          f"cap: 縮小{info['n_shrunk']}件(平均x{info['shrink_mean']:.2f}) 見送り{info['skip_cap']}件 "
          f"[{time.time()-t0:.0f}s]")
    return res, eq_of_k


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / grid {len(closes)} bars / mp={MAX_POS}")
    sanity_check(pool, closes)

    configs = [("baseline", np.inf, CUR_ALL)]
    for g in [1.0, 1.5, 2.0, 3.0]:
        configs.append((f"USD g={g}", g, ("USD",)))
    for g in [1.0, 1.5, 2.0, 3.0]:
        configs.append((f"ALL g={g}", g, CUR_ALL))

    print("\n=== 段階1: empirical + boot p95(seed0) + robust seed0 + IS/OOS ===")
    results = []
    eqfns = {}
    for label, g, cs in configs:
        res, eqfn = eval_config(pool, closes, label, g, cs)
        results.append(res)
        eqfns[label] = eqfn

    # 段階2: robust seed0 上位(>= STAGE2_GATE)の ≤3 構成に seeds 1,2 を追加
    cands = [r for r in results if r["label"] != "baseline"
             and r["rob_cagr_seed0"] >= STAGE2_GATE]
    cands = sorted(cands, key=lambda r: -r["rob_cagr_seed0"])[:3]
    print(f"\n=== 段階2: seeds 1,2 追加(対象 {len(cands)} 構成) ===")
    for r in cands:
        eqfn = eqfns[r["label"]]
        cs = [r["rob_cagr_seed0"]]
        for sd in (1, 2):
            k_r = calibrate_robust_seeded(eqfn, 0.20, n_boot=600, seed=sd, hi=32.0)
            cs.append(float(cagr_of(eqfn(k_r))))
        r["rob_cagr_seed1"], r["rob_cagr_seed2"] = cs[1], cs[2]
        r["rob_cagr_mean3"] = float(np.mean(cs))
        print(f"  {r['label']:14s} s0={cs[0]:+.2%} s1={cs[1]:+.2%} s2={cs[2]:+.2%} "
              f"-> mean3={r['rob_cagr_mean3']:+.2%} (合格線 {PASS_LINE:+.1%})")
    if not cands:
        print(f"  全構成 robust seed0 < {STAGE2_GATE:+.1%} -> 早期終了(reject)")

    df = pd.DataFrame(results)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\nsaved -> {OUT_JSON}")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
