"""exp42: ボラ第2因子サイジング f(z)×g(vol) — reports/15 実測勾配の配分変換検証。

仮説: vol_entry の within-instrument 因果ランクが高いトレードほど期待リターンが大きい
(reports/15: vol Q4=純益47.5%/PF2.19、z2.8+×高vol +54.8bps vs 低vol -25.8bps)。
ならば f(z)×g(vol_rank) の第2因子グレーディングで同テールのまま利益密度が上がるはず。
reports/09 の棄却は逆方向(高volを削る)だったため、本検証は未踏。

設計(exp37 の骨格を流用):
  1. 各銘柄 close から20本ボラを再計算し、エントリー時点の rolling(1000, min100) 因果
     パーセンタイルランクを全トレードに付与(先読み無し)。pool.vol_entry との整合を検算。
  2. 乗数 g(rank): べき (rank/v0)^Q (Q∈{0.5,1,2}, v0∈{0.4,0.5,0.6}) / 線形 1+a*(rank-0.5)
     (a∈{0.5,1}) / 二値 0.5 if rank<0.25。全て clip[0.3,3.0]→プール平均=1 正規化、f(z) に乗算。
  3. mp8 2段階プロトコル(seed0 スクリーニング → 上位≤3 を seeds0-4 ペアシード、
     ベースライン=champion_sizing(f(z)のみ, P=4.0))。最有力は mp11 でも測定。
  4. IS(<2022) robust seed0 較正で argmax 選択 → その OOS 素検証。
  5. 配分集中監査(最大単一重み/top10シェア/加重ワースト/skew/p1)+レバ偽装署名。
  6. 2022除外・全年プラス・低vol抑制 vs 高vol増し分解。

実行: PYTHONPATH=. uv run python research/experiments/exp42_gvol_sizing.py

結論(2026-06-11 実行): **reject**。
  因果ランクに置換すると reports/15 の勾配はほぼ消失(spearman full +0.09 / IS +0.06、
  五分位は非単調・IS Q2 は PF0.91)→ 解剖の vol Q4 勾配の大半は full-sample 分位
  (=事後レジーム知識)由来だった。g(vol) 全12構成が robust seed0 でベースライン以下、
  ペアシード5シードでも lin_a0.5 -0.43pp / bin_q1half -0.37pp(全シード負)。
  mp11 でも -0.65pp。唯一 empirical が伸びる bin_q1half(+1.6pp)は p95 -27.8→-29.9% の
  レバ偽装署名 + 改善の67%が2022(rob0-k では2022除外で符号反転)。IS-argmax も
  lin_a0.5(IS +13.6% < baseline +14.5%)で OOS もベースライン未満。
  低vol抑制/高vol増しの成分分解(seed0)も +16.47/+16.64% vs 基準 +16.40% で
  +0.5pp 基準に届かず p95 悪化 → 排他候補「Q1抑制フィルタ」も同根拠で見送り。
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
from mm_production import _fz, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab import universe as uni  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS_FULL = (0, 1, 2, 3, 4)
CLIP_LO, CLIP_HI = 0.3, 3.0
OUT_JSON = ROOT / "research" / "outputs" / "exp42_gvol_sizing.json"
OUT_CSV = ROOT / "research" / "outputs" / "exp42_gvol_sizing.csv"
OUT_DIAG = ROOT / "research" / "outputs" / "exp42_volrank_quintiles.csv"


# --- ボラ因果ランク -------------------------------------------------------
def entry_vol_rank(pool: pd.DataFrame, win=20, rank_win=1000, minp=100):
    """各トレードのエントリー時 20本ボラの within-instrument 因果ランク(過去 rank_win 本)。"""
    uni.register_cross_spreads(3.0)
    rank = np.full(len(pool), np.nan)
    vchk = np.full(len(pool), np.nan)
    for nm, idx in pool.groupby("instr").groups.items():
        c = uni.instrument_data(nm, "H4")["close"]
        vol = c.pct_change().rolling(win).std()
        vr = vol.rolling(rank_win, min_periods=minp).rank(pct=True)
        ii = np.asarray(idx)
        ent = pd.to_datetime(pool["entry"].iloc[ii])
        rank[ii] = vr.reindex(ent, method="ffill").to_numpy()
        vchk[ii] = vol.reindex(ent, method="ffill").to_numpy()
    return rank, vchk


def quintile_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    def pf(x):
        g = x[x > 0].sum()
        l = -x[x <= 0].sum()
        return g / l if l > 0 else np.inf

    d = df.dropna(subset=["vrank"]).copy()
    d["q"] = pd.qcut(d["vrank"], 5, labels=False)
    tab = d.groupby("q").agg(n=("ret", "size"), rk_lo=("vrank", "min"), rk_hi=("vrank", "max"),
                             mean_ret=("ret", "mean"), med_ret=("ret", "median"),
                             win=("ret", lambda x: (x > 0).mean()), PF=("ret", pf),
                             sum_ret=("ret", "sum"), mean_z=("z_entry", "mean"))
    tab.insert(0, "sample", label)
    return tab


# --- 乗数 → サイジング(per-trade キー紐付け, exp37 方式) -----------------
def normalize_mult(g_raw: np.ndarray) -> np.ndarray:
    """nan→1 → clip[0.3,3.0] → プール平均=1 に正規化(k 較正と直交)。"""
    g = np.where(np.isfinite(g_raw), g_raw, 1.0)
    g = np.clip(g, CLIP_LO, CLIP_HI)
    return g / g.mean()


def graded_sizing_factory(pool, mult: np.ndarray, max_pos=MAX_POS, fz=_fz):
    """champion z-power × per-trade 乗数 mult(正規化済)。"""
    fbar = float(np.mean([fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0
    key = {}
    instr = pool["instr"].to_numpy()
    ret = pool["ret"].to_numpy()
    bh = pool["bars_held"].to_numpy()
    for i in range(len(pool)):
        key[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = float(mult[i])

    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            gm = key.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), 1.0)
            return ctx["equity_real"] * base * gm * (fz(ctx["z"]) / fbar)
        return sizing
    return make_sizing


# --- 評価 -----------------------------------------------------------------
def eval_config(label, pool, closes, make_sizing, seeds=(0,), max_pos=MAX_POS,
                with_is=True) -> dict:
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, make_sizing(kk), max_pos=max_pos)
            cache[kk] = eqm
        return cache[kk]

    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    eq_emp = eq_of_k(res["emp_k"])
    yr_emp = yearly_returns(eq_emp)
    res["yr_emp"] = yr_emp
    res["worst_year_emp"] = float(yr_emp.min())
    res["neg_years_emp"] = int((yr_emp < 0).sum())
    r = eq_emp.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    res["skew_emp"] = float(r.skew())
    res["p1_emp"] = float(np.percentile(r, 1))
    k_r0 = res["rob"][0]["k"]
    eq_r0 = eq_of_k(k_r0)
    yr_r0 = yearly_returns(eq_r0)
    res["yr_rob0"] = yr_r0
    res["worst_year_rob0"] = float(yr_r0.min())
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())

    if with_is:
        is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
        oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
        is_cl = closes[closes.index < OOS_START]
        oos_cl = closes[closes.index >= OOS_START]
        cache_is = {}

        def eq_is(k):
            kk = round(float(k), 10)
            if kk not in cache_is:
                eqm, _, _ = mm.simulate(is_pool, is_cl, make_sizing(kk), max_pos=max_pos)
                cache_is[kk] = eqm
            return cache_is[kk]

        k_ie = calibrate_empirical(eq_is, 0.20)
        eqo, _, _ = mm.simulate(oos_pool, oos_cl, make_sizing(k_ie), max_pos=max_pos)
        res["k_is_emp"] = k_ie
        res["is_emp_cagr"] = cagr_of(eq_is(k_ie))
        res["oos_emp_cagr"] = cagr_of(eqo)
        res["oos_emp_dd"] = max_dd(eqo)
        k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
        eqo2, _, _ = mm.simulate(oos_pool, oos_cl, make_sizing(k_ir), max_pos=max_pos)
        res["k_is_rob0"] = k_ir
        res["is_rob0_cagr"] = cagr_of(eq_is(k_ir))
        res["oos_rob0_cagr"] = cagr_of(eqo2)
        res["oos_rob0_dd"] = max_dd(eqo2)
        print(f"      IS emp k={k_ie:5.2f} CAGR={res['is_emp_cagr']:+7.2%} -> OOS {res['oos_emp_cagr']:+7.2%} "
              f"(DD {res['oos_emp_dd']:+6.1%}) | IS rob0 k={k_ir:5.2f} CAGR={res['is_rob0_cagr']:+7.2%} "
              f"-> OOS {res['oos_rob0_cagr']:+7.2%}")
    return res


def concentration_audit(pool, mult, emp_k, label) -> dict:
    """理論上の per-trade 配分重み w = (k/mp)*g*f(z)/f̄(equity比)。"""
    fz_arr = np.array([_fz(z) for z in pool["z_entry"].to_numpy()])
    fbar = fz_arr.mean()
    w = (emp_k / MAX_POS) * mult * fz_arr / fbar
    contrib = w * pool["ret"].to_numpy()
    order = np.sort(w)[::-1]
    return {
        "label": label, "emp_k": float(emp_k),
        "max_w": float(w.max()), "mean_w": float(w.mean()),
        "n_w_gt1": int((w > 1.0).sum()), "n_w_gt2": int((w > 2.0).sum()),
        "top10_share_%": float(order[:10].sum() / order.sum() * 100),
        "worst_contrib_%eq": float(contrib.min() * 100),
        "best_contrib_%eq": float(contrib.max() * 100),
    }


def yearly_diff(yr_base: pd.Series, yr_cand: pd.Series, tag: str) -> dict:
    idx = sorted(set(yr_base.index) | set(yr_cand.index))
    d = pd.Series({y: np.log1p(yr_cand.get(y, 0.0)) - np.log1p(yr_base.get(y, 0.0)) for y in idx})
    total = float(d.sum())
    best = int(d.idxmax())
    excl = total - float(d.max())
    print(f"  [{tag}] 年次改善(log合計)={total:+.4f}  最良年={best}({d.max():+.4f})  "
          f"最良年除外後={excl:+.4f} ({'符号維持' if excl > 0 else '符号反転'})")
    print("    " + "  ".join(f"{y}:{v:+.3f}" for y, v in d.items()))
    return {"tag": tag, "total": total, "best_year": best, "best_val": float(d.max()),
            "excl_best": excl, "per_year": {int(y): float(v) for y, v in d.items()}}


def strip_series(res: dict) -> dict:
    return {k: ({str(s): w for s, w in v.items()} if k == "rob" else v)
            for k, v in res.items() if not isinstance(v, pd.Series)}


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / sum(ret)={pool['ret'].sum():+.4f} / grid {len(closes)} bars / max_pos={MAX_POS}")

    keys = list(zip(pool["instr"], pool["ret"].round(12), pool["bars_held"]))
    n_dup = len(keys) - len(set(keys))
    print(f"per-trade キー衝突: {n_dup} 件")

    # --- 1. 因果 vol ランク + 整合検算 ------------------------------------
    vrank, vchk = entry_vol_rank(pool)
    diff = np.abs(vchk - pool["vol_entry"].to_numpy())
    n_nan_rank = int(np.isnan(vrank).sum())
    print(f"\nvol 再計算整合: max|diff|={np.nanmax(diff):.3e} / "
          f"一致(<1e-12)率={(diff < 1e-12).mean():.1%} / rank NaN(序盤ウォームアップ)={n_nan_rank}件(g=1扱い)")

    df = pool.copy()
    df["vrank"] = vrank
    tabs = [quintile_table(df, "full"),
            quintile_table(df[df["entry"] < OOS_START], "IS"),
            quintile_table(df[df["entry"] >= OOS_START], "OOS")]
    diag = pd.concat(tabs)
    print("\n=== 診断: 因果volランク五分位 × プール成績(勾配は仮説方向=正か?) ===")
    print(diag.round(4).to_string())
    sp = df["vrank"].corr(df["ret"], method="spearman")
    sp_is = df[df["entry"] < OOS_START]["vrank"].corr(df[df["entry"] < OOS_START]["ret"], method="spearman")
    sp_oos = df[df["entry"] >= OOS_START]["vrank"].corr(df[df["entry"] >= OOS_START]["ret"], method="spearman")
    print(f"spearman(vrank, ret) full={sp:+.4f} IS={sp_is:+.4f} OOS={sp_oos:+.4f} (仮説は正)")
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(OUT_DIAG)

    # --- 2. 構成定義 -------------------------------------------------------
    g_of = {}
    configs: list[tuple[str, object, str]] = []
    configs.append(("baseline_mp8", champion_sizing(pool, max_pos=MAX_POS), "基準 f(z) P=4.0"))
    for q in [0.5, 1.0, 2.0]:
        for v0 in [0.4, 0.5, 0.6]:
            lab = f"pow_Q{q}_v{v0}"
            g = normalize_mult(np.power(np.clip(vrank / v0, 0.0, None), q))
            g_of[lab] = g
            configs.append((lab, graded_sizing_factory(pool, g), f"べき Q={q} v0={v0}"))
    for a in [0.5, 1.0]:
        lab = f"lin_a{a}"
        g = normalize_mult(1 + a * (vrank - 0.5))
        g_of[lab] = g
        configs.append((lab, graded_sizing_factory(pool, g), f"線形 a={a}"))
    g = normalize_mult(np.where(vrank < 0.25, 0.5, 1.0))
    g_of["bin_q1half"] = g
    configs.append(("bin_q1half", graded_sizing_factory(pool, g), "二値 Q1半減"))

    # --- 3. ステージ1: seed0 スクリーニング + IS/OOS ----------------------
    print("\n=== ステージ1: empirical較正 + robust seed0 + IS/OOS (mp8) ===")
    results = {}
    for label, mk, notes in configs:
        res = eval_config(label, pool, closes, mk, seeds=(0,))
        res["notes"] = notes
        results[label] = res
        OUT_JSON.write_text(json.dumps({k: strip_series(v) for k, v in results.items()},
                                       indent=2, default=float))

    base_s0 = results["baseline_mp8"]["rob"][0]["cagr"]
    cand = [(lab, r) for lab, r in results.items() if lab != "baseline_mp8"]
    cand.sort(key=lambda x: -x[1]["rob"][0]["cagr"])
    stage2 = [lab for lab, r in cand[:3] if r["rob"][0]["cagr"] >= base_s0 - 0.002]
    print(f"\nbaseline rob s0 = {base_s0:+.2%} / ステージ2対象 = {stage2 or 'なし(全候補が基準未満)'}")
    if not stage2:
        stage2 = [cand[0][0]]  # 最良1件はペアシードで確認(reject の根拠を5シードで固める)
        print(f"  → 最良 {stage2[0]} のみ確認のため seeds0-4 へ")

    # --- 4. ステージ2: ペアシード seeds 0-4 -------------------------------
    print("\n=== ステージ2: seeds(0..4) ペアシード較正 (mp8) ===")
    mk_of = {label: mk for label, mk, _ in configs}
    for label in ["baseline_mp8"] + stage2:
        res = eval_config(label, pool, closes, mk_of[label], seeds=SEEDS_FULL, with_is=False)
        results[label]["rob"] = res["rob"]
        results[label]["rob_cagr_mean"] = res["rob_cagr_mean"]
        results[label]["mean5"] = True

    base = results["baseline_mp8"]
    print("\n--- ペアシード per-seed 差(robust CAGR, 候補 - baseline) ---")
    for label in stage2:
        r = results[label]
        diffs = [r["rob"][s]["cagr"] - base["rob"][s]["cagr"] for s in SEEDS_FULL]
        print(f"  {label:18s} " + " ".join(f"s{s}:{d:+.2%}" for s, d in zip(SEEDS_FULL, diffs)) +
              f"  mean={np.mean(diffs):+.2%}")

    best_lab = max(stage2, key=lambda l: results[l]["rob_cagr_mean"])
    best = results[best_lab]
    print(f"\n最有力: {best_lab} rob mean5={best['rob_cagr_mean']:+.2%} "
          f"(baseline {base['rob_cagr_mean']:+.2%}, diff {best['rob_cagr_mean']-base['rob_cagr_mean']:+.2%})")

    # --- 5. mp11 でも測定 ---------------------------------------------------
    print("\n=== mp11 測定(ペアシード seeds0-4) ===")
    mp11_res = {}
    mk_b11 = champion_sizing(pool, max_pos=11)
    mp11_res["baseline_mp11"] = eval_config("baseline_mp11", pool, closes, mk_b11,
                                            seeds=SEEDS_FULL, max_pos=11, with_is=False)
    mk_c11 = graded_sizing_factory(pool, g_of[best_lab], max_pos=11)
    mp11_res[f"{best_lab}_mp11"] = eval_config(f"{best_lab}_mp11", pool, closes, mk_c11,
                                               seeds=SEEDS_FULL, max_pos=11, with_is=False)
    d11 = [mp11_res[f"{best_lab}_mp11"]["rob"][s]["cagr"] - mp11_res["baseline_mp11"]["rob"][s]["cagr"]
           for s in SEEDS_FULL]
    print(f"  mp11 per-seed diff: " + " ".join(f"s{s}:{d:+.2%}" for s, d in zip(SEEDS_FULL, d11)) +
          f"  mean={np.mean(d11):+.2%}")

    # --- 6. IS-argmax 監査 ---------------------------------------------------
    print("\n=== IS-argmax 監査(IS rob0 CAGR で選択 → OOS 素検証) ===")
    is_rank = sorted(((lab, r["is_rob0_cagr"]) for lab, r in results.items() if lab != "baseline_mp8"),
                     key=lambda x: -x[1])
    is_pick = is_rank[0][0]
    print("  IS rob0 上位5: " + " / ".join(f"{l}={c:+.2%}" for l, c in is_rank[:5]))
    print(f"  IS選択 = {is_pick} (フル期間選択 = {best_lab} → {'一致' if is_pick == best_lab else '不一致: IS選択構成を正とする'})")
    for lab in {is_pick, best_lab, "baseline_mp8"}:
        r = results[lab]
        print(f"    {lab:18s} OOS(emp-k): {r['oos_emp_cagr']:+7.2%} (DD {r['oos_emp_dd']:+6.1%}) | "
              f"OOS(rob0-k): {r['oos_rob0_cagr']:+7.2%} (DD {r['oos_rob0_dd']:+6.1%})")

    # --- 7. 配分集中の現実性監査 + レバ偽装署名 ----------------------------
    print("\n=== 配分集中監査(empirical k, mp8) ===")
    auds = [concentration_audit(pool, np.ones(len(pool)), base["emp_k"], "baseline_mp8"),
            concentration_audit(pool, g_of[best_lab], best["emp_k"], best_lab)]
    print(pd.DataFrame(auds).round(3).to_string(index=False))
    print(f"  skew/p1(MtM barリターン@emp k): baseline {base['skew_emp']:+.2f}/{base['p1_emp']:+.4%} "
          f"vs {best_lab} {best['skew_emp']:+.2f}/{best['p1_emp']:+.4%}")
    lev_disguise = (best["emp_cagr"] > base["emp_cagr"]) and (best["emp_p95"] < base["emp_p95"] - 0.005)
    print(f"  レバ偽装署名(emp CAGR↑かつp95悪化>0.5pp): {'あり=reject' if lev_disguise else 'なし'} "
          f"(p95 baseline {base['emp_p95']:+.1%} vs {best['emp_p95']:+.1%})")

    # --- 8. 2022除外・全年プラス・成分分解 ----------------------------------
    print("\n=== 年次分解(2022除外チェック) ===")
    yd_rob = yearly_diff(base["yr_rob0"], best["yr_rob0"], f"rob0-k {best_lab} vs baseline")
    yd_emp = yearly_diff(base["yr_emp"], best["yr_emp"], f"emp-k {best_lab} vs baseline")
    print(f"  負け年数(rob0): baseline {base['neg_years_rob0']} → 候補 {best['neg_years_rob0']} / "
          f"(emp): {base['neg_years_emp']} → {best['neg_years_emp']}")
    print(f"  最悪年(rob0): baseline {base['worst_year_rob0']:+.1%} → 候補 {best['worst_year_rob0']:+.1%}")

    print("\n=== 成分分解: 低vol抑制(g<1) vs 高vol増し(g>1) ===")
    g_best_clipped = np.clip(np.where(np.isfinite(g_of[best_lab]), g_of[best_lab], 1.0), None, None)
    g_supp = normalize_mult(np.minimum(g_best_clipped, 1.0))
    g_boost = normalize_mult(np.maximum(g_best_clipped, 1.0))
    decomp = {}
    for lab, g in [("supp_only(g<=1)", g_supp), ("boost_only(g>=1)", g_boost)]:
        r = eval_config(f"decomp_{lab}", pool, closes, graded_sizing_factory(pool, g),
                        seeds=(0,), with_is=False)
        decomp[lab] = {"rob_s0": r["rob"][0]["cagr"], "emp_cagr": r["emp_cagr"], "emp_p95": r["emp_p95"]}

    # --- 保存 ----------------------------------------------------------------
    rows = []
    for lab, r in {**results, **mp11_res}.items():
        rows.append({
            "label": lab, "notes": r.get("notes", ""),
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"], "worst_year_emp": r.get("worst_year_emp"),
            "neg_years_rob0": r.get("neg_years_rob0"),
            "rob_s0": r["rob"][0]["cagr"],
            "rob_mean": r.get("rob_cagr_mean") if (r.get("mean5") or "mp11" in lab) else None,
            "k_is_emp": r.get("k_is_emp"), "oos_emp_cagr": r.get("oos_emp_cagr"),
            "is_rob0_cagr": r.get("is_rob0_cagr"), "oos_rob0_cagr": r.get("oos_rob0_cagr"),
            "skew_emp": r.get("skew_emp"), "p1_emp": r.get("p1_emp"),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    payload = {k: strip_series(v) for k, v in {**results, **mp11_res}.items()}
    payload["_meta"] = {"yearly_diff_rob0": yd_rob, "yearly_diff_emp": yd_emp,
                        "decomp": decomp, "concentration": auds,
                        "spearman": {"full": float(sp), "IS": float(sp_is), "OOS": float(sp_oos)},
                        "best_lab": best_lab, "is_pick": is_pick,
                        "n_key_dup": n_dup, "n_nan_rank": n_nan_rank}
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n        -> {OUT_JSON}")
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
