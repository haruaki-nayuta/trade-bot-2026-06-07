"""敵対検証: 改善D(モメンタムbook long-only化 + band調整)を厳格に潰す。

exp61_momentum_longonly.py が報告した「long-only 化で baseline(both lb24 band0 w0.2)を上回る」
を 4 つの観点で攻撃する。championブックは exp60/exp61 と完全同一(build_pool_d1 + champion_sizing
max_pos8 + calibrate_robust target_dd0.20)。モメンタムbookのみ差し替え。

(1) leverage偽装の最終確認(same-tail署名):
    robust(p95再較正)だけでなく **経験的 maxDD=20% 較正** で合成し、
    - empirical CAGR が baseline比で上がるか
    - 経験的 maxDD が baseline比で悪化(深く)なるか
    - 固定レバ混入の署名(empCAGR↑ かつ DD↑/p95↑)が出るか
    p95(日次ブロックブート)も baseline比で悪化しないか。

(2) plateau_robust: 採用候補(long band0, w0.2)の近傍 OAT。
    band∈{0,0.001,0.002,0.003} と w∈{0.10,0.15,0.20,0.25,0.30} で delta が符号維持か、単一セルか。
    (exp61 の内点最大ではなく、固定 w=0.20 での band 安定性と、固定 band=0 での w 安定性を見る)

(3) oos_survives: IS(〜2021末)で構成決定(long-only band0, w0.2)→ レバを IS daily で固定 →
    OOS(2022-)に同じレバで素適用して baseline(both)との delta が正に残るか。
    モメンタムbook の long-only edge が JPY 円安ドリフト(2012-2024)に乗っていただけなら
    OOS でドリフトが鈍る/反転すれば delta は消える/負になるはず。

(4) seed_stable: 日次ブロックブートストラップの seed を 5 通り変えて、baseline比 delta(w0.2)が
    全 seed で正か。

NET(通常スプレッド)。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import universe as uni
from strategies import tsmom


def daily_block_bootstrap_p95(daily_ret, n_boot=2000, block=21, seed=0):
    r = np.asarray(daily_ret, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < block * 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block, size=(n_boot, n_blocks))
    dds = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts[i][:, None] + np.arange(block)).ravel()[:n]
        path = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(path)
        dds[i] = (path / peak - 1.0).min()
    return float(np.percentile(dds, 5))


def lever_to_p95(daily_ret, target=0.20, n_boot=2000, block=21, seed=0,
                 lo=0.05, hi=20.0, iters=30):
    def p95_at(L):
        return abs(daily_block_bootstrap_p95(daily_ret * L, n_boot=n_boot, block=block, seed=seed))
    if p95_at(hi) <= target:
        return hi, p95_at(hi)
    if p95_at(lo) > target:
        return lo, p95_at(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95_at(mid) > target:
            hi = mid
        else:
            lo = mid
    return lo, p95_at(lo)


def emp_maxdd(daily_ret):
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def lever_to_emp_dd(daily_ret, target=0.20, lo=0.05, hi=20.0, iters=40):
    """経験的(単一パス)maxDD == target に較正。"""
    def dd_at(L):
        return abs(emp_maxdd(daily_ret * L))
    if dd_at(hi) <= target:
        return hi
    if dd_at(lo) > target:
        return lo
    for _ in range(iters):
        mid = (lo + hi) / 2
        if dd_at(mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


def cagr_of(daily_ret, index):
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    years = (index[-1] - index[0]).days / 365.25
    final = path[-1]
    return (final ** (1 / years) - 1) if final > 0 else -1.0


def to_daily_ret(eqm):
    d = eqm.resample("1D").last().dropna()
    return d.pct_change().dropna()


def build_momentum_daily(lookback, band, side, n_boot=800):
    """USDJPY H1 tsmom の二口座用 eqm を p95=20% 較正で作り、日次リターンを返す。"""
    JPY = ["USDJPY"]
    tag = f"adv_tsmom_usdjpy_lb{lookback}_b{int(band*1000)}_{side}"
    pool_j = mm.build_pool_for(tsmom, {"lookback": lookback, "band": band}, tf="H1",
                               instruments=JPY, tag=tag, side=side, cache=False)
    closes_j = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()

    def mk_j(k):
        return lambda ctx: ctx["equity_real"] * k
    k_j, eqm_j, eqr_j, info_j, p95_j = mm.calibrate_robust(
        pool_j, closes_j, mk_j, target_dd=0.20, max_pos=1, n_boot=n_boot)
    return to_daily_ret(eqm_j)


def main():
    print("=== exp61d: 敵対検証 改善D long-only momentum ===\n")

    # ----- champion book (exp60/exp61 と完全同一) -----
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    rc_full = to_daily_ret(eqm_c)
    print(f"[champion] trades={len(pool_c)} daily_days={len(rc_full)} "
          f"{rc_full.index[0].date()}..{rc_full.index[-1].date()}")

    # ----- momentum books -----
    rj_both = build_momentum_daily(24, 0.0, "both")
    rj_long = {b: build_momentum_daily(24, b, "long") for b in [0.0, 0.001, 0.002, 0.003]}

    # 共通グリッド helper
    def align(rj):
        common = rc_full.index.intersection(rj.index)
        rc = rc_full.reindex(common).fillna(0.0).values
        rjj = rj.reindex(common).fillna(0.0).values
        return common, rc, rjj

    # =========================================================
    # (1) leverage偽装の最終確認: 経験的maxDD=20% 較正 + p95再較正の両方
    # =========================================================
    print("\n=== (1) leverage偽装 same-tail 確認 (champion単独 / baseline both / long variants) ===")

    def emp_and_p95_block(rj, w):
        common, rc, rjj = align(rj)
        # champion 単独(同一グリッド): 経験的DD較正 と p95較正
        Lc_emp = lever_to_emp_dd(rc, 0.20)
        cagr_c_emp = cagr_of(rc * Lc_emp, common)
        dd_c_emp = emp_maxdd(rc * Lc_emp)
        Lc_p95, p95_c = lever_to_p95(rc, 0.20)
        cagr_c_p95 = cagr_of(rc * Lc_p95, common)
        # blend
        blend = (1 - w) * rc + w * rjj
        L_emp = lever_to_emp_dd(blend, 0.20)
        cagr_emp = cagr_of(blend * L_emp, common)
        dd_emp = emp_maxdd(blend * L_emp)
        L_p95, p95_b = lever_to_p95(blend, 0.20)
        cagr_p95 = cagr_of(blend * L_p95, common)
        return dict(
            cagr_c_emp=cagr_c_emp, dd_c_emp=dd_c_emp, cagr_c_p95=cagr_c_p95, p95_c=p95_c,
            cagr_emp=cagr_emp, dd_emp=dd_emp, cagr_p95=cagr_p95, p95_b=p95_b)

    # baseline both w0.2
    b_base = emp_and_p95_block(rj_both, 0.20)
    print(f"  champion単独   : emp CAGR={b_base['cagr_c_emp']:+.2%} empDD={b_base['dd_c_emp']:+.1%} "
          f"| p95 CAGR={b_base['cagr_c_p95']:+.2%} p95={b_base['p95_c']:+.1%}")
    print(f"  baseline(both w0.2): emp CAGR={b_base['cagr_emp']:+.2%} empDD={b_base['dd_emp']:+.1%} "
          f"| p95 CAGR={b_base['cagr_p95']:+.2%} p95={b_base['p95_b']:+.1%}")
    base_emp_cagr = b_base["cagr_emp"]
    base_p95_cagr = b_base["cagr_p95"]

    # long-only variants @ w0.2 (採用候補域)
    print("\n  long-only @ w0.20:")
    print(f"    {'band':>6} {'empCAGR':>9} {'empDD':>7} {'Δemp_vs_base':>13} "
          f"{'p95CAGR':>9} {'p95':>7} {'Δp95_vs_base':>13} {'DD悪化?':>8} {'p95悪化?':>9}")
    longvar = {}
    for b in [0.0, 0.001, 0.002, 0.003]:
        r = emp_and_p95_block(rj_long[b], 0.20)
        d_emp = (r["cagr_emp"] - base_emp_cagr) * 100
        d_p95 = (r["cagr_p95"] - base_p95_cagr) * 100
        # same-tail署名: empCAGR上がりかつ(DD深化 or p95悪化)
        dd_worse = r["dd_emp"] < b_base["dd_emp"] - 0.005  # より深い(0.5pp超)
        p95_worse = r["p95_b"] > abs(b_base["p95_c"]) + 0.003
        longvar[b] = dict(d_emp=d_emp, d_p95=d_p95, dd_emp=r["dd_emp"],
                          p95_b=r["p95_b"], dd_worse=dd_worse, p95_worse=p95_worse)
        print(f"    {b:>6.3f} {r['cagr_emp']:>+9.2%} {r['dd_emp']:>+7.1%} {d_emp:>+13.2f} "
              f"{r['cagr_p95']:>+9.2%} {r['p95_b']:>+7.1%} {d_p95:>+13.2f} "
              f"{str(dd_worse):>8} {str(p95_worse):>9}")

    # =========================================================
    # (2) plateau_robust: 固定w=0.2でbandOAT, 固定band=0でwOAT (p95再較正)
    # =========================================================
    print("\n=== (2) plateau_robust (p95=20% 再較正, baseline=both w0.2) ===")
    # baseline robCAGR (p95再較正, both book)
    common_b, rc_b, rj_b = align(rj_both)
    Lb, _ = lever_to_p95((1 - 0.20) * rc_b + 0.20 * rj_b, 0.20)
    baseline_robCAGR = cagr_of(((1 - 0.20) * rc_b + 0.20 * rj_b) * Lb, common_b)
    print(f"  baseline robCAGR (both w0.2) = {baseline_robCAGR:+.3%}")

    def rob_blend(rj, w):
        common, rc, rjj = align(rj)
        blend = (1 - w) * rc + w * rjj
        L, p95 = lever_to_p95(blend, 0.20)
        return cagr_of(blend * L, common)

    print("\n  固定 w=0.20, band OAT:")
    band_delta_w02 = {}
    for b in [0.0, 0.001, 0.002, 0.003]:
        rob = rob_blend(rj_long[b], 0.20)
        d = (rob - baseline_robCAGR) * 100
        band_delta_w02[b] = d
        print(f"    band{b:.3f}: robCAGR={rob:+.3%} Δvs_baseline={d:+.2f}pp")
    print("\n  固定 band=0, w OAT (long-only):")
    w_delta_b0 = {}
    for w in [0.10, 0.15, 0.20, 0.25, 0.30]:
        rob = rob_blend(rj_long[0.0], w)
        d = (rob - baseline_robCAGR) * 100
        w_delta_b0[w] = d
        print(f"    w={w:.2f}: robCAGR={rob:+.3%} Δvs_baseline={d:+.2f}pp")
    # plateau: 採用域(band0 × w0.20)近傍で符号維持
    neighborhood = [band_delta_w02[0.0], band_delta_w02[0.001],
                    w_delta_b0[0.15], w_delta_b0[0.20], w_delta_b0[0.25]]
    plateau_robust = all(x > 0 for x in neighborhood)
    print(f"  近傍(band0/0.001 × w0.15/0.20/0.25) 全正? plateau_robust={plateau_robust}")

    # =========================================================
    # (3) oos_survives: IS(<=2021)で構成決定→レバ固定→OOS(2022-)素検証
    # =========================================================
    print("\n=== (3) oos_survives (IS<=2021 で較正, OOS 2022- 素適用) ===")
    SPLIT = pd.Timestamp("2022-01-01", tz="UTC")

    def is_oos_delta(rj_variant, w):
        # both baseline と long-only variant を同一 IS/OOS で比較
        common, rc, rjj = align(rj_variant)
        idx = pd.DatetimeIndex(common)
        is_mask = idx < SPLIT
        oos_mask = idx >= SPLIT
        # also baseline both on same grid
        common_bb = rc_full.index.intersection(rj_both.index)
        # restrict to identical common for fairness: use intersection of both books' grids
        gcommon = common.intersection(common_bb)
        gidx = pd.DatetimeIndex(gcommon)
        rc_g = rc_full.reindex(gcommon).fillna(0.0).values
        rv_g = rj_variant.reindex(gcommon).fillna(0.0).values
        rb_g = rj_both.reindex(gcommon).fillna(0.0).values
        is_m = gidx < SPLIT
        oos_m = gidx >= SPLIT

        def oos_cagr(rj_g):
            blend = (1 - w) * rc_g + w * rj_g
            # IS でレバ較正
            L_is, _ = lever_to_p95(blend[is_m], 0.20)
            # OOS に同じレバ適用
            return cagr_of(blend[oos_m] * L_is, gidx[oos_m]), L_is

        cagr_v, Lv = oos_cagr(rv_g)
        cagr_b, Lb_ = oos_cagr(rb_g)
        # champion 単独 OOS も
        Lc_is, _ = lever_to_p95(rc_g[is_m], 0.20)
        cagr_c = cagr_of(rc_g[oos_m] * Lc_is, gidx[oos_m])
        return dict(oos_champ=cagr_c, oos_base=cagr_b, oos_variant=cagr_v,
                    delta_oos=(cagr_v - cagr_b) * 100, n_oos=int(oos_m.sum()))

    oos_res = {}
    print(f"  {'band':>6} {'OOS_champ':>10} {'OOS_base(both)':>15} {'OOS_long':>10} {'Δoos_pp':>9} {'n_oos':>6}")
    for b in [0.0, 0.001, 0.002]:
        r = is_oos_delta(rj_long[b], 0.20)
        oos_res[b] = r
        print(f"  {b:>6.3f} {r['oos_champ']:>+10.2%} {r['oos_base']:>+15.2%} "
              f"{r['oos_variant']:>+10.2%} {r['delta_oos']:>+9.2f} {r['n_oos']:>6}")
    oos_survives = oos_res[0.0]["delta_oos"] > 0

    # =========================================================
    # (4) seed_stable: 5 seed で baseline比 delta(long band0 w0.2)
    # =========================================================
    print("\n=== (4) seed_stable (long band0 w0.2 の baseline比 delta, 5 seed) ===")
    common0, rc0, rj0 = align(rj_long[0.0])
    common_bb, rcb_, rjb_ = align(rj_both)
    gcommon = common0.intersection(common_bb)
    rc_g = rc_full.reindex(gcommon).fillna(0.0).values
    rl_g = rj_long[0.0].reindex(gcommon).fillna(0.0).values
    rb_g = rj_both.reindex(gcommon).fillna(0.0).values
    seed_deltas = []
    for sd in [0, 1, 2, 3, 4]:
        bl_long = 0.80 * rc_g + 0.20 * rl_g
        bl_base = 0.80 * rc_g + 0.20 * rb_g
        Ll, _ = lever_to_p95(bl_long, 0.20, seed=sd)
        Lb2, _ = lever_to_p95(bl_base, 0.20, seed=sd)
        c_long = cagr_of(bl_long * Ll, gcommon)
        c_base = cagr_of(bl_base * Lb2, gcommon)
        d = (c_long - c_base) * 100
        seed_deltas.append(d)
        print(f"  seed={sd}: long robCAGR={c_long:+.3%} base robCAGR={c_base:+.3%} Δ={d:+.2f}pp")
    seed_stable = all(d > 0 for d in seed_deltas)

    # =========================================================
    # 総合 delta_confirmed_pp = 採用域(band0 w0.2)の保守的 delta
    # robust(p95) baseline比, seed平均, OOS の最小を取る(最も保守的)
    # =========================================================
    robust_delta_w02_b0 = band_delta_w02[0.0]  # = w_delta_b0[0.20]
    seed_mean = float(np.mean(seed_deltas))
    oos_delta = oos_res[0.0]["delta_oos"]
    # confirmed = 採用域の robust delta だが OOS と seed が支えるか
    # 保守的に: full robust delta と OOS delta の小さい方(OOSが生存の証なら採用)
    delta_confirmed = min(robust_delta_w02_b0, oos_delta) if oos_survives else min(0.0, oos_delta)

    print("\n=== SUMMARY_JSON ===")
    summary = {
        "baseline_robCAGR": round(baseline_robCAGR, 5),
        "adopt_zone": "long band0 w0.20",
        "robust_delta_w02_b0_pp": round(robust_delta_w02_b0, 3),
        # (1) leverage偽装
        "emp_delta_w02_b0_pp": round(longvar[0.0]["d_emp"], 3),
        "emp_dd_base": round(b_base["dd_emp"], 4),
        "emp_dd_long_b0": round(longvar[0.0]["dd_emp"], 4),
        "dd_worse_b0": bool(longvar[0.0]["dd_worse"]),
        "p95_worse_b0": bool(longvar[0.0]["p95_worse"]),
        "same_tail_disguise": bool(longvar[0.0]["d_emp"] > 0 and
                                   (longvar[0.0]["dd_worse"] or longvar[0.0]["p95_worse"])),
        # (2) plateau
        "band_delta_w02_pp": {str(b): round(band_delta_w02[b], 3) for b in band_delta_w02},
        "w_delta_b0_pp": {str(w): round(w_delta_b0[w], 3) for w in w_delta_b0},
        "plateau_robust": bool(plateau_robust),
        # (3) oos
        "oos_delta_b0_pp": round(oos_delta, 3),
        "oos_champ_cagr": round(oos_res[0.0]["oos_champ"], 4),
        "oos_base_cagr": round(oos_res[0.0]["oos_base"], 4),
        "oos_long_cagr": round(oos_res[0.0]["oos_variant"], 4),
        "oos_survives": bool(oos_survives),
        # (4) seed
        "seed_deltas_pp": [round(d, 3) for d in seed_deltas],
        "seed_mean_pp": round(seed_mean, 3),
        "seed_stable": bool(seed_stable),
        # confirmed
        "delta_confirmed_pp": round(delta_confirmed, 3),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
