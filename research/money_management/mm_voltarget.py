"""資金管理: ボラティリティ・ターゲティング — チャンピオンv2 の口座レベル・サイジング。

狙い: 口座の直近実現ボラ recent_vol に反比例して建玉を調整し、高ボラ期(塩漬けが嵩む
2022 等)に自動で軽く、低ボラ期に厚く張る。同じ最大DD較正の下で平均レバを上げ CAGR を伸ばせるか。

サイジング:
    alloc = equity_real * (k / recent_vol) / max_pos
    上限 clip: alloc <= cap * (k * TARGET_VOL_REF / recent_vol を介さず) ... ※下記の線形性に注意

線形性(重要): make_sizing(k) は総建玉を k に対して厳密に線形でなければ calibrate の二分探索
(|MtM DD| が k に単調増加)の前提が壊れる。vol_target は recent_vol で割るだけなので
未clip部分は k に線形。clip 上限も k に比例させる(ceiling = cap_base * k * equity / max_pos)
ことで、clip 後も全体が k で線形にスケールする。すなわち
    alloc(k) = k * [ equity/max_pos * min( 1/recent_vol , cap_base ) ]
            (= k * shape(ctx)) という形にし、shape は k 非依存。これで完全線形。

recent_vol が nan の立ち上がり期は固定 weight でフォールバック: alloc = k * equity/max_pos
(= cap_base を 1.0 とした基準張り)。

スイープ: vol_win ∈ {60,120,240}(simulate へ渡す), cap_base ∈ {2,3,5}。
各構成を 20%DD 較正 → CAGR 比較。最良を n_boot=1500 で最終評価。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm

# vol を正規化する基準(年率)。recent_vol/REF が ~1 付近になるようにして cap_base の意味を
# 「基準張りの何倍まで許すか」に揃える。較正 k が吸収するので絶対値は重要でない(形だけ固定)。
TARGET_VOL_REF = 0.10


def make_voltarget(cap_base: float):
    """vol ターゲティングの make_sizing(k) を返す。

    shape(ctx) = equity/max_pos * min( TARGET_VOL_REF/recent_vol , cap_base )
    alloc(k)   = k * shape(ctx)            # k に完全線形
    立ち上がり(recent_vol=nan): shape = equity/max_pos * 1.0(基準張り)
    """
    def make_sizing(k: float):
        def sizing(ctx):
            base = ctx["equity_real"] / ctx["max_pos"]
            rv = ctx["recent_vol"]
            if rv is None or not np.isfinite(rv) or rv <= 0:
                mult = 1.0  # 立ち上がりフォールバック(固定 weight)
            else:
                mult = min(TARGET_VOL_REF / rv, cap_base)
            return k * base * mult
        return sizing
    return make_sizing


def run_sweep():
    instruments = mm.default_instruments()
    pool = mm.build_pool(instruments=instruments)
    closes = mm.load_closes(instruments=instruments)
    print(f"トレード総数 {len(pool)} / グリッド {len(closes)}本 / 対象 {len(instruments)}\n")

    print("=== ベースライン(固定比率)再掲 ===")
    eqm, eqr, info = mm.simulate(pool, closes, mm.fixed_fractional(1.0, 6))
    # ベースラインは make_sizing で較正して再現
    def base_make(k):
        return mm.fixed_fractional(k, 6)
    rb = mm.evaluate_method("fixed_fractional", pool, closes, base_make, n_boot=400)
    print(f"  k={rb['k']:.2f}x CAGR={rb['cagr']:+.2%} DD_mtm={rb['maxdd_mtm']:.2%} "
          f"Sharpe={rb['sharpe']:.2f} プラス年={rb['pos_year_rate']:.0%} "
          f"p95={rb['boot_p95']:.1%} OOS_CAGR={rb.get('oos_cagr',float('nan')):+.2%} "
          f"OOS_DD={rb.get('oos_maxdd_mtm',float('nan')):.2%}\n")

    print("=== vol ターゲティング スイープ(20%DD 較正 → CAGR) ===")
    print("  vol_win  cap   k       CAGR      DD_mtm   Sharpe  プラス年  p95     OOS_CAGR  OOS_DD")
    results = []
    for vol_win in [60, 120, 240]:
        for cap in [2.0, 3.0, 5.0]:
            mk = make_voltarget(cap)
            # vol_win は simulate 経由で recent_vol の窓を決める。evaluate_method は
            # max_pos しか転送しないので、vol_win を固定した薄いラッパで calibrate/simulate を回す。
            r = evaluate_with_volwin(f"vt_w{vol_win}_c{int(cap)}", pool, closes, mk,
                                     vol_win=vol_win, n_boot=400)
            results.append((vol_win, cap, r))
            print(f"  {vol_win:>5d}  {cap:>3.0f}  {r['k']:>5.2f}  {r['cagr']:>+7.2%}  "
                  f"{r['maxdd_mtm']:>7.2%}  {r['sharpe']:>5.2f}  {r['pos_year_rate']:>6.0%}  "
                  f"{r['boot_p95']:>6.1%}  {r.get('oos_cagr',float('nan')):>+7.2%}  "
                  f"{r.get('oos_maxdd_mtm',float('nan')):>7.2%}")

    best = max(results, key=lambda t: t[2]["cagr"])
    print(f"\n最良: vol_win={best[0]} cap={best[1]} CAGR={best[2]['cagr']:+.2%}")
    return results, best, pool, closes


# evaluate_method は vol_win を転送しないので、vol_win を注入した版を用意。
def _calibrate_vw(pool, closes, make_sizing, target_dd, max_pos, vol_win,
                  lo=0.02, hi=40.0, iters=26):
    def dd_of(k):
        eqm, _, _ = mm.simulate(pool, closes, make_sizing(k), max_pos=max_pos, vol_win=vol_win)
        return abs(mm._max_dd(eqm))
    dd_hi = dd_of(hi)
    if dd_hi <= target_dd:
        eqm, eqr, info = mm.simulate(pool, closes, make_sizing(hi), max_pos=max_pos, vol_win=vol_win)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        if dd_of(mid) > target_dd:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = mm.simulate(pool, closes, make_sizing(lo), max_pos=max_pos, vol_win=vol_win)
    return lo, eqm, eqr, info


def evaluate_with_volwin(name, pool, closes, make_sizing, *, vol_win=120,
                         target_dd=0.20, max_pos=6, tf="H4",
                         oos_start="2022-01-01", n_boot=1500):
    k, eqm, eqr, info = _calibrate_vw(pool, closes, make_sizing, target_dd, max_pos, vol_win)
    s = mm.stats(eqm, eqr, info, tf=tf)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)

    is_pool = pool[pool["entry"] < oos_start].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= oos_start].reset_index(drop=True)
    is_closes = closes[closes.index < oos_start]
    oos_closes = closes[closes.index >= oos_start]
    oos = {}
    try:
        k_is, *_ = _calibrate_vw(is_pool, is_closes, make_sizing, target_dd, max_pos, vol_win)
        eqm_o, eqr_o, info_o = mm.simulate(oos_pool, oos_closes, make_sizing(k_is),
                                           max_pos=max_pos, vol_win=vol_win)
        so = mm.stats(eqm_o, eqr_o, info_o, tf=tf)
        oos = {"k_is": k_is, "oos_cagr": so["cagr"], "oos_maxdd_mtm": so["maxdd_mtm"],
               "oos_pos_year": so["pos_year_rate"], "oos_sharpe": so["sharpe"]}
    except Exception as e:  # noqa: BLE001
        oos = {"error": str(e)}

    return {
        "method": name, "k": k,
        "cagr": s["cagr"], "total_return": s["total_return"],
        "maxdd_mtm": s["maxdd_mtm"], "maxdd_real": s["maxdd_real"],
        "sharpe": s["sharpe"], "sortino": s["sortino"],
        "pos_year_rate": s["pos_year_rate"], "worst_year": s["worst_year"],
        "boot_p95": bs["p95"], "boot_p99": bs["p99"], "boot_worst": bs["worst"],
        "max_conc": s["max_conc"], "n_taken": s["n_taken"], "skipped": s["skipped"],
        **oos,
    }


if __name__ == "__main__":
    run_sweep()
