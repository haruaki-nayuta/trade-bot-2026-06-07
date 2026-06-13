"""厳密 two-book robust 評価: champion(H4 平均回帰) と USDJPY H1 lb24 tsmom を別ブックとして
それぞれ独立に robust(bootstrap p95=20%)較正し、資本配分 w で合成 → 合成を p95=20% に
再較正した時の robust CAGR が champion 単独を上回るか。

手順:
  (1) champion d1 pool を build_pool_d1 + champion_sizing(max_pos=8) で calibrate_robust(target_dd=0.20)
      → eqm_champ(MtM equity, H4 grid)。
  (2) USDJPY H1 lb24 tsmom を build_pool_for で pool化(固定比率サイジング)→ 単独 simulate
      → eqm_jpy(MtM equity, H1 grid)。固定比率サイジングを calibrate_robust(p95=20%)。
  (3) 両 MtM equity を **日次** にそろえてリターン化、w で合成、合成系列を日次ブロックブートストラップで
      p95=20% に再レバ → CAGR。champion比 ΔCAGR(pp)。
  (4) 合成の boot p95 が champion単独のp95より悪化しないか(レバ偽装でないか)。
  (5) 合成の最悪年。

NET(通常スプレッド)で評価。tsmom は両建て。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import universe as uni
from strategies import tsmom

TRADING_DAYS = 252


# ---- 日次ブロックブートストラップで p95 maxDD を出す(日次粒度) ----------
def daily_block_bootstrap_p95(daily_ret: np.ndarray, n_boot=2000, block=21, seed=0):
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
    return float(np.percentile(dds, 5))  # p95 = 5th percentile (most negative tail at 95% conf)


def lever_to_p95(daily_ret: np.ndarray, target=0.20, n_boot=2000, block=21, seed=0,
                 lo=0.05, hi=20.0, iters=30):
    """日次リターン系列をレバ L 倍して p95 maxDD == target に較正。L を二分探索。返り値 (L, p95@L)。"""
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


def cagr_of(daily_ret: np.ndarray, index: pd.DatetimeIndex) -> float:
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    years = (index[-1] - index[0]).days / 365.25
    final = path[-1]
    return (final ** (1 / years) - 1) if final > 0 else -1.0


def worst_year(daily_ret: np.ndarray, index: pd.DatetimeIndex) -> float:
    eq = pd.Series(np.cumprod(1.0 + np.nan_to_num(daily_ret)), index=index)
    yearly = eq.groupby(eq.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] - 1.0  # path starts at 1.0
    return float(yr.min())


def to_daily_ret(eqm: pd.Series) -> pd.Series:
    """MtM equity → 日次 last → 日次リターン。"""
    d = eqm.resample("1D").last().dropna()
    return d.pct_change().dropna()


def main():
    print("=== exp60: 厳密 two-book robust (champion + USDJPY H1 lb24 tsmom) ===\n")

    # ---------- BOOK A: champion d1 (H4) ----------
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    s_c = mm.stats(eqm_c, eqr_c, info_c)
    bs_c = mm.bootstrap_maxdd(eqm_c, n_boot=1500)
    print(f"[A champion d1] k={k_c:.2f} CAGR={s_c['cagr']:+.2%} maxDD_mtm={s_c['maxdd_mtm']:+.1%} "
          f"boot_p95(H4)={bs_c['p95']:+.1%} trades={len(pool_c)}")

    # ---------- BOOK B: USDJPY H1 lb24 tsmom (fixed-fractional, single instrument) ----------
    JPY = ["USDJPY"]
    pool_j = mm.build_pool_for(tsmom, {"lookback": 24, "band": 0.0}, tf="H1",
                               instruments=JPY, tag="tsmom_usdjpy_lb24", side="both",
                               cache=False)
    closes_j = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()
    print(f"[B tsmom] pool trades={len(pool_j)} closes={closes_j.shape} "
          f"long={int((pool_j['dir']>0).sum())} short={int((pool_j['dir']<0).sum())}")

    # tsmom は固定比率サイジング(max_pos=1 = 常に単一ポジ; tsmom はドテンで常時建玉)。
    # fixed_fractional(deploy=k, max_pos=1) → alloc = equity * k。calibrate_robust で k を p95=20% に。
    def mk_j(k):
        return lambda ctx: ctx["equity_real"] * k
    k_j, eqm_j, eqr_j, info_j, p95_j_cal = mm.calibrate_robust(
        pool_j, closes_j, mk_j, target_dd=0.20, max_pos=1, n_boot=800)
    # stats は H4 BARS で Sharpe を出すので H1 用に直接計算は省略、CAGR/DD のみ参照
    dd_j = abs(float((eqm_j / eqm_j.cummax() - 1.0).min()))
    bs_j = mm.bootstrap_maxdd(eqm_j, n_boot=1500)
    cagr_j_raw = (eqm_j.iloc[-1] ** (1/((eqm_j.index[-1]-eqm_j.index[0]).days/365.25)) - 1) \
        if eqm_j.iloc[-1] > 0 else -1.0
    # eqm_j は init=10000 始まり
    cagr_j = (eqm_j.iloc[-1] / 10000.0) ** (1/((eqm_j.index[-1]-eqm_j.index[0]).days/365.25)) - 1
    print(f"[B tsmom] k={k_j:.2f} CAGR={cagr_j:+.2%} maxDD_mtm={-dd_j:+.1%} "
          f"boot_p95(H1)={bs_j['p95']:+.1%}")

    # ---------- 日次にそろえる ----------
    rc = to_daily_ret(eqm_c)   # champion daily ret (already p95=20% levered)
    rj = to_daily_ret(eqm_j)   # tsmom daily ret (already p95=20% levered)
    common = rc.index.intersection(rj.index)
    rc = rc.reindex(common).fillna(0.0)
    rj = rj.reindex(common).fillna(0.0)
    print(f"\n日次共通グリッド: {len(common)}日 {common[0].date()}..{common[-1].date()}")
    corr = float(np.corrcoef(rc.values, rj.values)[0, 1])
    print(f"日次相関(champion vs tsmom): {corr:+.3f}")

    # champion 単独 daily を p95=20% に再較正(基準線; 日次粒度ブートで champ の robCAGR)
    L_c, p95_c_d = lever_to_p95(rc.values, target=0.20, n_boot=2000, block=21)
    champ_robCAGR = cagr_of(rc.values * L_c, common)
    print(f"\n[基準] champion 単独 daily 再較正: L={L_c:.3f} p95={p95_c_d:+.1%} "
          f"robCAGR={champ_robCAGR:+.2%}")

    # ---------- w で合成 → 再較正 ----------
    print("\n=== 合成(w=tsmom配分) → 日次 p95=20% 再較正 ===")
    print(f"  {'w':>5} {'L':>7} {'p95':>8} {'robCAGR':>9} {'ΔCAGR_pp':>9} {'worst_yr':>9} {'p95_worse?':>10}")
    results = []
    for w in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        blend = (1 - w) * rc.values + w * rj.values
        L, p95 = lever_to_p95(blend, target=0.20, n_boot=2000, block=21)
        levered = blend * L
        cagr = cagr_of(levered, common)
        wy = worst_year(levered, common)
        drob = (cagr - champ_robCAGR) * 100
        p95_worse = p95 > abs(p95_c_d) + 0.003  # 0.3pp tolerance
        print(f"  {w:>5.2f} {L:>7.3f} {p95:>+8.1%} {cagr:>+9.2%} {drob:>+9.2f} {wy:>+9.1%} "
              f"{str(p95_worse):>10}")
        results.append(dict(w=w, L=L, p95=p95, cagr=cagr, drob=drob, worst_yr=wy,
                            p95_worse=p95_worse))

    # ベスト(robCAGR 最大の w>0)
    cand = [r for r in results if r["w"] > 0]
    best = max(cand, key=lambda r: r["cagr"])
    print(f"\n[BEST w={best['w']:.2f}] robCAGR={best['cagr']:+.2%} "
          f"ΔCAGR={best['drob']:+.2f}pp p95={best['p95']:+.1%} "
          f"(champ p95={p95_c_d:+.1%}) worst_yr={best['worst_yr']:+.1%} "
          f"p95_worsens={best['p95_worse']}")

    # サマリ行(機械可読)
    print("\n=== SUMMARY_JSON ===")
    import json
    print(json.dumps({
        "champion_robCAGR": round(champ_robCAGR, 5),
        "champ_p95_daily": round(p95_c_d, 5),
        "best_w": best["w"],
        "combined_robCAGR_best": round(best["cagr"], 5),
        "drob_pp": round(best["drob"], 3),
        "best_p95": round(best["p95"], 5),
        "p95_worsens": bool(best["p95_worse"]),
        "worst_year_combined": round(best["worst_yr"], 5),
        "daily_corr": round(corr, 4),
        "champ_H4_p95": round(bs_c["p95"], 5),
        "tsmom_H1_p95": round(bs_j["p95"], 5),
        "all_w": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                  for r in results],
    }, indent=2))


if __name__ == "__main__":
    main()
