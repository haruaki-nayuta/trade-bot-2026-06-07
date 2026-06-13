"""改善B: モメンタムbookを USDJPY単独 → USDJPY + GBPJPY の2通貨bookに拡張。

exp60 の枠組みを完全踏襲し、champion 側は一切いじらず(同じ build_pool_d1 / champion_sizing /
calibrate_robust)、**モメンタム book だけ差し替えて** baseline(USDJPY単独 w=0.2)との増分を測る。

2通貨bookの構成:
  各通貨 i のモメンタム日次リターン rj_i を、まず単独で p95=20% に再較正(USDJPYと同じ手順)。
  book内配分 g (GBPJPY のウェイト) で book_ret = (1-g)*rj_usdjpy + g*rj_gbpjpy。
  → これを「モメンタム book の日次リターン」として champion と w で合成、合成を p95=20% に再較正。

  g=0.0  → USDJPY単独(= exp60 baseline と一致するはず。健全性チェック)
  g=0.3/0.5 → 2通貨。plateau を見る。

検証規律:
  (1) leverage偽装でない: 合成 boot p95 が champion単独 p95 を悪化させない(再較正で常に20%に揃う設計だが
      念のため p95_worse フラグも出す。加えて経験的maxDDが baseline比で悪化しないか)。
  (2) plateau: g={0,0.3,0.5} で delta の符号が維持されるか。
  (3) increment は baseline(USDJPY lb24 w=0.2)との差で報告。

NET(通常スプレッド)。tsmom 両建て lookback24 band0.0。GBPJPY は uni 合成クロス(GBPUSD*USDJPY)。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mm_lab as mm
import mm_production as mp
from fxlab import universe as uni
from strategies import tsmom

# exp60 のヘルパをそのまま再利用(apples-to-apples)
from research.experiments.exp60_twobook_robust import (
    cagr_of, lever_to_p95, to_daily_ret, worst_year,
)

LB = 24
BAND = 0.0


def momentum_daily_ret(instrument: str) -> pd.Series:
    """単一通貨の tsmom(lb24,band0) を p95=20% に較正し、日次リターン(p95=20%レバ済)を返す。

    exp60 の BOOK B と同一手順(fixed-fractional, max_pos=1, calibrate_robust target_dd=0.20)。
    クロス(GBPJPY)も同じ run() 経路で扱える(close系戦略)。
    """
    pool = mm.build_pool_for(
        tsmom, {"lookback": LB, "band": BAND}, tf="H1",
        instruments=[instrument], tag=f"tsmom_{instrument}_lb{LB}", side="both",
        cache=False)
    closes = pd.DataFrame({instrument: uni.instrument_close(instrument, "H1")}).sort_index().ffill()

    def mk(k):
        return lambda ctx: ctx["equity_real"] * k

    k, eqm, eqr, info, p95cal = mm.calibrate_robust(
        pool, closes, mk, target_dd=0.20, max_pos=1, n_boot=800)
    dd = abs(float((eqm / eqm.cummax() - 1.0).min()))
    cagr = (eqm.iloc[-1] / 10000.0) ** (1 / ((eqm.index[-1] - eqm.index[0]).days / 365.25)) - 1
    print(f"  [{instrument}] pool={len(pool)} k={k:.3f} CAGR={cagr:+.2%} maxDD_mtm={-dd:+.1%} "
          f"long={int((pool['dir']>0).sum())} short={int((pool['dir']<0).sum())}")
    return to_daily_ret(eqm)


def empirical_maxdd(daily_ret: np.ndarray) -> float:
    path = np.cumprod(1.0 + np.nan_to_num(daily_ret))
    peak = np.maximum.accumulate(path)
    return float((path / peak - 1.0).min())


def main():
    print("=== exp61: 改善B GBPJPY 2通貨モメンタムbook (champion固定・apples-to-apples) ===\n")

    # ---------- BOOK A: champion d1 (H4) — exp60 と完全同一 ----------
    pool_c = mp.build_pool_d1()
    closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    k_c, eqm_c, eqr_c, info_c, p95_c_cal = mm.calibrate_robust(
        pool_c, closes_c, mk_c, target_dd=0.20, max_pos=8, n_boot=800)
    bs_c = mm.bootstrap_maxdd(eqm_c, n_boot=1500)
    print(f"[A champion d1] k={k_c:.2f} boot_p95(H4)={bs_c['p95']:+.1%} trades={len(pool_c)}")

    # ---------- モメンタム各通貨を単独 p95=20% 較正 → 日次 ----------
    print("\n[B momentum books — 各通貨単独 p95=20% 較正]")
    rj_usd = momentum_daily_ret("USDJPY")
    rj_gbp = momentum_daily_ret("GBPJPY")

    # ---------- 日次グリッドそろえ(champion ∩ USDJPY ∩ GBPJPY) ----------
    rc = to_daily_ret(eqm_c)
    common = rc.index.intersection(rj_usd.index).intersection(rj_gbp.index)
    rc = rc.reindex(common).fillna(0.0)
    ru = rj_usd.reindex(common).fillna(0.0)
    rg = rj_gbp.reindex(common).fillna(0.0)
    print(f"\n日次共通グリッド: {len(common)}日 {common[0].date()}..{common[-1].date()}")
    corr_uc = float(np.corrcoef(rc.values, ru.values)[0, 1])
    corr_gc = float(np.corrcoef(rc.values, rg.values)[0, 1])
    corr_ug = float(np.corrcoef(ru.values, rg.values)[0, 1])
    print(f"corr champ-USDJPY={corr_uc:+.3f}  champ-GBPJPY={corr_gc:+.3f}  USDJPY-GBPJPY={corr_ug:+.3f}")

    # ---------- 基準: champion 単独 daily を p95=20% 再較正 ----------
    L_c, p95_c_d = lever_to_p95(rc.values, target=0.20, n_boot=2000, block=21)
    champ_robCAGR = cagr_of(rc.values * L_c, common)
    print(f"\n[基準0] champion 単独: L={L_c:.3f} p95={p95_c_d:+.1%} robCAGR={champ_robCAGR:+.2%}")

    # ---------- baseline: USDJPY単独 book を w=0.2 で合成(g=0) ----------
    def blend_and_lever(book_ret_arr, w):
        blend = (1 - w) * rc.values + w * book_ret_arr
        L, p95 = lever_to_p95(blend, target=0.20, n_boot=2000, block=21)
        levered = blend * L
        return dict(L=L, p95=p95,
                    robCAGR=cagr_of(levered, common),
                    worst_yr=worst_year(levered, common),
                    emp_maxdd=empirical_maxdd(levered))

    # baseline = USDJPY単独 (g=0) w=0.2。これを基準にΔを測る。
    base = blend_and_lever(ru.values, 0.2)
    baseline_robCAGR = base["robCAGR"]
    print(f"\n[基準1=baseline] USDJPY単独 book w=0.2: robCAGR={baseline_robCAGR:+.2%} "
          f"p95={base['p95']:+.1%} emp_maxDD={base['emp_maxdd']:+.1%} (vs champ {champ_robCAGR:+.2%})")

    # ---------- 2通貨 book: g(GBPJPY配分) × w(モメンタム配分) ----------
    print("\n=== 2通貨 book (USDJPY+GBPJPY) g×w スイープ ===")
    print(f"  {'g':>5} {'w':>5} {'L':>7} {'p95':>7} {'robCAGR':>9} {'Δvs_base_pp':>11} "
          f"{'Δvs_champ_pp':>12} {'worst_yr':>9} {'empDD':>7} {'p95_worse?':>10}")
    g_grid = [0.0, 0.3, 0.5]
    w_grid = [0.15, 0.20, 0.25, 0.30, 0.40]
    rows = []
    for g in g_grid:
        book = (1 - g) * ru.values + g * rg.values
        for w in w_grid:
            r = blend_and_lever(book, w)
            d_base = (r["robCAGR"] - baseline_robCAGR) * 100
            d_champ = (r["robCAGR"] - champ_robCAGR) * 100
            # leverage偽装: 再較正で p95 は 20% に揃う設計だが念のため + 経験的DDが baseline比悪化か
            p95_worse = r["p95"] > abs(p95_c_d) + 0.003
            emp_worse = r["emp_maxdd"] < base["emp_maxdd"] - 0.003  # より深い(負方向)
            print(f"  {g:>5.2f} {w:>5.2f} {r['L']:>7.3f} {r['p95']:>+7.1%} {r['robCAGR']:>+9.2%} "
                  f"{d_base:>+11.2f} {d_champ:>+12.2f} {r['worst_yr']:>+9.1%} "
                  f"{r['emp_maxdd']:>+7.1%} {str(p95_worse or emp_worse):>10}")
            rows.append(dict(g=g, w=w, L=r["L"], p95=r["p95"], robCAGR=r["robCAGR"],
                             d_base_pp=d_base, d_champ_pp=d_champ, worst_yr=r["worst_yr"],
                             emp_maxdd=r["emp_maxdd"], p95_worse=bool(p95_worse or emp_worse)))

    # g=0 健全性チェック(exp60 baseline と一致するか)
    g0w20 = next(r for r in rows if r["g"] == 0.0 and r["w"] == 0.20)
    print(f"\n[健全性] g=0,w=0.2 robCAGR={g0w20['robCAGR']:+.4%} "
          f"(exp60 baseline 21.85%と整合か)")

    # 2通貨(g>0)で baseline を超える最良
    cand = [r for r in rows if r["g"] > 0]
    best2 = max(cand, key=lambda r: r["robCAGR"])
    # 全体最良(g任意)
    best_all = max(rows, key=lambda r: r["robCAGR"])
    print(f"\n[BEST 2通貨 g>0] g={best2['g']:.2f} w={best2['w']:.2f} "
          f"robCAGR={best2['robCAGR']:+.2%} Δvs_base={best2['d_base_pp']:+.2f}pp "
          f"Δvs_champ={best2['d_champ_pp']:+.2f}pp p95_worse={best2['p95_worse']}")
    print(f"[BEST all] g={best_all['g']:.2f} w={best_all['w']:.2f} "
          f"robCAGR={best_all['robCAGR']:+.2%} Δvs_base={best_all['d_base_pp']:+.2f}pp")

    # plateau 判定: 同一 w(=0.2)で g={0,0.3,0.5} の Δvs_base 符号
    w20 = sorted([r for r in rows if r["w"] == 0.20], key=lambda r: r["g"])
    signs = [r["d_base_pp"] for r in w20]
    print(f"\n[plateau @w=0.2] g=0,0.3,0.5 のΔvs_base: {[round(s,3) for s in signs]}")

    print("\n=== SUMMARY_JSON ===")
    print(json.dumps({
        "champ_robCAGR": round(champ_robCAGR, 5),
        "baseline_usdjpy_w20_robCAGR": round(baseline_robCAGR, 5),
        "g0_w20_robCAGR": round(g0w20["robCAGR"], 5),
        "corr_champ_usdjpy": round(corr_uc, 4),
        "corr_champ_gbpjpy": round(corr_gc, 4),
        "corr_usdjpy_gbpjpy": round(corr_ug, 4),
        "best2_g": best2["g"], "best2_w": best2["w"],
        "best2_robCAGR": round(best2["robCAGR"], 5),
        "best2_d_base_pp": round(best2["d_base_pp"], 3),
        "best2_d_champ_pp": round(best2["d_champ_pp"], 3),
        "best2_p95_worse": best2["p95_worse"],
        "plateau_w20_dbase": [round(s, 3) for s in signs],
        "all_rows": [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}
                     for r in rows],
    }, indent=2))


if __name__ == "__main__":
    main()
