"""trend_insurance ベスト構成の持続性検証 + 統合DDテスト。

(1) 失血窓を「年/クラスタ」別に分解し、ベスト構成が 2022以外の窓(2017/2018/2021/2023)でも
    稼ぐかを確認(=2022一発まぐれの排除)。
(2) er_hi の高原性(閾値を少しずらしても窓貢献が滑らかに保たれるか)。
(3) integrated_dd_test で champion+overlay@20%DD の CAGR がチャンピオン単独 +21.6% を上回るか(最終判定)。

実行: uv run python ti_verify.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm
import strategies.trend_insurance as ti

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 50)

BEST = {"er_win": 40, "er_hi": 0.45, "er_lo": 0.25, "ma_win": 100}
BEST_SIDE = "both"


def main():
    eqm, eqr, pool_c, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)

    mp = bl.strategy_monthly_pnl("trend_insurance", params=BEST, side=BEST_SIDE)
    s = mp.reindex(mask.index).fillna(0.0)

    print(f"=== ベスト構成 {BEST} side={BEST_SIDE} ===")
    print(f"単体通算PnL: ${mp.sum():,.0f}(負=保険コスト) / 失血窓 {int(mask.sum())}ヶ月\n")

    # (1) 失血窓の各月での貢献(年別)
    bleed_months = mask.index[mask.values]
    print(">>> 失血窓 各月での trend_insurance のPnL($)と年別合計:")
    tbl = pd.DataFrame({"dd": dd.reindex(bleed_months).values,
                        "ti_pnl": s.reindex(bleed_months).values},
                       index=[str(p) for p in bleed_months])
    print(tbl.round(1).to_string())
    by_year = pd.Series(s.reindex(bleed_months).values,
                        index=[p.year for p in bleed_months]).groupby(level=0).sum()
    cnt = pd.Series(1, index=[p.year for p in bleed_months]).groupby(level=0).sum()
    yr = pd.DataFrame({"窓月数": cnt, "TI窓内PnL合計": by_year.round(0)})
    yr["プラス"] = by_year > 0
    print("\n>>> 失血窓・年別の trend_insurance 貢献(2022以外でも稼ぐか):")
    print(yr.to_string())
    non2022 = by_year.drop(2022, errors="ignore")
    print(f"\n  2022を除く失血窓年の通算: ${non2022.sum():,.0f}  "
          f"(プラスの年 {int((non2022>0).sum())}/{len(non2022)})")

    # (2) er_hi 高原性
    print("\n>>> er_hi 高原性(er_win=40, ma_win=100, both で er_hi を振る):")
    for hi in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        p = dict(BEST); p["er_hi"] = hi
        m2 = bl.strategy_monthly_pnl("trend_insurance", params=p, side=BEST_SIDE)
        sc = bl.conditional_score(m2, mask)
        print(f"  er_hi={hi:.2f}  mean_in_bleed={sc['mean_in_bleed']:+7.1f}  "
              f"IS={sc['mean_in_bleed_IS']:+7.1f}  OOS={sc['mean_in_bleed_OOS']:+7.1f}  "
              f"hedge_edge={sc['hedge_edge']:+7.1f}  total_all={sc['total_all']:+9.0f}")

    # (3) 統合DDテスト — 最終判定
    print("\n>>> 統合DDテスト(champion + trend_insurance overlay, DD=20%較正)")
    print("    比較基準: チャンピオン単独 CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / 理論DD p95 -28.5%")
    ov = mm.build_pool_for(ti, BEST, tag="trendins_best", side=BEST_SIDE)
    print(f"    overlay トレード数: {len(ov)}")
    print("    weight   k     CAGR    maxDD(MtM)  Sharpe  ブートp95  プラス年")
    for w in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        if w == 0.0:
            # overlay 無し = champion 単独(参照)
            from mm_production import champion_sizing
            pool = mm.build_pool(); cl = mm.load_closes()
            mk = champion_sizing(pool, max_pos=8)
            k, em, er, info = mm.calibrate(pool, cl, mk, target_dd=0.20, max_pos=8)
            st = mm.stats(em, er, info); bs = mm.bootstrap_maxdd(em, n_boot=800)
            print(f"    {w:>4.2f}(単独) {k:>5.2f}  {st['cagr']:>+6.1%}  {st['maxdd_mtm']:>9.1%}  "
                  f"{st['sharpe']:>6.2f}  {bs['p95']:>7.1%}  {st['pos_year_rate']:>5.0%}")
            continue
        r = bl.integrated_dd_test(ov, overlay_weight=w, max_pos=8)
        print(f"    {w:>4.2f}     {r['k']:>5.2f}  {r['cagr']:>+6.1%}  {r['maxdd_mtm']:>9.1%}  "
              f"{r['sharpe']:>6.2f}  {r['boot_p95']:>7.1%}  {r['pos_year_rate']:>5.0%}")


if __name__ == "__main__":
    main()
