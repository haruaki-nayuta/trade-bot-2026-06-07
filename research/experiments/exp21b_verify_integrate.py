"""イテレーション21b: ベスト失血窓ヘッジ候補の (a)持続性の精査 (b)統合DDテスト(最終判定)。

exp21 で抽出した持続ヘッジ候補について:
  1) 失血窓PnLの年別内訳 — 2022一発でなく複数の失血年(2017/18/20/21/23)で稼ぐか
  2) integrated_dd_test — champion + overlay を1口座(MtM)統合し DD=20% 較正 → CAGR が
     champion単独 +21.6% を上回るか(=保険のドラッグを払ってなお失血窓DD低減でレバ余地が純増)
  overlay_weight を複数振り、保険の最適投入量を探す。

実行: uv run python exp21b_verify_integrate.py
"""

from __future__ import annotations

import warnings

import pandas as pd

import bleed_lab as bl
import mm_lab as mm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)


# exp21 の上位・持続ヘッジ候補(両側H4中心 + ADXゲート対比)
CANDIDATES = [
    ("adx_trend", {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 20}, "both", "H4"),
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "both", "H4"),
    ("ma_cross", {"fast": 30, "slow": 100}, "both", "H4"),
    ("ma_cross", {"fast": 20, "slow": 100}, "both", "H4"),
    ("adx_trend", {"fast": 20, "slow": 100, "adx_period": 14, "adx_th": 20}, "both", "H4"),
    ("adx_trend", {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "short", "H4"),
]


def bleed_pnl_by_year(mp: pd.Series, mask: pd.Series) -> pd.Series:
    """失血窓に該当する月の候補PnLを年別合計(=失血年ごとの貢献)。"""
    s = mp.reindex(mask.index).fillna(0.0)
    inb = s[mask.values]
    by_year = inb.groupby([p.year for p in inb.index]).sum()
    return by_year


def main():
    print("=== 失血窓マスク ===")
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)
    # 失血年(該当月>0の年)
    bleed_years = sorted({p.year for p in mask.index[mask.values]})
    print(f"失血年: {bleed_years}\n")

    print("=== 候補ごと: 失血窓PnLの年別内訳(持続性の精査)===")
    for name, params, side, tf in CANDIDATES:
        mp = bl.strategy_monthly_pnl(name, params=params, side=side, tf=tf)
        by = bleed_pnl_by_year(mp, mask)
        sc = bl.conditional_score(mp, mask)
        pstr = "_".join(f"{k}{v}" for k, v in params.items() if k in ("fast", "slow", "adx_th"))
        # 失血年のうち何年でプラスか
        bleed_yr_pnl = by[by.index.isin(bleed_years)]
        pos_yr = int((bleed_yr_pnl > 0).sum())
        n_yr = len(bleed_yr_pnl)
        # 2022を除いた失血窓平均(2022一発でないか)
        non2022 = mp.reindex(mask.index).fillna(0.0)[mask.values]
        non2022 = non2022[[p.year != 2022 for p in non2022.index]]
        print(f"\n{name} {pstr} side={side} {tf}: "
              f"mean_bleed={sc['mean_in_bleed']:.0f} IS={sc['mean_in_bleed_IS']:.0f} "
              f"OOS={sc['mean_in_bleed_OOS']:.0f} | 失血年プラス {pos_yr}/{n_yr} | "
              f"2022除外の窓内平均={non2022.mean():.0f}")
        print("   年別失血窓PnL: " + "  ".join(f"{y}:{v:+.0f}" for y, v in bleed_yr_pnl.items()))

    print("\n\n=== 統合DDテスト(最終判定) ===")
    print("基準: champion単独 z-size mp8 DD=20% → CAGR +21.6% / Sharpe 1.21 / 100%プラス年 / boot_p95 -28.5%")
    print("\n候補 × overlay_weight で integrated_dd_test(champion+overlay@20%DD)\n")
    print(f"{'候補':<46}{'w':>5}{'k':>7}{'CAGR':>9}{'maxDD':>9}{'Sharpe':>8}{'boot95':>9}{'+年率':>7}")
    print("-" * 100)

    for name, params, side, tf in CANDIDATES:
        ovl = mm.build_pool_for(__import__(f"strategies.{name}", fromlist=["x"]),
                                params, tf=tf, side=side,
                                tag=f"{name}_{'_'.join(str(v) for v in params.values())}_{side}")
        if ovl.empty:
            print(f"{name}: overlay空")
            continue
        pstr = "_".join(f"{k}{v}" for k, v in params.items() if k in ("fast", "slow", "adx_th"))
        label = f"{name} {pstr} {side}"
        for w in [0.3, 0.5, 0.75, 1.0]:
            r = bl.integrated_dd_test(ovl, overlay_weight=w, max_pos=8)
            flag = "  <<" if r["cagr"] > 0.216 else ""
            print(f"{label:<46}{w:>5.2f}{r['k']:>7.2f}{r['cagr']:>+9.1%}{r['maxdd_mtm']:>+9.1%}"
                  f"{r['sharpe']:>8.2f}{r['boot_p95']:>+9.1%}{r['pos_year_rate']:>6.0%}{flag}")


if __name__ == "__main__":
    main()
