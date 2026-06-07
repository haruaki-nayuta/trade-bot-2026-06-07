"""敵対的検証: breakout_trend H4 short e80/x20/t200 overlay が本当にチャンピオンを底上げするか反証。

1. ヘッジ頑健性:
   (a) bleed閾値 -3%/-5%/-8% で mean_in_bleed が依然プラスか
   (b) IS/OOS 両方プラスか(再確認)
   (c) パラメータ近傍(entry/exit/trend)で mean_in_bleed が崩れないか
   (d) 2022年(USDラリー)を窓から除外しても mean_in_bleed がプラスか
2. 統合DD低weight細粒度(w=0.05..0.20)で champion +21.6% を超える点があるか確定。

実行: uv run python exp_breakout_adversarial.py
"""
from __future__ import annotations

import importlib
import itertools

import numpy as np
import pandas as pd

import bleed_lab as bl
import mm_lab as mm

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def cond_excl_year(cand_monthly, bleed_mask, excl_years):
    """指定年を窓から除外した mean_in_bleed と窓内合計。"""
    keep = pd.Series([p.year not in excl_years for p in bleed_mask.index], index=bleed_mask.index)
    m2 = bleed_mask & keep
    s = cand_monthly.reindex(m2.index).fillna(0.0)
    inb = s[m2.values]
    return {
        "n": int(m2.sum()),
        "mean": float(inb.mean()) if len(inb) else float("nan"),
        "total": float(inb.sum()),
        "wr": float((inb > 0).mean()) if len(inb) else float("nan"),
    }


def main():
    eqm, eqr, pool, closes = bl.champion_mtm(max_pos=8)

    BEST = {"entry": 80, "exit": 20, "trend": 200}
    mp_best = bl.strategy_monthly_pnl("breakout_trend", params=BEST, side="short", tf="H4")

    # ===== (a) bleed閾値感度 =====
    print("=== (a) bleed閾値感度: best config の mean_in_bleed (短) ===")
    for thr in [0.03, 0.05, 0.08]:
        mask, dd = bl.bleed_mask_monthly(eqm, thresh=thr)
        sc = bl.conditional_score(mp_best, mask)
        print(f"  thr=-{thr:.0%}  n_bleed={sc['n_bleed_months']:>2d}  "
              f"mean_bleed={sc['mean_in_bleed']:+.1f}  mean_norm={sc['mean_normal']:+.1f}  "
              f"edge={sc['hedge_edge']:+.1f}  IS={sc['mean_in_bleed_IS']:+.1f}  OOS={sc['mean_in_bleed_OOS']:+.1f}  "
              f"wr={sc['winrate_in_bleed']:.0%}")

    # 標準 -5% マスクで以降を実施
    mask, dd = bl.bleed_mask_monthly(eqm, thresh=0.05)

    # ===== (d) 2022除外 =====
    print("\n=== (d) 年除外感度: best config の窓内貢献 ===")
    print(f"  全窓        : {cond_excl_year(mp_best, mask, set())}")
    print(f"  2022除外    : {cond_excl_year(mp_best, mask, {2022})}")
    print(f"  2021+22除外 : {cond_excl_year(mp_best, mask, {2021, 2022})}")
    print(f"  2022+23除外 : {cond_excl_year(mp_best, mask, {2022, 2023})}")
    # 窓の年別分布
    yr = pd.Series([p.year for p in mask.index[mask.values]])
    print(f"  失血窓の年別月数: {yr.value_counts().sort_index().to_dict()}")
    # 窓内 overlay 月次PnL を年別合計
    s = mp_best.reindex(mask.index).fillna(0.0)
    bdf = pd.DataFrame({"year": [p.year for p in mask.index], "pnl": s.values, "bleed": mask.values})
    bdf = bdf[bdf["bleed"]]
    print("  窓内 overlay 年別合計PnL:")
    print(bdf.groupby("year")["pnl"].agg(["sum", "count"]).round(0).to_string())

    # ===== (c) パラメータ近傍 =====
    print("\n=== (c) パラメータ近傍: mean_in_bleed / IS / OOS (-5%窓) ===")
    grid_e = [55, 70, 80, 90]
    grid_x = [10, 20, 30]
    grid_t = [150, 200, 250]
    rows = []
    for e, x, t in itertools.product(grid_e, grid_x, grid_t):
        if x >= e:
            continue
        try:
            mp = bl.strategy_monthly_pnl("breakout_trend",
                                         params={"entry": e, "exit": x, "trend": t},
                                         side="short", tf="H4")
        except Exception:  # noqa: BLE001
            continue
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({"cfg": f"e{e}/x{x}/t{t}", "mean_bleed": sc["mean_in_bleed"],
                     "IS": sc["mean_in_bleed_IS"], "OOS": sc["mean_in_bleed_OOS"],
                     "edge": sc["hedge_edge"], "wr": sc["winrate_in_bleed"]})
    nb = pd.DataFrame(rows)
    nb["persistent"] = (nb["IS"] > 0) & (nb["OOS"] > 0)
    print(nb.round(1).to_string(index=False))
    print(f"\n  近傍 {len(nb)} 構成中 mean_bleed>0: {(nb['mean_bleed']>0).sum()} / IS&OOS両プラス: {nb['persistent'].sum()}")

    # ===== 統合DD 低weight 細粒度 =====
    print("\n=== 統合DD 低weight 細粒度 (champ単独基準 +21.6% / Sharpe1.21 / p95 -28.7%) ===")
    bt = importlib.import_module("strategies.breakout_trend")
    overlay = mm.build_pool_for(bt, BEST, side="short", tag="bt_e80x20t200_S")
    for w in [0.05, 0.10, 0.15, 0.20]:
        r = bl.integrated_dd_test(overlay, overlay_weight=w, max_pos=8)
        flag = "  <== beats champ" if r["cagr"] > 0.216 else ""
        print(f"  w={w:>4.2f}  k={r['k']:.3f}  CAGR={r['cagr']:+.1%}  Sharpe={r['sharpe']:.2f}  "
              f"maxDD={r['maxdd_mtm']:.1%}  p95={r['boot_p95']:.1%}  プラス年={r['pos_year_rate']:.0%}  "
              f"worstYr={r['worst_year']:+.1%}{flag}")


if __name__ == "__main__":
    main()
