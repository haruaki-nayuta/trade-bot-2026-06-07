"""敵対的検証: vol_breakout(bb_breakout p60 m1.5 short H4)の失血窓ヘッジが本物か反証する。

角度:
 (a) bleed閾値 -3%/-5%/-8% でも失血窓貢献がプラスか
 (b) IS(2017-2021)/OOS(2022-)両方でプラスか(=2022一発でないか)
 (c) パラメータ近傍(period 40/60/80, mult 1.5/2.0/2.5)で滑らかに正か
 (d) 2022を除いた失血窓(2021/2023/2018)でもプラスか(=核心の反証)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import bleed_lab as bl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def conditional_by_mask(mp: pd.Series, mask: pd.Series):
    """与えた失血窓マスクで mean_in_bleed / total / winrate を返す。"""
    s = mp.reindex(mask.index).fillna(0.0)
    inb = s[mask.values]
    out = s[~mask.values]
    return {
        "n_bleed": int(mask.sum()),
        "mean_in_bleed": float(inb.mean()) if len(inb) else float("nan"),
        "mean_normal": float(out.mean()) if len(out) else float("nan"),
        "total_in_bleed": float(inb.sum()),
        "wr_bleed": float((inb > 0).mean()) if len(inb) else float("nan"),
    }


def main():
    eqm, eqr, pool, closes = bl.champion_mtm()

    # === (a) bleed閾値変動 ===
    print("=" * 80)
    print("(a) bleed閾値変動: bb p60 m1.5 short H4")
    print("=" * 80)
    mp_best = bl.strategy_monthly_pnl("bb_breakout", params={"period": 60, "mult": 1.5}, side="short", tf="H4")
    for thr in [0.03, 0.05, 0.08]:
        mask, dd = bl.bleed_mask_monthly(eqm, thresh=thr)
        sc = conditional_by_mask(mp_best, mask)
        print(f"  thr=-{thr:.0%}: n_bleed={sc['n_bleed']:2d}  mean_in_bleed={sc['mean_in_bleed']:+8.1f}  "
              f"mean_normal={sc['mean_normal']:+8.1f}  wr_bleed={sc['wr_bleed']:.2f}  tot_bleed={sc['total_in_bleed']:+9.0f}")

    # 基準マスク(-5%)
    mask5, dd5 = bl.bleed_mask_monthly(eqm, thresh=0.05)

    # === (b) IS/OOS 月次内訳 + 年別 ===
    print("\n" + "=" * 80)
    print("(b) IS(2017-2021)/OOS(2022-) と 失血窓の年別内訳")
    print("=" * 80)
    bleed_months = [p for p, b in mask5.items() if b]
    print(f"  失血窓 {len(bleed_months)}ヶ月: {[str(p) for p in bleed_months]}")
    by_year = pd.Series(mask5.values, index=[p.year for p in mask5.index]).groupby(level=0).sum()
    print(f"  年別失血窓月数:\n{by_year[by_year>0].to_string()}")

    # 各失血窓月での候補PnL
    s = mp_best.reindex(mask5.index).fillna(0.0)
    print("\n  各失血窓月での候補(bb p60m1.5 short)PnL:")
    rows = []
    for p in bleed_months:
        rows.append((str(p), p.year, float(s.loc[p])))
    bd = pd.DataFrame(rows, columns=["month", "year", "pnl"])
    print(bd.to_string(index=False))

    print("\n  年別 失血窓内 候補PnL合計:")
    print(bd.groupby("year")["pnl"].agg(["sum", "mean", "count"]).round(1).to_string())

    # === (d) 核心: 2022を除いた失血窓でもプラスか ===
    print("\n" + "=" * 80)
    print("(d) 核心反証: 2022を除いた失血窓だけでプラスか")
    print("=" * 80)
    mask_no2022 = mask5.copy()
    for p in mask_no2022.index:
        if p.year == 2022:
            mask_no2022.loc[p] = False
    sc_no22 = conditional_by_mask(mp_best, mask_no2022)
    print(f"  2022除外失血窓 n={sc_no22['n_bleed']}: mean_in_bleed={sc_no22['mean_in_bleed']:+.1f}  "
          f"total={sc_no22['total_in_bleed']:+.0f}  wr={sc_no22['wr_bleed']:.2f}")

    # 各年だけの失血窓
    for yr in [2018, 2021, 2023]:
        mask_yr = mask5.copy()
        for p in mask_yr.index:
            mask_yr.loc[p] = (mask5.loc[p] and p.year == yr)
        if mask_yr.sum() == 0:
            print(f"  {yr}年のみ: 失血窓なし")
            continue
        sc_yr = conditional_by_mask(mp_best, mask_yr)
        print(f"  {yr}年のみ失血窓 n={sc_yr['n_bleed']}: mean_in_bleed={sc_yr['mean_in_bleed']:+.1f}  "
              f"total={sc_yr['total_in_bleed']:+.0f}  wr={sc_yr['wr_bleed']:.2f}")

    # === (c) パラメータ近傍 ===
    print("\n" + "=" * 80)
    print("(c) パラメータ近傍(period x mult, short H4)— mean_in_bleed と IS/OOS")
    print("=" * 80)
    rows = []
    for period in [40, 60, 80]:
        for mult in [1.5, 2.0, 2.5]:
            mp = bl.strategy_monthly_pnl("bb_breakout", params={"period": period, "mult": mult}, side="short", tf="H4")
            if len(mp) == 0:
                continue
            sc = bl.conditional_score(mp, mask5)
            # 2022除外
            sc_no22 = conditional_by_mask(mp, mask_no2022)
            rows.append({
                "period": period, "mult": mult,
                "mean_in_bleed": sc["mean_in_bleed"],
                "mIS": sc["mean_in_bleed_IS"], "mOOS": sc["mean_in_bleed_OOS"],
                "wr_bleed": sc["winrate_in_bleed"],
                "no2022_mean": sc_no22["mean_in_bleed"],
                "total_all": sc["total_all"],
            })
    nb = pd.DataFrame(rows)
    print(nb.round(1).to_string(index=False))
    print("\n  -> period 60周辺 / mult 1.5-2.0 で mean_in_bleed が滑らかに正かを見る。"
          "\n     no2022_mean が負に転じるなら『2022依存』の証拠。")


if __name__ == "__main__":
    main()
