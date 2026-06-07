"""ボラ・ブレイクアウト族の失血窓貢献スイープ。

チャンピオンv2の失血窓(21ヶ月, 2017-2023, 高ER=トレンド継続レジーム)で
持続的に(IS/OOS両プラス)稼げる順張りエッジを探す。

候補: bb_breakout / squeeze_breakout を period/mult/squeeze/side/TF で振る。
評価軸: conditional_score の mean_in_bleed / hedge_edge / IS・OOS窓内平均。

実行: uv run python exp_vol_breakout.py
"""

from __future__ import annotations

import itertools
import warnings

import pandas as pd

import bleed_lab as bl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def run_sweep():
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)

    rows = []

    # --- bb_breakout: period x mult x side x tf ---
    bb_periods = [10, 20, 40, 60]
    bb_mults = [1.5, 2.0, 2.5, 3.0]
    sides = ["both", "long", "short"]
    tfs = ["H4", "D1"]

    print("=== bb_breakout sweep ===")
    for tf in tfs:
        for period, mult, side in itertools.product(bb_periods, bb_mults, sides):
            params = {"period": period, "mult": mult}
            try:
                mp = bl.strategy_monthly_pnl("bb_breakout", params=params, side=side, tf=tf)
            except Exception as e:  # noqa: BLE001
                print(f"  ERR bb p={period} m={mult} {side} {tf}: {e}")
                continue
            if len(mp) == 0:
                continue
            sc = bl.conditional_score(mp, mask)
            rows.append({
                "fam": "bb", "tf": tf, "period": period, "mult": mult, "side": side,
                "n_tr_months": len(mp),
                "total_all": sc["total_all"],
                "mean_in_bleed": sc["mean_in_bleed"],
                "mean_normal": sc["mean_normal"],
                "hedge_edge": sc["hedge_edge"],
                "wr_bleed": sc["winrate_in_bleed"],
                "mIS": sc["mean_in_bleed_IS"],
                "mOOS": sc["mean_in_bleed_OOS"],
                "tot_bleed": sc["total_in_bleed"],
            })

    # --- squeeze_breakout: period x mult x squeeze x side x tf ---
    sq_periods = [20, 40]
    sq_mults = [2.0, 2.5]
    squeezes = [50, 100, 200]
    print("=== squeeze_breakout sweep ===")
    for tf in tfs:
        for period, mult, squeeze, side in itertools.product(sq_periods, sq_mults, squeezes, sides):
            params = {"period": period, "mult": mult, "squeeze": squeeze}
            try:
                mp = bl.strategy_monthly_pnl("squeeze_breakout", params=params, side=side, tf=tf)
            except Exception as e:  # noqa: BLE001
                print(f"  ERR sq p={period} m={mult} sq={squeeze} {side} {tf}: {e}")
                continue
            if len(mp) == 0:
                continue
            sc = bl.conditional_score(mp, mask)
            rows.append({
                "fam": "sq", "tf": tf, "period": period, "mult": mult, "side": side,
                "squeeze": squeeze,
                "n_tr_months": len(mp),
                "total_all": sc["total_all"],
                "mean_in_bleed": sc["mean_in_bleed"],
                "mean_normal": sc["mean_normal"],
                "hedge_edge": sc["hedge_edge"],
                "wr_bleed": sc["winrate_in_bleed"],
                "mIS": sc["mean_in_bleed_IS"],
                "mOOS": sc["mean_in_bleed_OOS"],
                "tot_bleed": sc["total_in_bleed"],
            })

    df = pd.DataFrame(rows)
    df.to_csv("exp_vol_breakout_results.csv", index=False)
    return df, mask


def main():
    df, mask = run_sweep()
    print(f"\n=== 全 {len(df)} 構成 ===")

    # 持続ヘッジ条件: IS/OOS 両方の窓内平均がプラス
    persist = df[(df["mIS"] > 0) & (df["mOOS"] > 0)].copy()
    print(f"\n=== 持続ヘッジ候補 (mIS>0 かつ mOOS>0): {len(persist)} 構成 ===")
    persist = persist.sort_values("mean_in_bleed", ascending=False)
    cols = ["fam", "tf", "period", "mult", "side", "squeeze", "mean_in_bleed",
            "mean_normal", "hedge_edge", "wr_bleed", "mIS", "mOOS", "total_all", "tot_bleed"]
    cols = [c for c in cols if c in persist.columns]
    print(persist[cols].round(2).to_string(index=False))

    print("\n=== 全構成 mean_in_bleed トップ15 ===")
    top = df.sort_values("mean_in_bleed", ascending=False).head(15)
    print(top[cols].round(2).to_string(index=False))

    print("\n=== 全構成 hedge_edge トップ15 ===")
    top2 = df.sort_values("hedge_edge", ascending=False).head(15)
    print(top2[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
