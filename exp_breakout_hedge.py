"""補完エッジ探索: ドンチャン/ブレイクアウト・トレンド族の「失血窓貢献」最大化。

戦略フレーム(bleed_lab): チャンピオンv2 が失血しているまさにその窓(トレンド継続レジーム)で
稼ぐエッジだけが DD≤20% に効く。breakout_trend が窓IS+79/窓OOS+41(持続ヘッジ)と予備調査で判明。
ここを起点に entry期間 / trendフィルタ / side / TF を振り、conditional_score 全項目を最大化する。

実行: uv run python exp_breakout_hedge.py
"""

from __future__ import annotations

import itertools

import pandas as pd

import bleed_lab as bl

pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def main():
    # --- チャンピオンの失血窓マスク(H4 標準較正) ---
    eqm, eqr, pool, closes = bl.champion_mtm(max_pos=8)
    mask, dd = bl.bleed_mask_monthly(eqm)
    print(f"失血窓マスク: {len(mask)}ヶ月中 {int(mask.sum())}ヶ月が失血窓\n")

    rows = []

    # ===== A) breakout_trend (entry/exit/trend × side × TF) =====
    bt_entry = [20, 40, 55, 80]
    bt_exit = [10, 20, 40]
    bt_trend = [100, 200]
    sides = ["both", "long", "short"]
    tfs = ["H4", "D1"]
    print("=== A) breakout_trend スイープ ===")
    for tf, side, entry, exit, trend in itertools.product(tfs, sides, bt_entry, bt_exit, bt_trend):
        if exit >= entry:
            continue
        try:
            mp = bl.strategy_monthly_pnl("breakout_trend",
                                         params={"entry": entry, "exit": exit, "trend": trend},
                                         side=side, tf=tf)
        except Exception as e:  # noqa: BLE001
            print(f"  skip breakout_trend e{entry} x{exit} t{trend} {side} {tf}: {e}")
            continue
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({
            "fam": "breakout_trend", "tf": tf, "side": side,
            "cfg": f"e{entry}/x{exit}/t{trend}",
            "mean_bleed": sc["mean_in_bleed"], "mean_norm": sc["mean_normal"],
            "edge": sc["hedge_edge"], "wr_bleed": sc["winrate_in_bleed"],
            "tot_bleed": sc["total_in_bleed"], "tot_all": sc["total_all"],
            "is_bleed": sc["mean_in_bleed_IS"], "oos_bleed": sc["mean_in_bleed_OOS"],
        })

    # ===== B) donchian_breakout (entry/exit × side × TF) =====
    dc_entry = [20, 40, 55, 80]
    dc_exit = [10, 20, 40]
    print("=== B) donchian_breakout スイープ ===")
    for tf, side, entry, exit in itertools.product(tfs, sides, dc_entry, dc_exit):
        if exit >= entry:
            continue
        try:
            mp = bl.strategy_monthly_pnl("donchian_breakout",
                                         params={"entry": entry, "exit": exit},
                                         side=side, tf=tf)
        except Exception as e:  # noqa: BLE001
            print(f"  skip donchian e{entry} x{exit} {side} {tf}: {e}")
            continue
        if mp.empty:
            continue
        sc = bl.conditional_score(mp, mask)
        rows.append({
            "fam": "donchian_breakout", "tf": tf, "side": side,
            "cfg": f"e{entry}/x{exit}",
            "mean_bleed": sc["mean_in_bleed"], "mean_norm": sc["mean_normal"],
            "edge": sc["hedge_edge"], "wr_bleed": sc["winrate_in_bleed"],
            "tot_bleed": sc["total_in_bleed"], "tot_all": sc["total_all"],
            "is_bleed": sc["mean_in_bleed_IS"], "oos_bleed": sc["mean_in_bleed_OOS"],
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/exp_breakout_hedge.csv", index=False)

    # 持続ヘッジ条件: IS と OOS の窓内平均が両方プラス
    df["persistent"] = (df["is_bleed"] > 0) & (df["oos_bleed"] > 0)

    print("\n=== 失血窓貢献(mean_in_bleed)降順 トップ25 ===")
    show = df.sort_values("mean_bleed", ascending=False).head(25)
    print(show[["fam", "tf", "side", "cfg", "mean_bleed", "mean_norm", "edge",
                "wr_bleed", "is_bleed", "oos_bleed", "persistent", "tot_all"]].round(1).to_string(index=False))

    print("\n=== 持続ヘッジ(IS/OOS両プラス)のみ、mean_in_bleed 降順 ===")
    pp = df[df["persistent"]].sort_values("mean_bleed", ascending=False)
    print(pp[["fam", "tf", "side", "cfg", "mean_bleed", "mean_norm", "edge",
              "wr_bleed", "is_bleed", "oos_bleed", "tot_all"]].round(1).to_string(index=False))

    print("\n=== hedge_edge 降順 トップ15 ===")
    show2 = df.sort_values("edge", ascending=False).head(15)
    print(show2[["fam", "tf", "side", "cfg", "mean_bleed", "mean_norm", "edge",
                 "is_bleed", "oos_bleed", "persistent"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
