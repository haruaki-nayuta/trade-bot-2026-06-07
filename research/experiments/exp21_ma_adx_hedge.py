"""イテレーション21: 移動平均/ADXトレンド を「チャンピオンの失血窓ヘッジ(保険)」として最適化。

戦略フレーム(bleed_lab): 補完エッジは「平均相関の低さ」でなく **チャンピオンが失血している
まさにその窓で稼ぐか** で測る。チャンピオンの失血窓は高ER(トレンド継続レジーム)に集中
(2021-2023, 最深=2022 USDラリー)。トレンド/ブレイク族はそこで稼ぐはず。

本実験: strategies.ma_cross と adx_trend を fast/slow/adx_period/adx_th/side/TF で広く振り、
各構成の月次PnLストリームを失血窓 vs 平時で評価(conditional_score)。
評価軸の優先順:
  1) hedge_edge = mean_in_bleed - mean_normal が大きい(失血窓で相対的に稼ぐ)
  2) mean_in_bleed_IS と mean_in_bleed_OOS が **両方プラス** = 持続ヘッジ(2022一発でない)
  3) mean_in_bleed > 0(窓内で絶対的にプラス=純粋な保険)
ADXゲートが「高ER=失血窓」に絞り込めるか(=保険の効率化)を ma_cross と対比して確認。

実行: uv run python exp21_ma_adx_hedge.py
"""

from __future__ import annotations

import warnings

import pandas as pd

import bleed_lab as bl

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)


def main():
    print("=== チャンピオン失血窓マスク取得 ===")
    eqm, eqr, pool, closes = bl.champion_mtm()
    mask, dd = bl.bleed_mask_monthly(eqm)
    nb = int(mask.sum())
    print(f"月数 {len(mask)} / 失血窓 {nb}ヶ月 / 窓範囲 {mask.index.min()}..{mask.index.max()}")
    is_b = int((mask & (mask.index < pd.Period('2022-01', 'M'))).sum())
    oos_b = int((mask & (mask.index >= pd.Period('2022-01', 'M'))).sum())
    print(f"  失血窓 IS(<2022)={is_b}ヶ月 / OOS(>=2022)={oos_b}ヶ月\n")

    configs = []

    # --- ma_cross グリッド ---
    for tf in ["H4", "D1"]:
        for fast, slow in [(10, 50), (20, 50), (10, 100), (20, 100), (30, 100),
                           (50, 200), (20, 150), (10, 200)]:
            for side in ["both", "long", "short"]:
                configs.append(("ma_cross", {"fast": fast, "slow": slow}, side, tf))

    # --- adx_trend グリッド(トレンド強度ゲート) ---
    for tf in ["H4", "D1"]:
        for fast, slow in [(10, 50), (20, 50), (10, 100), (20, 100), (30, 100), (50, 200)]:
            for adx_th in [20, 25, 30, 35]:
                for side in ["both", "long", "short"]:
                    configs.append(("adx_trend",
                                    {"fast": fast, "slow": slow, "adx_period": 14, "adx_th": adx_th},
                                    side, tf))

    print(f"=== 候補 {len(configs)} 構成を評価中 ===")
    rows = []
    for i, (name, params, side, tf) in enumerate(configs):
        try:
            mp = bl.strategy_monthly_pnl(name, params=params, side=side, tf=tf)
        except Exception as e:  # noqa: BLE001
            continue
        if mp.empty or len(mp) < 12:
            continue
        sc = bl.conditional_score(mp, mask)
        pstr = "_".join(f"{k}{v}" for k, v in params.items()
                        if k in ("fast", "slow", "adx_th"))
        rows.append({
            "fam": name, "cfg": pstr, "side": side, "tf": tf,
            "standalone": round(sc["total_all"], 0),
            "mean_bleed": round(sc["mean_in_bleed"], 1),
            "mean_norm": round(sc["mean_normal"], 1),
            "edge": round(sc["hedge_edge"], 1),
            "wr_bleed": round(sc["winrate_in_bleed"], 2),
            "bleed_IS": round(sc["mean_in_bleed_IS"], 1),
            "bleed_OOS": round(sc["mean_in_bleed_OOS"], 1),
            "tot_bleed": round(sc["total_in_bleed"], 0),
            "persist": (sc["mean_in_bleed_IS"] > 0 and sc["mean_in_bleed_OOS"] > 0),
        })
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(configs)}")

    df = pd.DataFrame(rows)
    df.to_csv("/Users/yutootsuka/Documents/economy/results/exp21_hedge_sweep.csv", index=False)

    print("\n=== 全構成: hedge_edge 降順 トップ25 ===")
    print(df.sort_values("edge", ascending=False).head(25).to_string(index=False))

    print("\n=== 持続ヘッジ(IS&OOS両プラス)のみ: mean_bleed 降順 トップ20 ===")
    pers = df[df["persist"]].sort_values("mean_bleed", ascending=False)
    print(pers.head(20).to_string(index=False))

    print("\n=== 持続ヘッジ かつ 窓内絶対プラス(mean_bleed>0): edge 降順 ===")
    ins = df[(df["persist"]) & (df["mean_bleed"] > 0)].sort_values("edge", ascending=False)
    print(ins.head(20).to_string(index=False))

    # ファミリ別ベスト(持続ヘッジ前提で edge 最大)
    print("\n=== ファミリ別ベスト(持続ヘッジ前提・edge最大)===")
    for fam in ["ma_cross", "adx_trend"]:
        sub = df[(df["fam"] == fam) & (df["persist"]) & (df["mean_bleed"] > 0)]
        if sub.empty:
            sub = df[(df["fam"] == fam) & (df["persist"])]
        if sub.empty:
            print(f"  {fam}: 持続ヘッジ構成なし")
            continue
        b = sub.sort_values("edge", ascending=False).iloc[0]
        print(f"  {fam}: {b['cfg']} side={b['side']} tf={b['tf']} | "
              f"edge={b['edge']} mean_bleed={b['mean_bleed']} "
              f"IS={b['bleed_IS']} OOS={b['bleed_OOS']} standalone={b['standalone']}")

    return df


if __name__ == "__main__":
    main()
