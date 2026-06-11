"""敵対検証: グロスエッジ/コスト/方向反転の主張を独立再計算する。

主張(mechanism Q1):
- グロス +17.9bps/トレード、コストはグロスの11.9%(純益=88.1%)
- 反転 -20.0bps/トレード・勝率28%、方向スプレッド +35.7bps
- sum(gross)=+2.1673 / sum(cost)=+0.2587 / sum(ret)=+1.9086
- cost 全件>=0、平均2.1bps(メジャー1.0 / クロス2.7)
- 反転純益 sum=-2.4260
"""
import numpy as np
import pandas as pd

from fxlab import universe as uni

MAJORS = {"EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"}

POOL = "results/mm_pool_v2_H4_19.parquet"


def main():
    uni.register_cross_spreads(3.0)
    df = pd.read_parquet(POOL)
    n = len(df)
    print(f"n={n} sum(ret)={df['ret'].sum():+.4f} mean={df['ret'].mean()*1e4:+.2f}bps "
          f"winrate={(df['ret'] > 0).mean()*100:.1f}%")

    gross = np.full(n, np.nan)
    bars_match = 0
    for instr, g in df.groupby("instr"):
        close = uni.instrument_close(instr, "H4")
        idx = close.index
        e_pos = idx.get_indexer(g["entry"])
        x_pos = idx.get_indexer(g["exit"])
        assert (e_pos >= 0).all() and (x_pos >= 0).all(), f"{instr}: timestamp not found"
        bars_match += int((x_pos - e_pos == g["bars_held"].to_numpy()).sum())
        gr = g["dir"].to_numpy() * (close.to_numpy()[x_pos] / close.to_numpy()[e_pos] - 1.0)
        gross[df.index.get_indexer(g.index)] = gr
    print(f"bars_held一致率: {bars_match}/{n} = {bars_match/n*100:.1f}%")

    df = df.assign(gross=gross)
    df["cost"] = df["gross"] - df["ret"]
    df["rev"] = -df["gross"] - df["cost"]  # 方向反転の純益(コストは同額払う)

    sg, sc, sr = df["gross"].sum(), df["cost"].sum(), df["ret"].sum()
    print(f"\nsum(gross)={sg:+.4f}  sum(cost)={sc:+.4f}  sum(ret)={sr:+.4f}")
    print(f"検算 gross-cost-ret = {sg - sc - sr:+.2e}")
    print(f"グロス平均 {df['gross'].mean()*1e4:+.2f}bps/トレード")
    print(f"コスト/グロス = {sc/sg*100:.1f}%  (純益/グロス = {sr/sg*100:.1f}%)")

    neg_cost = int((df["cost"] < 0).sum())
    is_major = df["instr"].isin(MAJORS)
    print(f"\ncost<0 の件数: {neg_cost}/{n}")
    print(f"cost平均 {df['cost'].mean()*1e4:.2f}bps "
          f"(メジャー {df.loc[is_major,'cost'].mean()*1e4:.2f}bps n={is_major.sum()} / "
          f"クロス {df.loc[~is_major,'cost'].mean()*1e4:.2f}bps n={(~is_major).sum()})")

    rev_mean = df["rev"].mean() * 1e4
    rev_win = (df["rev"] > 0).mean() * 100
    print(f"\n反転: sum={df['rev'].sum():+.4f}  平均{rev_mean:+.2f}bps  勝率{rev_win:.1f}%")
    spread = (df["ret"].mean() - df["rev"].mean()) * 1e4
    print(f"方向スプレッド(順-逆) = {spread:+.2f}bps")


if __name__ == "__main__":
    main()
