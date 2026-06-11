"""敵対検証: MAE浅瀬集中の主張を独立再計算で検証する。

主張: MAEが-0.5%以内の713件(58.7%)が総純益の+207.7%(+3.964)を供給し、
-0.5%超に沈んだ501件は合計-107.7%(-2.055)。MAE≤-1%の273件だけで-120.0%。

方法: pool の各トレードについて H4 close 経路で
path = dir * (close/entry_close - 1) を構築、MAE = min(path[1:], 0)。
排他ビンで sum(ret) を分解し全体と照合する。
"""

import numpy as np
import pandas as pd

from fxlab import universe as uni


def main():
    pool = pd.read_parquet("results/mm_pool_v2_H4_19.parquet")
    print(f"pool: n={len(pool)} sum(ret)={pool['ret'].sum():+.4f}")

    uni.register_cross_spreads(3.0)
    closes = {instr: uni.instrument_close(instr, "H4") for instr in pool["instr"].unique()}

    maes = np.empty(len(pool))
    path_mismatch = 0
    for i, row in enumerate(pool.itertuples(index=False)):
        c = closes[row.instr].loc[row.entry : row.exit]
        # 経路長の確認: entry〜exit のバー数は bars_held+1 のはず
        if len(c) != row.bars_held + 1:
            path_mismatch += 1
        path = row.dir * (c.values / c.values[0] - 1.0)
        maes[i] = min(path[1:].min(), 0.0) if len(path) > 1 else 0.0

    print(f"path length mismatches: {path_mismatch}")

    df = pool.copy()
    df["mae_pct"] = maes * 100.0

    # 排他ビン（左開右閉。MAE<=0 なので上端0を含む）
    edges = [-np.inf, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0]
    labels = ["<=-3%", "-2~-3%", "-1.5~-2%", "-1~-1.5%", "-0.5~-1%", "0~-0.5%"]
    df["bucket"] = pd.cut(df["mae_pct"], bins=edges, labels=labels, include_lowest=True)

    total = df["ret"].sum()
    print(f"\n=== MAE排他バケツ別分解 (total={total:+.4f}) ===")
    rows = []
    for lab in labels[::-1]:  # 浅い方から
        g = df[df["bucket"] == lab]
        rows.append(
            dict(
                bucket=lab,
                n=len(g),
                sum_ret=g["ret"].sum(),
                win=(g["ret"] > 0).mean() * 100,
                avg_bars=g["bars_held"].mean(),
                pct_of_total=g["ret"].sum() / total * 100,
            )
        )
    rep = pd.DataFrame(rows)
    print(rep.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"バケツ合計 sum(ret) = {rep['sum_ret'].sum():+.4f}  (全体 {total:+.4f})")

    # 主張の集約値
    shallow = df[df["mae_pct"] >= -0.5]
    deep = df[df["mae_pct"] < -0.5]
    deep1 = df[df["mae_pct"] <= -1.0]
    print("\n=== 主張の集約値 ===")
    print(
        f"MAE 0〜-0.5%: n={len(shallow)} ({len(shallow)/len(df)*100:.1f}%) "
        f"sum={shallow['ret'].sum():+.4f} ({shallow['ret'].sum()/total*100:+.1f}% of total)"
    )
    print(
        f"MAE <-0.5% : n={len(deep)} sum={deep['ret'].sum():+.4f} "
        f"({deep['ret'].sum()/total*100:+.1f}% of total)"
    )
    print(
        f"MAE <=-1%  : n={len(deep1)} sum={deep1['ret'].sum():+.4f} "
        f"({deep1['ret'].sum()/total*100:+.1f}% of total)"
    )


if __name__ == "__main__":
    main()
