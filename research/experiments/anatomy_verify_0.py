"""敵対検証: 勝ちトレード集中度の主張を独立再計算する。

主張(concentration):
- 勝ちn=850
- 上位1%(9件)=グロス利益の4.9% / 純益の11.9%
- 上位5%(43件)=15.4% / 37.1%
- 上位10%(85件)=24.8% / 60.0%
- 純益ゼロ化 k=177件 (勝ちの20.8%, 全1214件の14.6%)
- 最大単一勝ち AUDCHF 2020-03-18 +3.59% (9本保有)
"""
import math

import pandas as pd

POOL = "results/mm_pool_v2_H4_19.parquet"


def main():
    df = pd.read_parquet(POOL)

    n_all = len(df)
    net = df["ret"].sum()
    wins = df.loc[df["ret"] > 0, "ret"].sort_values(ascending=False)
    n_win = len(wins)
    gross = wins.sum()
    gross_loss = df.loc[df["ret"] <= 0, "ret"].sum()

    print(f"n_all={n_all}  net={net:+.4f}  gross_win={gross:+.4f}  gross_loss={gross_loss:+.4f}")
    print(f"wins n={n_win}")

    # 検算: gross + loss = net
    assert abs((gross + gross_loss) - net) < 1e-9

    cum = wins.cumsum()

    for p in (0.01, 0.05, 0.10):
        k = math.ceil(n_win * p)
        top_sum = cum.iloc[k - 1]
        print(
            f"top {p*100:.0f}% -> k={k}  sum={top_sum:+.4f}  "
            f"share_gross={top_sum/gross*100:.1f}%  share_net={top_sum/net*100:.1f}%"
        )

    # 純益ゼロ化: 上位勝ちを何件除外すれば sum(ret) <= 0 になるか
    k_zero = int((cum >= net).idxmax() if (cum >= net).any() else -1)
    # idxmax はラベルを返すので位置で計算し直す
    k_zero = int((cum.values >= net).argmax() + 1) if (cum.values >= net).any() else -1
    print(
        f"net-zero k={k_zero}  (= wins {k_zero/n_win*100:.1f}% / all {k_zero/n_all*100:.1f}%)  "
        f"removed_sum={cum.iloc[k_zero-1]:+.4f} vs net={net:+.4f}"
    )
    # 検算: k件除外でゼロ以下、k-1件除外ではまだプラス
    print(
        f"  check: net - top{k_zero} = {net - cum.iloc[k_zero-1]:+.6f} (<=0?), "
        f"net - top{k_zero-1} = {net - cum.iloc[k_zero-2]:+.6f} (>0?)"
    )

    # 最大単一勝ち
    top1 = df.loc[df["ret"].idxmax()]
    print(
        f"max win: instr={top1['instr']}  entry={top1['entry']}  exit={top1['exit']}  "
        f"ret={top1['ret']*100:+.2f}%  bars_held={top1['bars_held']}  dir={top1['dir']}"
    )


if __name__ == "__main__":
    main()
