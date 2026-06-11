"""敵対検証: 勝ち/負け集中の非対称性の独立再計算。

主張:
- ベスト10%(ceil=122件) がグロス利益の 32.0%
- ワースト10%(122件) が総損失の 73.3% (floor=121件なら73.0%, 既知72.5%と整合)
- 最大単一損失 = NZDCAD 2022-11-14, -7.55%, 140本, 純益の-4.0%
"""
import math

import pandas as pd

POOL = "results/mm_pool_v2_H4_19.parquet"


def main() -> None:
    df = pd.read_parquet(POOL)

    n = len(df)
    total = df["ret"].sum()
    print(f"n={n}  sum(ret)={total:+.4f}  mean={df['ret'].mean()*1e4:.1f}bps  "
          f"win_rate={(df['ret'] > 0).mean():.3f}")

    gross_profit = df.loc[df["ret"] > 0, "ret"].sum()
    gross_loss = -df.loc[df["ret"] < 0, "ret"].sum()  # 正の値で表現
    pf = gross_profit / gross_loss
    print(f"gross_profit={gross_profit:+.4f}  gross_loss={-gross_loss:+.4f}  PF={pf:.3f}")
    # 検算: グロス利益 - グロス損失 = 純益
    assert abs((gross_profit - gross_loss) - total) < 1e-9

    s = df["ret"].sort_values()
    k_ceil = math.ceil(0.10 * n)
    k_floor = math.floor(0.10 * n)
    print(f"k_ceil={k_ceil}  k_floor={k_floor}")

    best_ceil = s.iloc[-k_ceil:].sum()
    worst_ceil = s.iloc[:k_ceil].sum()
    best_floor = s.iloc[-k_floor:].sum()
    worst_floor = s.iloc[:k_floor].sum()

    print(f"best10% (ceil {k_ceil}件): sum={best_ceil:+.4f} -> グロス利益の "
          f"{best_ceil / gross_profit * 100:.1f}%")
    print(f"worst10% (ceil {k_ceil}件): sum={worst_ceil:+.4f} -> 総損失の "
          f"{-worst_ceil / gross_loss * 100:.1f}%")
    print(f"best10% (floor {k_floor}件): グロス利益の {best_floor / gross_profit * 100:.1f}%")
    print(f"worst10% (floor {k_floor}件): 総損失の {-worst_floor / gross_loss * 100:.1f}%")

    ratio = (-worst_ceil / gross_loss) / (best_ceil / gross_profit)
    print(f"集中度比 (worst側/best側) = {ratio:.2f}x")

    # 最大単一損失
    w = df.loc[df["ret"].idxmin()]
    print("\n最大単一損失:")
    print(f"  instr={w['instr']}  entry={w['entry']}  exit={w['exit']}  dir={w['dir']}")
    print(f"  ret={w['ret']*100:+.2f}%  bars_held={w['bars_held']}  "
          f"純益比={w['ret'] / total * 100:+.1f}%")


if __name__ == "__main__":
    main()
