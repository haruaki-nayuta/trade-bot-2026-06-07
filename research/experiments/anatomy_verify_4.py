"""敵対検証: per-tradeコスト再構成の主張(出典: cost)を独立再計算する。

主張:
  sum(gross)=+2.1673, sum(cost)=+0.2587 (グロスの11.9% / 純益の13.6%),
  全1214件で cost>0, cost中央値 2.05bps = 半スプレッド中央値 1.02bps x 2,
  net(x1.5)=+1.7792 マイナス年0 / net(x2.0)=+1.6499 マイナス年0。

方法(独立実装):
  gross = dir * (exit_close/entry_close - 1)   # H4 close を uni.instrument_close で取得
  cost  = gross - ret
  net(k) = gross - k*cost を決済年(exit年)で集計。
"""
import numpy as np
import pandas as pd

from fxlab import universe as uni

POOL = "results/mm_pool_v2_H4_19.parquet"


def main() -> None:
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL)
    n = len(pool)
    sum_ret = pool["ret"].sum()
    print(f"n={n}  sum(ret)={sum_ret:+.4f}  (baseline: 1214 / +1.9086)")

    # --- H4 close を銘柄ごとに取得し、entry/exit タイムスタンプの完全一致を確認 ---
    closes = {ins: uni.instrument_close(ins, "H4") for ins in pool["instr"].unique()}

    miss = 0
    entry_close = np.full(n, np.nan)
    exit_close = np.full(n, np.nan)
    for i, row in enumerate(pool.itertuples()):
        s = closes[row.instr]
        try:
            entry_close[i] = s.at[row.entry]
            exit_close[i] = s.at[row.exit]
        except KeyError:
            miss += 1
    print(f"timestamp exact-match failures: {miss}")

    d = pool.copy()
    d["entry_close"] = entry_close
    d["exit_close"] = exit_close
    d["gross"] = d["dir"] * (d["exit_close"] / d["entry_close"] - 1.0)
    d["cost"] = d["gross"] - d["ret"]

    sum_gross = d["gross"].sum()
    sum_cost = d["cost"].sum()
    print(f"\nsum(gross) = {sum_gross:+.4f}   (claim +2.1673)")
    print(f"sum(cost)  = {sum_cost:+.4f}   (claim +0.2587)")
    print(f"cost / gross = {sum_cost / sum_gross * 100:.1f}%   (claim 11.9%)")
    print(f"cost / net   = {sum_cost / sum_ret * 100:.1f}%   (claim 13.6%)")
    print(f"checksum gross - cost - ret = {(d['gross'] - d['cost'] - d['ret']).abs().max():.2e}")

    n_pos = int((d["cost"] > 0).sum())
    print(f"\ncost>0 の件数: {n_pos}/{n}   (claim 1214/1214)")
    print(f"cost 最小値: {d['cost'].min() * 1e4:.3f} bps")
    med_cost_bps = d["cost"].median() * 1e4
    print(f"cost 中央値: {med_cost_bps:.2f} bps   (claim 2.05)")

    # 半スプレッド: entry_price は close に半スプレッド分のスリッページ込み
    # long: entry_price = close*(1+hs) / short: entry_price = close*(1-hs)
    hs = d["dir"] * (d["entry_price"] / d["entry_close"] - 1.0)
    print(f"エントリー側 半スプレッド中央値: {hs.median() * 1e4:.2f} bps   (claim 1.02)")
    print(f"半スプレッド<0 の件数: {int((hs < 0).sum())}")

    # --- コスト倍率ストレス: net(k) = gross - k*cost を決済年で集計 ---
    d["year"] = d["exit"].dt.year
    print("\nyear  n     net(x1.0)  net(x1.5)  net(x2.0)")
    for k in (1.0, 1.5, 2.0):
        d[f"net_{k}"] = d["gross"] - k * d["cost"]
    yearly = d.groupby("year").agg(
        n=("ret", "size"),
        net10=("net_1.0", "sum"),
        net15=("net_1.5", "sum"),
        net20=("net_2.0", "sum"),
    )
    for y, r in yearly.iterrows():
        print(f"{y}  {int(r['n']):4d}  {r['net10']:+9.4f}  {r['net15']:+9.4f}  {r['net20']:+9.4f}")
    t15, t20 = d["net_1.5"].sum(), d["net_2.0"].sum()
    print(f"\nnet(x1.5) 合計 = {t15:+.4f} ({t15 / sum_ret * 100:.1f}% of base)   (claim +1.7792 / 93.2%)")
    print(f"net(x2.0) 合計 = {t20:+.4f} ({t20 / sum_ret * 100:.1f}% of base)   (claim +1.6499 / 86.4%)")
    print(f"マイナス年: x1.5 = {int((yearly['net15'] < 0).sum())} 年, "
          f"x2.0 = {int((yearly['net20'] < 0).sum())} 年   (claim 両方 0)")


if __name__ == "__main__":
    main()
