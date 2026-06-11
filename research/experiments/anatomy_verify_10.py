"""敵対検証: キャリー/スワップマークアップ主張の独立再計算 (claim: cost).

主張:
  (a) キャリー合計 -0.0365 = 純益の -1.9%
  (b) 負キャリー方向保有 51.9% (ゼロ 3.3%)
  (c) 2016-21: -0.0066 (-0.10bps/件) -> 2022+: -0.0299 (-0.54bps/件) 約5.5倍悪化、最悪 2024 -0.0122
  (d) マークアップ片側年率0.5% -> -0.0832 (純益の-4.4%)、1.0% -> -0.1664 (-8.7%)

方法論(主張側の前提に従い独立実装):
  carry = dir * carry_annual(instr, entry年)/100 * 暦日保有/365   (fxlab.carry の RATES、事後会計)
  markup = -保有暦日/365 * 年率 を全トレードに加算(方向によらず)
"""

from __future__ import annotations

import pandas as pd

from fxlab.carry import carry_annual

POOL = "results/mm_pool_v2_H4_19.parquet"


def main() -> None:
    df = pd.read_parquet(POOL)
    n = len(df)
    total = df["ret"].sum()
    print(f"baseline check: n={n}  sum(ret)={total:+.4f}  mean={total/n*1e4:+.2f}bps  "
          f"win={float((df['ret']>0).mean()):.3f}")

    days = (pd.to_datetime(df["exit"]) - pd.to_datetime(df["entry"])).dt.total_seconds() / 86400.0
    print(f"holding calendar days: median={days.median():.2f}  mean={days.mean():.2f}  sum={days.sum():.1f}")

    entry_year = pd.to_datetime(df["entry"]).dt.year
    car_ann = pd.Series(
        [carry_annual(i, int(y)) for i, y in zip(df["instr"], entry_year)], index=df.index
    )
    carry = df["dir"].astype(float) * (car_ann / 100.0) * (days / 365.0)

    # (a) 合計
    csum = carry.sum()
    print(f"\n(a) carry sum = {csum:+.4f}  = {csum/total*100:+.2f}% of net profit (+{total:.4f})")

    # (b) 負/ゼロキャリー方向の比率
    neg = float((carry < 0).mean())
    zero = float((carry == 0).mean())
    pos = float((carry > 0).mean())
    print(f"(b) negative-carry holds = {neg*100:.1f}%  zero = {zero*100:.1f}%  positive = {pos*100:.1f}%")

    # (c) 期間分割(entry年基準)+年次
    pre = carry[entry_year <= 2021]
    post = carry[entry_year >= 2022]
    print(f"(c) 2016-21: sum={pre.sum():+.4f}  per-trade={pre.mean()*1e4:+.3f}bps  n={len(pre)}")
    print(f"    2022+ : sum={post.sum():+.4f}  per-trade={post.mean()*1e4:+.3f}bps  n={len(post)}")
    if pre.mean() != 0:
        print(f"    per-trade worsening ratio = {post.mean()/pre.mean():.2f}x")
    yearly = carry.groupby(entry_year).sum()
    print("    yearly carry sums (entry-year):")
    for y, v in yearly.items():
        print(f"      {y}: {v:+.4f}")
    # exit年基準でも(年次集計の定義差をチェック)
    exit_year = pd.to_datetime(df["exit"]).dt.year
    pre_x = carry[exit_year <= 2021]
    post_x = carry[exit_year >= 2022]
    print(f"    [exit-year basis] 2016-21 sum={pre_x.sum():+.4f}  2022+ sum={post_x.sum():+.4f}  "
          f"worst year = {carry.groupby(exit_year).sum().idxmin()} {carry.groupby(exit_year).sum().min():+.4f}")

    # (d) マークアップドラッグ(方向によらず、保有日数 x 年率)
    for rate in (0.005, 0.010):
        drag = -(days / 365.0 * rate).sum()
        print(f"(d) markup {rate*100:.1f}%/yr: drag = {drag:+.4f}  = {drag/total*100:+.2f}% of net profit")

    # 比較: キャリー vs マークアップ どちらが支配的か
    print(f"\ncompare: |carry|={abs(csum):.4f}  vs markup0.5%={abs((days/365*0.005).sum()):.4f}  "
          f"markup1.0%={abs((days/365*0.010).sum()):.4f}")


if __name__ == "__main__":
    main()
