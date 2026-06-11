"""敵対検証: 20:00バーエントリーの寄与主張を独立再計算する。

主張 (出典: time):
- 20:00エントリーは n=89/1214 (7.3%), sum=+0.2741 (総純益の14.4%), mean +30.8bps (全体+15.7bpsの約2倍)
- long n=52 +22.0bps / short n=37 +43.2bps・勝率86.5%
- z-power本番加重では寄与12.9% (加重総益+2.2649中+0.2928)

実行: uv run python -m research.experiments.anatomy_verify_2
"""
import numpy as np
import pandas as pd

POOL = "results/mm_pool_v2_H4_19.parquet"


def main():
    df = pd.read_parquet(POOL)

    # --- ベースライン検算 ---
    n = len(df)
    total = df["ret"].sum()
    mean_bps = df["ret"].mean() * 1e4
    win = (df["ret"] > 0).mean()
    gross_win = df.loc[df["ret"] > 0, "ret"].sum()
    gross_loss = -df.loc[df["ret"] < 0, "ret"].sum()
    pf = gross_win / gross_loss
    print(f"[baseline] n={n} sum={total:+.4f} mean={mean_bps:+.1f}bps "
          f"win={win:.1%} PF={pf:.3f}")

    # --- 時刻別集計 (entryバー開始ラベルの hour, UTC) ---
    hour = pd.to_datetime(df["entry"]).dt.hour
    by_hour = df.groupby(hour)["ret"].agg(["count", "sum", "mean"])
    checksum = by_hour["sum"].sum()
    print(f"[checksum] 時刻別sum合計={checksum:+.4f} (全体{total:+.4f}と一致: "
          f"{np.isclose(checksum, total)})")

    h20 = df[hour == 20]
    n20 = len(h20)
    s20 = h20["ret"].sum()
    m20 = h20["ret"].mean() * 1e4
    print(f"[20:00] n={n20}/{n} ({n20/n:.1%}) sum={s20:+.4f} "
          f"(寄与 {s20/total:+.1%}) mean={m20:+.1f}bps "
          f"(全体比 x{m20/mean_bps:.2f})")

    for d, label in [(1, "long"), (-1, "short")]:
        sub = h20[h20["dir"] == d]
        print(f"[20:00 {label}] n={len(sub)} mean={sub['ret'].mean()*1e4:+.1f}bps "
              f"win={ (sub['ret']>0).mean():.1%} sum={sub['ret'].sum():+.4f}")

    # --- z-power 本番加重 ---
    f = np.clip((df["z_entry"].abs() / 2.2) ** 4.0, 0.3, 3.0)
    ret_w = df["ret"] * f / f.mean()
    total_w = ret_w.sum()
    s20_w = ret_w[hour == 20].sum()
    print(f"[z-power加重] 加重総益={total_w:+.4f} 20:00寄与={s20_w:+.4f} "
          f"({s20_w/total_w:+.1%})")

    # --- 周辺時刻との比較 (20時だけ突出か、頑健性の参考) ---
    by_hour["mean_bps"] = by_hour["mean"] * 1e4
    by_hour["share_%"] = by_hour["sum"] / total * 100
    print("\n[時刻別一覧]")
    print(by_hour[["count", "sum", "mean_bps", "share_%"]].round(3).to_string())


if __name__ == "__main__":
    main()
