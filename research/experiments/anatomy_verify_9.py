# anatomy_verify_9.py — 敵対検証: 「エッジは減衰していない」主張の独立再計算
# 主張(regime): 2022-2026 が 2016-2021 より全指標で良い / ret~時間の線形トレンドはゼロ
# 独立再計算のみ。anatomy_*.py は読まない・流用しない。
import pandas as pd
import numpy as np
from scipy import stats

PATH = "results/mm_pool_v2_H4_19.parquet"

df = pd.read_parquet(PATH)

# ---------- 0. ベースライン検算 ----------
n_all = len(df)
tot = df["ret"].sum()
print("== baseline ==")
print(f"n={n_all}  sum(ret)={tot:+.4f}  mean={df['ret'].mean()*1e4:.1f}bps  "
      f"win%={(df['ret']>0).mean()*100:.1f}")

# ---------- 1. 時代分割（決済日基準, 2022-01-01） ----------
ex = pd.to_datetime(df["exit"])
cut = pd.Timestamp("2022-01-01", tz=ex.dt.tz) if ex.dt.tz is not None else pd.Timestamp("2022-01-01")
era1 = df[ex < cut]
era2 = df[ex >= cut]

def pf(r):
    g = r[r > 0].sum()
    l = -r[r <= 0].sum()
    return g / l if l > 0 else np.inf

YEAR = 365.25 * 24 * 3600

def era_stats(name, sub, t0, t1):
    span_y = (t1 - t0).total_seconds() / YEAR
    r = sub["ret"]
    print(f"{name}: n={len(sub)}  mean={r.mean()*1e4:.1f}bps  PF={pf(r):.2f}  "
          f"sum={r.sum():+.4f}  span={span_y:.2f}y  net/yr={r.sum()/span_y:.3f}  "
          f"trades/yr={len(sub)/span_y:.1f}")
    return r.sum()

print("\n== era split (exit-date basis, cut=2022-01-01) ==")
e1_first = ex[ex < cut].min()
e2_last = ex[ex >= cut].max()
print(f"era1 exit range: {e1_first} .. {ex[ex<cut].max()}")
print(f"era2 exit range: {ex[ex>=cut].min()} .. {e2_last}")
s1 = era_stats("2016-2021", era1, e1_first, cut)
s2 = era_stats("2022-2026", era2, cut, e2_last)
print(f"era sums total = {s1 + s2:+.4f}  (baseline {tot:+.4f}, diff={s1+s2-tot:+.2e})")

# 参考: 期間スパンの別定義（エントリー初日起点など）の感度
ent = pd.to_datetime(df["entry"])
alt0 = ent.min()
print(f"(ref) first entry={alt0}  -> era1 span from first entry = "
      f"{(cut - alt0).total_seconds()/YEAR:.2f}y")

# ---------- 2. ret ~ 時間 の線形回帰 + Spearman ----------
print("\n== ret vs time (exit time, decimal years) ==")
t = ex.astype("int64") / 1e9 / YEAR  # 秒→年（オフセットは回帰のslope/pに無影響）
res = stats.linregress(t.values, df["ret"].values)
print(f"linregress: slope={res.slope*1e4:+.3f} bps/yr  p={res.pvalue:.3f}  "
      f"R2={res.rvalue**2:.5f}")
rho, p_s = stats.spearmanr(t.values, df["ret"].values)
print(f"spearman: rho={rho:+.4f}  p={p_s:.3f}")

# ---------- 3. 年次（決済年）mean bps — 最悪年の確認 ----------
print("\n== yearly (exit-year) ==")
yr = ex.dt.year
g = df.groupby(yr)["ret"]
yearly = pd.DataFrame({"n": g.size(), "sum": g.sum(), "bps": g.mean() * 1e4})
print(yearly.round(3).to_string())
worst = yearly["bps"].sort_values().head(3)
print(f"worst-3 years by mean bps: {[(int(i), round(v,1)) for i, v in worst.items()]}")
print(f"all years positive (sum>0): {(yearly['sum'] > 0).all()}")
