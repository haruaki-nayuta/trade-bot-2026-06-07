"""敵対検証: 「利益はエントリー時絶対ボラ最上位四分位(Q4)に集中」主張の独立再計算.

主張(regime, Q1節):
  pool.vol_entry を qcut(4) →
  Q1: n=304, sum=+0.074(3.9%), 2.4bps, 勝率65.5%, PF1.11
  Q2: +0.505(26.4%), 16.7bps / Q3: +0.424(22.2%), 14.0bps
  Q4: n=304, sum=+0.906(47.5%), 29.8bps, 勝率72.0%, PF2.19
  Q4は決済年ベース全11年プラス、時代別 2016-21=31.0 / 2022-26=28.7bps
  Q4銘柄分散良好(最大 AUDCHF +0.138)。四分位合計=+1.9086
"""
import numpy as np
import pandas as pd

pool = pd.read_parquet("results/mm_pool_v2_H4_19.parquet")

# ---- ベースライン検算 ----
total = pool["ret"].sum()
print(f"baseline: n={len(pool)}  sum(ret)={total:+.4f}  "
      f"mean={pool['ret'].mean()*1e4:.1f}bps  win={(pool['ret']>0).mean()*100:.1f}%")

def pf(x: pd.Series) -> float:
    g = x[x > 0].sum()
    l = -x[x < 0].sum()
    return g / l if l > 0 else np.inf

# ---- 四分位分解 ----
q = pd.qcut(pool["vol_entry"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
rows = []
for lab in ["Q1", "Q2", "Q3", "Q4"]:
    r = pool.loc[q == lab, "ret"]
    rows.append({
        "Q": lab, "n": len(r), "sum": r.sum(), "share_%": r.sum() / total * 100,
        "mean_bps": r.mean() * 1e4, "win_%": (r > 0).mean() * 100, "PF": pf(r),
    })
tab = pd.DataFrame(rows).set_index("Q")
print("\n--- vol_entry qcut(4) ---")
print(tab.round(3).to_string())
print(f"四分位合計 = {tab['sum'].sum():+.4f}  (全体 {total:+.4f})")

# ---- Q4 の年次(決済年)・時代別 ----
exit_year = pd.to_datetime(pool["exit"]).dt.year
q4 = pool[q == "Q4"]
y4 = q4.groupby(exit_year[q == "Q4"])["ret"]
print("\n--- Q4 決済年別 sum / mean(bps) / n ---")
yt = pd.DataFrame({"sum": y4.sum(), "mean_bps": y4.mean() * 1e4, "n": y4.size()})
print(yt.round(3).to_string())
print(f"Q4 全年プラス? {bool((yt['sum'] > 0).all())}  (年数={len(yt)})")
era = np.where(yt.index <= 2021, "2016-21", "2022-26")
q4y = q4.assign(era=np.where(pd.to_datetime(q4["exit"]).dt.year <= 2021, "2016-21", "2022-26"))
print("Q4 時代別 mean(bps):")
print((q4y.groupby("era")["ret"].mean() * 1e4).round(1).to_string())

# ---- Q1 も年次で確認(「ほぼ損益分岐」の妥当性 + 勾配の時代横断性) ----
q1 = pool[q == "Q1"]
q1y = q1.assign(era=np.where(pd.to_datetime(q1["exit"]).dt.year <= 2021, "2016-21", "2022-26"))
print("\nQ1 時代別 mean(bps):")
print((q1y.groupby("era")["ret"].mean() * 1e4).round(1).to_string())

# ---- Q4 銘柄分散 ----
inst = q4.groupby("instr")["ret"].agg(["sum", "size"]).sort_values("sum", ascending=False)
print("\n--- Q4 銘柄別 sum(上位8) ---")
print(inst.head(8).round(4).to_string())
print(f"Q4 銘柄数={len(inst)}  Q4内最大銘柄寄与={inst['sum'].max():+.4f} "
      f"({inst['sum'].idxmax()})")

# ---- 敵対チェック: vol_entry の単位と銘柄組成バイアス ----
med = pool.groupby("instr")["vol_entry"].median().sort_values(ascending=False)
print("\n--- 銘柄別 vol_entry 中央値(プール横断qcutの組成バイアス確認) ---")
print((med * 1e4).round(1).to_string())
comp = pool.assign(q=q).pivot_table(index="instr", columns="q", values="ret",
                                    aggfunc="size", observed=True).fillna(0).astype(int)
print("\n--- 四分位×銘柄 件数(Q4が特定銘柄に偏っていないか) ---")
print(comp.to_string())

# ---- 敵対チェック: 銘柄内(within-instrument)ボラ四分位でも勾配が残るか ----
def within_q(g):
    try:
        return pd.qcut(g, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        return pd.Series(pd.NA, index=g.index)
wq = pool.groupby("instr")["vol_entry"].transform(within_q)
rows2 = []
for lab in ["Q1", "Q2", "Q3", "Q4"]:
    r = pool.loc[wq == lab, "ret"]
    rows2.append({"Q": lab, "n": len(r), "sum": r.sum(),
                  "share_%": r.sum() / total * 100, "mean_bps": r.mean() * 1e4,
                  "win_%": (r > 0).mean() * 100, "PF": pf(r)})
print("\n--- 銘柄内ボラ四分位(組成効果を除いた勾配) ---")
print(pd.DataFrame(rows2).set_index("Q").round(3).to_string())
