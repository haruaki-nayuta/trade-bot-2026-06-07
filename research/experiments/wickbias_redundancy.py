"""
wickbias_redundancy.py — Does the wick signal add anything beyond what the
champion ALREADY uses (z / RSI / ER)?

The champion (confluence_meanrev_v2) enters at z-extremes (|z|>2, RSI<35 / >65, ER<=0.55).
Entering at a z-extreme mechanically pins the close near the bar extreme (low CLV for longs),
which mechanically correlates with wick geometry. So before believing a "wick filter" helps,
we must show wick carries information ORTHOGONAL to the z/RSI/ER confluence the champion
already conditions on.

Pool: research/outputs/wickbias_champion_pool.csv (436 champion trades, 7 majors, H4).
  return_pct = per-trade % return (size_mode=value 10000).
  Direction-aligned wick features:
    wick_support = wick_diff1(long) / -wick_diff1(short)   (+ = wick imbalance "supports" the bounce)
    clv_support  = -clv1(long)      /  clv1(short)         (+ = close pinned at extreme = max stretch, per reports/12)
  Raw confluence inputs at entry: z, rsi, er.

Tests:
 (1) Spearman corr of clv_support & wick_support vs z, rsi, er  (full + direction-aligned magnitudes).
 (2) Hold z fixed (narrow band) -> does wick_support still separate winners/losers, or collapse?
 (3) Rank-regress return_pct on [z, rsi, er] FIRST; take residual; ask whether wick_support
     has incremental rank-correlation with that residual (partial Spearman).
 (4) Conclude: redundant vs orthogonal.
"""
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

CSV = "/Users/yutootsuka/Documents/economy/.claude/worktrees/great-williamson-f8ced5/research/outputs/wickbias_champion_pool.csv"
df = pd.read_csv(CSV)
N = len(df)
df["is_long"] = df["dir"].str.lower().eq("long")
df["win"] = (df["return_pct"] > 0).astype(int)

# Direction-aligned magnitude of the confluence trigger:
#   long entries have z very negative, short entries z very positive.
#   z_support = how stretched in the trade's own direction (bigger = more extreme).
df["z_support"] = np.where(df["is_long"], -df["z"], df["z"])          # both ~ +2..+4
df["rsi_support"] = np.where(df["is_long"], 35.0 - df["rsi"], df["rsi"] - 65.0)  # bigger = more extreme RSI in trade dir
# er is direction-agnostic (lower = choppier/more mean-reverting); keep raw.

def sp(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = stats.spearmanr(a[m], b[m])
    return r, p, int(m.sum())

print("=" * 92)
print(f"WICK REDUNDANCY AUDIT  |  champion pool n={N}  (long={df['is_long'].sum()} short={(~df['is_long']).sum()})")
print(f"overall win_rate={100*df['win'].mean():.2f}%  total_return(sum ret_pct)={df['return_pct'].sum():+.2f}  mean={df['return_pct'].mean():+.4f}%")
print("=" * 92)

# ----------------------------------------------------------------------------------
# (1) Spearman: do the wick features just echo z / rsi / er at entry?
# ----------------------------------------------------------------------------------
print("\n[1] SPEARMAN CORR of wick features vs the champion's existing inputs (at entry)")
print("    direction-aligned: z_support / rsi_support bigger = more extreme in trade direction; er raw.")
wick_feats = ["wick_support", "clv_support"]
conf_feats = ["z_support", "rsi_support", "er"]
print(f"\n  {'wick_feat':<13}{'vs':<5}" + "".join(f"{c:>16}" for c in conf_feats))
for wf in wick_feats:
    cells = []
    for cf in conf_feats:
        r, p, n = sp(df[wf].values, df[cf].values)
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        cells.append(f"{r:>+7.3f}{star:<3}")
    print(f"  {wf:<13}{'':5}" + "".join(f"{c:>16}" for c in cells))

# Also the within-feature correlation: are the two wick features themselves redundant?
r_ww, p_ww, _ = sp(df["wick_support"].values, df["clv_support"].values)
print(f"\n  wick_support vs clv_support (are the two wick feats themselves redundant?): rho={r_ww:+.3f} (p={p_ww:.1e})")

# ----------------------------------------------------------------------------------
# (2) Hold z fixed (narrow band) -> does wick_support still separate winners/losers?
# ----------------------------------------------------------------------------------
print("\n[2] PARTIAL CHECK — within narrow z bands, does wick_support still separate W/L?")
print("    z_support band held fixed; compare wick_support of winners vs losers (Mann-Whitney),")
print("    and net return of high-wick vs low-wick half (median split) inside the band.")

def band_report(sub, label):
    if len(sub) < 20:
        print(f"    {label:<22} n={len(sub):<4} (too few, skip)")
        return
    w = sub[sub["win"] == 1]["wick_support"].values
    l = sub[sub["win"] == 0]["wick_support"].values
    if len(w) >= 3 and len(l) >= 3:
        u, pu = stats.mannwhitneyu(w, l, alternative="two-sided")
    else:
        pu = np.nan
    # median split on wick_support within band
    med = sub["wick_support"].median()
    hi = sub[sub["wick_support"] >= med]
    lo = sub[sub["wick_support"] < med]
    # spearman wick_support vs return within band
    rho, prho, _ = sp(sub["wick_support"].values, sub["return_pct"].values)
    print(f"    {label:<22} n={len(sub):<4} "
          f"win_wick_mean={np.mean(w):+.3f} los_wick_mean={np.mean(l):+.3f} MW_p={pu:.3f} | "
          f"hi-half ret={hi['return_pct'].mean():+.3f}(wr{100*hi['win'].mean():.0f}%) "
          f"lo-half ret={lo['return_pct'].mean():+.3f}(wr{100*lo['win'].mean():.0f}%) "
          f"Δret={hi['return_pct'].mean()-lo['return_pct'].mean():+.3f} | "
          f"rho(wick,ret)={rho:+.3f}(p{prho:.2f})")

# Longs and shorts share z_support scale (both ~ +2..+4). Bands on z_support magnitude.
for lo_b, hi_b in [(2.0, 2.25), (2.25, 2.5), (2.5, 3.0), (3.0, 9.0)]:
    sub = df[(df["z_support"] >= lo_b) & (df["z_support"] < hi_b)]
    band_report(sub, f"z_support[{lo_b:.2f},{hi_b:.2f})")
# whole pool baseline for comparison
band_report(df, "ALL (no z band)")

# Same but for clv_support (the reports/12 'stretch' feature), since z and CLV are the
# mechanically-linked pair we most worry about.
print("\n    [2b] same bands but testing clv_support (close-pinned-at-extreme = the z-mechanical twin):")
def band_report_clv(sub, label):
    if len(sub) < 20:
        print(f"    {label:<22} n={len(sub):<4} (too few, skip)")
        return
    med = sub["clv_support"].median()
    hi = sub[sub["clv_support"] >= med]
    lo = sub[sub["clv_support"] < med]
    rho, prho, _ = sp(sub["clv_support"].values, sub["return_pct"].values)
    print(f"    {label:<22} n={len(sub):<4} "
          f"hi-half ret={hi['return_pct'].mean():+.3f}(wr{100*hi['win'].mean():.0f}%) "
          f"lo-half ret={lo['return_pct'].mean():+.3f}(wr{100*lo['win'].mean():.0f}%) "
          f"Δret={hi['return_pct'].mean()-lo['return_pct'].mean():+.3f} | "
          f"rho(clv,ret)={rho:+.3f}(p{prho:.2f})")
for lo_b, hi_b in [(2.0, 2.25), (2.25, 2.5), (2.5, 3.0), (3.0, 9.0)]:
    sub = df[(df["z_support"] >= lo_b) & (df["z_support"] < hi_b)]
    band_report_clv(sub, f"z_support[{lo_b:.2f},{hi_b:.2f})")
band_report_clv(df, "ALL (no z band)")

# ----------------------------------------------------------------------------------
# (3) Rank-regress return on [z, rsi, er]; does wick add incremental rank info on residual?
# ----------------------------------------------------------------------------------
print("\n[3] PARTIAL SPEARMAN — rank-regress return on (z_support,rsi_support,er) FIRST,")
print("    then test wick features against the RESIDUAL (incremental rank info).")

def rankit(x):
    x = np.asarray(x, float)
    r = stats.rankdata(x)
    return (r - r.mean()) / (r.std() + 1e-12)

base_cols = ["z_support", "rsi_support", "er"]
work = df.dropna(subset=base_cols + ["return_pct"] + wick_feats).copy()
print(f"    rows used (complete cases): {len(work)}")

Y = rankit(work["return_pct"].values)
Xb = np.column_stack([rankit(work[c].values) for c in base_cols])
Xb = np.column_stack([np.ones(len(work)), Xb])
beta_y, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
res_Y = Y - Xb @ beta_y  # part of return-rank NOT explained by z/rsi/er rank
R2_base = 1 - np.var(res_Y) / np.var(Y)
print(f"    rank-R^2 of return explained by (z,rsi,er): {R2_base:.4f}  (how much of return rank the champion inputs already grab)")

for wf in wick_feats:
    # raw rank-corr of wick vs return (no control)
    r_raw, p_raw, _ = sp(work[wf].values, work["return_pct"].values)
    # residualize wick on the same base, then corr residual-wick vs residual-return = partial Spearman
    Wf = rankit(work[wf].values)
    beta_w, *_ = np.linalg.lstsq(Xb, Wf, rcond=None)
    res_W = Wf - Xb @ beta_w
    r_par, p_par = stats.pearsonr(res_W, res_Y)  # pearson on residual ranks = partial spearman
    print(f"    {wf:<13} raw rho(vs ret)={r_raw:+.3f}(p{p_raw:.2f})   "
          f"PARTIAL rho(|z,rsi,er)={r_par:+.3f}(p{p_par:.2f})   "
          f"shrink={abs(r_par)-abs(r_raw):+.3f}")

# How much of each wick feature is itself explained by z/rsi/er (collinearity)?
print("\n    collinearity: rank-R^2 of each wick feature explained by (z,rsi,er):")
for wf in wick_feats:
    Wf = rankit(work[wf].values)
    beta_w, *_ = np.linalg.lstsq(Xb, Wf, rcond=None)
    res_W = Wf - Xb @ beta_w
    R2 = 1 - np.var(res_W) / np.var(Wf)
    print(f"      {wf:<13} rank-R^2 = {R2:.4f}  -> {100*R2:.1f}% of its variation is already in z/rsi/er")

# ----------------------------------------------------------------------------------
# (4) Is there a wick-DEFINED REMOVABLE LOSING COHORT that z/ER can't already kill?
#     i.e. would a wick veto remove losers that survive the z/ER confluence?
# ----------------------------------------------------------------------------------
print("\n[4] REMOVABLE-COHORT probe — would a wick veto kill losers the champion keeps?")
print("    Test simple vetoes: drop trades with low wick_support / low clv_support quantiles.")
for feat in wick_feats:
    print(f"    -- veto on low {feat}:")
    for q in [0.20, 0.33, 0.50]:
        thr = df[feat].quantile(q)
        kept = df[df[feat] >= thr]
        dropped = df[df[feat] < thr]
        print(f"       drop bottom {int(q*100):>2}% ({feat}<{thr:+.3f}): "
              f"dropped n={len(dropped):>3} net={dropped['return_pct'].sum():+.2f} wr={100*dropped['win'].mean():.0f}% mean={dropped['return_pct'].mean():+.4f} | "
              f"kept n={len(kept):>3} net={kept['return_pct'].sum():+.2f} mean={kept['return_pct'].mean():+.4f}")

print("\n" + "=" * 92)
print("DONE.")
