"""Does the entry-candle WICK split champion winners from losers?

Pool: research/outputs/wickbias_champion_pool.csv (436 champion trades, 7 majors, H4).
Outcome = return_pct (per-trade %, size_mode=value 10000). Win = return_pct > 0.

Tests:
 (1) Mann-Whitney AUC = P(feature higher among winners) for direction-aligned
     wick_support, clv_support and raw wick_diff1, clv1, body1, range_rel20.
 (2) Quintiles by wick_support and by clv_support: mean return_pct, win rate, n -> monotonic?
 (3) ER-STYLE REMOVABLE COHORT: remove trades where wick_support is in its lowest
     quantile (wick OPPOSES the trade). n, share, net return, win rate of that cohort.
     Does removing it improve aggregate PF / expectancy?
 (4) IS (year<2022) vs OOS (>=2022): does the split hold in BOTH or single-period?
 (5) Per-pair: consistent across 7 majors or one-pair-driven?

Run: uv run python -m research.experiments.wickbias_cond_split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

CSV = "research/outputs/wickbias_champion_pool.csv"
FEATURES = ["wick_support", "clv_support", "wick_diff1", "clv1", "body1", "range_rel20"]


def pf_expectancy(r: pd.Series) -> tuple[float, float, float, float]:
    """profit factor, expectancy(mean return_pct), win_rate%, n."""
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else np.inf
    return pf, r.mean(), 100.0 * (r > 0).mean(), len(r)


def auc_winners(feat: pd.Series, win: pd.Series) -> tuple[float, float]:
    """Mann-Whitney AUC = P(feat higher in winners than losers) + two-sided p."""
    a = feat[win].dropna()
    b = feat[~win].dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    auc = u / (len(a) * len(b))  # P(winner feat > loser feat)
    return auc, p


def quintile_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    q = pd.qcut(df[col], 5, labels=[f"Q{i}" for i in range(1, 6)], duplicates="drop")
    rows = []
    for lab, g in df.groupby(q, observed=True):
        pf, exp, wr, n = pf_expectancy(g["return_pct"])
        rows.append(
            dict(quint=lab, n=n, mean_ret=exp, win_rate=wr, pf=pf,
                 feat_lo=g[col].min(), feat_hi=g[col].max(), net=g["return_pct"].sum())
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    df["win"] = df["return_pct"] > 0
    print(f"Loaded {len(df)} trades. overall win_rate={100*df['win'].mean():.2f}%  "
          f"net_return={df['return_pct'].sum():+.2f}  mean={df['return_pct'].mean():+.4f}")
    pf, exp, wr, n = pf_expectancy(df["return_pct"])
    print(f"BASELINE: PF={pf:.3f}  expectancy={exp:+.4f}  win={wr:.2f}%  n={n}\n")

    # ---------- (1) Mann-Whitney AUC ----------
    print("=" * 78)
    print("(1) Mann-Whitney AUC  P(feature higher among WINNERS)   [0.5=no signal]")
    print("=" * 78)
    for c in FEATURES:
        auc, p = auc_winners(df[c], df["win"])
        # also point-biserial-ish: mean feat in winners vs losers
        mw = df.loc[df["win"], c].mean()
        ml = df.loc[~df["win"], c].mean()
        ic = df[c].rank().corr(df["return_pct"].rank())  # spearman feat vs outcome
        print(f"  {c:<14} AUC={auc:.3f} (p={p:.3f})  winMean={mw:+.4f} loseMean={ml:+.4f} "
              f"diff={mw-ml:+.4f}  spearmanIC={ic:+.3f}")

    # ---------- (2) Quintiles ----------
    for col in ["wick_support", "clv_support"]:
        print("\n" + "=" * 78)
        print(f"(2) Quintiles by {col}  (Q1=lowest feat .. Q5=highest)")
        print("=" * 78)
        qt = quintile_table(df, col)
        print(qt.to_string(index=False,
              formatters={"mean_ret": "{:+.4f}".format, "win_rate": "{:.1f}".format,
                          "pf": "{:.2f}".format, "feat_lo": "{:+.3f}".format,
                          "feat_hi": "{:+.3f}".format, "net": "{:+.2f}".format}))
        # monotonic check on mean_ret
        mr = qt["mean_ret"].values
        mono_up = all(mr[i] <= mr[i + 1] for i in range(len(mr) - 1))
        mono_dn = all(mr[i] >= mr[i + 1] for i in range(len(mr) - 1))
        sp = np.corrcoef(range(len(mr)), mr)[0, 1]
        print(f"  -> monotonic_up={mono_up} monotonic_dn={mono_dn}  "
              f"rank-corr(quintile,mean_ret)={sp:+.3f}")

    # ---------- (3) ER-style removable cohort ----------
    print("\n" + "=" * 78)
    print("(3) REMOVABLE COHORT: remove trades where wick_support in lowest quantile")
    print("    (wick OPPOSES the trade). Test several cutoffs.")
    print("=" * 78)
    base_pf, base_exp, base_wr, base_n = pf_expectancy(df["return_pct"])
    for qcut in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        thr = df["wick_support"].quantile(qcut)
        removed = df[df["wick_support"] <= thr]
        kept = df[df["wick_support"] > thr]
        rpf, rexp, rwr, rn = pf_expectancy(removed["return_pct"])
        kpf, kexp, kwr, kn = pf_expectancy(kept["return_pct"])
        print(f"  cut q<={qcut:.2f} (thr={thr:+.4f}): REMOVED n={rn} "
              f"({100*rn/base_n:.1f}%) net={removed['return_pct'].sum():+.2f} "
              f"win={rwr:.1f}% mean={rexp:+.4f} PF={rpf:.2f}   ||  "
              f"KEPT n={kn} net={kept['return_pct'].sum():+.2f} win={kwr:.1f}% "
              f"PF={kpf:.2f} exp={kexp:+.4f} (base PF={base_pf:.2f} exp={base_exp:+.4f})")

    # also test removing lowest clv_support quantile and lowest wick_diff1
    print("\n  -- same removal logic for clv_support and raw wick_diff1 --")
    for col in ["clv_support", "wick_diff1"]:
        for qcut in [0.20, 0.30]:
            thr = df[col].quantile(qcut)
            removed = df[df[col] <= thr]
            kept = df[df[col] > thr]
            rpf, rexp, rwr, rn = pf_expectancy(removed["return_pct"])
            kpf, kexp, kwr, kn = pf_expectancy(kept["return_pct"])
            print(f"  {col} q<={qcut:.2f}: REMOVED n={rn} net={removed['return_pct'].sum():+.2f} "
                  f"win={rwr:.1f}% mean={rexp:+.4f}  || KEPT PF={kpf:.2f} exp={kexp:+.4f} "
                  f"(base PF={base_pf:.2f} exp={base_exp:+.4f})")

    # ---------- (4) IS vs OOS ----------
    print("\n" + "=" * 78)
    print("(4) IS (year<2022) vs OOS (>=2022): does the lowest-wick cohort lose in BOTH?")
    print("=" * 78)
    # Use the IS-derived threshold applied to OOS (no leakage), like the ER protocol.
    df_is = df[df["year"] < 2022]
    df_oos = df[df["year"] >= 2022]
    print(f"  IS n={len(df_is)}  OOS n={len(df_oos)}")
    for col in ["wick_support", "clv_support"]:
        print(f"\n  -- {col} --")
        for qcut in [0.20, 0.30]:
            thr_is = df_is[col].quantile(qcut)  # threshold fit on IS only
            for tag, d in [("IS", df_is), ("OOS", df_oos)]:
                rem = d[d[col] <= thr_is]
                kep = d[d[col] > thr_is]
                rpf, rexp, rwr, rn = pf_expectancy(rem["return_pct"])
                kpf, kexp, kwr, kn = pf_expectancy(kep["return_pct"])
                print(f"    [{tag}] thr_IS(q{qcut:.2f})={thr_is:+.4f}: "
                      f"REMOVED n={rn} net={rem['return_pct'].sum():+.2f} win={rwr:.1f}% "
                      f"mean={rexp:+.4f}  | KEPT exp={kexp:+.4f} PF={kpf:.2f}")

    # ---------- (5) Per-pair ----------
    print("\n" + "=" * 78)
    print("(5) Per-pair: lowest-wick_support cohort (global q<=0.30 thr) net & win")
    print("=" * 78)
    thr = df["wick_support"].quantile(0.30)
    print(f"  global wick_support q0.30 thr = {thr:+.4f}")
    print(f"  {'pair':<8}{'n':>4}{'rem_n':>6}{'rem_net':>9}{'rem_win':>8}{'rem_mean':>10}"
          f"{'kept_mean':>10}{'kept_pf':>8}{'base_mean':>10}")
    for pair, g in df.groupby("pair"):
        rem = g[g["wick_support"] <= thr]
        kep = g[g["wick_support"] > thr]
        _, rexp, rwr, rn = pf_expectancy(rem["return_pct"]) if len(rem) else (np.nan,)*4
        kpf, kexp, _, _ = pf_expectancy(kep["return_pct"]) if len(kep) else (np.nan,)*4
        _, bexp, _, _ = pf_expectancy(g["return_pct"])
        print(f"  {pair:<8}{len(g):>4}{rn:>6}{rem['return_pct'].sum():>+9.2f}"
              f"{rwr:>7.1f}%{rexp:>+10.4f}{kexp:>+10.4f}{kpf:>8.2f}{bexp:>+10.4f}")

    # spearman of wick_support vs return within each pair (sign consistency)
    print("\n  per-pair spearman(wick_support, return_pct):")
    signs = []
    for pair, g in df.groupby("pair"):
        ic = g["wick_support"].rank().corr(g["return_pct"].rank())
        signs.append(ic)
        print(f"    {pair}: {ic:+.3f}")
    signs = np.array(signs)
    print(f"  -> {(signs>0).sum()}/{len(signs)} pairs positive  "
          f"mean={signs.mean():+.3f}  (sign consistency check)")

    # ---------- redundancy: wick_support vs z/rsi/er at entry ----------
    print("\n" + "=" * 78)
    print("(redundancy) corr of wick_support with z-stretch / rsi / er at entry")
    print("=" * 78)
    # direction-align z and rsi so 'more stretched' is comparable across long/short
    zr = np.where(df["dir"] == "Long", -df["z"], df["z"])     # +ve = more stretched against entry
    rr = np.where(df["dir"] == "Long", 35 - df["rsi"], df["rsi"] - 65)  # +ve = more extreme
    dd = df.copy()
    dd["z_stretch"] = zr
    dd["rsi_stretch"] = rr
    for c in ["z_stretch", "rsi_stretch", "er", "clv_support", "range_rel20"]:
        sp = dd["wick_support"].corr(dd[c], method="spearman")
        print(f"  spearman(wick_support, {c:<12}) = {sp:+.3f}")


if __name__ == "__main__":
    main()
