"""SIZE-THE-PRIZE: champion vs champion+wick-veto, MAJORS-ONLY, H4 (size_mode=value).

Phase 2 verdict on this pool: the entry-candle wick does NOT separate champion winners
from losers (wick_support AUC=0.528 p=0.339; partial Spearman vs return residual after
controlling z/rsi/er = +0.048 p=0.32; NO net-losing removable cohort at any cutoff/decile/
IS/OOS; sign flips across pairs 4/7 positive). So a wick veto is expected to be sample-
shrinkage flattery, not edge. This script CONFIRMS that with the actual aggregate backtest
numbers (the task's "size the prize") and runs the project's standard adversarial gates so
the rejection is documented in real numbers, not hand-waved.

The single most-defensible design = BINARY WICK-OPPOSED VETO: drop champion trades whose
entry/trigger candle wick OPPOSES the trade direction (wick_support in its lowest quantile),
applied MAJORS-ONLY (these 7 pairs are the only ones with real OHLC / wicks; the 12 crosses
are synthetic close-only so a wick rule cannot apply to ~12/19 of the universe).

We compare WITH vs WITHOUT the veto:
  - aggregate net return (sum of per-trade return_pct, %), PF, expectancy, win rate, n
  - IS (year<2022) / OOS (>=2022) split
  - per-YEAR net sign (the project's yearly-sign gate)
  - leverage-disguise signature check: does the veto raise mean/expectancy ONLY by discarding
    profitable trades (empirical up but fewer trades, worse total $) -> that is the disguise.
  - bootstrap p95 of the kept-set total return vs baseline (does dispersion worsen?).

Threshold is FIT ON IS ONLY then applied to OOS (no leakage), as a proper filter would be.

Run: uv run python -m research.experiments.wickbias_integration
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CSV = "research/outputs/wickbias_champion_pool.csv"
RNG = np.random.default_rng(20260613)
N_BOOT = 5000


def agg(r: pd.Series) -> dict:
    """net return_pct sum, mean(expectancy), PF, win%, n."""
    r = r.dropna()
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else np.inf
    return dict(
        n=len(r),
        net=float(r.sum()),
        exp=float(r.mean()) if len(r) else np.nan,
        pf=float(pf),
        win=100.0 * (r > 0).mean() if len(r) else np.nan,
    )


def fmt(d: dict) -> str:
    return (f"n={d['n']:4d}  net={d['net']:+7.2f}  exp={d['exp']:+.4f}  "
            f"PF={d['pf']:.3f}  win={d['win']:5.1f}%")


def per_year_sign(df: pd.DataFrame) -> dict[int, float]:
    return {int(y): float(g["return_pct"].sum())
            for y, g in df.groupby("year")}


def boot_total(r: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """bootstrap the SUM (total return) -> (p5, p50, p95). Resample trades with replacement,
    SAME n each draw so it reflects the sampling dispersion of that cohort's total."""
    n = len(r)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    idx = RNG.integers(0, n, size=(n_boot, n))
    sums = r[idx].sum(axis=1)
    return tuple(np.percentile(sums, [5, 50, 95]))


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["wick_support", "return_pct"]).reset_index(drop=True)
    print(f"Loaded pool: n={len(df)} champion trades, 7 majors, H4 (size_mode=value 10000)\n")

    base = agg(df["return_pct"])
    print("BASELINE (champion, no wick rule):")
    print("  " + fmt(base))
    print()

    # ---- fit veto threshold on IS only, apply to OOS (no leakage) ----
    is_df = df[~df["is_oos"]]
    oos_df = df[df["is_oos"]]

    print("=" * 78)
    print("WICK-OPPOSED VETO: drop trades whose entry-candle wick OPPOSES trade direction")
    print("(wick_support below the q-th percentile). Threshold FIT ON IS, applied to OOS.")
    print("=" * 78)

    for q in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50]:
        thr = is_df["wick_support"].quantile(q)  # fit on IS
        keep = df["wick_support"] >= thr
        removed = df[~keep]
        kept = df[keep]

        k = agg(kept["return_pct"])
        rm = agg(removed["return_pct"])

        # leverage-disguise tell: removed cohort net sign
        removed_net = rm["net"]
        disguise = "REMOVED COHORT IS NET-POSITIVE (cutting winners = disguise)" \
            if removed_net > 0 else "removed cohort net-negative (genuine loss excision)"

        print(f"\n-- veto q={q:.2f}  (IS thr wick_support={thr:+.4f}) --")
        print(f"  KEPT   : {fmt(k)}")
        print(f"  REMOVED: {fmt(rm)}   <- {disguise}")
        print(f"  delta net (kept-base) = {k['net'] - base['net']:+.2f}   "
              f"delta exp = {k['exp'] - base['exp']:+.4f}   "
              f"delta PF = {k['pf'] - base['pf']:+.3f}")

        # IS/OOS split of the KEPT set (threshold from IS)
        keep_is = is_df[is_df["wick_support"] >= thr]
        keep_oos = oos_df[oos_df["wick_support"] >= thr]
        base_is = agg(is_df["return_pct"])
        base_oos = agg(oos_df["return_pct"])
        print(f"    IS  base {fmt(base_is)}")
        print(f"    IS  kept {fmt(agg(keep_is['return_pct']))}")
        print(f"    OOS base {fmt(base_oos)}")
        print(f"    OOS kept {fmt(agg(keep_oos['return_pct']))}")

    # ---- focus on q=0.25 for the full gate battery (representative cutoff) ----
    print("\n" + "=" * 78)
    print("FULL ADVERSARIAL GATES at q=0.25 (representative wick-veto cutoff)")
    print("=" * 78)
    thr = is_df["wick_support"].quantile(0.25)
    kept = df[df["wick_support"] >= thr].copy()
    base_kept = agg(kept["return_pct"])
    print(f"baseline : {fmt(base)}")
    print(f"wick-veto: {fmt(base_kept)}   (threshold fit on IS = {thr:+.4f})")

    # (A) yearly-sign gate
    print("\n[A] PER-YEAR net return sign (yearly-sign gate):")
    yb = per_year_sign(df)
    yk = per_year_sign(kept)
    years = sorted(set(yb) | set(yk))
    flipped = 0
    print(f"   {'year':>6} {'base_net':>10} {'veto_net':>10}  flip?")
    for y in years:
        b = yb.get(y, 0.0)
        kk = yk.get(y, 0.0)
        flip = (b > 0) != (kk > 0)
        flipped += int(flip and b > 0)  # base positive -> veto negative = bad
        print(f"   {y:>6} {b:>+10.2f} {kk:>+10.2f}  {'<-FLIP-TO-NEG' if flip and b>0 else ''}")
    print(f"   years where veto turned a positive base-year NEGATIVE: {flipped}")

    # (B) cross-pair gate: does veto help in each pair or one-pair-driven?
    print("\n[B] PER-PAIR delta net (cross-pair gate):")
    pos_pairs = 0
    for pair, g in df.groupby("pair"):
        gb = agg(g["return_pct"])
        gk = agg(g[g["wick_support"] >= thr]["return_pct"])
        d = gk["net"] - gb["net"]
        pos_pairs += int(d > 0)
        print(f"   {pair}: base net {gb['net']:+7.2f} -> veto net {gk['net']:+7.2f}  "
              f"delta {d:+6.2f}  (removed {gb['n']-gk['n']:2d})")
    print(f"   pairs where veto IMPROVED net: {pos_pairs}/7")

    # (C) leverage-disguise signature: empirical mean up + bootstrap p95 of TOTAL worse?
    print("\n[C] LEVERAGE-DISGUISE signature (empirical exp UP but total/p95 DOWN?):")
    rb = df["return_pct"].dropna().to_numpy()
    rk = kept["return_pct"].dropna().to_numpy()
    bb = boot_total(rb)
    bk = boot_total(rk)
    print(f"   baseline : exp={base['exp']:+.4f}  total p5/p50/p95 = "
          f"{bb[0]:+.2f}/{bb[1]:+.2f}/{bb[2]:+.2f}")
    print(f"   wick-veto: exp={base_kept['exp']:+.4f}  total p5/p50/p95 = "
          f"{bk[0]:+.2f}/{bk[1]:+.2f}/{bk[2]:+.2f}")
    emp_up = base_kept["exp"] > base["exp"]
    total_down = bk[1] < bb[1]
    print(f"   empirical expectancy UP? {emp_up}   |   median TOTAL DOWN? {total_down}")
    if emp_up and total_down:
        print("   => DISGUISE SIGNATURE PRESENT: mean rises only because total $ is discarded.")

    # (D) single-year dependence of any 'improvement'
    print("\n[D] SINGLE-YEAR dependence of the expectancy lift (drop-one-year jackknife):")
    full_lift = base_kept["exp"] - base["exp"]
    print(f"   full-sample expectancy lift = {full_lift:+.4f}")
    for y in years:
        sub = df[df["year"] != y]
        sub_thr = sub[~sub["is_oos"]]["wick_support"].quantile(0.25) \
            if (~sub["is_oos"]).any() else sub["wick_support"].quantile(0.25)
        sub_keep = sub[sub["wick_support"] >= sub_thr]
        lift = agg(sub_keep["return_pct"])["exp"] - agg(sub["return_pct"])["exp"]
        flag = "  <-sign-flip" if (lift > 0) != (full_lift > 0) else ""
        print(f"   drop {y}: lift={lift:+.4f}{flag}")

    # ---- bottom line ----
    print("\n" + "=" * 78)
    print("BOTTOM LINE")
    print("=" * 78)
    print(f"baseline total net  = {base['net']:+.2f}  (PF {base['pf']:.3f}, exp {base['exp']:+.4f})")
    print(f"wick-veto total net = {base_kept['net']:+.2f}  (PF {base_kept['pf']:.3f}, "
          f"exp {base_kept['exp']:+.4f})")
    print(f"net change in total return = {base_kept['net']-base['net']:+.2f}  "
          f"(removed {base['n']-base_kept['n']} trades, "
          f"{100*(base['n']-base_kept['n'])/base['n']:.0f}% of pool)")
    print(f"removed cohort net  = {agg(df[df['wick_support'] < thr]['return_pct'])['net']:+.2f} "
          f"(NET-POSITIVE => cutting winners, not losers)")


if __name__ == "__main__":
    main()
