"""wickbias_m5_eurusd — Re-confirm the "wick bias" next-bar edge on EURUSD M5.

ANGLE: M5 EURUSD reproduction. Re-test the textbook hammer hypothesis
("long lower wick => bounce next bar", i.e. wick_diff high) against the rival
"no-wick close pinned at the low" hypothesis (CLV low = max downward stretch),
exactly as the prior reports/12 work framed it, using the shared next-bar
evaluation protocol (research/lab/nextbar_common.py).

Standalone next-bar signals evaluated:
  wick_diff1, wick_diff3, clv1, clv3, lower_wick1, upper_wick1.

For each: Spearman IC train/test + extreme-bin (train-quantile) conditional
mean pips + t, at q in {0.02, 0.05, 0.10}. Reported BOTH on the full sample
and with UTC 20-23 entries EXCLUDED (decontaminated, per the rollover-BID
artifact discipline). Horizon decay h1..h20 for the best feature, the
hammer-vs-CLV showdown, a yearly breakdown for the winner, and comparison
against the ~0.6 pip EURUSD round-trip cost.

Run: uv run python -m research.experiments.wickbias_m5_eurusd
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.config import spread_pips
from research.lab.nextbar_common import (
    SPLIT,
    eval_signal,
    load_xy,
)

PAIR = "EURUSD"
TF = "M5"
COST = spread_pips(PAIR)  # 0.6 pips round-trip
ROLLOVER_HOURS = [20, 21, 22, 23]  # decontamination window (UTC 20-23)
QS = (0.02, 0.05, 0.10)
HZ = list(range(1, 21))  # h1..h20


# ---------------------------------------------------------------------------
# Features (identical defs to nb_candle_anatomy.py / repo WICK FEATURE DEFS)
# ---------------------------------------------------------------------------
def wick_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    rng_ok = rng.where(rng > 0)  # zero-range bars -> NaN
    clv = (c - l) / rng_ok                       # close location value
    up_w = (h - np.maximum(o, c)) / rng_ok       # upper wick fraction
    lo_w = (np.minimum(o, c) - l) / rng_ok       # lower wick fraction
    wick_diff = lo_w - up_w                       # +: lower wick dominates (hammer)
    return {
        "wick_diff1": wick_diff,
        "wick_diff3": wick_diff.rolling(3).mean(),
        "clv1": clv,
        "clv3": clv.rolling(3).mean(),
        "lower_wick1": lo_w,
        "upper_wick1": up_w,
    }


def horizon_targets_local(df: pd.DataFrame, pip: float, horizons) -> dict[int, pd.Series]:
    c = df["close"]
    return {h: c.diff(h).shift(-h) / pip for h in horizons}


def decon_mask(index: pd.DatetimeIndex) -> pd.Series:
    """True where the entry hour is NOT in the rollover window UTC 20-23."""
    return pd.Series(~np.isin(index.hour, ROLLOVER_HOURS), index=index)


def signed_bin_mean(feat: pd.Series, tgt: pd.Series, q: float):
    """For a feature where we suspect HIGH end predicts a bounce (hammer logic),
    return the 'hi' extreme bin; the eval_signal helper already gives both ends.
    This wrapper just runs eval_signal on the (optionally) masked series."""
    return eval_signal(feat, tgt, q=q, name="")


# ---------------------------------------------------------------------------
def battery(df, tgt, feats, label: str) -> None:
    print("=" * 96)
    print(f"M5 EURUSD wickbias battery — {label}")
    print("  IC tr/te | LO-bin (low feat) te mean/t/n | HI-bin (high feat) te mean/t/n")
    print(f"  cost reference (round-trip) = {COST:.2f} pips")
    print("=" * 96)
    for name, f in feats.items():
        for q in QS:
            r = eval_signal(f, tgt, q=q, name=f"{name:<11} q={q:.2f}")
            spd = r.get("lo_sig_per_day", float("nan")), r.get("hi_sig_per_day", float("nan"))
            print(
                f"{r['name']:<18} IC {r['ic_train']:+.4f}/{r['ic_test']:+.4f} | "
                f"LO {r['lo_test_mean_pips']:+.3f}p t={r['lo_test_t']:+.1f} n={r['lo_test_n']:>5} "
                f"({spd[0]:.1f}/d) | "
                f"HI {r['hi_test_mean_pips']:+.3f}p t={r['hi_test_t']:+.1f} n={r['hi_test_n']:>5} "
                f"({spd[1]:.1f}/d)"
            )
        print()


def hammer_vs_clv(df, tgt, feats, label: str) -> None:
    """Direct showdown: does the next-bar bounce come from
       (A) HAMMER  = wick_diff1 HIGH  (long lower wick), or
       (B) NO-WICK = clv1 LOW (close pinned at the low, max stretch)?
    Both are 'expect a long/bounce' setups; print the conditional next-bar mean
    of the relevant extreme bin (HI for wick_diff, LO for clv)."""
    print("=" * 96)
    print(f"HAMMER (wick_diff1 HIGH) vs NO-WICK-AT-LOW (clv1 LOW) — {label}")
    print("  positive next-bar mean pips => predicts a bounce. Beat +{:.2f}p to clear cost.".format(COST))
    print("=" * 96)
    for q in QS:
        rwd = eval_signal(feats["wick_diff1"], tgt, q=q, name="wick_diff1")
        rclv = eval_signal(feats["clv1"], tgt, q=q, name="clv1")
        # Hammer = high wick_diff (HI bin). No-wick-at-low = low clv (LO bin).
        print(
            f"q={q:.2f} | HAMMER  wick_diff1>=q{1-q:.2f}: "
            f"{rwd['hi_test_mean_pips']:+.3f}p t={rwd['hi_test_t']:+.1f} n={rwd['hi_test_n']:>5}"
        )
        print(
            f"        | NO-WICK clv1     <=q{q:.2f}: "
            f"{rclv['lo_test_mean_pips']:+.3f}p t={rclv['lo_test_t']:+.1f} n={rclv['lo_test_n']:>5}"
        )
        # also show the OTHER end of each, to expose any sign confusion
        print(
            f"        | (ref) wick_diff1<=q{q:.2f}: "
            f"{rwd['lo_test_mean_pips']:+.3f}p t={rwd['lo_test_t']:+.1f} | "
            f"clv1>=q{1-q:.2f}: {rclv['hi_test_mean_pips']:+.3f}p t={rclv['hi_test_t']:+.1f}"
        )
    print()


def horizon_decay(df, pip, feat: pd.Series, tag: str, q: float, label: str) -> None:
    """h1..h20 cumulative mean pips for the chosen extreme bin (train-quantile),
    measured on the TEST set only."""
    hts = horizon_targets_local(df, pip, HZ)
    m = feat.notna() & np.isfinite(feat)
    f = feat[m]
    tr = f.index < SPLIT
    lo, hi = f[tr].quantile(q), f[tr].quantile(1 - q)
    te = f.index[~tr]
    sel = (f[te] <= lo) if tag == "lo" else (f[te] >= hi)
    idx = sel[sel].index
    print("=" * 96)
    print(f"HORIZON DECAY h1..h20 — {label} | {tag}-bin q={q:.2f}, n={len(idx)} (test)")
    print(f"  cost reference = {COST:.2f}p")
    print("=" * 96)
    row = []
    for h in HZ:
        v = float(hts[h].reindex(idx).mean())
        row.append(f"h{h:>2}={v:+.2f}")
    # print in two rows of 10 for readability
    print("  " + "  ".join(row[:10]))
    print("  " + "  ".join(row[10:]))
    print()


def yearly_breakdown(df, tgt, feat: pd.Series, tag: str, q: float, label: str) -> None:
    """Per-year next-bar conditional mean pips for the winner's extreme bin,
    using the train-quantile threshold (fixed, no per-year refit)."""
    m = feat.notna() & np.isfinite(feat) & tgt.notna()
    f, y = feat[m], tgt[m]
    tr = f.index < SPLIT
    lo, hi = f[tr].quantile(q), f[tr].quantile(1 - q)
    sel = (f <= lo) if tag == "lo" else (f >= hi)
    yy = y[sel]
    print("=" * 96)
    print(f"YEARLY BREAKDOWN — {label} | {tag}-bin q={q:.2f} (threshold fixed on train)")
    print(f"  cost reference = {COST:.2f}p  (years < 2023 are in-sample)")
    print("=" * 96)
    for yr, g in yy.groupby(yy.index.year):
        n = len(g)
        t = g.mean() / (g.std() / np.sqrt(n)) if n > 2 and g.std() > 0 else float("nan")
        flag = "IS" if yr < 2023 else "OOS"
        print(f"  {yr} [{flag}]: mean={g.mean():+.3f}p t={t:+.2f} n={n}")
    print()


def main() -> None:
    df, tgt, pip = load_xy(PAIR, TF)
    feats = wick_features(df)

    # Decontaminated copies (UTC 20-23 entries excluded)
    keep = decon_mask(df.index)
    feats_dec = {k: v[keep] for k, v in feats.items()}
    tgt_dec = tgt[keep]
    df_dec = df[keep]

    n_total = int(df.shape[0])
    n_drop = int((~keep).sum())
    print(f"\nEURUSD {TF}: {n_total} bars, {df.index[0]} .. {df.index[-1]}")
    print(f"SPLIT={SPLIT.date()} | round-trip cost = {COST:.2f} pips")
    print(f"decontamination: dropping {n_drop} bars in UTC {ROLLOVER_HOURS} "
          f"({100*n_drop/n_total:.1f}% of bars)\n")

    # 1) Full-sample battery
    battery(df, tgt, feats, "FULL SAMPLE")

    # 2) Decontaminated battery
    battery(df_dec, tgt_dec, feats_dec, "DECONTAMINATED (UTC 20-23 excluded)")

    # 3) Hammer vs CLV showdown (both samples)
    hammer_vs_clv(df, tgt, feats, "FULL SAMPLE")
    hammer_vs_clv(df_dec, tgt_dec, feats_dec, "DECONTAMINATED")

    # 4) Pick the best feature by |decontaminated test mean pips| at a sensible q,
    #    among the relevant directional bins, then show horizon decay for it.
    #    We scan all (feat, bin, q) on the DECON sample and pick the largest
    #    statistically-meaningful conditional bounce magnitude.
    print("=" * 96)
    print("WINNER SELECTION (by decontaminated test conditional mean, |t|>=2 preferred)")
    print("=" * 96)
    cands = []
    for name, f in feats_dec.items():
        for q in QS:
            r = eval_signal(f, tgt_dec, q=q, name=name)
            for tag in ("lo", "hi"):
                mean = r[f"{tag}_test_mean_pips"]
                t = r[f"{tag}_test_t"]
                n = r[f"{tag}_test_n"]
                cands.append((name, tag, q, mean, t, n))
    # rank by |mean| but require |t|>=2 and n>=200; we want the bounce side (mean>0
    # is a long-profitable signal; mean<0 is a short-profitable signal)
    ranked = sorted(
        cands,
        key=lambda x: (abs(x[3]) if (abs(x[4]) >= 2 and x[5] >= 200) else -1),
        reverse=True,
    )
    print("  top 8 by |mean| (with |t|>=2 & n>=200):")
    for name, tag, q, mean, t, n in ranked[:8]:
        print(f"    {name:<11} {tag}-bin q={q:.2f}: {mean:+.3f}p t={t:+.1f} n={n}")
    win_name, win_tag, win_q, win_mean, win_t, win_n = ranked[0]
    print(f"\n  => WINNER: {win_name} {win_tag}-bin q={win_q:.2f} "
          f"({win_mean:+.3f}p, t={win_t:+.1f}, n={win_n}, decontaminated)\n")

    # 5) Horizon decay for the winner (both full and decon)
    horizon_decay(df, pip, feats[win_name], win_tag, win_q, f"{win_name} FULL")
    horizon_decay(df_dec, pip, feats_dec[win_name], win_tag, win_q,
                  f"{win_name} DECONTAMINATED")

    # 6) Yearly breakdown for the winner (decon)
    yearly_breakdown(df_dec, tgt_dec, feats_dec[win_name], win_tag, win_q,
                     f"{win_name} DECONTAMINATED")

    # 7) Explicit verdict line vs cost
    print("=" * 96)
    print("VERDICT vs COST")
    print("=" * 96)
    print(f"  EURUSD M5 round-trip cost = {COST:.2f} pips.")
    print(f"  Winner decontaminated edge = {win_mean:+.3f} pips/trade (t={win_t:+.1f}).")
    net = abs(win_mean) - COST
    print(f"  |edge| - cost = {abs(win_mean):.3f} - {COST:.2f} = {net:+.3f} pips/trade "
          f"({'NET POSITIVE' if net > 0 else 'NET NEGATIVE — cost-dominated'}).")


if __name__ == "__main__":
    main()
