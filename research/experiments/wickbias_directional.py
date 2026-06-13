"""wickbias_directional — literal test of the user's "wick imbalance precedes direction" hypothesis.

User's mental model: BEFORE a directional move, upper/lower wick amounts become imbalanced.
  - upper-wick excess  (wick_diff < 0)  precedes DOWN moves
  - lower-wick excess   (wick_diff > 0)  precedes UP moves
So wick_diff = lo_w - up_w should be a POSITIVE-slope SIGNED predictor of forward return.
We also test clv (close location value) to disambiguate "wick imbalance" from "close pinned at extreme".

We test on EURUSD at M5, M15, H1, H4 over horizons h1,h3,h5,h10 (cumulative forward close-to-close pips).
Two regimes:
  (a) UNCONDITIONAL across all bars.
  (b) CONDITIONED on a turning point: |zscore(close,50)| > 1.5 (where the champion acts).
      Within that set we ALSO test the SIGNED contribution: does wick_diff predict direction
      beyond the z-extreme itself? We orient by z-sign: at z<<0 (oversold) a lower-wick excess
      should predict a stronger bounce; at z>>0 (overbought) an upper-wick excess a sharper drop.

Anti-artifact: on M5/M15 we ALSO report the decontaminated number with UTC 20-23 entries excluded,
and judge on that. Weekend-gap bars excluded already by horizon construction (step>3*median -> NaN).

Run: uv run python -m research.experiments.wickbias_directional

Feature defs (fraction of bar range rng=h-l, NaN if rng==0; bar t fully formed, no look-ahead):
  clv=(c-l)/rng ; up_w=(h-max(o,c))/rng ; lo_w=(min(o,c)-l)/rng ; wick_diff=lo_w-up_w
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load
from fxlab.config import pip_size

SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
PAIR = "EURUSD"
TFS = ("M5", "M15", "H1", "H4")
HORIZONS = (1, 3, 5, 10)
ROLLOVER_HOURS = [20, 21, 22, 23]
Q = 0.05  # tail fraction for extreme-bin conditional means (signed predictor: lo bin vs hi bin)
DECON_TFS = {"M5", "M15"}


def wick_features(df: pd.DataFrame) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).where((h - l) > 0)
    clv = (c - l) / rng
    up_w = (h - np.maximum(o, c)) / rng
    lo_w = (np.minimum(o, c) - l) / rng
    wick_diff = lo_w - up_w
    return {
        "wick_diff": wick_diff,
        "wick_diff3": wick_diff.rolling(3).mean(),
        "clv": clv,
        "clv3": clv.rolling(3).mean(),
    }


def fwd_pips(df: pd.DataFrame, pair: str, horizons=HORIZONS) -> dict[int, pd.Series]:
    """h-bar cumulative forward close-to-close change (pips). A window [t -> t+h] is set NaN
    if ANY of its h forward steps is an abnormal gap (weekend etc.), so no edge bleeds across gaps."""
    pip = pip_size(pair)
    c = df["close"]
    step = df.index.to_series().diff()  # step[i] = index[i]-index[i-1]
    med = step.median()
    big = (step > med * 3).astype(float)  # big[i]=1 if the step INTO bar i is abnormal
    out = {}
    for h in horizons:
        y = c.diff(h).shift(-h) / pip  # return from close[t] to close[t+h], indexed at t
        # forward steps inside window starting at t are big[t+1..t+h]; flag if any is abnormal.
        # big.rolling(h).sum() at i = sum(big[i-h+1..i]); .shift(-h) brings t+h -> t = sum(big[t+1..t+h]).
        fwd_big = big.rolling(h).sum().shift(-h)
        out[h] = y.where(fwd_big.fillna(1.0) == 0)
    return out


def signed_eval(feat: pd.Series, tgt: pd.Series, q: float, mask: pd.Series | None = None) -> dict:
    """IC (train/test, Spearman) + extreme tail-bin conditional mean pips & t (train q-thresholds).

    For a SIGNED predictor the hypothesis is: hi-bin mean > 0, lo-bin mean < 0, IC > 0.
    'spread' = hi_mean - lo_mean (directional separation; >0 supports hypothesis)."""
    m = feat.notna() & tgt.notna() & np.isfinite(feat) & np.isfinite(tgt)
    if mask is not None:
        m = m & mask.reindex(feat.index, fill_value=False).astype(bool)
    f, y = feat[m], tgt[m]
    tr = f.index < SPLIT
    te = ~tr
    out = {"n_tr": int(tr.sum()), "n_te": int(te.sum())}
    if tr.sum() < 50 or te.sum() < 50:
        out.update(ic_tr=np.nan, ic_te=np.nan, lo_te=np.nan, lo_t=np.nan, lo_n=0,
                   hi_te=np.nan, hi_t=np.nan, hi_n=0, spread=np.nan)
        return out
    out["ic_tr"] = float(f[tr].rank().corr(y[tr].rank()))
    out["ic_te"] = float(f[te].rank().corr(y[te].rank()))
    lo, hi = f[tr].quantile(q), f[tr].quantile(1 - q)
    for tag, mb in [("lo", f <= lo), ("hi", f >= hi)]:
        yy = y[mb & te]
        n = len(yy)
        out[f"{tag}_te"] = float(yy.mean()) if n else np.nan
        out[f"{tag}_t"] = float(yy.mean() / (yy.std() / np.sqrt(n))) if n > 2 and yy.std() > 0 else np.nan
        out[f"{tag}_n"] = n
    out["spread"] = (out["hi_te"] - out["lo_te"]) if np.isfinite(out["hi_te"]) and np.isfinite(out["lo_te"]) else np.nan
    return out


def fmt(r: dict) -> str:
    return (
        f"IC tr/te {r['ic_tr']:+.4f}/{r['ic_te']:+.4f} | "
        f"lo {r['lo_te']:+.2f}p(t={r['lo_t']:+.1f},n={r['lo_n']}) "
        f"hi {r['hi_te']:+.2f}p(t={r['hi_t']:+.1f},n={r['hi_n']}) | "
        f"spread {r['spread']:+.2f}p"
    )


def run_tf(tf: str) -> None:
    df = load(PAIR, tf)
    feats = wick_features(df)
    ys = fwd_pips(df, PAIR)
    c = df["close"]
    sma, std = c.rolling(50).mean(), c.rolling(50).std()
    z = (c - sma) / std.where(std > 0)
    norol = pd.Series(~np.isin(df.index.hour, ROLLOVER_HOURS), index=df.index)
    decon = tf in DECON_TFS

    print("\n" + "=" * 96)
    print(f"{PAIR} {tf}  (range {df.index[0].date()}..{df.index[-1].date()}, n={len(df)})  "
          f"q={Q} signed tail bins; horizons {HORIZONS}")
    print("=" * 96)

    # ---------- (a) UNCONDITIONAL ----------
    print("(a) UNCONDITIONAL — all bars")
    for fn in ("wick_diff", "wick_diff3", "clv", "clv3"):
        print(f"  [{fn}]")
        for h in HORIZONS:
            r = signed_eval(feats[fn], ys[h], Q)
            print(f"    h{h:<2} {fmt(r)}")
            if decon:
                rd = signed_eval(feats[fn], ys[h], Q, mask=norol)
                print(f"    h{h:<2} ex20-23 {fmt(rd)}")

    # ---------- (b) CONDITIONED on turning point |z|>1.5 ----------
    extreme = z.abs() > 1.5
    n_ext = int((extreme & (df.index >= SPLIT)).sum())
    print(f"\n(b) CONDITIONED — |zscore(close,50)|>1.5 turning points (test bars n={n_ext})")
    print("    Does wick_diff add DIRECTION beyond the z-extreme? raw signed predictor within the set:")
    for fn in ("wick_diff", "clv"):
        print(f"  [{fn} | |z|>1.5]")
        for h in HORIZONS:
            r = signed_eval(feats[fn], ys[h], Q, mask=extreme)
            print(f"    h{h:<2} {fmt(r)}")
            if decon:
                rd = signed_eval(feats[fn], ys[h], Q, mask=extreme & norol)
                print(f"    h{h:<2} ex20-23 {fmt(rd)}")

    # z-oriented test: the champion is mean-reversion. At z<<0 expect bounce up; z>>0 expect drop.
    # User hypothesis at a turning point: a lower-wick excess (wick_diff>0) at an OVERSOLD extreme
    # should foreshadow the up-move; an upper-wick excess at OVERBOUGHT the down-move.
    # We test wick_diff as predictor of forward pips SEPARATELY in the oversold and overbought sets.
    print("    z-oriented (mean-reversion context): wick_diff -> forward pips, split by z sign")
    os_mask = z < -1.5   # oversold: bounce-up expected; hi wick_diff (lower-wick) should help
    ob_mask = z > 1.5    # overbought: drop expected; lo wick_diff (upper-wick) should help
    for label, mk in [("oversold z<-1.5", os_mask), ("overbought z>1.5", ob_mask)]:
        print(f"  [wick_diff | {label}]")
        for h in (1, 5):
            r = signed_eval(feats["wick_diff"], ys[h], Q, mask=mk)
            print(f"    h{h:<2} {fmt(r)}")
            if decon:
                rd = signed_eval(feats["wick_diff"], ys[h], Q, mask=mk & norol)
                print(f"    h{h:<2} ex20-23 {fmt(rd)}")


def main() -> None:
    for tf in TFS:
        run_tf(tf)
    print("\n" + "=" * 96)
    print("READING GUIDE: hypothesis holds at a (TF,horizon) iff IC_te>0 AND hi-bin mean>0 AND "
          "lo-bin mean<0 (spread>0) with |t|>=2 on BOTH tails, surviving ex20-23 on M5/M15.")
    print("=" * 96)


if __name__ == "__main__":
    main()
