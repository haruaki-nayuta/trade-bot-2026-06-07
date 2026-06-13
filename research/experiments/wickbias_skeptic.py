"""wickbias_skeptic — adversarial referee for the M5 wick next-bar edge.

Angle: REFUTE. The strongest wick signal on EURUSD M5 (from nb_candle_anatomy)
is the close-location-value family: LOW clv (close pinned at the bar low =
"no-wick stretch down") predicts a next-bar bounce; HIGH clv predicts a drop.
Textbook long-lower-wick (wick_diff / lower_wick) is weak/dead. We attack the
best (clv3 + clv1 + wick_diff1) with four stress tests and quantify how much of
the apparent bin-mean-pips edge is artifact vs real.

Stress tests (all on TEST half, 2023-01-01+, thresholds frozen on TRAIN):
  (1) Rollover decomposition  — share of edge living in UTC 20-23 vs outside.
  (2) Weekend/Monday-open dependence — drop Friday-late (Fri >=20 UTC) and
      Sunday/Monday-open (first ~2h of the trading week) bars; recompute.
  (3) Cross-pair same-sign signature — does the SAME clv rule give same-sign
      edges on ALL 7 majors AND both USD directions at once (=artifact) or is it
      idiosyncratic (some pairs flip)?
  (4) Train-argmax -> test shrinkage — pick best (feature,q,horizon) on TRAIN
      only, report the TEST number. How much edge survives selection?

Cost ruler: a next-bar directional trade pays ~1 spread roundtrip on the pair
(EURUSD 0.6p). The lo/hi bins must clear that to be tradable.

Run: uv run python -m research.experiments.wickbias_skeptic
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.config import spread_pips
from research.lab.nextbar_common import (
    SPLIT,
    horizon_targets,
    load_xy,
)

PAIR = "EURUSD"
TF = "M5"
ROLLOVER_HOURS = [20, 21, 22, 23]  # decontamination window (NY17 rollover ±)
MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
# USD direction of a +clv-low long: long the base currency vs USD.
# For USDxxx pairs (USD is base), "long base" = long USD. We track the sign so we
# can detect "all pairs + both USD directions same sign" = artifact signature.
USD_IS_BASE = {"USDJPY": True, "USDCHF": True, "USDCAD": True}  # else USD is quote


def candle_clv(df: pd.DataFrame):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).where((h - l) > 0)
    clv = (c - l) / rng
    up_w = (h - np.maximum(o, c)) / rng
    lo_w = (np.minimum(o, c) - l) / rng
    wick_diff = lo_w - up_w
    return {
        "clv1": clv,
        "clv3": clv.rolling(3).mean(),
        "wick_diff1": wick_diff,
        "wick_diff3": wick_diff.rolling(3).mean(),
    }


def _stats(yy: pd.Series) -> tuple[float, float, int]:
    yy = yy.dropna()
    n = len(yy)
    if n < 3 or yy.std() == 0:
        return (float(yy.mean()) if n else np.nan, np.nan, n)
    return float(yy.mean()), float(yy.mean() / (yy.std() / np.sqrt(n))), n


def _bins(feat: pd.Series, tgt: pd.Series, q: float):
    """Freeze lo/hi thresholds on TRAIN; return boolean masks lo/hi (full index)."""
    m = feat.notna() & np.isfinite(feat)
    f = feat[m]
    tr = f.index < SPLIT
    lo_thr, hi_thr = f[tr].quantile(q), f[tr].quantile(1 - q)
    lo = feat <= lo_thr
    hi = feat >= hi_thr
    return lo.fillna(False), hi.fillna(False), float(lo_thr), float(hi_thr)


def _long_short_edge(tgt: pd.Series, lo: pd.Series, hi: pd.Series, mask: pd.Series):
    """Directional edge: go LONG in lo bin (+tgt), SHORT in hi bin (-tgt).
    Returns combined per-trade mean pips, t, n over the masked subset."""
    te = (tgt.index >= SPLIT) & mask
    long_pnl = tgt[lo & te]
    short_pnl = -tgt[hi & te]
    combined = pd.concat([long_pnl, short_pnl]).dropna()
    lm, lt, ln = _stats(long_pnl)
    sm, st, sn = _stats(short_pnl)
    cm, ct, cn = _stats(combined)
    return dict(long=(lm, lt, ln), short=(sm, st, sn), combined=(cm, ct, cn))


def main() -> None:
    df, tgt, pip = load_xy(PAIR, TF)
    feats = candle_clv(df)
    cost = spread_pips(PAIR)
    te = df.index >= SPLIT
    allmask = pd.Series(True, index=df.index)

    print("=" * 90)
    print(f"BASELINE: {PAIR} {TF} next-bar directional wick edge (TEST half, cost~{cost}p roundtrip)")
    print("=" * 90)
    print("rule: LONG in low-bin, SHORT in high-bin. q=0.02 unless noted.")
    base = {}
    for nm in ("clv1", "clv3", "wick_diff1", "wick_diff3"):
        lo, hi, lt, ht = _bins(feats[nm], tgt, 0.02)
        e = _long_short_edge(tgt, lo, hi, allmask)
        base[nm] = e
        lm, ltv, ln = e["long"]; sm, stv, sn = e["short"]; cm, ctv, cn = e["combined"]
        print(f"  {nm:<11} long {lm:+.3f}p(t={ltv:+.1f},n={ln})  "
              f"short {sm:+.3f}p(t={stv:+.1f},n={sn})  "
              f"COMBINED {cm:+.3f}p(t={ctv:+.1f},n={cn})  net-of-cost {cm-cost:+.3f}p")
    best = "clv3"
    print(f"\n>>> strongest = {best}. Attacking it.\n")

    # ===== (1) ROLLOVER DECOMPOSITION =====
    print("=" * 90)
    print("(1) ROLLOVER DECOMPOSITION — where does the bin-mean edge live? (TEST)")
    print("=" * 90)
    norol = pd.Series(~np.isin(df.index.hour, ROLLOVER_HOURS), index=df.index)
    rolonly = ~norol
    rollover_share = {}
    for nm in ("clv3", "clv1", "wick_diff1"):
        lo, hi, _, _ = _bins(feats[nm], tgt, 0.02)
        e_all = _long_short_edge(tgt, lo, hi, allmask)["combined"]
        e_out = _long_short_edge(tgt, lo, hi, norol)["combined"]
        e_in = _long_short_edge(tgt, lo, hi, rolonly)["combined"]
        # share of total edge-pips coming from rollover bars:
        tot_pips = e_all[0] * e_all[2]
        in_pips = e_in[0] * e_in[2] if not np.isnan(e_in[0]) else 0.0
        share = in_pips / tot_pips if tot_pips not in (0, np.nan) else np.nan
        rollover_share[nm] = share
        print(f"  {nm:<11} ALL {e_all[0]:+.3f}p(n={e_all[2]})  "
              f"ex-rollover {e_out[0]:+.3f}p(t={e_out[1]:+.1f},n={e_out[2]})  "
              f"rollover-only {e_in[0]:+.3f}p(t={e_in[1]:+.1f},n={e_in[2]})")
        print(f"              -> share of total edge-pips in UTC{ROLLOVER_HOURS}: {share*100:.1f}%  "
              f"(decontaminated net-of-cost {e_out[0]-cost:+.3f}p)")

    # ===== (2) WEEKEND / MONDAY-OPEN DEPENDENCE =====
    print("\n" + "=" * 90)
    print("(2) WEEKEND / MONDAY-OPEN DEPENDENCE (TEST, on top of ex-rollover)")
    print("=" * 90)
    dow = df.index.dayofweek  # Mon=0 .. Sun=6
    hour = df.index.hour
    fri_late = pd.Series((dow == 4) & (hour >= 20), index=df.index)
    # week open: Sunday bars + Monday before 02 UTC (Sydney/Tokyo thin open)
    week_open = pd.Series((dow == 6) | ((dow == 0) & (hour < 2)), index=df.index)
    clean = norol & ~fri_late & ~week_open
    weekend_share = {}
    for nm in ("clv3", "clv1", "wick_diff1"):
        lo, hi, _, _ = _bins(feats[nm], tgt, 0.02)
        e_norol = _long_short_edge(tgt, lo, hi, norol)["combined"]
        e_clean = _long_short_edge(tgt, lo, hi, clean)["combined"]
        e_friweek = _long_short_edge(tgt, lo, hi, norol & (fri_late | week_open))["combined"]
        base_pips = e_norol[0] * e_norol[2]
        fw_pips = e_friweek[0] * e_friweek[2] if not np.isnan(e_friweek[0]) else 0.0
        share = fw_pips / base_pips if base_pips not in (0, np.nan) else np.nan
        weekend_share[nm] = share
        print(f"  {nm:<11} ex-rol {e_norol[0]:+.3f}p(n={e_norol[2]})  "
              f"+drop Fri-late&week-open {e_clean[0]:+.3f}p(t={e_clean[1]:+.1f},n={e_clean[2]})  "
              f"net-of-cost {e_clean[0]-cost:+.3f}p")
        print(f"              -> share of ex-rollover edge-pips in Fri-late/week-open: {share*100:.1f}%")

    # ===== (3) CROSS-PAIR SAME-SIGN SIGNATURE =====
    print("\n" + "=" * 90)
    print("(3) CROSS-PAIR SAME-SIGN SIGNATURE — artifact if ALL majors + both USD dirs same sign")
    print("=" * 90)
    print("clv3 q=0.02, ex-rollover, directional combined edge per pair:")
    signs = []
    usd_signed = []
    for p in MAJORS:
        dfx, tgtx, pipx = load_xy(p, TF)
        fx = candle_clv(dfx)["clv3"]
        norolx = pd.Series(~np.isin(dfx.index.hour, ROLLOVER_HOURS), index=dfx.index)
        lo, hi, _, _ = _bins(fx, tgtx, 0.02)
        e = _long_short_edge(tgtx, lo, hi, norolx)["combined"]
        cm, ctv, cn = e
        cst = spread_pips(p)
        sgn = np.sign(cm) if not np.isnan(cm) else 0
        signs.append(sgn)
        # USD-direction view: a clv-low LONG = long base. Express as USD long/short sign.
        # If USD is base (USDxxx): long base = long USD -> usd_sign = +sgn (edge favors USD)
        # If USD is quote (xxxUSD): long base = short USD -> usd_sign = -sgn
        usd_sign = sgn if USD_IS_BASE.get(p, False) else -sgn
        usd_signed.append((p, usd_sign))
        flag = "PASS-cost" if (cm - cst) > 0 else "below-cost"
        print(f"  {p}: combined {cm:+.3f}p (t={ctv:+.1f}, n={cn}) net-of-cost {cm-cst:+.3f}p [{flag}]  "
              f"dir-sign={int(sgn):+d} usd-sign={int(usd_sign):+d}")
    same_sign_frac = np.mean([s == signs[0] for s in signs if s != 0])
    usd_long = [p for p, s in usd_signed if s > 0]
    usd_short = [p for p, s in usd_signed if s < 0]
    print(f"\n  same-direction-sign fraction across majors: {same_sign_frac*100:.0f}%")
    print(f"  USD-favoring pairs: {usd_long}")
    print(f"  USD-opposing pairs: {usd_short}")
    print("  -> ARTIFACT signature requires same dir-sign on all majors AND USD pulling "
          "the same way in both base/quote groups. Idiosyncratic (mixed) = real-ish.")

    # ===== (4) TRAIN-ARGMAX -> TEST SHRINKAGE =====
    print("\n" + "=" * 90)
    print("(4) TRAIN-ARGMAX -> TEST SHRINKAGE — pick best (feature,q,horizon) on TRAIN, read TEST")
    print("=" * 90)
    hts = horizon_targets(df, PAIR)
    feat_names = ["clv1", "clv3", "clv5_window", "wick_diff1", "wick_diff3",
                  "lower_wick1", "upper_wick1"]
    # add a couple more for a fair selection menu
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = (h - l).where((h - l) > 0)
    hh5, ll5 = h.rolling(5).max(), l.rolling(5).min()
    extra = {
        "clv5_window": (c - ll5) / (hh5 - ll5).where((hh5 - ll5) > 0),
        "lower_wick1": (np.minimum(o, c) - l) / rng,
        "upper_wick1": (h - np.maximum(o, c)) / rng,
    }
    menu = {**feats, **extra}
    tr = df.index < SPLIT
    candidates = []
    for nm in feat_names:
        f = menu[nm]
        for q in (0.01, 0.02, 0.05, 0.10):
            mm = f.notna() & np.isfinite(f)
            ff = f[mm]
            trm = ff.index < SPLIT
            lo_thr, hi_thr = ff[trm].quantile(q), ff[trm].quantile(1 - q)
            lo = (f <= lo_thr).fillna(False)
            hi = (f >= hi_thr).fillna(False)
            for hz, htgt in hts.items():
                # TRAIN directional edge (per "trade", normalized per bar by /hz to compare)
                long_tr = htgt[lo & tr]
                short_tr = -htgt[hi & tr]
                comb_tr = pd.concat([long_tr, short_tr]).dropna()
                m_tr, t_tr, n_tr = _stats(comb_tr)
                per_bar_tr = m_tr / hz if not np.isnan(m_tr) else np.nan
                if n_tr < 500:
                    continue
                candidates.append((nm, q, hz, per_bar_tr, m_tr, t_tr, n_tr, lo, hi))
    # argmax on TRAIN per-bar edge
    candidates.sort(key=lambda x: (x[3] if not np.isnan(x[3]) else -9), reverse=True)
    print("Top-5 TRAIN-selected (by per-bar combined edge), then their TEST number:")
    for nm, q, hz, pb_tr, m_tr, t_tr, n_tr, lo, hi in candidates[:5]:
        htgt = hts[hz]
        long_te = htgt[lo & te]
        short_te = -htgt[hi & te]
        comb_te = pd.concat([long_te, short_te]).dropna()
        m_te, t_te, n_te = _stats(comb_te)
        per_bar_te = m_te / hz if not np.isnan(m_te) else np.nan
        if np.isfinite(pb_tr) and pb_tr != 0:
            shrink = 100.0 * per_bar_te / pb_tr
        else:
            shrink = float("nan")
        h1_net = (m_te - cost) if hz == 1 else float("nan")
        print(f"  {nm:<12} q={q:<4} h={hz:<2}  TRAIN/bar {pb_tr:+.3f}p (h-sum {m_tr:+.2f}p,t={t_tr:+.1f})"
              f"  ->  TEST/bar {per_bar_te:+.3f}p (h-sum {m_te:+.2f}p,t={t_te:+.1f},n={n_te})"
              f"  retained {shrink:.0f}%  TEST h1-net-of-cost {h1_net:+.3f}p")
    # also: the single winner's h1 directional TEST edge net-of-cost
    win = candidates[0]
    nm, q, hz, pb_tr, _, _, _, lo, hi = win
    e_h1 = _long_short_edge(tgt, lo, hi, allmask)["combined"]
    print(f"\n  TRAIN winner = {nm} q={q} h={hz}.  h1 directional TEST combined "
          f"{e_h1[0]:+.3f}p (t={e_h1[1]:+.1f}, n={e_h1[2]})  net-of-cost {e_h1[0]-cost:+.3f}p")

    # ===== SUMMARY ATTRIBUTION =====
    print("\n" + "=" * 90)
    print("SUMMARY — edge attribution (clv3, the strongest)")
    print("=" * 90)
    lo, hi, _, _ = _bins(feats["clv3"], tgt, 0.02)
    e_all = _long_short_edge(tgt, lo, hi, allmask)["combined"]
    e_out = _long_short_edge(tgt, lo, hi, norol)["combined"]
    e_clean = _long_short_edge(tgt, lo, hi, clean)["combined"]
    print(f"  raw combined edge            : {e_all[0]:+.3f}p (t={e_all[1]:+.1f}, n={e_all[2]})")
    print(f"  share in rollover UTC20-23   : {rollover_share['clv3']*100:.1f}%")
    print(f"  ex-rollover edge             : {e_out[0]:+.3f}p (t={e_out[1]:+.1f}, n={e_out[2]})")
    print(f"  share in Fri-late/week-open  : {weekend_share['clv3']*100:.1f}% (of ex-rollover)")
    print(f"  fully decontaminated edge    : {e_clean[0]:+.3f}p (t={e_clean[1]:+.1f}, n={e_clean[2]})")
    print(f"  roundtrip cost (EURUSD)      : {cost:.3f}p")
    print(f"  decontaminated NET of cost   : {e_clean[0]-cost:+.3f}p")
    print(f"  cross-pair same-sign frac    : {same_sign_frac*100:.0f}% (USD-favoring={len(usd_long)}/7)")
    verdict = "TRADABLE" if (e_clean[0] - cost) > 0 and e_clean[1] and e_clean[1] > 2 else "NOT TRADABLE (artifact/cost-dominated)"
    print(f"\n  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
