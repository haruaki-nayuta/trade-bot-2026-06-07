"""wickbias_h4_majors — does the wick/CLV directional edge survive at H4 across the 7 majors?

Angle: H4 is the CHAMPION's timeframe (confluence_meanrev_v2). We test whether a
directional wick edge exists at H4 on the 7 REAL-OHLC majors (cross pairs are
synthetic close-only = no wicks, excluded).

Features (fraction of bar range, bar t fully formed, NO look-ahead):
  clv1   = (c-l)/rng                    (close location value; low=pinned at low)
  clv3   = clv1.rolling(3).mean()
  wick_diff1 = lo_w - up_w              (+ = lower wick dominates / hammer-ish)
  wick_diff3 = wick_diff1.rolling(3).mean()

Target: next-bar close-to-close (pips), plus multi-horizon h1/3/5/10/20.

Reported per pair:
  - Spearman IC train/test
  - extreme-bin (q=0.05 and q=0.10) conditional mean pips + t (lo bin & hi bin), test
Then:
  - sign-consistency table across the 7 majors (is it systematic or random?)
  - POOLED: z-score each feature WITHIN pair, concat all 7, report pooled IC + bin t
  - 20-23 UTC concentration check (H4 has few rollover bars but flag anyway)
  - compare bin mean pips to each pair's round-trip spread cost (spread_pips)

Run: uv run python -m research.experiments.wickbias_h4_majors
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab.config import pip_size, spread_pips
from research.lab.nextbar_common import (
    SPLIT,
    horizon_targets,
    load_xy,
)

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "H4"
FEATS = ["wick_diff1", "wick_diff3", "clv1", "clv3"]
QS = (0.05, 0.10)
# H4 bars open at UTC 0,4,8,12,16,20. The 20:00 bar spans the NY 17:00 rollover.
ROLLOVER_HOURS = [20]


def wick_features(df: pd.DataFrame) -> dict[str, pd.Series]:
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


def bin_stats(feat: pd.Series, tgt: pd.Series, q: float):
    """train-quantile extreme bins, test-period conditional mean pips + t.

    Returns dict with lo/hi test mean pips, t, n.  Sign convention: positive
    feat (high clv / lower-wick-dominant) is the 'hi' bin.
    """
    m = feat.notna() & tgt.notna() & np.isfinite(feat) & np.isfinite(tgt)
    f, y = feat[m], tgt[m]
    tr = f.index < SPLIT
    te = ~tr
    lo_thr, hi_thr = f[tr].quantile(q), f[tr].quantile(1 - q)
    out = {}
    for tag, mb in [("lo", f <= lo_thr), ("hi", f >= hi_thr)]:
        yy = y[mb & te]
        n = len(yy)
        mean = float(yy.mean()) if n else np.nan
        t = float(yy.mean() / (yy.std() / np.sqrt(n))) if n > 2 and yy.std() > 0 else np.nan
        out[f"{tag}_mean"] = mean
        out[f"{tag}_t"] = t
        out[f"{tag}_n"] = n
    return out


def ic(feat: pd.Series, tgt: pd.Series):
    m = feat.notna() & tgt.notna() & np.isfinite(feat) & np.isfinite(tgt)
    f, y = feat[m], tgt[m]
    tr = f.index < SPLIT
    te = ~tr
    ic_tr = float(f[tr].rank().corr(y[tr].rank())) if tr.sum() > 2 else np.nan
    ic_te = float(f[te].rank().corr(y[te].rank())) if te.sum() > 2 else np.nan
    return ic_tr, ic_te


def main() -> None:
    # collect per-pair data once
    data = {}
    for pair in MAJORS:
        df, tgt, pip = load_xy(pair, TF)
        feats = wick_features(df)
        data[pair] = {"df": df, "tgt": tgt, "pip": pip, "feats": feats}

    span = next(iter(data.values()))["df"]
    print("=" * 100)
    print(f"WICK BIAS @ {TF} — 7 MAJORS (real OHLC). span {span.index[0].date()}..{span.index[-1].date()}, "
          f"train<{SPLIT.date()}<=test")
    print(f"H4 bars/pair ~ {len(span):,}. Round-trip cost = spread_pips (entry+exit = 1 spread).")
    print("=" * 100)

    # ---------- per feature: per-pair IC + extreme bins ----------
    sign_table = {}  # feat -> list of (pair, lo_t, hi_t) for consistency
    for feat_name in FEATS:
        print(f"\n{'#'*100}\n## FEATURE: {feat_name}   (sign: +feat = high CLV / lower-wick-dominant)\n{'#'*100}")
        for q in QS:
            print(f"\n--- extreme-bin q={q} (lo bin = bottom {q:.0%}, hi bin = top {q:.0%}) ---")
            print(f"{'pair':<8} {'cost':>5} | {'IC_tr':>7} {'IC_te':>7} | "
                  f"{'lo_mean':>8} {'lo_t':>6} {'lo_n':>5} | {'hi_mean':>8} {'hi_t':>6} {'hi_n':>5}")
            rows = []
            for pair in MAJORS:
                d = data[pair]
                f = d["feats"][feat_name]
                ic_tr, ic_te = ic(f, d["tgt"])
                bs = bin_stats(f, d["tgt"], q)
                cost = spread_pips(pair)
                rows.append((pair, q, bs))
                print(f"{pair:<8} {cost:>5.1f} | {ic_tr:>+7.4f} {ic_te:>+7.4f} | "
                      f"{bs['lo_mean']:>+8.3f} {bs['lo_t']:>+6.2f} {bs['lo_n']:>5d} | "
                      f"{bs['hi_mean']:>+8.3f} {bs['hi_t']:>+6.2f} {bs['hi_n']:>5d}")
            # store q=0.05 row signs for the consistency summary
            if q == 0.05:
                sign_table[feat_name] = rows

    # ---------- sign consistency across the 7 majors ----------
    print(f"\n{'='*100}\nSIGN CONSISTENCY across 7 majors (q=0.05, test) — systematic or random?\n{'='*100}")
    print("For a real mean-reversion bounce after a low-CLV/upper-wick close, expect LO bin > 0.")
    print(f"{'feature':<12} | {'lo>0 cnt':>9} {'lo mean-t':>10} | {'hi>0 cnt':>9} {'hi mean-t':>10} | {'lo-hi spread':>12}")
    for feat_name in FEATS:
        rows = sign_table[feat_name]
        lo_ts = [r[2]["lo_t"] for r in rows if not np.isnan(r[2]["lo_t"])]
        hi_ts = [r[2]["hi_t"] for r in rows if not np.isnan(r[2]["hi_t"])]
        lo_pos = sum(1 for t in lo_ts if t > 0)
        hi_pos = sum(1 for t in hi_ts if t > 0)
        lo_means = [r[2]["lo_mean"] for r in rows if not np.isnan(r[2]["lo_mean"])]
        hi_means = [r[2]["hi_mean"] for r in rows if not np.isnan(r[2]["hi_mean"])]
        spread = np.mean(lo_means) - np.mean(hi_means)
        print(f"{feat_name:<12} | {lo_pos:>3}/{len(lo_ts):<5} {np.mean(lo_ts):>+10.2f} | "
              f"{hi_pos:>3}/{len(hi_ts):<5} {np.mean(hi_ts):>+10.2f} | {spread:>+12.3f}")
    print("(lo>0 cnt = how many of 7 pairs have positive lo-bin mean; 7/7 or 0/7 = systematic, ~3-4/7 = random)")
    print("(lo-hi spread = avg lo mean minus avg hi mean across pairs; the directional spread the feature captures)")

    # ---------- POOLED: z-score within pair, concat ----------
    print(f"\n{'='*100}\nPOOLED (z-score each feature within pair, concat all 7 majors)\n{'='*100}")
    print(f"{'feature':<12} | {'IC_tr':>8} {'IC_te':>8} | "
          f"{'lo_mean':>8} {'lo_t':>6} {'lo_n':>6} | {'hi_mean':>8} {'hi_t':>6} {'hi_n':>6}")
    for feat_name in FEATS:
        pooled_f = []
        pooled_y = []
        for pair in MAJORS:
            d = data[pair]
            f = d["feats"][feat_name]
            y = d["tgt"]
            m = f.notna() & y.notna() & np.isfinite(f) & np.isfinite(y)
            fz = f[m]
            # z-score within pair (full-sample standardization is fine; it's monotone, IC unaffected;
            # for bins we need comparable scale across pairs, so z within pair)
            fz = (fz - fz.mean()) / fz.std()
            yy = y[m]
            pf = pd.Series(fz.values, index=fz.index)
            py = pd.Series(yy.values, index=yy.index)
            pooled_f.append(pf)
            pooled_y.append(py)
        # concat preserving timestamps so train/test split still works
        PF = pd.concat(pooled_f)
        PY = pd.concat(pooled_y)
        ic_tr, ic_te = ic(PF, PY)
        bs = bin_stats(PF, PY, 0.05)
        print(f"{feat_name:<12} | {ic_tr:>+8.4f} {ic_te:>+8.4f} | "
              f"{bs['lo_mean']:>+8.3f} {bs['lo_t']:>+6.2f} {bs['lo_n']:>6d} | "
              f"{bs['hi_mean']:>+8.3f} {bs['hi_t']:>+6.2f} {bs['hi_n']:>6d}")
    print("Note: pooled bins mix pairs; mean pips are in raw pips (cross-pair pip values differ; "
          "interpret t & sign, not the absolute pip magnitude).")

    # ---------- multi-horizon decay (pooled-ish: show EURUSD + GBPUSD, the cleanest) ----------
    print(f"\n{'='*100}\nMULTI-HORIZON decay (test, q=0.05 lo & hi bins) — does any edge persist or fade?\n{'='*100}")
    for pair in ["EURUSD", "GBPUSD", "USDJPY"]:
        d = data[pair]
        hts = horizon_targets(d["df"], pair)
        print(f"\n[{pair}] (cost rt = {spread_pips(pair):.1f}p)")
        for feat_name in FEATS:
            f = d["feats"][feat_name]
            m = f.notna() & np.isfinite(f)
            ff = f[m]
            tr = ff.index < SPLIT
            lo_thr, hi_thr = ff[tr].quantile(0.05), ff[tr].quantile(0.95)
            te_idx = ff.index[~tr]
            lo_idx = ff[te_idx][ff[te_idx] <= lo_thr].index
            hi_idx = ff[te_idx][ff[te_idx] >= hi_thr].index
            lo_h = {f"h{h}": round(float(ht.reindex(lo_idx).mean()), 2) for h, ht in hts.items()}
            hi_h = {f"h{h}": round(float(ht.reindex(hi_idx).mean()), 2) for h, ht in hts.items()}
            print(f"  {feat_name:<12} lo {lo_h}")
            print(f"  {feat_name:<12} hi {hi_h}")

    # ---------- 20-23 UTC concentration (the 20:00 H4 bar spans rollover) ----------
    print(f"\n{'='*100}\nROLLOVER CHECK — is any edge concentrated in the UTC-20 H4 bar? (decontaminate)\n{'='*100}")
    print("Compares q=0.05 lo & hi bin test mean pips: ALL bars vs EXCLUDING the 20:00 bar.")
    print(f"{'pair':<8} {'feat':<12} | {'lo all':>8} {'lo ex20':>8} | {'hi all':>8} {'hi ex20':>8}")
    for pair in MAJORS:
        d = data[pair]
        for feat_name in FEATS:
            f = d["feats"][feat_name]
            tgt = d["tgt"]
            norol = ~np.isin(f.index.hour, ROLLOVER_HOURS)
            # all
            bs_all = bin_stats(f, tgt, 0.05)
            # ex-20 (apply mask to both feat and tgt)
            fno = f.where(norol)
            bs_no = bin_stats(fno, tgt, 0.05)
            print(f"{pair:<8} {feat_name:<12} | "
                  f"{bs_all['lo_mean']:>+8.3f} {bs_no['lo_mean']:>+8.3f} | "
                  f"{bs_all['hi_mean']:>+8.3f} {bs_no['hi_mean']:>+8.3f}")

    # ---------- COST comparison summary ----------
    print(f"\n{'='*100}\nEDGE vs COST — best directional bin mean pips per pair vs round-trip spread\n{'='*100}")
    print("A directional wick trade pays the round-trip spread. |bin mean| must clear it to matter.")
    print(f"{'pair':<8} {'cost_rt':>8} | {'best |bin mean| (q05)':>22} {'feature/bin':>16} {'clears cost?':>13}")
    for pair in MAJORS:
        d = data[pair]
        cost = spread_pips(pair)
        best_abs = 0.0
        best_desc = ""
        for feat_name in FEATS:
            bs = bin_stats(d["feats"][feat_name], d["tgt"], 0.05)
            for tag in ("lo", "hi"):
                v = bs[f"{tag}_mean"]
                if not np.isnan(v) and abs(v) > best_abs:
                    best_abs = abs(v)
                    best_desc = f"{feat_name}/{tag}"
        clears = "YES" if best_abs > cost else "no"
        print(f"{pair:<8} {cost:>8.1f} | {best_abs:>22.3f} {best_desc:>16} {clears:>13}")


if __name__ == "__main__":
    main()
