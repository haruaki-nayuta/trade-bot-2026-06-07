"""Independent re-implementation audit of exp65 theta=0.5% pyramid tranches.

Written from the spec, NOT copied from exp65.build_addons. Run:
  PYTHONPATH=. uv run python research/outputs/verify65_indep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing, _fz  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
)
from fxlab import universe as uni  # noqa: E402

THETA = 0.005
MAX_POS = 8
OOS = pd.Timestamp("2022-01-01", tz="UTC")

t0 = time.time()
uni.register_cross_spreads(3.0)
pool = pd.read_parquet(ROOT / "results" / "mm_pool_v2d1_H4_19.parquet")
pool = pool.sort_values("entry").reset_index(drop=True)
print(f"pool: {len(pool)} trades, {pool['instr'].nunique()} instruments")

# ---------- Part 1: independent tranche reconstruction --------------------
closes_by = {i: uni.instrument_data(i, "H4")["close"] for i in sorted(pool["instr"].unique())}
zfull_by = {i: ((c - c.rolling(50).mean()) / c.rolling(50).std())
            for i, c in closes_by.items()}

recs = []
viol = {"trigger_not_interior": 0, "earlier_hit_missed": 0, "below_theta": 0}
for ti in range(len(pool)):
    row = pool.iloc[ti]
    c = closes_by[row["instr"]]
    e = c.index.get_loc(row["entry"])
    x = c.index.get_loc(row["exit"])
    assert x > e, f"trade {ti}: exit not after entry"
    d = float(row["dir"])
    ce, cx = float(c.iloc[e]), float(c.iloc[x])
    cost = d * (cx / ce - 1.0) - float(row["ret"])  # parent round-trip cost (back out)
    if x - e < 2:
        continue  # no interior bar
    interior = c.to_numpy()[e + 1: x]               # bars strictly between entry and exit
    prof = d * (interior / ce - 1.0)                # unrealized P&L path on closes
    hits = np.flatnonzero(prof >= THETA)
    if hits.size == 0:
        continue
    j = e + 1 + int(hits[0])                        # first bar reaching +theta
    # empirical causality checks
    if not (e < j < x):
        viol["trigger_not_interior"] += 1
    if hits[0] > 0 and (prof[:hits[0]] >= THETA).any():
        viol["earlier_hit_missed"] += 1
    if prof[hits[0]] < THETA:
        viol["below_theta"] += 1
    cj = float(c.iloc[j])
    fwd = d * (cx / cj - 1.0) - cost                # forward net return, full round-trip cost
    slip = float(row["entry_price"]) / ce
    recs.append({
        "instr": row["instr"], "entry": c.index[j], "exit": row["exit"],
        "dir": int(d), "entry_price": cj * slip, "ret": fwd,
        "bars_held": int(x - j),
        "z_entry": abs(float(zfull_by[row["instr"]].iloc[j])),
        "vol_entry": np.nan,
    })

add = pd.DataFrame(recs)
is_m = add["entry"] < OOS
print("\n--- Part 1: tranche reconstruction (independent) ---")
print(f"n        = {len(add)}    (expected 220)")
print(f"sum      = {add['ret'].sum():+.6f} (expected +0.359125)")
print(f"IS / OOS = {add['ret'][is_m].sum():+.6f} / {add['ret'][~is_m].sum():+.6f}"
      f" (expected +0.176447 / +0.182678)")
print(f"win rate = {(add['ret'] > 0).mean():.4f} (expected 0.7909)")
print(f"mean bps = {add['ret'].mean() * 1e4:+.2f} (expected +16.32)")
print(f"causality violations: {viol}")

# ---------- Part 2: account-level re-evaluation ---------------------------
closes = mm.load_closes()
aug = pd.concat([pool, add], ignore_index=True).sort_values("entry").reset_index(drop=True)


def make_eq(pl):
    mk = champion_sizing(pl, max_pos=MAX_POS)
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    return eq_of_k


print(f"\n--- Part 2: account stage  [{time.time()-t0:.0f}s] ---")
out = {}
for tag, pl in (("base", pool), ("pyr0.5%", aug)):
    f = make_eq(pl)
    k_e = calibrate_empirical(f, 0.20)
    eq_e = f(k_e)
    res = {"emp_k": k_e, "emp_cagr": cagr_of(eq_e), "emp_dd": max_dd(eq_e),
           "emp_p95": boot_dd(eq_e, n_boot=1500, seed=0)["p95"]}
    for sd in (0, 1):
        k_r = calibrate_robust_seeded(f, 0.20, seed=sd)
        res[f"rob{sd}_k"] = k_r
        res[f"rob{sd}_cagr"] = cagr_of(f(k_r))
    out[tag] = res
    print(f"  {tag:8s} emp k={k_e:.3f} CAGR={res['emp_cagr']:+.4%} dd={res['emp_dd']:+.2%} "
          f"p95={res['emp_p95']:+.2%} | rob s0 k={res['rob0_k']:.3f} {res['rob0_cagr']:+.4%} "
          f"| rob s1 k={res['rob1_k']:.3f} {res['rob1_cagr']:+.4%}  [{time.time()-t0:.0f}s]")

b, p = out["base"], out["pyr0.5%"]
print(f"  pair diff rob s0: {(p['rob0_cagr']-b['rob0_cagr'])*100:+.3f}pp (expected +0.546)")
print(f"  pair diff rob s1: {(p['rob1_cagr']-b['rob1_cagr'])*100:+.3f}pp (expected +0.108)")
print(f"  emp: k {b['emp_k']:.2f}->{p['emp_k']:.2f} (expected 8.89->7.20), "
      f"CAGR {b['emp_cagr']:.2%}->{p['emp_cagr']:.2%} (expected 27.41->25.66)")
print(f"  p95: {b['emp_p95']:.2%}->{p['emp_p95']:.2%} (expected -27.3->-25.3)")

# ---------- Part 4: leverage / margin audit (replay with exposure track) ---
print(f"\n--- Part 4: exposure audit at emp_k of augmented pool  [{time.time()-t0:.0f}s] ---")
aug2 = aug.copy()
# flag tranches: mark rows whose (instr, entry, ret) match the addon table
key_add = set(zip(add["instr"], add["entry"], add["ret"]))
aug2["is_tranche"] = [
    (r.instr, r.entry, r.ret) in key_add for r in aug2.itertuples()]
assert aug2["is_tranche"].sum() == len(add)

k_use = p["emp_k"]
mk = champion_sizing(aug2, max_pos=MAX_POS)
sizing = mk(k_use)
fbar = float(np.mean([_fz(z) for z in aug2["z_entry"].to_numpy()]))

grid = closes.index
gi = grid.to_numpy()
carr = closes.to_numpy()
col_of = {c: i for i, c in enumerate(closes.columns)}
n = len(grid)
entry_pos = np.clip(np.searchsorted(gi, aug2["entry"].to_numpy(), side="left"), 0, n - 1)
exit_pos = np.clip(np.searchsorted(gi, aug2["exit"].to_numpy(), side="left"), 0, n - 1)
by_entry = {}
for ti in range(len(aug2)):
    by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

equity, peak = 10_000.0, 10_000.0
open_pos = []
max_expo_mtm = max_expo_real = 0.0
expo_at_max = None
alloc_frac = {"tranche": [], "parent": []}
fz_mult = {"tranche": [], "parent": []}
skipped = 0
for bbar in range(n):
    if open_pos:
        open_pos2 = []
        for pp in open_pos:
            if pp["exit_pos"] <= bbar:
                equity += pp["alloc"] * pp["ret"]
            else:
                open_pos2.append(pp)
        open_pos = open_pos2
    unreal = sum(pp["alloc"] * pp["dir"] * (carr[bbar, pp["col"]] / pp["eprice"] - 1.0)
                 for pp in open_pos)
    mtm = equity + unreal
    peak = max(peak, mtm)
    if bbar in by_entry:
        for ti in by_entry[bbar]:
            if len(open_pos) >= MAX_POS:
                skipped += 1
                continue
            ctx = {"equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak,
                   "dd_mtm": mtm / peak - 1.0, "n_open": len(open_pos),
                   "max_pos": MAX_POS, "recent_vol": float("nan"),
                   "z": float(aug2.at[ti, "z_entry"]), "instr": aug2.at[ti, "instr"],
                   "ret": float(aug2.at[ti, "ret"]), "bars_held": int(aug2.at[ti, "bars_held"])}
            alloc = float(sizing(ctx))
            if alloc <= 0:
                skipped += 1
                continue
            open_pos.append({"col": col_of[aug2.at[ti, "instr"]],
                             "dir": float(aug2.at[ti, "dir"]),
                             "eprice": float(aug2.at[ti, "entry_price"]),
                             "alloc": alloc, "exit_pos": int(exit_pos[ti]),
                             "ret": float(aug2.at[ti, "ret"])})
            kind = "tranche" if aug2.at[ti, "is_tranche"] else "parent"
            alloc_frac[kind].append(alloc / equity)
            fz_mult[kind].append(_fz(ctx["z"]) / fbar)
    gross = sum(pp["alloc"] for pp in open_pos)
    if mtm > 0 and gross / mtm > max_expo_mtm:
        max_expo_mtm = gross / mtm
        expo_at_max = (grid[bbar], len(open_pos), gross / equity)
    if equity > 0:
        max_expo_real = max(max_expo_real, gross / equity)

print(f"  k(emp, aug) = {k_use:.3f}, fbar(aug) = {fbar:.4f}")
print(f"  max gross exposure / MtM equity  = {max_expo_mtm:.2f}x "
      f"(at {expo_at_max[0]}, {expo_at_max[1]} slots, /real={expo_at_max[2]:.2f}x)")
print(f"  max gross exposure / real equity = {max_expo_real:.2f}x  (JP retail cap = 25x)")
print(f"  theoretical cap k*f_hi/fbar      = {k_use * 3.0 / fbar:.2f}x")
nt, npar = len(alloc_frac['tranche']), len(alloc_frac['parent'])
print(f"  entries taken: parent {npar}, tranche {nt} (skipped total {skipped})")
print(f"  tranche alloc/equity: mean {np.mean(alloc_frac['tranche']):.3f} "
      f"median {np.median(alloc_frac['tranche']):.3f} max {np.max(alloc_frac['tranche']):.3f}")
print(f"  parent  alloc/equity: mean {np.mean(alloc_frac['parent']):.3f} "
      f"median {np.median(alloc_frac['parent']):.3f} max {np.max(alloc_frac['parent']):.3f}")
print(f"  f(z)/fbar multiplier: tranche mean {np.mean(fz_mult['tranche']):.3f} "
      f"vs parent mean {np.mean(fz_mult['parent']):.3f}")
at_lo = np.mean([abs(m * fbar - 0.3) < 1e-9 for m in fz_mult["tranche"]])
print(f"  tranches at clip floor f=0.3: {at_lo:.1%}")
print(f"\ndone [{time.time()-t0:.0f}s]")
