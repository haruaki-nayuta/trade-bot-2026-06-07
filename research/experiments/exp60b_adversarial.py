"""exp60 の敵対検証: (a)合成の経験的maxDD、(b)bootシード安定性、(c)same-tail署名
(固定レバで empirical CAGR↑ かつ p95悪化 ならレバ偽装)、(d)IS/OOS頑健性、
(e)H1テール耐性チェック(block長感度)。
"""
from __future__ import annotations
import numpy as np, pandas as pd
import mm_lab as mm, mm_production as mp
from fxlab import universe as uni
from strategies import tsmom


def daily_block_bootstrap(daily_ret, n_boot=2000, block=21, seed=0, pct=5):
    r = np.asarray(daily_ret, float); r = r[np.isfinite(r)]; n = len(r)
    if n < block * 2: return float("nan")
    rng = np.random.default_rng(seed); nb = int(np.ceil(n / block))
    st = rng.integers(0, n - block, size=(n_boot, nb)); dds = np.empty(n_boot)
    for i in range(n_boot):
        idx = (st[i][:, None] + np.arange(block)).ravel()[:n]
        p = np.cumprod(1 + r[idx]); pk = np.maximum.accumulate(p); dds[i] = (p / pk - 1).min()
    return float(np.percentile(dds, pct))


def lever_to_p95(daily, target=0.20, n_boot=2000, block=21, seed=0, lo=0.05, hi=20.0, it=30):
    f = lambda L: abs(daily_block_bootstrap(daily * L, n_boot, block, seed))
    if f(hi) <= target: return hi
    if f(lo) > target: return lo
    for _ in range(it):
        m = (lo + hi) / 2
        if f(m) > target: hi = m
        else: lo = m
    return lo


def emp_maxdd(daily):
    eq = np.cumprod(1 + np.nan_to_num(daily)); return float((eq / np.maximum.accumulate(eq) - 1).min())
def cagr(daily, idx):
    eq = np.cumprod(1 + np.nan_to_num(daily)); yrs = (idx[-1] - idx[0]).days / 365.25
    return (eq[-1] ** (1 / yrs) - 1) if eq[-1] > 0 else -1.0
def to_daily(eqm): return eqm.resample("1D").last().dropna().pct_change().dropna()


def build():
    pool_c = mp.build_pool_d1(); closes_c = mm.load_closes()
    mk_c = mp.champion_sizing(pool_c, max_pos=8)
    _, eqm_c, _, _, _ = mm.calibrate_robust(pool_c, closes_c, mk_c, 0.20, 8, n_boot=800)
    pool_j = mm.build_pool_for(tsmom, {"lookback": 24, "band": 0.0}, tf="H1",
                               instruments=["USDJPY"], tag="tsmom_usdjpy_lb24", side="both", cache=False)
    closes_j = pd.DataFrame({"USDJPY": uni.instrument_close("USDJPY", "H1")}).sort_index().ffill()
    _, eqm_j, _, _, _ = mm.calibrate_robust(pool_j, closes_j, lambda k: (lambda c: c["equity_real"] * k),
                                            0.20, 1, n_boot=800)
    rc = to_daily(eqm_c); rj = to_daily(eqm_j)
    cm = rc.index.intersection(rj.index)
    return rc.reindex(cm).fillna(0.0), rj.reindex(cm).fillna(0.0), cm


def main():
    rc, rj, idx = build()
    rcv, rjv = rc.values, rj.values

    print("=== (a) 合成の経験的maxDD(p95=20%再較正後)===")
    print(f"  {'w':>5} {'L':>7} {'emp_maxDD':>10} {'robCAGR':>9}")
    base_L = lever_to_p95(rcv); base = cagr(rcv * base_L, idx)
    for w in [0.0, 0.2, 0.4, 0.5]:
        b = (1 - w) * rcv + w * rjv; L = lever_to_p95(b)
        print(f"  {w:>5.2f} {L:>7.3f} {emp_maxdd(b*L):>+10.1%} {cagr(b*L, idx):>+9.2%}")

    print("\n=== (b) bootシード安定性(w=0.4, 5シード)===")
    for sd in range(5):
        b = 0.6 * rcv + 0.4 * rjv; L = lever_to_p95(b, seed=sd)
        Lc = lever_to_p95(rcv, seed=sd)
        print(f"  seed{sd}: champ robCAGR={cagr(rcv*Lc, idx):+.2%}  "
              f"combined={cagr(b*L, idx):+.2%}  Δ={(cagr(b*L,idx)-cagr(rcv*Lc,idx))*100:+.2f}pp")

    print("\n=== (c) same-tail署名: 固定レバ(champの較正L)で empirical CAGR↑ かつ p95悪化か ===")
    Lc = lever_to_p95(rcv)
    for w in [0.0, 0.2, 0.4]:
        b = (1 - w) * rcv + w * rjv
        empc = cagr(b * Lc, idx); p95 = abs(daily_block_bootstrap(b * Lc))
        empdd = emp_maxdd(b * Lc)
        print(f"  w={w:.2f} @L={Lc:.3f}: empCAGR={empc:+.2%} emp_maxDD={empdd:+.1%} p95={p95:+.1%}")
    print("  (champ比でempCAGR↑&p95↑=偽装 / empCAGR↑&p95↓ or 不変=真の前進)")

    print("\n=== (d) IS(〜2021)較正→OOS(2021〜)素検証(w=0.4)===")
    cut = pd.Timestamp("2021-01-01", tz="UTC")
    mis = idx < cut; moos = idx >= cut
    for w in [0.0, 0.4]:
        b = (1 - w) * rcv + w * rjv
        L_is = lever_to_p95(b[mis])
        oos = b[moos] * L_is
        print(f"  w={w:.2f}: L_is={L_is:.3f}  OOS robCAGR(素)={cagr(oos, idx[moos]):+.2%}  "
              f"OOS emp_maxDD={emp_maxdd(oos):+.1%}")

    print("\n=== (e) block長感度(w=0.4 robCAGR; block=10/21/42/63)===")
    for blk in [10, 21, 42, 63]:
        b = 0.6 * rcv + 0.4 * rjv; L = lever_to_p95(b, block=blk)
        Lc = lever_to_p95(rcv, block=blk)
        print(f"  block={blk}: champ={cagr(rcv*Lc,idx):+.2%} combined={cagr(b*L,idx):+.2%} "
              f"Δ={(cagr(b*L,idx)-cagr(rcv*Lc,idx))*100:+.2f}pp")


if __name__ == "__main__":
    main()
