"""短期足ブレイク/チャネル(tsmom以外の順張り)が短期モメンタムを取れるか。

検証対象(H1主):
  A) Donchian ブレイク(entry本高安更新で順張り + 短期 exit本でトレーリング手仕舞い)
  B) 前日(D1)高安ブレイク = opening-range/前日レンジブレイク的

7メジャー、GROSS / NET(通常スプレッド)/ NET(半スプレッド)を全部測る。
plateau(entry帯×exit帯で滑らかに正か)と breadth(pos>=5/7)を見る。
アーティファクト除染: UTC20-23の新規エントリー禁止版でも生き残るか。

実行: cwd から  uv run python -m research.experiments.exp_stf_breakout
"""

from __future__ import annotations

import copy
import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
LOW_SPREAD3 = ["EURUSD", "USDJPY", "GBPUSD"]  # seedで正に出た3ペア

_ORIG_SPREADS = copy.deepcopy(C.SPREADS_PIPS)
_ORIG_COMM = C.COMMISSION_FRACTION


def set_cost(mode: str):
    """mode: gross / full / half  (グローバルなコスト設定を切替)"""
    global _ORIG_SPREADS, _ORIG_COMM
    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in _ORIG_SPREADS}
        C.COMMISSION_FRACTION = 0.0
    elif mode == "half":
        C.SPREADS_PIPS = {k: v * 0.5 for k, v in _ORIG_SPREADS.items()}
        C.COMMISSION_FRACTION = _ORIG_COMM
    elif mode == "full":
        C.SPREADS_PIPS = copy.deepcopy(_ORIG_SPREADS)
        C.COMMISSION_FRACTION = _ORIG_COMM
    else:
        raise ValueError(mode)


# ---------------------------------------------------------------------------
# A) Donchian ブレイク(短期足の順張り)
# ---------------------------------------------------------------------------
def donchian_signals(data: pd.DataFrame, entry: int = 24, exit: int = 8,
                     no_late: bool = False):
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    exit_upper = high.rolling(exit).max().shift()
    exit_lower = low.rolling(exit).min().shift()

    long_e = close > upper
    short_e = close < lower
    long_x = close < exit_lower
    short_x = close > exit_upper

    if no_late:
        hour = data.index.hour
        ok = ~((hour >= 20) & (hour <= 23))
        oks = pd.Series(ok, index=data.index)
        long_e = long_e & oks
        short_e = short_e & oks
    return long_e, long_x, short_e, short_x


# ---------------------------------------------------------------------------
# B) 前日高安ブレイク(prev-day range breakout)
#    上位足D1の確定済み高安を当日H1バーが抜けたら順張り。出口は当日レンジ反対側 or 日替わり。
# ---------------------------------------------------------------------------
def prevday_signals(data: pd.DataFrame, exit: int = 8, no_late: bool = False):
    high, low, close = data["high"], data["low"], data["close"]
    # 各バーの「前営業日」高安(その日の確定値を使うため日付シフト)
    day = data.index.normalize()
    dayhigh = high.groupby(day).transform("max")
    daylow = low.groupby(day).transform("min")
    # 前日の高安: 日単位の値を1日ずらす
    daily_high = high.groupby(day).max()
    daily_low = low.groupby(day).min()
    prev_high = daily_high.shift(1)
    prev_low = daily_low.shift(1)
    pday_high = day.map(prev_high)
    pday_low = day.map(prev_low)
    pday_high = pd.Series(pday_high, index=data.index)
    pday_low = pd.Series(pday_low, index=data.index)

    long_e = close > pday_high
    short_e = close < pday_low
    # 出口: 短期 exit 本の反対極値割れ(donchianと同じトレーリング)
    exit_upper = high.rolling(exit).max().shift()
    exit_lower = low.rolling(exit).min().shift()
    long_x = close < exit_lower
    short_x = close > exit_upper

    if no_late:
        hour = data.index.hour
        ok = ~((hour >= 20) & (hour <= 23))
        oks = pd.Series(ok, index=data.index)
        long_e = long_e & oks
        short_e = short_e & oks
    return long_e, long_x, short_e, short_x


def sharpe_one(pair, tf, sigfn, params, data, side="both"):
    pf = run(pair, tf, sigfn, params, data=data, size_mode="value", side=side)
    m = metrics(pf)
    return float(m["sharpe"].iloc[0]), int(m["num_trades"].iloc[0])


def eval_grid(sigfn, grid, tf="H1", no_late=False, pairs=PAIRS):
    """grid = list of param dicts. 各セル×各ペアの GROSS Sharpe を返す。"""
    datacache = {p: load(p, tf) for p in pairs}
    rows = []
    set_cost("gross")
    for params in grid:
        p2 = dict(params)
        if no_late:
            p2["no_late"] = True
        sh = {}
        nt = {}
        for p in pairs:
            s, n = sharpe_one(p, tf, sigfn, p2, datacache[p])
            sh[p] = s
            nt[p] = n
        vals = np.array([sh[p] for p in pairs])
        rows.append({
            "params": params,
            "mean": float(vals.mean()),
            "pos": int((vals > 0).sum()),
            "low3_mean": float(np.mean([sh[p] for p in LOW_SPREAD3])),
            "low3_pos": int(sum(sh[p] > 0 for p in LOW_SPREAD3)),
            **{f"sh_{p}": sh[p] for p in pairs},
            "trades_mean": float(np.mean([nt[p] for p in pairs])),
        })
    return pd.DataFrame(rows), datacache


def eval_net(sigfn, params, datacache, tf="H1", pairs=PAIRS, mode="full"):
    set_cost(mode)
    sh = {}
    for p in pairs:
        s, _ = sharpe_one(p, tf, sigfn, params, datacache[p])
        sh[p] = s
    vals = np.array([sh[p] for p in pairs])
    return {
        "mean": float(vals.mean()),
        "pos": int((vals > 0).sum()),
        "low3_mean": float(np.mean([sh[p] for p in LOW_SPREAD3])),
        "low3_pos": int(sum(sh[p] > 0 for p in LOW_SPREAD3)),
        **{p: sh[p] for p in pairs},
    }


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("=" * 90)
    print("A) DONCHIAN BREAKOUT H1 — GROSS sweep (entry x exit)")
    print("=" * 90)
    donch_grid = []
    for entry in [12, 20, 24, 36, 48, 72]:
        for ex in [6, 8, 12]:
            donch_grid.append({"entry": entry, "exit": ex})
    dfA, cacheA = eval_grid(donchian_signals, donch_grid, tf="H1")
    show = dfA[["params", "mean", "pos", "low3_mean", "low3_pos", "trades_mean"]]
    print(show.to_string(index=False))
    bestA = dfA.sort_values("mean", ascending=False).iloc[0]
    bestA_low3 = dfA.sort_values("low3_mean", ascending=False).iloc[0]
    print("\nBEST by 7-pair mean:", bestA["params"], "mean=%.3f pos=%d" % (bestA["mean"], bestA["pos"]))
    print("BEST by low3 mean :", bestA_low3["params"], "low3=%.3f pos=%d" % (bestA_low3["low3_mean"], bestA_low3["low3_pos"]))

    print("\n" + "=" * 90)
    print("B) PREV-DAY RANGE BREAKOUT H1 — GROSS sweep (exit only)")
    print("=" * 90)
    pday_grid = [{"exit": ex} for ex in [4, 6, 8, 12, 24]]
    dfB, cacheB = eval_grid(prevday_signals, pday_grid, tf="H1")
    showB = dfB[["params", "mean", "pos", "low3_mean", "low3_pos", "trades_mean"]]
    print(showB.to_string(index=False))
    bestB = dfB.sort_values("mean", ascending=False).iloc[0]
    bestB_low3 = dfB.sort_values("low3_mean", ascending=False).iloc[0]
    print("\nBEST by 7-pair mean:", bestB["params"], "mean=%.3f pos=%d" % (bestB["mean"], bestB["pos"]))
    print("BEST by low3 mean :", bestB_low3["params"], "low3=%.3f pos=%d" % (bestB_low3["low3_mean"], bestB_low3["low3_pos"]))

    # ---- pick the best donchian cell, run NET full/half + decontam ----
    print("\n" + "=" * 90)
    print("NET + DECONTAM on best Donchian cells")
    print("=" * 90)
    cand = [bestA["params"], bestA_low3["params"]]
    # unique
    seen = set()
    uniq = []
    for c in cand:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    for params in uniq:
        print(f"\n--- Donchian {params} ---")
        for mode in ["gross", "full", "half"]:
            r = eval_net(donchian_signals, params, cacheA, mode=mode)
            print(f"  [{mode:5}] 7pair mean={r['mean']:+.3f} pos={r['pos']}/7 | "
                  f"low3 mean={r['low3_mean']:+.3f} pos={r['low3_pos']}/3")
        # decontam (UTC20-23 exclude) GROSS + NET
        print("  -- UTC20-23 entry-excluded (decontam) --")
        for mode in ["gross", "full", "half"]:
            p2 = dict(params); p2["no_late"] = True
            r = eval_net(donchian_signals, p2, cacheA, mode=mode)
            print(f"  [{mode:5}] 7pair mean={r['mean']:+.3f} pos={r['pos']}/7 | "
                  f"low3 mean={r['low3_mean']:+.3f} pos={r['low3_pos']}/3")

    # ---- long/short split + rollover signature on the 7-pair-best donchian ----
    print("\n" + "=" * 90)
    print("LONG/SHORT split (GROSS) on best Donchian — rollover signature check")
    print("=" * 90)
    set_cost("gross")
    bp = bestA["params"]
    for side in ["long", "short"]:
        sh = {}
        for p in PAIRS:
            s, _ = sharpe_one(p, "H1", donchian_signals, bp, cacheA[p], side=side)
            sh[p] = s
        vals = np.array([sh[p] for p in PAIRS])
        print(f"  {side:5}: mean={vals.mean():+.3f} pos={(vals>0).sum()}/7  " +
              " ".join(f"{p}={sh[p]:+.2f}" for p in PAIRS))

    print("\n--- per-pair GROSS detail (best 7-pair donchian) ---")
    set_cost("gross")
    detail = eval_net(donchian_signals, bp, cacheA, mode="gross")
    print("  " + " ".join(f"{p}={detail[p]:+.3f}" for p in PAIRS))

    set_cost("full")  # restore
    print("\nDONE")
