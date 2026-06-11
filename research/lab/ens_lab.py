"""ens_lab — 複数ストリーム(時間足・戦略変種)を1口座に統合する MtM シミュ基盤。

mm_lab.simulate の拡張版。プールに stream(ストリーム名)と w(トレード相対ウェイト)列を
持たせ、ストリーム別の同時建玉上限(per-stream budget)と z-power サイジングで統合口座を回す。
用途: 時間足アンサンブル(H4+D1)、分割系の変種比較など「チャンピオン族の枠を増やす」検証。

契約:
  pool 列: instr, entry, exit, dir, entry_price, ret, z_entry, stream, w
    - entry/exit はシミュレーショングリッド(closes.index)に searchsorted で写像される。
      時間足が違うストリームは事前に「実約定時刻に最も近いグリッド時刻」へシフトしておくこと
      (例: D1 ラベル 00:00 の約定は翌日 00:00 = H4 ラベル 20:00 の終値 → +20h シフト)。
  sizing: alloc = equity_real * (k / slots_total) * w * f(z)/f̄_stream
    f(z) = clip((|z|/2.2)^2.0, 0.3, 3.0)(mm_production と同一)
  budgets: {stream: max_pos}。合計が slots_total。

実行はスクリプト側から(これはライブラリ)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z0, P, CLIP_LO, CLIP_HI = 2.2, 2.0, 0.3, 3.0


def fz(z: float) -> float:
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def stream_fbars(pool: pd.DataFrame) -> dict:
    out = {}
    for s, g in pool.groupby("stream"):
        out[s] = float(np.mean([fz(z) for z in g["z_entry"].to_numpy()])) or 1.0
    return out


def simulate_streams(pool: pd.DataFrame, closes: pd.DataFrame, k: float, budgets: dict,
                     *, fbars: dict | None = None, init: float = 10_000.0, gate=None):
    """ストリーム別建玉枠つき統合口座シミュ(複利・MtM)。

    gate: 任意。gate(stream, dd_mtm) -> サイズ乗数(0で見送り)。
    例: DD連動トレンド = lambda s, dd: (1.0 if dd < -0.05 else 0.0) if s == "trend" else 1.0
    """
    fbars = fbars or stream_fbars(pool)
    slots_total = int(sum(budgets.values()))
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    e_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    x_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry: dict[int, list[int]] = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(e_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    ep_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    w_arr = pool["w"].to_numpy().astype(float)
    s_arr = pool["stream"].to_numpy()

    equity = init
    open_pos: list[dict] = []
    n_open_by = {s: 0 for s in budgets}
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    skipped = 0
    conc = []
    peak_mtm = init

    base = k / max(slots_total, 1)

    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
                    n_open_by[p["stream"]] -= 1
                else:
                    still.append(p)
            open_pos = still
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            unreal += p["alloc"] * (p["dir"] * (px / p["eprice"] - 1.0))
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity
        peak_mtm = max(peak_mtm, mtm)
        dd_mtm = mtm / peak_mtm - 1.0

        if b in by_entry:
            for ti in by_entry[b]:
                s = s_arr[ti]
                if s not in budgets or n_open_by[s] >= budgets[s]:
                    skipped += 1
                    continue
                gm = 1.0 if gate is None else float(gate(s, dd_mtm))
                if gm <= 0:
                    skipped += 1
                    continue
                alloc = equity * base * w_arr[ti] * gm * (fz(float(z_arr[ti])) / fbars[s])
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({"col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": ep_arr[ti], "alloc": alloc,
                                 "exit_pos": int(x_pos[ti]), "ret": float(ret_arr[ti]),
                                 "stream": s})
                n_open_by[s] += 1
                conc.append(len(open_pos))

    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0,
            "avg_conc": float(np.mean(conc)) if conc else 0.0}
    return pd.Series(eq_mtm, index=grid), pd.Series(eq_real, index=grid), info


def calibrate_streams(pool, closes, budgets, *, fbars=None, target_dd=0.20,
                      lo=0.02, hi=16.0, iters=22, gate=None):
    fbars = fbars or stream_fbars(pool)

    def dd_of(k):
        eqm, _, _ = simulate_streams(pool, closes, k, budgets, fbars=fbars, gate=gate)
        return abs(float((eqm / eqm.cummax() - 1.0).min()))

    if dd_of(hi) <= target_dd:
        eqm, eqr, info = simulate_streams(pool, closes, hi, budgets, fbars=fbars, gate=gate)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        if dd_of(mid) > target_dd:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate_streams(pool, closes, lo, budgets, fbars=fbars, gate=gate)
    return lo, eqm, eqr, info
