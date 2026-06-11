"""exp24: M1 粒度の MtM ドローダウン監査(誠実性チェック)。

mm_lab/mm_production の「MtM DD=20%」は H4 終値グリッドで測っている。だが口座の真の谷は
バー内(分単位)の含み損の谷であり、H4 終値はそれを過小評価しうる。本実験は較正済み口座
(乖離連動z, mp8, k=7.98 ≒ empirical 20%)の建玉ログを再生し、M1 終値グリッドで MtM equity を
再構成して真の最大DDを測る。次に「M1 粒度 DD=20%」になるよう k を再較正し、正直なベースライン
CAGR を更新する。

実行: PYTHONPATH=. uv run python research/experiments/exp24_m1_dd_audit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 200)


def simulate_with_log(pool, closes, sizing, *, init=10_000.0, max_pos=8):
    """mm_lab.simulate と同一ロジック + 建玉ログ(alloc/entry/exit)を返す。"""
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)

    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()

    equity = init
    peak_mtm = init
    open_pos = []
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    log = []
    skipped = 0

    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                if p["exit_pos"] <= b:
                    equity += p["alloc"] * p["ret"]
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
                if len(open_pos) >= max_pos:
                    skipped += 1
                    continue
                ctx = {"equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                       "dd_mtm": dd_mtm, "n_open": len(open_pos), "max_pos": max_pos,
                       "recent_vol": float("nan"), "z": float(z_arr[ti]),
                       "instr": instr_arr[ti], "ret": float(ret_arr[ti]), "bars_held": 0}
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    skipped += 1
                    continue
                open_pos.append({"ti": ti, "col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": eprice_arr[ti], "alloc": alloc,
                                 "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti])})
                log.append({"ti": ti, "instr": instr_arr[ti], "dir": dir_arr[ti],
                            "eprice": eprice_arr[ti], "alloc": alloc,
                            "entry_pos": b, "exit_pos": int(exit_pos[ti]),
                            "ret": float(ret_arr[ti])})
    return (pd.Series(eq_mtm, index=grid), pd.Series(eq_real, index=grid),
            pd.DataFrame(log), {"skipped": skipped})


def m1_close(instr: str) -> pd.Series:
    if instr in CROSS_DEFS:
        a, op, b = CROSS_DEFS[instr]
        df = pd.concat([load_m1(a)["close"], load_m1(b)["close"]], axis=1, keys=["a", "b"]).ffill().dropna()
        return (df["a"] / df["b"]) if op == "/" else (df["a"] * df["b"])
    return load_m1(instr)["close"]


def m1_dd_of_log(log: pd.DataFrame, closes: pd.DataFrame, base_grid_m1: np.ndarray,
                 init=10_000.0) -> tuple[float, pd.Series]:
    """建玉ログから M1 グリッド上の MtM equity を再構成し最大DDを返す。"""
    gi = closes.index.tz_localize(None).to_numpy()
    n_m1 = len(base_grid_m1)
    unreal = np.zeros(n_m1)
    # 実現 equity: exit バー終端時刻にステップ
    real = np.full(n_m1, init)
    exit_times = gi[log["exit_pos"].to_numpy()] + np.timedelta64(4, "h")
    pnl = (log["alloc"] * log["ret"]).to_numpy()
    order = np.argsort(exit_times)
    step_pos = np.searchsorted(base_grid_m1, exit_times[order], side="left")
    cum = 0.0
    last = 0
    for sp, v in zip(step_pos, pnl[order]):
        sp = min(sp, n_m1)
        if sp > last:
            real[last:sp] = init + cum
            last = sp
        cum += v
    real[last:] = init + cum

    for instr, gl in log.groupby("instr"):
        c = m1_close(instr)
        # 非破壊で tz-naive 化(load_m1 キャッシュの index を汚染しない)
        c = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
        px = c.reindex(pd.DatetimeIndex(base_grid_m1), method="ffill").to_numpy()
        for _, p in gl.iterrows():
            t_in = gi[int(p["entry_pos"])] + np.timedelta64(4, "h")
            t_out = gi[int(p["exit_pos"])] + np.timedelta64(4, "h")
            a = np.searchsorted(base_grid_m1, t_in, side="left")
            b = np.searchsorted(base_grid_m1, t_out, side="left")
            if b <= a:
                continue
            unreal[a:b] += p["alloc"] * (p["dir"] * (px[a:b] / p["eprice"] - 1.0))
    eq = pd.Series(real + unreal, index=pd.DatetimeIndex(base_grid_m1))
    dd = float((eq / eq.cummax() - 1.0).min())
    return dd, eq


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    mk = champion_sizing(pool, max_pos=8)

    # M1 共通グリッド = EURUSD M1(最密)。週末等の欠けは各銘柄 ffill で吸収。
    base_grid_m1 = load_m1("EURUSD").index.tz_localize(None).to_numpy()
    print(f"M1 grid: {len(base_grid_m1):,} bars")

    k = 7.98  # mm_production の empirical 20% 較正値
    for it in range(3):
        eqm, eqr, log, info = simulate_with_log(pool, closes, mk(k), max_pos=8)
        dd_h4 = float((eqm / eqm.cummax() - 1.0).min())
        dd_m1, eq_m1 = m1_dd_of_log(log, closes, base_grid_m1)
        years = (eqm.index[-1] - eqm.index[0]).days / 365.25
        cagr = (eqm.iloc[-1] / 10_000.0) ** (1 / years) - 1
        print(f"[iter{it}] k={k:.2f}  DD(H4終値)={dd_h4:+.1%}  DD(M1)={dd_m1:+.1%}  CAGR={cagr:+.2%}")
        if abs(abs(dd_m1) - 0.20) < 0.002:
            break
        k = k * (0.20 / abs(dd_m1))
    print(f"\n→ 正直ベースライン(M1粒度 DD=20%): k={k:.2f}, CAGR={cagr:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
