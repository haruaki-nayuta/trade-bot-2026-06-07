"""exp29: 位相オフセット H4 ストリーム — 同品質シグナルを「グリッドの位相」で増やす。

mp11 で取りこぼし(skip)はほぼ 0 = 枠はもう余っている。CAGR@DD20 をさらに上げるには
「同品質のシグナル数」を増やすしかない。H4 バーを +1h/+2h/+3h ずらした 3 つのグリッドで
同一チャンピオン構造を回せば、終値確認のタイミングが変わり別のエントリー集合が得られる
(パラメータは一切再最適化しない=カーブフィット無し)。

検証手順:
  1. 各オフセットのストリームを単体検証(PF/年次/相関行列)— 品質が落ちないか
  2. ens_lab で 2〜4 ストリーム統合 → empirical DD=20% 較正 CAGR を基準(mp11 +23.8%)と比較
  3. IS較正→OOS素検証 + ブートストラップ

実行: PYTHONPATH=. uv run python research/experiments/exp29_phase_offsets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
import ens_lab as ens  # noqa: E402
import strategies.confluence_meanrev_v2 as v2  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from fxlab.universe import CROSS_DEFS, register_cross_spreads  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
COLS = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def h4_offset(pair: str, offset_h: int) -> pd.DataFrame:
    """M1 → H4(offset 時間ずらし)リサンプル。"""
    df = load_m1(pair)
    out = df.resample("4h", label="left", closed="left", offset=f"{offset_h}h").agg(AGG)
    return out.dropna(subset=["close"])


_m1_cross_cache: dict[str, pd.Series] = {}


def instrument_h4_offset(name: str, offset_h: int) -> pd.DataFrame:
    """オフセット H4 の OHLCV(クロスは M1 合成 close から)。"""
    if name not in CROSS_DEFS:
        return h4_offset(name, offset_h)
    if name in _m1_cross_cache:
        c1 = _m1_cross_cache[name]
    else:
        a, op, b = CROSS_DEFS[name]
        df = pd.concat([load_m1(a)["close"], load_m1(b)["close"]], axis=1,
                       keys=["a", "b"]).ffill().dropna()
        c1 = (df["a"] / df["b"]) if op == "/" else (df["a"] * df["b"])
        _m1_cross_cache[name] = c1
    c = c1.resample("4h", label="left", closed="left", offset=f"{offset_h}h").last().dropna()
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=c.index)


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_pool_offset(offset_h: int, cache=True) -> pd.DataFrame:
    register_cross_spreads(3.0)
    instruments = mm.default_instruments()
    params = dict(v2.PARAMS)
    win = params.get("window", 50)
    cache_path = config.RESULTS_DIR / f"mm_pool_v2_H4off{offset_h}_19.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    frames = []
    for nm in instruments:
        data = instrument_h4_offset(nm, offset_h)
        pf = run(nm, "H4", v2.generate_signals, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        close = data["close"]
        z = _zscore(close, win)
        vol = close.pct_change().rolling(20).std()
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(z.reindex(tt["entry"]).to_numpy()),
            "vol_entry": vol.reindex(tt["entry"]).to_numpy(),
        }))
        print(f"    off+{offset_h}h {nm} done")
    pool = pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)
    if cache:
        pool.to_parquet(cache_path)
    return pool


def monthly_pnl(pool):
    m = pd.to_datetime(pool["exit"]).dt.tz_localize(None).dt.to_period("M")
    return pool.groupby(m)["ret"].sum()


def pool_quick(pool, label):
    r = pool["ret"]
    gp = r[r > 0].sum(); gl = -r[r < 0].sum()
    is_r = pool.loc[pool["entry"] < OOS_START, "ret"]
    oos_r = pool.loc[pool["entry"] >= OOS_START, "ret"]
    def pf(x):
        g = x[x > 0].sum(); l = -x[x < 0].sum()
        return g / l if l > 0 else float("inf")
    print(f"  [{label}] n={len(pool)} ΣR={r.sum():+.3f} PF={gp/gl:.3f} "
          f"IS_PF={pf(is_r):.3f} OOS_PF={pf(oos_r):.3f} 平均={r.mean()*1e4:+.1f}bps")


def evaluate(pool, closes, budgets, label, target_dd=0.20, n_boot=800):
    fbars = ens.stream_fbars(pool)
    k, eqm, eqr, info = ens.calibrate_streams(pool, closes, budgets, fbars=fbars,
                                              target_dd=target_dd)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = ens.calibrate_streams(is_pool, is_cl, budgets, fbars=fbars, target_dd=target_dd)
    eqo, ero, io = ens.simulate_streams(oos_pool, oos_cl, k_is, budgets, fbars=fbars)
    so = mm.stats(eqo, ero, io)
    print(f"  {label:40s} k={k:5.2f} CAGR={s['cagr']:+7.2%} DD={s['maxdd_mtm']:+6.1%} "
          f"Sh={s['sharpe']:4.2f} boot95={bs['p95']:+6.1%} +年={s['pos_year_rate']:3.0%} "
          f"worst={s['worst_year']:+5.1%} skip={s['skipped']:4d} | "
          f"OOS CAGR={so['cagr']:+7.2%} DD={so['maxdd_mtm']:+6.1%} +年={so['pos_year_rate']:3.0%}")
    return {"label": label, "cagr": s["cagr"], "boot95": bs["p95"], "oos_cagr": so["cagr"],
            "worst": s["worst_year"]}


def main() -> int:
    closes = mm.load_closes()

    print("=== ストリーム構築(オフセット 0/+1h/+2h/+3h) ===")
    pools = {0: mm.build_pool().copy()}
    for off in [1, 2, 3]:
        pools[off] = build_pool_offset(off)

    print("\n=== 単体品質 ===")
    for off, p in pools.items():
        pool_quick(p, f"offset+{off}h")

    print("\n=== 月次PnL相関行列 ===")
    ms = {off: monthly_pnl(p) for off, p in pools.items()}
    idx = None
    for s in ms.values():
        idx = s.index if idx is None else idx.union(s.index)
    M = pd.DataFrame({off: s.reindex(idx).fillna(0) for off, s in ms.items()})
    print(M.corr().round(3).to_string())

    for off, p in pools.items():
        p["stream"] = f"o{off}"
        p["w"] = 1.0

    print("\n=== 基準(offset0 単独) ===")
    evaluate(pools[0][COLS], closes, {"o0": 11}, "o0 mp11(基準)")

    print("\n=== 統合 ===")
    results = []
    duo = pd.concat([pools[0][COLS], pools[2][COLS]], ignore_index=True).sort_values("entry").reset_index(drop=True)
    for b in [(8, 8), (11, 11), (6, 6)]:
        results.append(evaluate(duo, closes, {"o0": b[0], "o2": b[1]}, f"o0+o2 mp{b[0]}/{b[1]}"))

    quad = pd.concat([pools[o][COLS] for o in [0, 1, 2, 3]], ignore_index=True).sort_values("entry").reset_index(drop=True)
    for b in [(4, 4, 4, 4), (6, 6, 6, 6), (8, 8, 8, 8), (11, 11, 11, 11)]:
        results.append(evaluate(quad, closes, {f"o{o}": bb for o, bb in zip([0, 1, 2, 3], b)},
                                f"o0..o3 mp{b[0]}x4"))

    best = max(results, key=lambda r: r["cagr"])
    print(f"\nベスト: {best['label']}  CAGR={best['cagr']:+.2%} (OOS {best['oos_cagr']:+.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
