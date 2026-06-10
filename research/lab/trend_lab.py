"""trend_lab — トレンドフォロー検証の共有基盤(低メモリ・並列エージェント安全)。

M1 からの再リサンプルはプロセスごとに ~2GB 食うため、事前構築した H4 OHLCV キャッシュ
(results/h4_cache/{instr}.parquet)だけを読む。D1/W1 は H4 から集約(規則は fxlab.data と同一)。

API:
  load_tf(instr, tf)         : H4/D1/W1 の OHLCV(クロスは close 系列の複製 OHLC)
  build_pool(gen, params, tf, side, instruments) : mm_lab 互換のトレードプール
                               (instr, entry, exit, dir, entry_price, ret, bars_held, z_entry, vol_entry)
  pool_stats(pool)           : PF / IS / OOS / 年次 / bps 等の要約 dict
  cache_h4()                 : キャッシュ事前構築(リポジトリ直下で1回実行)

実行(キャッシュ構築): PYTHONPATH=. uv run python research/lab/trend_lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fxlab import config  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

CACHE = config.RESULTS_DIR / "h4_cache"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]


def default_instruments() -> list[str]:
    """19対象(メジャー7+クロス12, AUDJPY除外)+ XAUUSD があれば追加。"""
    from fxlab.universe import CROSS_DEFS
    out = MAJORS + [c for c in CROSS_DEFS if c != "AUDJPY"]
    if (CACHE / "XAUUSD.parquet").exists():
        out.append("XAUUSD")
    return out


def cache_h4() -> None:
    """H4 OHLCV キャッシュを構築(メジャー=実OHLC, クロス=合成close, 金=実OHLC)。"""
    from fxlab.data import load
    from fxlab.universe import CROSS_DEFS, instrument_data

    CACHE.mkdir(parents=True, exist_ok=True)
    for nm in MAJORS:
        load(nm, "H4").to_parquet(CACHE / f"{nm}.parquet")
        print(f"  {nm} cached")
    for nm in CROSS_DEFS:
        instrument_data(nm, "H4").to_parquet(CACHE / f"{nm}.parquet")
        print(f"  {nm} cached")
    gold_m1 = config.DATA_DIR / "XAUUSD_M1.parquet"
    if gold_m1.exists():
        from fxlab.data import resample
        df = pd.read_parquet(gold_m1)
        resample(df, "H4").to_parquet(CACHE / "XAUUSD.parquet")
        print("  XAUUSD cached")


def load_tf(instr: str, tf: str = "H4") -> pd.DataFrame:
    """H4 は h4_cache、M15/M30/H1 は tf_cache、D1/W1 は H4 から集約。"""
    if tf in ("M15", "M30", "H1"):
        return pd.read_parquet(config.RESULTS_DIR / "tf_cache" / tf / f"{instr}.parquet")
    df = pd.read_parquet(CACHE / f"{instr}.parquet")
    if tf == "H4":
        return df
    rule = {"D1": "1D", "W1": "1W"}[tf]
    out = df.resample(rule, label="left", closed="left").agg(AGG)
    return out.dropna(subset=["close"])


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def register_spreads() -> None:
    from fxlab.universe import register_cross_spreads
    register_cross_spreads(3.0)
    config.SPREADS_PIPS["XAUUSD"] = 0.40 / 0.0001  # $0.40 フルスプレッド(pip=0.0001換算)


def build_pool(gen, params: dict, tf: str = "H4", side: str = "both",
               instruments: list[str] | None = None) -> pd.DataFrame:
    """mm_lab 互換プール。gen は generate_signals 互換関数。"""
    register_spreads()
    instruments = instruments or default_instruments()
    frames = []
    for nm in instruments:
        data = load_tf(nm, tf)
        try:
            pf = run(nm, tf, gen, params, data=data, size_mode="value", side=side)
            tt = trade_table(pf, data)
        except Exception:  # noqa: BLE001
            continue
        if tt.empty:
            continue
        close = data["close"]
        z = _zscore(close, int(params.get("window", 50)) or 50)
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
    if not frames:
        return pd.DataFrame(columns=["instr", "entry", "exit", "dir", "entry_price", "ret",
                                     "bars_held", "z_entry", "vol_entry"])
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


def pool_stats(pool: pd.DataFrame) -> dict:
    if pool.empty:
        return {"n": 0}
    r = pool["ret"]

    def pf(x):
        g = x[x > 0].sum()
        l = -x[x < 0].sum()
        return float(g / l) if l > 0 else float("inf")

    is_r = pool.loc[pool["entry"] < OOS_START, "ret"]
    oos_r = pool.loc[pool["entry"] >= OOS_START, "ret"]
    yearly = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    years = max((pool["exit"].max() - pool["entry"].min()).days / 365.25, 1e-9)
    return {
        "n": int(len(pool)), "trades_per_year": round(len(pool) / years, 1),
        "sum_ret": round(float(r.sum()), 4), "pool_pf": round(pf(r), 3),
        "is_pf": round(pf(is_r), 3), "oos_pf": round(pf(oos_r), 3),
        "is_sum": round(float(is_r.sum()), 4), "oos_sum": round(float(oos_r.sum()), 4),
        "mean_bps": round(float(r.mean() * 1e4), 2), "win_rate": round(float((r > 0).mean()), 3),
        "avg_bars": round(float(pool["bars_held"].mean()), 1),
        "yearly_pos": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "worst_year": round(float(yearly.min()), 4),
    }


if __name__ == "__main__":
    cache_h4()
    print("done:", sorted(p.name for p in CACHE.glob("*.parquet")))
