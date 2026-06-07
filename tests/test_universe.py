"""ユニバース(合成クロス + ポートフォリオ集計)の正しさを担保するテスト。

ダウンロード済みデータに依存しないよう、load を合成OHLCVに差し替える。
クロス合成の式・OHLC代用・年次集計の整合を検証する。
"""

import numpy as np
import pandas as pd

from fxlab import universe as uni
from strategies.confluence_meanrev import generate_signals as cmr


def _synth(pair, tf="H1"):
    """ペアごとに決定的に異なる H1 合成OHLCV。"""
    n = 6000
    idx = pd.date_range("2019-01-01", periods=n, freq="1h", tz="UTC")
    seed = sum(ord(c) for c in pair)
    rng = np.random.RandomState(seed)
    close = 1.10 + np.cumsum(rng.randn(n) * 0.0008)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.randn(n)) * 0.0003
    low = np.minimum(open_, close) - np.abs(rng.randn(n)) * 0.0003
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": 1.0}, index=idx)


def test_cross_close_composition(monkeypatch):
    monkeypatch.setattr(uni, "load", _synth)
    c = uni.instrument_close("EURGBP", "H1")           # EURUSD / GBPUSD
    eu = _synth("EURUSD")["close"]
    gu = _synth("GBPUSD")["close"]
    expected = (eu / gu)
    assert np.allclose(c.values, expected.reindex(c.index).values)
    # 積タイプ
    cj = uni.instrument_close("EURJPY", "H1")          # EURUSD * USDJPY
    assert np.allclose(cj.values, (_synth("EURUSD")["close"] * _synth("USDJPY")["close"]).values)


def test_instrument_data_is_ohlcv(monkeypatch):
    monkeypatch.setattr(uni, "load", _synth)
    d = uni.instrument_data("AUDCAD", "H1")
    assert list(d.columns) == ["open", "high", "low", "close", "volume"]
    # クロスは close で OHLC 代用
    assert (d["high"] == d["close"]).all() and (d["low"] == d["close"]).all()


def test_universe_includes_crosses(monkeypatch):
    monkeypatch.setattr(uni, "available_pairs", lambda: ["EURUSD", "GBPUSD"])
    u = uni.universe(crosses=True)
    assert "EURGBP" in u and "AUDCAD" in u and "EURUSD" in u
    assert uni.universe(crosses=False) == ["EURUSD", "GBPUSD"]


def test_portfolio_yearly_aggregates(monkeypatch):
    """ポートフォリオ年次の pnl 合算 = 各対象の年次 pnl 合算。"""
    monkeypatch.setattr(uni, "load", _synth)
    instruments = ["EURGBP", "AUDCAD"]                 # クロスのみ(data= 経路でDL不要)
    port = uni.portfolio_yearly("H1", cmr, {"slow_z": 0, "vol_pct": 1.0},
                                instruments=instruments)
    # 各対象の年次を直に合算して一致を確認
    total = 0.0
    for nm in instruments:
        y = uni.yearly(nm, "H1", cmr, {"slow_z": 0, "vol_pct": 1.0},
                       data=uni.instrument_data(nm, "H1"))
        total += y["pnl"].sum()
    assert np.isclose(port["pnl"].sum(), total, rtol=1e-6)
