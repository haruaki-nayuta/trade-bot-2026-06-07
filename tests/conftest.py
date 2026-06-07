"""テスト共通フィクスチャ。

ダウンロード済みデータに依存せず高速に回るよう、合成OHLCVを使う。
run()/sweep() は data= でこの合成データを差し込める。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth_h1():
    """1000本の H1 合成OHLCV(再現性のためseed固定)。"""
    n = 1000
    idx = pd.date_range("2020-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)
    close = 1.10 + np.cumsum(rng.randn(n) * 0.0008)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + np.abs(rng.randn(n)) * 0.0003
    low = np.minimum(open_, close) - np.abs(rng.randn(n)) * 0.0003
    vol = rng.randint(100, 1000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


@pytest.fixture
def synth_m1():
    """120本の M1 合成OHLCV(リサンプル検証用、値は決定的)。"""
    n = 120
    idx = pd.date_range("2020-01-01 00:00", periods=n, freq="1min", tz="UTC")
    base = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": 1.0 + base,
            "high": 1.0 + base + 0.5,
            "low": 1.0 + base - 0.5,
            "close": 1.0 + base + 0.1,
            "volume": np.ones(n),
        },
        index=idx,
    )
