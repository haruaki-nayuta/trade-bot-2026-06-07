"""データ読込・リサンプル・キャッシュのテスト。"""

import pandas as pd
import pytest

from fxlab import data


def test_resample_ohlc(synth_m1):
    """M1→H1 の集約規則: open=first, high=max, low=min, close=last, volume=sum。"""
    h1 = data.resample(synth_m1, "H1")
    assert len(h1) == 2  # 120分 = 2時間
    first_hour = synth_m1.iloc[:60]
    assert h1.iloc[0]["open"] == first_hour.iloc[0]["open"]
    assert h1.iloc[0]["close"] == first_hour.iloc[-1]["close"]
    assert h1.iloc[0]["high"] == first_hour["high"].max()
    assert h1.iloc[0]["low"] == first_hour["low"].min()
    assert h1.iloc[0]["volume"] == first_hour["volume"].sum()


def test_resample_m1_identity(synth_m1):
    assert data.resample(synth_m1, "M1") is synth_m1


def test_resample_unknown_tf(synth_m1):
    with pytest.raises(ValueError):
        data.resample(synth_m1, "X9")


def test_load_missing_pair_raises():
    with pytest.raises(FileNotFoundError):
        data.load_m1("NOTAPAIR")


def test_cache_returns_same_object():
    """ダウンロード済みデータがあればキャッシュ同一性を確認(無ければスキップ)。"""
    pairs = data.available_pairs()
    if not pairs:
        pytest.skip("価格データ未取得")
    p = pairs[0]
    a = data.load(p, "H1")
    b = data.load(p, "H1")
    assert a is b
    data.clear_cache()
    c = data.load(p, "H1")
    assert c is not a  # クリア後は別オブジェクト
    assert c.equals(a)  # 中身は同じ
