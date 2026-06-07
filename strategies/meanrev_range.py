"""レンジ・レジーム限定の RSI 平均回帰(ADX フィルタ付き)。

平均回帰は「レンジ相場」で機能し「トレンド相場」で大きく負ける。素の rsi_meanrev が
トレンド期に崩れる弱点を、**ADX が低い(=トレンドが弱い=レンジ)ときだけ建玉**する
ことで除外する。経済合理性のあるフィルタで、パラメータ穿りではない。

  * レンジ判定: ADX(adx_period) < adx_max
  * 買い: レンジ かつ RSI が low を下から回復(売られすぎ転換)
  * 売り: レンジ かつ RSI が high を上から反落(買われすぎ転換)
  * 手仕舞い: RSI が中央(50)へ回帰
先読みなし(RSI・ADX とも確定バー)。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt
from ta.trend import ADXIndicator

PARAMS = {"period": 14, "low": 30, "high": 70, "adx_period": 14, "adx_max": 25}
PARAM_GRID = {
    "period": [7, 14, 21],
    "low": [25, 30],
    "high": [70, 75],
    "adx_period": [14],
    "adx_max": [20, 25, 30],
}


def generate_signals(data: pd.DataFrame, period: int = 14, low: float = 30, high: float = 70,
                     adx_period: int = 14, adx_max: float = 25):
    close = data["close"]
    rsi = vbt.RSI.run(close, period).rsi
    adx = ADXIndicator(data["high"], data["low"], close, window=adx_period).adx()
    ranging = adx < adx_max

    long_entries = (rsi < low) & (rsi.shift() >= low) & ranging
    long_exits = rsi > 50
    short_entries = (rsi > high) & (rsi.shift() <= high) & ranging
    short_exits = rsi < 50
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
