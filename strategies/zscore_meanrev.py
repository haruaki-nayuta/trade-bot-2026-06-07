"""Zスコア平均回帰(自己正規化=どの通貨ペアにも適応)。

固定しきい値(RSI 30/70 など)は通貨ごとのボラ・値幅の違いに合わず「通貨依存」を生む。
ここでは終値の移動平均からの乖離を**標準偏差で正規化した Zスコア**でエントリーする。
Z は各ペア・各局面のボラで自動スケールされるため、同一パラメータが通貨横断で通用しやすい。

  * Z = (close - SMA(window)) / STD(window)
  * 買い: Z が -entry_z を下抜け(平均から大きく下振れ=売られすぎ)
  * 売り: Z が +entry_z を上抜け(買われすぎ)
  * 手仕舞い: Z が平均近傍(±exit_z)へ回帰
先読みなし(SMA・STD とも確定バー)。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"window": 50, "entry_z": 2.0, "exit_z": 0.5}
PARAM_GRID = {"window": [20, 50, 100], "entry_z": [1.5, 2.0, 2.5], "exit_z": [0.0, 0.5]}


def generate_signals(data: pd.DataFrame, window: int = 50, entry_z: float = 2.0, exit_z: float = 0.5):
    close = data["close"]
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std()
    z = (close - ma) / sd

    long_entries = (z < -entry_z) & (z.shift() >= -entry_z)
    long_exits = z > -exit_z
    short_entries = (z > entry_z) & (z.shift() <= entry_z)
    short_exits = z < exit_z
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
