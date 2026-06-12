"""ビル・ウィリアムズ ゾーントレーディング — 『Trading Chaos』のAO+AC同色ゾーン。

AC(Accelerator) = AO − SMA(AO, 5)。
緑ゾーン = AO・AC が共に増加、赤ゾーン = 共に減少。
n 本連続の同色ゾーンでエントリー、反対色が exit_m 本連続で手仕舞い。
teeth_filter=1 で「終値がアリゲーターの歯(teeth)の正しい側」のときのみ入る。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"n": 3, "exit_m": 1, "teeth_filter": 1}
PARAM_GRID = {"n": [2, 3, 5], "exit_m": [1, 2], "teeth_filter": [0, 1]}


def generate_signals(data: pd.DataFrame, n: int = 3, exit_m: int = 1, teeth_filter: int = 1):
    high, low, close = data["high"], data["low"], data["close"]
    median = (high + low) / 2.0
    ao = median.rolling(5).mean() - median.rolling(34).mean()
    ac = ao - ao.rolling(5).mean()

    green = ((ao.diff() > 0) & (ac.diff() > 0)).astype(int)
    red = ((ao.diff() < 0) & (ac.diff() < 0)).astype(int)

    green_run = green.rolling(n).sum() == n
    red_run = red.rolling(n).sum() == n
    long_entries = green_run & ~green_run.shift(fill_value=False)
    short_entries = red_run & ~red_run.shift(fill_value=False)

    long_exits = red.rolling(exit_m).sum() == exit_m
    short_exits = green.rolling(exit_m).sum() == exit_m

    if teeth_filter:
        teeth = median.ewm(alpha=1.0 / 8, adjust=False).mean().shift(5)
        long_entries &= close > teeth
        short_entries &= close < teeth

    return (
        long_entries.fillna(False),
        long_exits.fillna(False),
        short_entries.fillna(False),
        short_exits.fillna(False),
    )
