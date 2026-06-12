"""ビル・ウィリアムズ AO(Awesome Oscillator)モメンタム — 『Trading Chaos』第2の賢者。

AO = SMA(中値, fast) − SMA(中値, slow)。書籍の既定は 5/34。
signal=0 (ゼロクロス): AO が 0 を上抜けで買い/下抜けで売り(ドテン)。
signal=1 (ソーサー): AO がゼロより上で 赤2本(減少)→緑1本(増加) で買い。
                     ゼロより下で 緑2本→赤1本 で売り。手仕舞いは反対側へのゼロクロス。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"fast": 5, "slow": 34, "signal": 0}
PARAM_GRID = {"fast": [3, 5, 8], "slow": [21, 34, 55], "signal": [0, 1]}


def generate_signals(data: pd.DataFrame, fast: int = 5, slow: int = 34, signal: int = 0):
    median = (data["high"] + data["low"]) / 2.0
    ao = median.rolling(fast).mean() - median.rolling(slow).mean()

    cross_up = (ao > 0) & (ao.shift() <= 0)
    cross_dn = (ao < 0) & (ao.shift() >= 0)

    if int(signal) == 0:
        long_entries, short_entries = cross_up, cross_dn
        long_exits, short_exits = cross_dn, cross_up
    else:
        d = ao.diff()
        long_entries = (ao > 0) & (d > 0) & (d.shift(1) < 0) & (d.shift(2) < 0)
        short_entries = (ao < 0) & (d < 0) & (d.shift(1) > 0) & (d.shift(2) > 0)
        long_exits, short_exits = cross_dn, cross_up

    return (
        long_entries.fillna(False),
        long_exits.fillna(False),
        short_entries.fillna(False),
        short_exits.fillna(False),
    )
