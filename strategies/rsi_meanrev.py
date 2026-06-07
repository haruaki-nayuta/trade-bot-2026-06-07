"""RSI 逆張り(売られすぎで買い／買われすぎで売り、中央回帰で手仕舞い)。レンジ向け。"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"period": 14, "low": 30, "high": 70}
PARAM_GRID = {"period": [7, 14, 21], "low": [20, 25, 30], "high": [70, 75, 80]}


def generate_signals(data: pd.DataFrame, period: int = 14, low: float = 30, high: float = 70):
    close = data["close"]
    rsi = vbt.RSI.run(close, period).rsi

    long_entries = (rsi < low) & (rsi.shift() >= low)      # 売られすぎ転換で買い
    long_exits = rsi > 50                                  # 中央回帰で利確
    short_entries = (rsi > high) & (rsi.shift() <= high)   # 買われすぎ転換で売り
    short_exits = rsi < 50
    return long_entries, long_exits, short_entries, short_exits
