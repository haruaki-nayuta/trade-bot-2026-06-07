"""移動平均クロス(ゴールデン/デッドクロスで両建て)。トレンドフォローの基本形。"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"fast": 20, "slow": 50}
PARAM_GRID = {"fast": [10, 20, 30, 50], "slow": [100, 150, 200]}


def generate_signals(data: pd.DataFrame, fast: int = 20, slow: int = 50):
    close = data["close"]
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma

    golden = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())  # 上抜け
    dead = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())    # 下抜け

    long_entries = golden
    long_exits = dead
    short_entries = dead
    short_exits = golden
    return long_entries, long_exits, short_entries, short_exits
