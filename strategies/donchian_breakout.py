"""ドンチャン・ブレイクアウト(N本高値更新で買い／N本安値更新で売り)。タートルズ系。"""

from __future__ import annotations

import pandas as pd

PARAMS = {"entry": 20, "exit": 10}
PARAM_GRID = {"entry": [20, 40, 55], "exit": [10, 20]}


def generate_signals(data: pd.DataFrame, entry: int = 20, exit: int = 10):
    high, low, close = data["high"], data["low"], data["close"]

    # 直前バーまでの極値(自バーを含めないため shift で先読み回避)
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    exit_upper = high.rolling(exit).max().shift()
    exit_lower = low.rolling(exit).min().shift()

    long_entries = close > upper       # 高値ブレイクで買い
    short_entries = close < lower      # 安値ブレイクで売り
    long_exits = close < exit_lower    # 短期安値割れで手仕舞い
    short_exits = close > exit_upper   # 短期高値抜けで手仕舞い
    return long_entries, long_exits, short_entries, short_exits
