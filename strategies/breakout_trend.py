"""トレンド・フィルタ付きドンチャン・ブレイクアウト(順張り、損小利大)。

素のドンチャン(donchian_breakout)はレンジで往復負けしやすい。ここでは
**長期SMAの方向にのみ**ブレイクを取る(逆張り的なダマシを除外)。手仕舞いは
短期ドンチャン(=トレーリング相当)で、勝ちを伸ばし負けを早く切る非対称ペイオフを狙う。

  * 買い: 終値が entry 本高値を上抜け、かつ 終値 > 長期SMA(上昇トレンド)
  * 売り: 終値が entry 本安値を下抜け、かつ 終値 < 長期SMA(下降トレンド)
  * 手仕舞い: 反対側の exit 本極値(順張りトレーリング)
先読み防止のため極値・SMA は確定バーまでで判断(rolling 極値は .shift())。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"entry": 40, "exit": 20, "trend": 200}
PARAM_GRID = {"entry": [20, 40, 55], "exit": [10, 20], "trend": [100, 200]}


def generate_signals(data: pd.DataFrame, entry: int = 40, exit: int = 20, trend: int = 200):
    high, low, close = data["high"], data["low"], data["close"]

    upper = high.rolling(entry).max().shift()       # 直前までの entry 本高値
    lower = low.rolling(entry).min().shift()
    exit_lower = low.rolling(exit).min().shift()     # 手仕舞い用の短期極値
    exit_upper = high.rolling(exit).max().shift()
    sma = close.rolling(trend).mean()                # 自バー終値で判断する長期トレンド

    uptrend = close > sma
    downtrend = close < sma

    long_entries = (close > upper) & uptrend
    short_entries = (close < lower) & downtrend
    long_exits = close < exit_lower
    short_exits = close > exit_upper
    return long_entries, long_exits, short_entries, short_exits
