"""Connors RSI(2) 押し目買い(長期トレンド方向への逆張り)。

「上昇トレンド中の短期的な売られすぎを買う」= トレンドフォローと逆張りの合わせ技。
Larry Connors の RSI(2) 戦略。長期SMAでトレンド方向を固定し、その方向にだけ
短期 RSI の行き過ぎを取る。勝率が高くなりやすい。

  * 買い: 終値 > 長期SMA(上昇) かつ RSI(2) < low(短期売られすぎ)
  * 売り: 終値 < 長期SMA(下降) かつ RSI(2) > high(短期買われすぎ)
  * 手仕舞い: RSI が中央(50)へ回帰
先読みなし(SMA・RSI とも確定バー)。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"trend": 200, "rsi_p": 2, "low": 10, "high": 90}
PARAM_GRID = {"trend": [100, 200], "rsi_p": [2, 3], "low": [5, 10, 15], "high": [85, 90, 95]}


def generate_signals(data: pd.DataFrame, trend: int = 200, rsi_p: int = 2,
                     low: float = 10, high: float = 90):
    close = data["close"]
    sma = close.rolling(trend).mean()
    rsi = vbt.RSI.run(close, rsi_p).rsi

    up = close > sma
    down = close < sma

    long_entries = up & (rsi < low)
    long_exits = rsi > 50
    short_entries = down & (rsi > high)
    short_exits = rsi < 50
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
