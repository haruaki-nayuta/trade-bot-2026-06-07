"""ボリンジャーバンド・ブレイクアウト(順張り=ボラ拡張に乗る)。

逆張り(rsi_meanrev / 古典的BB逆張り)とは逆に、**バンド上抜けで買い・下抜けで売り**。
ボラティリティ拡張(=新しいトレンドの初動)に乗り、中央線回帰で手仕舞いする。
損切りは中央線割れで早め、利は伸ばす設計。

  * 買い: 終値が上バンドを上抜け
  * 売り: 終値が下バンドを下抜け
  * 手仕舞い: 中央線(移動平均)を逆向きに跨いだら
先読み防止: バンドは確定バーの終値から計算し、ブレイク判定は前バンドとの比較。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"period": 20, "mult": 2.0}
PARAM_GRID = {"period": [10, 20, 40], "mult": [1.5, 2.0, 2.5]}


def generate_signals(data: pd.DataFrame, period: int = 20, mult: float = 2.0):
    close = data["close"]
    bb = vbt.BBANDS.run(close, window=period, alpha=mult)
    upper, mid, lower = bb.upper, bb.middle, bb.lower

    long_entries = (close > upper) & (close.shift() <= upper.shift())
    short_entries = (close < lower) & (close.shift() >= lower.shift())
    long_exits = close < mid
    short_exits = close > mid
    return long_entries, long_exits, short_entries, short_exits
