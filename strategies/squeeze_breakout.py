"""ボラティリティ・スクイーズ → 拡張ブレイクアウト。

「ボラが縮小(スクイーズ)した後の放れ」を取る古典的手法(TTM Squeeze 系)。
バンド幅が過去に対して低水準=エネルギー蓄積。そこからの上抜け/下抜けに乗ると、
ダマシが減り損小利大になりやすい。

  * スクイーズ判定: BB バンド幅 < 過去 squeeze 本の中央値(=低ボラ局面)
  * 買い: スクイーズ中に上バンドを上抜け
  * 売り: スクイーズ中に下バンドを下抜け
  * 手仕舞い: 中央線回帰
先読み防止: スクイーズ条件・バンドはすべて確定バー(.shift())で判断。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"period": 20, "mult": 2.0, "squeeze": 100}
PARAM_GRID = {"period": [20, 40], "mult": [2.0, 2.5], "squeeze": [50, 100, 200]}


def generate_signals(data: pd.DataFrame, period: int = 20, mult: float = 2.0, squeeze: int = 100):
    close = data["close"]
    bb = vbt.BBANDS.run(close, window=period, alpha=mult)
    upper, mid, lower = bb.upper, bb.middle, bb.lower

    width = (upper - lower) / mid
    is_squeeze = (width < width.rolling(squeeze).median()).shift()  # 直前バーが低ボラ

    long_entries = is_squeeze & (close > upper.shift())
    short_entries = is_squeeze & (close < lower.shift())
    long_exits = close < mid
    short_exits = close > mid
    return long_entries.fillna(False), long_exits, short_entries.fillna(False), short_exits
