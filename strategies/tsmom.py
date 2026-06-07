"""タイムシリーズ・モメンタム(過去リターンの符号で順張り)。

Moskowitz らの time-series momentum。「直近 lookback 期間が上昇なら買い、下落なら売り」を
維持し、符号が反転したらドテン。資産横断で頑健な数少ないアノマリーの一つ。
トレンドの大きな波を取りにいくため非対称ペイオフ(損小利大)になりやすい。

  * 買い: 過去 lookback 本リターン > 閾値(プラス転換)
  * 売り: 過去 lookback 本リターン < -閾値(マイナス転換)
  * 手仕舞い: 反対シグナルでドテン
先読みなし(過去リターンは確定値)。取引頻度は低めなので上位足(H4/D1)向き。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"lookback": 100, "band": 0.0}
PARAM_GRID = {"lookback": [50, 100, 200], "band": [0.0, 0.002]}


def generate_signals(data: pd.DataFrame, lookback: int = 100, band: float = 0.0):
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0       # 過去 lookback 本の累積リターン

    long_state = mom > band
    short_state = mom < -band

    long_entries = long_state & ~long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    long_exits = short_entries                       # 反対転換でドテン
    short_exits = long_entries
    return long_entries, long_exits, short_entries, short_exits
