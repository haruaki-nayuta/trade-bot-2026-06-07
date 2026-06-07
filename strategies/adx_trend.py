"""ADX フィルタ付き移動平均トレンドフォロー(強トレンド時のみ順張り)。

MA クロスは「トレンドが無い局面」で往復負けする。ADX(トレンドの強さ)が閾値超の
ときだけ建玉することで、レンジのダマシを除外し PF を底上げする狙い。

  * 買い: 短期MA > 長期MA かつ ADX > th(強い上昇トレンド)
  * 売り: 短期MA < 長期MA かつ ADX > th(強い下降トレンド)
  * 手仕舞い: MA が逆クロス(トレンド転換)
先読み防止: MA・ADX とも確定バーで計算。状態ベースなので shift 済みクロスで建玉。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt
from ta.trend import ADXIndicator

PARAMS = {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 25}
PARAM_GRID = {"fast": [10, 20], "slow": [50, 100], "adx_period": [14], "adx_th": [20, 25, 30]}


def generate_signals(data: pd.DataFrame, fast: int = 20, slow: int = 50,
                     adx_period: int = 14, adx_th: float = 25):
    close = data["close"]
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma
    adx = ADXIndicator(data["high"], data["low"], close, window=adx_period).adx()

    strong = adx > adx_th
    up = fast_ma > slow_ma
    down = fast_ma < slow_ma

    long_entries = up & strong & ~(up.shift(fill_value=False) & strong.shift(fill_value=False))
    short_entries = down & strong & ~(down.shift(fill_value=False) & strong.shift(fill_value=False))
    long_exits = down       # トレンド転換で手仕舞い
    short_exits = up
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
