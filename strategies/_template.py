"""戦略テンプレート — これをコピーして新しい手法を作る。

ルール:
  * generate_signals(data, **params) を 1 つだけ定義する。
  * data は OHLCV の DataFrame(columns: open/high/low/close/volume, UTC index)。
  * 返り値は data.index に整列した bool の pd.Series:
        ロングのみ      -> (long_entries, long_exits)
        ロング+ショート -> (long_entries, long_exits, short_entries, short_exits)
  * 先読み(look-ahead)を避けるため、シグナルは「確定したバー」で判断すること。
  * PARAMS にデフォルト値、PARAM_GRID に探索範囲を書いておくと runner が使える。

検証:
  uv run python run_backtest.py <このファイル名(拡張子なし)> --pair EURUSD --tf H1
  uv run python run_backtest.py <名前> --pair EURUSD --tf H1 --sweep
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt  # vbt.MA / vbt.RSI / vbt.BBANDS / vbt.ATR などが使える

# 単発検証で使うデフォルトパラメータ
PARAMS = {"fast": 20, "slow": 50}

# sweep（パラメータ探索）で使う範囲
PARAM_GRID = {"fast": [10, 20, 30], "slow": [50, 100, 200]}


def generate_signals(data: pd.DataFrame, fast: int = 20, slow: int = 50):
    close = data["close"]

    # --- ここに手法のロジックを書く ---
    fast_ma = vbt.MA.run(close, fast).ma
    slow_ma = vbt.MA.run(close, slow).ma

    long_entries = (fast_ma > slow_ma) & (fast_ma.shift() <= slow_ma.shift())
    long_exits = (fast_ma < slow_ma) & (fast_ma.shift() >= slow_ma.shift())
    short_entries = long_exits
    short_exits = long_entries
    # ----------------------------------

    return long_entries, long_exits, short_entries, short_exits
