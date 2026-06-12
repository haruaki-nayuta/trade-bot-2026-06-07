"""chaos_zone v2 — ビル・ウィリアムズ ゾーントレーディングのショート専用版。

改善イテレーション(2026-06-12):
- 元の chaos_zone は H1 で全面マイナス(OOS Sharpe -1.03)。
- H4 再評価(eval_chaos_zone_EURUSD_H4.md)でロング/ショート分離が
  long Sharpe -0.39 / short +0.42(PF 1.18)とショート片側のみ有効と判明。
- evaluate の自動提案「ショート専用にする」に従い片側化した変種。
- ロジック本体(AO+AC同色ゾーン、teethフィルタ)は chaos_zone と同一。
  ロング側のシグナルを常に False にしただけ。グリッドは狭めていない。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"n": 5, "exit_m": 1, "teeth_filter": 1}
PARAM_GRID = {"n": [2, 3, 5], "exit_m": [1, 2], "teeth_filter": [0, 1]}


def generate_signals(data: pd.DataFrame, n: int = 5, exit_m: int = 1, teeth_filter: int = 1):
    high, low, close = data["high"], data["low"], data["close"]
    median = (high + low) / 2.0
    ao = median.rolling(5).mean() - median.rolling(34).mean()
    ac = ao - ao.rolling(5).mean()

    green = ((ao.diff() > 0) & (ac.diff() > 0)).astype(int)
    red = ((ao.diff() < 0) & (ac.diff() < 0)).astype(int)

    red_run = red.rolling(n).sum() == n
    short_entries = red_run & ~red_run.shift(fill_value=False)
    short_exits = green.rolling(exit_m).sum() == exit_m

    if teeth_filter:
        teeth = median.ewm(alpha=1.0 / 8, adjust=False).mean().shift(5)
        short_entries &= close < teeth

    no_long = pd.Series(False, index=data.index)
    return (
        no_long,
        no_long,
        short_entries.fillna(False),
        short_exits.fillna(False),
    )
