"""ER連動トレンド保険 — チャンピオンv2(平均回帰・無ストップ)の失血窓専用ヘッジ。

設計意図(bleed_lab.py の実測に基づく):
  チャンピオンの失血窓 21ヶ月は **効率比ER(trendiness)が高い**(0.185 vs 平時0.172)。
  最深窓=2022 USDラリー(ER 0.19-0.22)。=チャンピオンは「一直線トレンド継続レジーム」で
  逆張りが轢かれて失血する。チャンピオンのERフィルタ(er_max=0.55で逆張り回避)の**裏返し**として、
  ここでは「**ERが高い(=一直線)時だけ順張りトレンド追随する**」専用の小戦略を作る。
  失血窓に的を絞った保険であり、単体PFが負でも「失血窓で稼ぐ(IS/OOS両プラス)」なら価値がある。

ロジック(closeベース・先読みなし):
  ER(er_win) = |close - close.shift(er_win)| / Σ_{er_win}|close.diff()|  ∈ [0,1]
    1 に近い=一直線(高効率なトレンド) / 0 に近い=往復ばかり(レンジ)。
  買い: ER >= er_hi  かつ  close > 中期MA(ma_win)   (=強い上昇トレンドに順張り)
  売り: ER >= er_hi  かつ  close < 中期MA(ma_win)   (=強い下降トレンドに順張り)
  手仕舞い: ER < er_lo(トレンド効率が落ちた=失血窓が終わりつつある)
           または MA を逆方向に割った(トレンド転換)。
  ※ ER・MA とも自バー終値(確定値)で判断。rolling 内に未来は入らない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = {"er_win": 40, "er_hi": 0.45, "er_lo": 0.25, "ma_win": 50}
PARAM_GRID = {
    "er_win": [20, 40, 60],
    "er_hi": [0.35, 0.45, 0.55],
    "er_lo": [0.20, 0.30],
    "ma_win": [50, 100, 200],
}


def _efficiency_ratio(close: pd.Series, w: int) -> pd.Series:
    """Kaufman 効率比 = |w本の正味変化| / w本の総移動量。bleed_lab._efficiency_ratio と同定義。"""
    direction = (close - close.shift(w)).abs()
    vol = close.diff().abs().rolling(w).sum()
    return (direction / vol).replace([np.inf, -np.inf], np.nan)


def generate_signals(data: pd.DataFrame, er_win: int = 40, er_hi: float = 0.45,
                     er_lo: float = 0.25, ma_win: int = 50):
    close = data["close"]
    er = _efficiency_ratio(close, er_win)
    ma = close.rolling(ma_win).mean()

    strong = er >= er_hi
    weak = er < er_lo
    above = close > ma
    below = close < ma

    long_entries = strong & above
    short_entries = strong & below
    # トレンド効率が落ちた or MA を逆方向に割ったら手仕舞い
    long_exits = weak | below
    short_exits = weak | above
    return long_entries, long_exits, short_entries, short_exits
