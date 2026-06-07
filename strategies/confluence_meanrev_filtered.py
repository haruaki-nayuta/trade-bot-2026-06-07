"""チャンピオン confluence_meanrev に「エントリー時点(先読みなし)フィルタ」を載せた検証用変種。

目的: ワーストトレード(=戻らずに長期保有で失血する塩漬け)を**入る前に避ける**。
出口を切るのは平均回帰では逆効果と実証済み(exp13/14)。残る打ち手はエントリー側。

全フィルタは既定 OFF で、その場合 confluence_meanrev と完全一致する(apples-to-apples)。
すべて確定バーの因果指標のみ(先読みなし)。close ベースなので合成クロスにも適用可。

追加フィルタ(各 None/無効値で OFF):
  er_max     : Kaufman 効率比 ER(er_win) ≤ er_max のときだけ建玉(強トレンド=失敗回帰を回避)
  adx_max    : ADX(adx_p) ≤ adx_max のときだけ建玉(トレンド強度上限)
  slow_z_cap : 長期Z |zs| ≤ slow_z_cap(長期乖離が過大=レジーム転換の疑い→見送り)。
               チャンピオンは |zs|>slow_z の下限。これは上限を足して「適度な乖離」帯に絞る。
  atr_cap    : ATR%(atr_p) ≤ atr_cap(急変時の逆張りを回避)
  slope_max  : 直前 slope_win 本の終値回帰傾き(%/bar)の逆行成分 ≤ slope_max
               (ロングなら下落の傾きが急すぎる"落ちるナイフ"を回避)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt

# チャンピオン本番パラメータ + フィルタ(既定 OFF)
PARAMS = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35,
          "rsi_high": 65, "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75,
          "er_win": 20, "er_max": None, "adx_p": 14, "adx_max": None,
          "slow_z_cap": None, "atr_p": 14, "atr_cap": None,
          "slope_win": 10, "slope_max": None}

# 探索は Workflow 側で動的に指定する。ここは最小限。
PARAM_GRID = {"er_max": [0.3, 0.4, 0.5, 1.01]}


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _efficiency_ratio(close: pd.Series, w: int) -> pd.Series:
    """Kaufman 効率比: |Δw| / Σ|Δ1|。1に近いほど一直線=強トレンド。close のみ=クロス対応。"""
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


def _adx(data: pd.DataFrame, p: int) -> pd.Series:
    from ta.trend import ADXIndicator
    return ADXIndicator(data["high"], data["low"], data["close"], window=p,
                        fillna=False).adx()


def _atr_pct(data: pd.DataFrame, p: int) -> pd.Series:
    h, l, c = data["high"], data["low"], data["close"]
    pc = c.shift()
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(p).mean() / c * 100


def _slope_pct_per_bar(close: pd.Series, w: int) -> pd.Series:
    """直前 w 本の終値を正規化回帰した傾き(%/bar)。負=下落トレンド中。因果(rolling)。"""
    x = np.arange(w)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def _s(arr):
        y = arr / arr[0]
        return float(((x - xm) * (y - y.mean())).sum() / denom * 100)

    return close.rolling(w).apply(_s, raw=True)


def generate_signals(data: pd.DataFrame, window=50, entry_z=2.0, exit_z=0.5,
                     rsi_p=14, rsi_low=35, rsi_high=65, vol_win=100, vol_pct=0.70,
                     slow_win=250, slow_z=1.75,
                     er_win=20, er_max=None, adx_p=14, adx_max=None,
                     slow_z_cap=None, atr_p=14, atr_cap=None,
                     slope_win=10, slope_max=None):
    close = data["close"]
    z = _zscore(close, window)
    rsi = vbt.RSI.run(close, rsi_p).rsi

    if vol_pct >= 1.0:
        calm = pd.Series(True, index=close.index)
    else:
        vol = close.pct_change().rolling(20).std()
        calm = vol <= vol.rolling(vol_win).quantile(vol_pct)

    if slow_z > 0:
        zs = _zscore(close, slow_win)
        long_ok = (zs < -slow_z).fillna(False)
        short_ok = (zs > slow_z).fillna(False)
    else:
        zs = _zscore(close, slow_win)
        long_ok = short_ok = pd.Series(True, index=close.index)

    base_long = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm & long_ok
    base_short = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm & short_ok

    # --- エントリー時点フィルタ(両方向共通 + 方向別) ---
    keep_long = pd.Series(True, index=close.index)
    keep_short = pd.Series(True, index=close.index)

    if er_max is not None and er_max < 1.0:
        er = _efficiency_ratio(close, er_win)
        ok = (er <= er_max).fillna(False)
        keep_long &= ok; keep_short &= ok
    if adx_max is not None:
        ok = (_adx(data, adx_p) <= adx_max).fillna(False)
        keep_long &= ok; keep_short &= ok
    if slow_z_cap is not None:
        ok = (zs.abs() <= slow_z_cap).fillna(False)
        keep_long &= ok; keep_short &= ok
    if atr_cap is not None:
        ok = (_atr_pct(data, atr_p) <= atr_cap).fillna(False)
        keep_long &= ok; keep_short &= ok
    if slope_max is not None:
        sl = _slope_pct_per_bar(close, slope_win)
        # ロング: 下落の傾き(=負)が急すぎる(落ちるナイフ)を除外 → slope >= -slope_max
        keep_long &= (sl >= -slope_max).fillna(False)
        # ショート: 上昇の傾きが急すぎるを除外 → slope <= slope_max
        keep_short &= (sl <= slope_max).fillna(False)

    long_entries = base_long & keep_long
    short_entries = base_short & keep_short
    long_exits = z > -exit_z
    short_exits = z < exit_z
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
