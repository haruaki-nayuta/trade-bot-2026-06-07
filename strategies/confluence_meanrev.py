"""コンフルエンス平均回帰(Zスコア × RSI × ボラフィルタ × マルチウィンドウ)= 本リポジトリのチャンピオン。

これまでの検証で「メジャー/クロスFXの頑健なエッジは平均回帰のみ(順張りは net マイナス)」
「固定しきい値は通貨依存を生む」と判明。本戦略は複数の**独立な自己正規化条件**が同時に
行き過ぎを示す高確信の局面だけ建玉する。すべて経済合理で、パラメータ穿りをしていない。

4つの確認(すべて満たすときのみエントリー):
  1. 短期Zスコア:  Z = (close - SMA(window)) / STD(window) が ±entry_z を突破(行き過ぎ)
  2. RSI:          RSI(rsi_p) も同方向に行き過ぎ(売られすぎ/買われすぎ)
  3. 長期Zスコア:  長期窓 slow_win のZが ±slow_z 超(=より大きな時間軸でも持続的に乖離)
  4. 平穏レジーム:  直近実現ボラ ≤ 過去 vol_win 本の vol_pct 分位(危機/急変はスキップ)
手仕舞い: 短期Zが平均近傍(±exit_z)へ回帰。損切り不要(回帰で自然手仕舞い、DD 浅め)。

**検証結果(H4, size=value, カーブフィットなし。reports/04):**
  * 20対象(メジャー7+クロス13)等加重ポートフォリオで **プラス年率 100%(11/11年, 2024含む)**
  * PF中央 ≈1.55 / 年取引 ≈150(合算)/ 19-20対象が通算プラス(AUDJPYのみ弱い)
  * 100%プラス年は slow_z∈[1.5,1.75]×slow_win∈{200,250,300}×vol_pct∈{0.65,0.75} の高原で安定
    (=偶然の1点でない)。クロスのスプレッドを4pipsに上げても 100% を維持。
  * 個別は AUDCAD PF5.1 / EURGBP4.1 / EURUSD2.0 など PF2.0超も多数。

先読みなし(全指標が確定バー)。close のみ使用=クロス合成系列にもそのまま適用可(exp05.build_crosses)。
slow_z=0 で長期Z条件を無効、vol_pct>=1.0 でボラフィルタを無効にできる(アブレーション用)。
"""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt

PARAMS = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35,
          "rsi_high": 65, "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.5}
PARAM_GRID = {
    "window": [30, 50, 100],
    "entry_z": [1.5, 2.0, 2.5],
    "exit_z": [0.0, 0.5],
    "rsi_p": [14],
    "rsi_low": [30, 35],
    "rsi_high": [65, 70],
    "vol_win": [100],
    "vol_pct": [0.70],
    "slow_win": [200, 250, 300],
    "slow_z": [1.0, 1.5, 1.75],
}


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def generate_signals(data: pd.DataFrame, window: int = 50, entry_z: float = 2.0, exit_z: float = 0.5,
                     rsi_p: int = 14, rsi_low: float = 35, rsi_high: float = 65,
                     vol_win: int = 100, vol_pct: float = 0.70,
                     slow_win: int = 250, slow_z: float = 1.5):
    close = data["close"]
    z = _zscore(close, window)
    rsi = vbt.RSI.run(close, rsi_p).rsi

    # 平穏レジーム(vol_pct>=1.0 で無効)
    if vol_pct >= 1.0:
        calm = pd.Series(True, index=close.index)
    else:
        vol = close.pct_change().rolling(20).std()
        calm = vol <= vol.rolling(vol_win).quantile(vol_pct)

    # 長期Z(マルチウィンドウ)合流(slow_z<=0 で無効)
    if slow_z > 0:
        zs = _zscore(close, slow_win)
        long_ok = (zs < -slow_z).fillna(False)
        short_ok = (zs > slow_z).fillna(False)
    else:
        long_ok = short_ok = pd.Series(True, index=close.index)

    long_entries = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm & long_ok
    short_entries = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm & short_ok
    long_exits = z > -exit_z
    short_exits = z < exit_z
    return long_entries.fillna(False), long_exits.fillna(False), \
        short_entries.fillna(False), short_exits.fillna(False)
