"""nextbar_revscalp — M5 次足反転スキャルプ(EURUSD 専用設計)。

「直近100本の価格変動 → 次足エッジ」リサーチ(reports/12)の生き残り仕様:
  ロング:   z10 急落 × 高 ER10(ほぼ一直線の効率的な急落) × 広レンジ regime → 翌足決済
  ショート: z50 / ret3 急騰 × 高値ベタ引け(CLV) × 広レンジ regime → 5本保有
  共通:     UTC 20-23 時はエントリー禁止(ロールオーバーの BID スプレッド拡大
            アーティファクト。reports/12 §罠 参照)

閾値定数は train(2016-06〜2022-12, EURUSD M5)の分位から固定したもの。
  z10_thr   = train で ER10>er_thr に限定した z10 の 2% 分位(条件付き分位である点に注意)
  er_thr    = ER10 の train 2/3 分位
  rng_thr   = 100本レンジ幅(pips)の train 2/3 分位
  z50_thr / ret3_thr = train 98% 分位、clv_thr = ゾーン内 train 75% 分位
2023 年以降はこの定数にとって OOS。他ペアはロールオーバー除外後にコストを
超えないことが検証済み(EURUSD 限定で使う)。

検証:
  uv run python run_backtest.py nextbar_revscalp --pair EURUSD --tf M5
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = {
    "z10_thr": -2.302,
    "er_thr": 0.3785,
    "rng_thr": 48.0,
    "z50_thr": 2.686,
    "ret3_thr": 2.286,
    "clv_thr": 0.925,
    "long_hold": 1,
    "short_hold": 5,
}

PARAM_GRID = {
    "z10_thr": [-2.0, -2.302, -2.6],
    "rng_thr": [40.0, 48.0, 60.0],
    "long_hold": [1, 3],
    "short_hold": [3, 5],
}


def generate_signals(
    data: pd.DataFrame,
    z10_thr: float = -2.302,
    er_thr: float = 0.3785,
    rng_thr: float = 48.0,
    z50_thr: float = 2.686,
    ret3_thr: float = 2.286,
    clv_thr: float = 0.925,
    long_hold: int = 1,
    short_hold: int = 5,
):
    close, high, low = data["close"], data["high"], data["low"]
    # pip 幅は価格水準から推定(JPY ペア対策。本戦略は EURUSD 専用設計だが
    # クロスペア検証時に rng_thr の pips 換算が壊れないように)
    pip = 0.01 if float(close.median()) > 20 else 0.0001

    # --- ロング部品: 効率的な急落 ---
    z10 = (close - close.rolling(10).mean()) / close.rolling(10).std()
    er10 = (close - close.shift(10)).abs() / close.diff().abs().rolling(10).sum()

    # --- ショート部品: 急騰 × 高値ベタ引け ---
    z50 = (close - close.rolling(50).mean()) / close.rolling(50).std()
    ret3_norm = close.diff(3) / (close.diff().rolling(100).std() * np.sqrt(3))
    rng_bar = (high - low).replace(0, np.nan)
    clv = (close - low) / rng_bar

    # --- regime フィルタ: 直近100本レンジ幅(pips) ---
    rng100 = (high.rolling(100).max() - low.rolling(100).min()) / pip
    wide = rng100 >= rng_thr

    # --- ロールオーバー帯(UTC 20-23)はエントリーしない ---
    hour_ok = ~pd.Series(data.index.hour, index=data.index).isin([20, 21, 22, 23])

    long_entries = (z10 <= z10_thr) & (er10 > er_thr) & wide & hour_ok
    short_entries = (
        ((z50 >= z50_thr) | (ret3_norm >= ret3_thr))
        & (clv >= clv_thr)
        & wide
        & hour_ok
    )
    long_entries = long_entries.fillna(False)
    short_entries = short_entries.fillna(False)

    # 時間出口: エントリーの N 本後に決済
    long_exits = long_entries.shift(long_hold, fill_value=False)
    short_exits = short_entries.shift(short_hold, fill_value=False)

    return long_entries, long_exits, short_entries, short_exits
