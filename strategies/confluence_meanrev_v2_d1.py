"""チャンピオン confluence_meanrev_v2 のエントリー1バー遅延変種(d1)— 採用候補の本番実装。

機構(reports/15 / exp47):
  v2 のエントリー直後 1〜3 本は平均で逆行する(条件付き収束プレミアムの「第1波」)。
  エントリーをシグナルバーの次の H4 バー close まで 1 本待つことで
    (a) 平均的に僅かに有利な価格で建玉できる(プール段 +2.8%)
    (b) 逆行第1波を MtM パスから外す → DD 形状が浅くなり、ブート p95=20% 較正の
        掛け目 k が上がる(これが口座レベル利得の主因)
  exit ルールは不変(z が exit 閾値へ戻ったら決済)。

ルール:
  ロング  : le_d1 = le_v2.shift(1) & (z <= -exit_z)
  ショート: se_d1 = se_v2.shift(1) & (z >=  exit_z)
  つまり「v2 が前バーでエントリーシグナルを出し、かつ遅延先バーで z がまだ exit 閾値を
  超えたまま(=トレードがまだ生きている)」ときだけ次バー close で建玉する。
  遅延先バーで z がすでに exit 域へ戻っていたら見送り(そのトレードは消滅)。

先読みなし:
  ・le/se は v2 がシグナルバー t の確定値のみで生成(v2 自体が因果)。shift(1) で
    バー t+1 に持ち越すため、t+1 時点では完全に既知。
  ・ゲートに使う z はバー t+1 の close までで計算した同じ window=50 の z スコア。
    エントリーは t+1 の close 執行(vectorbt 既定)なので、その時点で z[t+1] は確定済み。
  ・exit シグナルは v2 のものをそのまま使用(z > -exit_z / z < exit_z、確定バーのみ)。

検証経緯:
  exp47: 再構成方式(ベースプールの entry を1本後ろへシフト)で全6ゲート通過。
         robust(ブートp95 DD=20%, シード0-4平均) +16.41% → +18.63%(全シード正)、
         empirical +24.64% → +27.50%(p95 は悪化せず=レバ偽装でない)、全年プラス、
         IS較正→OOS rob +24.79% vs base +21.04%。d2/d3 は単年依存・署名で reject。
  exp51: 本ファイル(シグナルレベル+vectorbt 約定・コスト処理)による独立再実装で
         exp47 の数値を突合・再現監査(research/experiments/exp51_d1_reimpl.py)。

PARAMS / PARAM_GRID は v2 と同一。遅延幅は 1 固定(d2/d3 は exp47 で棄却済みのため
パラメータ化しない=逐次探索バイアスの再導入を防ぐ)。
"""

from __future__ import annotations

import pandas as pd

from .confluence_meanrev_v2 import PARAM_GRID, PARAMS  # noqa: F401 (CLI が参照)
from .confluence_meanrev_v2 import generate_signals as _generate_signals_v2


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def generate_signals(data: pd.DataFrame, **params):
    """v2 のシグナルを生成し、エントリーのみ 1 バー遅延 + z ゲートを適用する。

    返り値は v2 と同じ 4 要素 (long_entries, long_exits, short_entries, short_exits)。
    exit は v2 のまま(不変)。
    """
    window = params.get("window", PARAMS["window"])
    exit_z = params.get("exit_z", PARAMS["exit_z"])

    le, lx, se, sx = _generate_signals_v2(data, **params)

    # v2 内部と同一定義の短期 z(close のみ・因果)。遅延先バー t+1 の確定 close まで
    # で計算されるため、t+1 close 執行のエントリー判断に使ってよい(先読みなし)。
    z = _zscore(data["close"], window)

    # エントリーを 1 バー持ち越し、遅延先バーで z がまだ exit 閾値の外側にある
    # (=exit シグナルが立っていない)ときだけ建玉。NaN 比較は False = 見送り。
    long_entries = le.shift(1, fill_value=False) & (z <= -exit_z)
    short_entries = se.shift(1, fill_value=False) & (z >= exit_z)

    return long_entries.fillna(False), lx, short_entries.fillna(False), sx
