"""アジア時間レンジ・ブレイクアウト(ロンドン時間の放れに乗る日中戦略)。

FX は流動性・ボラがセッション境界に構造的に集中する。低ボラのアジア時間(0–7 UTC)に
形成されたレンジを、ロンドン時間(7–12 UTC)に上抜け/下抜けした方向へ順張りする。
オーバーナイトはギャップ риск回避のため建玉しない(夕方にフラット化)。
これは時間帯の構造特性に基づく経済合理的なエッジで、価格パラメータ穿りではない。

  * アジアレンジ: その日の 0:00–asian_end UTC の高値/安値(asian_end 時点で確定)
  * 買い: entry 窓(asian_end–entry_end)で 終値 > 当日アジア高値(その日最初の放れのみ)
  * 売り: entry 窓で 終値 < 当日アジア安値(同上)
  * 手仕舞い: exit_hour 以降でフラット(日跨ぎしない)
先読み防止: アジアレンジは確定後(hour>=asian_end)のみ参照。エントリーは窓内・日内初回のみ。
時間足は時刻粒度が要るので H1 以下で使う。
"""

from __future__ import annotations

import pandas as pd

PARAMS = {"asian_end": 7, "entry_end": 12, "exit_hour": 20}
PARAM_GRID = {"asian_end": [6, 7, 8], "entry_end": [11, 12, 14], "exit_hour": [18, 20, 22]}


def generate_signals(data: pd.DataFrame, asian_end: int = 7, entry_end: int = 12,
                     exit_hour: int = 20):
    high, low, close = data["high"], data["low"], data["close"]
    idx = data.index
    hour = idx.hour
    day = idx.floor("D")

    # その日のアジア時間(0:00–asian_end)の高値/安値 → 全バーへマップ(asian_end 時点で確定)
    asian = (hour < asian_end)
    asian_high_by_day = high[asian].groupby(day[asian]).max()
    asian_low_by_day = low[asian].groupby(day[asian]).min()
    day_s = pd.Series(day, index=idx)
    ah = day_s.map(asian_high_by_day)
    al = day_s.map(asian_low_by_day)

    entry_win = pd.Series((hour >= asian_end) & (hour < entry_end), index=idx)
    long_break = entry_win & (close > ah) & ah.notna()
    short_break = entry_win & (close < al) & al.notna()

    # その日最初の放れだけを採用(cumsum==1)
    day_idx = pd.Series(day, index=idx)
    first_long = long_break & (long_break.groupby(day_idx).cumsum() == 1)
    first_short = short_break & (short_break.groupby(day_idx).cumsum() == 1)

    eod = pd.Series(hour >= exit_hour, index=idx)  # 夕方以降でフラット化(日跨ぎ回避)

    long_entries = first_long.fillna(False)
    short_entries = first_short.fillna(False)
    long_exits = eod
    short_exits = eod
    return long_entries, long_exits, short_entries, short_exits
