"""trend2: セッション・オープニングレンジ・ブレイクアウト(ORB)。

文献的本命のイントラデイ・トレンド戦略。UTC バーをそのまま使用。
  London: 04:00-06:45 UTC のレンジ(close ベース)を 07:00-10:00 UTC のバーの
          close 上抜けでロング / 下抜けでショート。16:00 UTC 以降の最初のバーで全決済。
  NY:     09:00-11:45 UTC のレンジ → 12:00-15:00 UTC でブレイク判定 → 20:00 UTC 以降で決済。
  フィルタ変種: 当日レンジ幅 < 過去20営業日のレンジ幅中央値(shift(1) で当日除外)の
          日のみエントリー(収縮→拡張)。

先読み防止:
  - レンジ窓(〜06:45/11:45)はブレイク窓(07:00〜/12:00〜)より厳密に前に終わる
    (06:45 M15 バーは 07:00 に確定)。assert で窓の非重複を保証。
  - レンジは close のみから構築(クロス12は close 複製 OHLC のため)。
  - 20日中央値フィルタは日次レンジ幅 series の rolling(20).median().shift(1)。
    当日のレンジ幅自体は 07:00(ブレイク窓開始)時点で確定済みなので比較に使ってよい。
  - エントリーは日内・方向ごとに最初のブレイクのみ(cumsum==1)。

構成(固定グリッド・追い込みなし): {LDN, NY} × {raw, narrow} × {M15, M30} = 8、side=both。
対象: FX19(XAUUSD 除外)。net / gross(コスト0)両方を計測。

実行: リポジトリ直下で
  PYTHONPATH=. uv run python research/experiments/trend2/exp_orb_session.py
"""

from __future__ import annotations

import sys
import time

import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

# セッション定義(分単位, UTC)
SESSIONS = {
    "LDN": dict(rs=4 * 60, re=6 * 60 + 45, bs=7 * 60, be=10 * 60, ex=16 * 60),
    "NY": dict(rs=9 * 60, re=11 * 60 + 45, bs=12 * 60, be=15 * 60, ex=20 * 60),
}
FILT_DAYS = 20


def orb_signals(data: pd.DataFrame, session: str = "LDN", use_filter: int = 0):
    s = SESSIONS[session]
    assert s["bs"] > s["re"], "ブレイク窓はレンジ窓より後(先読み防止)"
    close = data["close"]
    idx = data.index
    tmin = idx.hour * 60 + idx.minute
    day = pd.Series(idx.floor("D"), index=idx)

    in_rng = pd.Series((tmin >= s["rs"]) & (tmin <= s["re"]), index=idx)
    in_brk = pd.Series((tmin >= s["bs"]) & (tmin <= s["be"]), index=idx)

    # 当日レンジ(close のみ)。ブレイク窓のバーから見れば全て確定済みバー。
    c_rng = close.where(in_rng)
    day_hi = c_rng.groupby(day).transform("max")
    day_lo = c_rng.groupby(day).transform("min")

    long_break = in_brk & (close > day_hi)   # NaN 比較は False
    short_break = in_brk & (close < day_lo)

    if use_filter:
        # 日次レンジ幅 < 過去20営業日(当日除く)の中央値 → 収縮日のみ
        width_bar = day_hi - day_lo
        daily_w = width_bar.groupby(day).first().dropna()
        med = daily_w.rolling(FILT_DAYS, min_periods=FILT_DAYS).median().shift(1)
        narrow_day = daily_w < med
        narrow = day.map(narrow_day).fillna(False).astype(bool)
        long_break &= narrow
        short_break &= narrow

    # 日内・方向ごとに最初のブレイクのみ
    le = long_break & (long_break.groupby(day).cumsum() == 1)
    se = short_break & (short_break.groupby(day).cumsum() == 1)

    # 手仕舞い: exit 時刻以降の最初のバーで全決済(オーバーナイト無し)
    eod = pd.Series(tmin >= s["ex"], index=idx)
    return le.fillna(False), eod, se.fillna(False), eod.copy()


CONFIGS = [
    (sess, filt, tf)
    for sess in ("LDN", "NY")
    for filt in (0, 1)
    for tf in ("M15", "M30")
]


def main() -> None:
    fx19 = [i for i in tl.default_instruments() if i != "XAUUSD"]
    print(f"instruments ({len(fx19)}): {fx19}")

    rows = []

    # ---- net(実コスト) ----
    for sess, filt, tf in CONFIGS:
        t0 = time.time()
        params = {"session": sess, "use_filter": filt}
        pool = tl.build_pool(orb_signals, params, tf=tf, side="both", instruments=fx19)
        st = tl.pool_stats(pool)
        st.update(label=f"{sess}_{'narrow' if filt else 'raw'}_{tf}", tf=tf,
                  session=sess, filt=filt, cost="net")
        rows.append(st)
        print(f"[net]   {st['label']:<18} n={st.get('n', 0):>6} "
              f"pf={st.get('pool_pf')} oos={st.get('oos_pf')} "
              f"mean_bps={st.get('mean_bps')} ({time.time() - t0:.0f}s)")

    # ---- gross(コスト0)診断 ----
    tl.register_spreads()
    from fxlab import config
    for k in list(config.SPREADS_PIPS):
        config.SPREADS_PIPS[k] = 0.0
    _orig = tl.register_spreads
    tl.register_spreads = lambda: None  # build_pool 内のリセット無効化
    try:
        for sess, filt, tf in CONFIGS:
            t0 = time.time()
            params = {"session": sess, "use_filter": filt}
            pool = tl.build_pool(orb_signals, params, tf=tf, side="both", instruments=fx19)
            st = tl.pool_stats(pool)
            st.update(label=f"{sess}_{'narrow' if filt else 'raw'}_{tf}", tf=tf,
                      session=sess, filt=filt, cost="gross")
            rows.append(st)
            print(f"[gross] {st['label']:<18} n={st.get('n', 0):>6} "
                  f"pf={st.get('pool_pf')} oos={st.get('oos_pf')} "
                  f"mean_bps={st.get('mean_bps')} ({time.time() - t0:.0f}s)")
    finally:
        tl.register_spreads = _orig

    df = pd.DataFrame(rows)
    out = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/outputs/trend2_orb_session.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
