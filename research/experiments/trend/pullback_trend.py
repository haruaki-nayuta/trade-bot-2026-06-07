"""押し目トレンドフォロー(close のみ=クロス対応)— 固定グリッド検証。

設計:
  長期トレンド: z250 = (close - SMA250) / STD250 を .shift(1)(確定バーのみ)。
    z250 > +zth で上昇トレンド / z250 < -zth で下降トレンド。
  エントリー: 上昇トレンド中に RSI(14) < rsi_lo(確定バー)でロング。
              下降トレンド中に RSI(14) > 100 - rsi_lo でショート。
  手仕舞い: RSI が 50 を回復(ロング: rsi>50 / ショート: rsi<50)or トレンド条件消滅。

固定グリッド(これ以外は回さない):
  H4: (zth, rsi_lo) in {(0.5,40), (1.0,40), (1.0,35)}
  D1: (zth, rsi_lo) in {(0.5,40), (1.0,40), (1.0,35)}
  side=both, 19対象(メジャー7+クロス12)。

実行: PYTHONPATH=. uv run python research/experiments/trend/pullback_trend.py
"""

import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")

import vectorbt as vbt  # noqa: E402

import trend_lab as tl  # noqa: E402


def generate_signals(data, zth=1.0, rsi_lo=40, z_win=250, rsi_win=14):
    """押し目トレンドフォロー。close のみ使用(クロスの合成OHLCでも正しい)。"""
    close = data["close"]
    sma = close.rolling(z_win).mean()
    std = close.rolling(z_win).std()
    z = ((close - sma) / std).shift(1)  # 確定バーのみ(先読み防止)
    rsi = vbt.RSI.run(close, rsi_win).rsi  # 自バー close 確定値で判定→同バー close 執行(標準プロトコル)

    up = (z > zth).fillna(False)
    dn = (z < -zth).fillna(False)

    long_entries = up & (rsi < rsi_lo)
    long_exits = (rsi > 50) | ~up
    short_entries = dn & (rsi > (100 - rsi_lo))
    short_exits = (rsi < 50) | ~dn
    return (long_entries.fillna(False), long_exits.fillna(False),
            short_entries.fillna(False), short_exits.fillna(False))


GRID = [
    ("H4", 0.5, 40),
    ("H4", 1.0, 40),
    ("H4", 1.0, 35),
    ("D1", 0.5, 40),
    ("D1", 1.0, 40),
    ("D1", 1.0, 35),
]


def main():
    rows = []
    for tf, zth, rsi_lo in GRID:
        label = f"pullback_z{zth}_rsi{rsi_lo}_{tf}_both"
        params = {"zth": zth, "rsi_lo": rsi_lo}
        pool = tl.build_pool(generate_signals, params, tf=tf, side="both")
        st = tl.pool_stats(pool)
        rows.append((label, tf, params, st))
        print(f"\n=== {label} ===")
        for k, v in st.items():
            print(f"  {k}: {v}")
    print("\n--- summary ---")
    for label, tf, params, st in rows:
        print(f"{label}: n={st.get('n')} pf={st.get('pool_pf')} "
              f"is={st.get('is_pf')} oos={st.get('oos_pf')} "
              f"mean_bps={st.get('mean_bps')} yearly={st.get('yearly_pos')} "
              f"worst={st.get('worst_year')}")


if __name__ == "__main__":
    main()
