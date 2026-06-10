"""critic_pocket_xau — pocket_donch_e55x20_H4_XAUUSD_long への敵対監査。

対象(報告値): donchian_close e=55/x=20, H4, XAUUSD, long-only
  n=154, sum=0.7744, PF=1.776, IS=1.262, OOS=2.454

監査:
  1. 独立再現: vectorbt を使わない手書きループ(同バー終値約定+半スプレッド片道)
     + 発見者パイプライン(tl.build_pool side='long')の両方で再計算。
  2. 先読み監査: ロジック検査(levels は .shift(1) 済みか)+ エントリー/エグジット
     1バー遅延(シグナル翌バー終値約定)での再計算。
  3. コスト 1.5 倍($0.40 → $0.60 フルスプレッド)。
  4. 2022年除外 sum(exit 年基準 / entry 年基準の両方)。

実行: PYTHONPATH=. uv run python research/experiments/trend2/critic_pocket_xau.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

from fxlab import config  # noqa: E402

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
E, X = 55, 20
INSTR = "XAUUSD"


def donchian_close(data: pd.DataFrame, entry_window: int = 55, exit_window: int = 20):
    """発見者と同一ロジック(critic_audit.py から固定再掲)。"""
    close = data["close"]
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(entry_window).min().shift(1)
    exit_upper = close.rolling(exit_window).max().shift(1)
    exit_lower = close.rolling(exit_window).min().shift(1)
    return close > upper, close < exit_lower, close < lower, close > exit_upper


def donchian_close_delay1(data: pd.DataFrame, entry_window: int = 55, exit_window: int = 20):
    """全シグナルを1バー遅延(シグナル確定の翌バー終値で約定)。"""
    sigs = donchian_close(data, entry_window, exit_window)
    return tuple(s.shift(1, fill_value=False) for s in sigs)


# ------------------------------------------------------------------
# 1. 手書きループ(vectorbt 非依存の独立実装)
# ------------------------------------------------------------------
def hand_loop_long(data: pd.DataFrame, half_spread: float, delay: int = 0) -> pd.DataFrame:
    """long-only Donchian。delay=0: シグナルバー終値約定 / delay=1: 翌バー終値約定。"""
    close = data["close"].to_numpy()
    idx = data.index
    c = data["close"]
    upper = c.rolling(E).max().shift(1)
    exit_lower = c.rolling(X).min().shift(1)
    le = (c > upper).to_numpy()
    lx = (c < exit_lower).to_numpy()

    n = len(close)
    trades = []
    pos = 0
    entry_i = -1
    entry_fill = np.nan

    for i in range(n):
        j = i + delay  # 約定バー
        if j >= n:
            break
        if pos == 1 and lx[i]:
            xp = close[j] - half_spread
            trades.append((idx[entry_i], idx[j], entry_fill, xp, xp / entry_fill - 1.0))
            pos = 0
        elif pos == 0 and le[i]:
            pos, entry_i, entry_fill = 1, j, close[j] + half_spread
    if pos == 1:  # 建玉中のまま終了 → 最終バー終値(スリッページなし)で評価
        trades.append((idx[entry_i], idx[-1], entry_fill, close[-1],
                       close[-1] / entry_fill - 1.0))
    return pd.DataFrame(trades, columns=["entry", "exit", "entry_fill", "exit_fill", "ret"])


def stats(df: pd.DataFrame, ret_col: str = "ret") -> dict:
    r = df[ret_col]

    def pf(x):
        g = x[x > 0].sum()
        l = -x[x < 0].sum()
        return float(g / l) if l > 0 else float("inf")

    is_r = r[df["entry"] < OOS_START]
    oos_r = r[df["entry"] >= OOS_START]
    yr_exit = r.groupby(pd.to_datetime(df["exit"]).dt.year).sum()
    return {
        "n": int(len(df)),
        "sum_ret": round(float(r.sum()), 4),
        "pf": round(pf(r), 3),
        "is_pf": round(pf(is_r), 3),
        "oos_pf": round(pf(oos_r), 3),
        "is_sum": round(float(is_r.sum()), 4),
        "oos_sum": round(float(oos_r.sum()), 4),
        "ex2022_sum_exityear": round(float(yr_exit.drop(2022, errors="ignore").sum()), 4),
        "yearly": {int(k): round(float(v), 4) for k, v in yr_exit.items()},
    }


def pool_to_df(pool: pd.DataFrame) -> pd.DataFrame:
    out = pool[["entry", "exit", "ret"]].copy()
    out["entry"] = pd.to_datetime(out["entry"])
    out["exit"] = pd.to_datetime(out["exit"])
    return out


def main() -> None:
    tl.register_spreads()
    data = tl.load_tf(INSTR, "H4")
    print(f"data: {INSTR} H4 {data.index[0]} .. {data.index[-1]}  bars={len(data)}")
    full_spread = config.spread_pips(INSTR) * config.pip_size(INSTR)
    print(f"full spread = ${full_spread:.2f}")

    # ---- 1. 再現 ----
    print("\n=== 1. 再現 ===")
    hand = hand_loop_long(data, full_spread / 2.0, delay=0)
    st_hand = stats(hand)
    print("hand-loop (独立実装):", st_hand)

    pool = tl.build_pool(donchian_close, {"entry_window": E, "exit_window": X},
                         tf="H4", side="long", instruments=[INSTR])
    st_pool = stats(pool_to_df(pool))
    print("tl.build_pool side=long:", st_pool)
    print("tl.pool_stats:", tl.pool_stats(pool))

    # both プールの long 側(発見元の集計方法)も確認
    pool_b = tl.build_pool(donchian_close, {"entry_window": E, "exit_window": X},
                           tf="H4", side="both", instruments=[INSTR])
    pool_bl = pool_b[pool_b["dir"] == 1]
    st_bl = stats(pool_to_df(pool_bl))
    print("both プールの dir==1 側:", st_bl)

    # ---- 2. 先読み監査: 1バー遅延 ----
    print("\n=== 2. 1バー遅延(シグナル翌バー終値約定) ===")
    hand_d1 = hand_loop_long(data, full_spread / 2.0, delay=1)
    st_d1 = stats(hand_d1)
    print("hand-loop delay=1:", st_d1)
    pool_d1 = tl.build_pool(donchian_close_delay1, {"entry_window": E, "exit_window": X},
                            tf="H4", side="long", instruments=[INSTR])
    st_pool_d1 = stats(pool_to_df(pool_d1))
    print("build_pool delay=1:", st_pool_d1)

    # ---- 3. コスト 1.5 倍 ----
    print("\n=== 3. コスト1.5倍 (full $0.60) ===")
    hand_c15 = hand_loop_long(data, full_spread * 1.5 / 2.0, delay=0)
    st_c15 = stats(hand_c15)
    print("hand-loop cost x1.5:", st_c15)
    saved = dict(config.SPREADS_PIPS)
    _orig = tl.register_spreads
    try:
        config.SPREADS_PIPS[INSTR] = saved[INSTR] * 1.5
        tl.register_spreads = lambda: None
        pool_c15 = tl.build_pool(donchian_close, {"entry_window": E, "exit_window": X},
                                 tf="H4", side="long", instruments=[INSTR])
    finally:
        tl.register_spreads = _orig
        config.SPREADS_PIPS.update(saved)
        tl.register_spreads()
    st_pool_c15 = stats(pool_to_df(pool_c15))
    print("build_pool cost x1.5:", st_pool_c15)

    # ---- 4. 2022年除外 sum ----
    print("\n=== 4. 2022年除外 ===")
    h = pool_to_df(pool)
    ex_exit = float(h.loc[pd.to_datetime(h["exit"]).dt.year != 2022, "ret"].sum())
    ex_entry = float(h.loc[pd.to_datetime(h["entry"]).dt.year != 2022, "ret"].sum())
    y2022 = float(h.loc[pd.to_datetime(h["exit"]).dt.year == 2022, "ret"].sum())
    print(f"baseline sum={h['ret'].sum():.4f}  2022(exit年)寄与={y2022:.4f}")
    print(f"ex2022 (exit年基準) ={ex_exit:.4f}")
    print(f"ex2022 (entry年基準)={ex_entry:.4f}")
    print("年別:", st_pool["yearly"])

    # ---- 突き合わせ(hand vs pipeline) ----
    print("\n=== 突き合わせ ===")
    m = pool_to_df(pool).merge(hand, on="entry", suffixes=("_vbt", "_hand"))
    print(f"matched {len(m)}/{len(pool)}  max|ret_diff|="
          f"{(m['ret_vbt'] - m['ret_hand']).abs().max():.6f}  "
          f"exit不一致={(m['exit_vbt'] != m['exit_hand']).sum()}")


if __name__ == "__main__":
    main()
