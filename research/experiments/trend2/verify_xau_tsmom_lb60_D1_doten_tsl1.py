"""敵対検証: xau_tsmom_lb60_D1_doten_tsl1 (tsmom lb60 D1 doten + tsl_stop=0.01, XAUUSD単独)。

報告値: n=127, sum=0.5808, PF=1.992, IS=1.568, OOS=2.374
発見者: research/experiments/trend2/exp_asym_exit.py

検証項目:
  A. 独立実装での再現(sum_ret ±20%)
  B. 先読み監査: シグナルの因果性(truncation test)
  C. エントリー1バー遅延(全シグナル shift(1)、tsl はそのまま)
  D. コスト1.5倍(XAUUSD $0.40 → $0.60 フルスプレッド)
  E. 2022年除外 sum(exit年 / entry年 両方)
  F. 追加診断: ストップ約定モデル
     - stop_exit_price='close'(レベル約定でなく当日終値約定 = ギャップ楽観の除去)
     - high/low を渡して日中値でトレーリング(実ブローカーのトレーリング挙動に近い)

実行: PYTHONPATH=. uv run python research/experiments/trend2/verify_xau_tsmom_lb60_D1_doten_tsl1.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

ROOT = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/research/lab")

import vectorbt as vbt  # noqa: E402

import trend_lab as tl  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

OOS = pd.Timestamp("2022-01-01", tz="UTC")


# --- 独立実装(strategies.tsmom は import しない) -------------------------
def my_tsmom(data: pd.DataFrame, lookback: int = 60):
    """tsmom ドテン。過去 lookback 本リターンの符号転換でエントリー/ドテン。"""
    c = data["close"]
    mom = c / c.shift(lookback) - 1.0
    long_state = mom > 0.0
    short_state = mom < 0.0
    le = long_state & ~long_state.shift(fill_value=False)
    se = short_state & ~short_state.shift(fill_value=False)
    return le, se, se, le  # (long_entries, long_exits=se, short_entries, short_exits=le)


def my_tsmom_delay1(data: pd.DataFrame, lookback: int = 60):
    le, lx, se, sx = my_tsmom(data, lookback)
    s = lambda x: x.shift(1, fill_value=False)  # noqa: E731
    return s(le), s(lx), s(se), s(sx)


# --- 統計 ------------------------------------------------------------------
def stats(tt: pd.DataFrame) -> dict:
    r = tt["return_pct"] / 100.0

    def pf(x):
        g = x[x > 0].sum()
        l = -x[x < 0].sum()
        return round(float(g / l), 3) if l > 0 else float("inf")

    is_r = r[tt["entry"] < OOS]
    oos_r = r[tt["entry"] >= OOS]
    yearly = r.groupby(tt["exit"].dt.year).sum()
    return {
        "n": int(len(tt)),
        "sum_ret": round(float(r.sum()), 4),
        "pool_pf": pf(r),
        "is_pf": pf(is_r),
        "oos_pf": pf(oos_r),
        "is_sum": round(float(is_r.sum()), 4),
        "oos_sum": round(float(oos_r.sum()), 4),
        "win_rate": round(float((r > 0).mean()), 3),
        "yearly": {int(k): round(float(v), 4) for k, v in yearly.items()},
    }


def run_my(data, gen, tsl, *, stop_exit_price=None, use_hl=False):
    """fxlab.run と同条件の独立呼び出し(stop 約定モデルを差し替え可能)。"""
    le, lx, se, sx = gen(data, lookback=60)
    close = data["close"]
    half = config.spread_pips("XAUUSD") * config.pip_size("XAUUSD") / 2.0
    kw = {}
    if use_hl:
        kw["high"] = data["high"]
        kw["low"] = data["low"]
    if stop_exit_price is not None:
        kw["stop_exit_price"] = stop_exit_price
    pf = vbt.Portfolio.from_signals(
        close, entries=le, exits=lx, short_entries=se, short_exits=sx,
        slippage=half / close, fees=config.COMMISSION_FRACTION,
        init_cash=10_000, freq=config.TIMEFRAMES["D1"],
        size=10_000, size_type="value",
        sl_stop=tsl, sl_trail=True, **kw,
    )
    return trade_table(pf, data)


def main() -> None:
    tl.register_spreads()  # XAUUSD = $0.40 フル
    data = tl.load_tf("XAUUSD", "D1")
    print(f"data: {data.index[0]} .. {data.index[-1]}  rows={len(data)}")
    out = {}

    # --- A. 再現(fxlab.run 経由 = 発見者と同経路だが自前シグナル) ---
    pf = run("XAUUSD", "D1", my_tsmom, {"lookback": 60}, data=data,
             size_mode="value", side="both", tsl_stop=0.01)
    tt_base = trade_table(pf, data)
    out["A_base_fxlab_run"] = stats(tt_base)

    # 完全自前経路(vbt 直叩き)でも一致するか
    tt_mine = run_my(data, my_tsmom, 0.01)
    out["A_base_direct_vbt"] = stats(tt_mine)

    # --- B. 先読み監査: truncation test --------------------------------
    le_f, lx_f, se_f, sx_f = my_tsmom(data, 60)
    trunc_ok = True
    for k in (300, 1000, 2000, len(data) - 5):
        sub = data.iloc[:k]
        le_s, lx_s, se_s, sx_s = my_tsmom(sub, 60)
        for full, part in ((le_f, le_s), (lx_f, lx_s), (se_f, se_s), (sx_f, sx_s)):
            if not full.iloc[:k].equals(part):
                trunc_ok = False
    out["B_truncation_causal"] = bool(trunc_ok)

    # --- C. エントリー1バー遅延 ----------------------------------------
    pf_d = run("XAUUSD", "D1", my_tsmom_delay1, {"lookback": 60}, data=data,
               size_mode="value", side="both", tsl_stop=0.01)
    out["C_delay1"] = stats(trade_table(pf_d, data))

    # --- D. コスト1.5倍 ($0.60 フル) ------------------------------------
    config.SPREADS_PIPS["XAUUSD"] = 0.60 / 0.0001
    pf_c = run("XAUUSD", "D1", my_tsmom, {"lookback": 60}, data=data,
               size_mode="value", side="both", tsl_stop=0.01)
    out["D_cost15"] = stats(trade_table(pf_c, data))
    config.SPREADS_PIPS["XAUUSD"] = 0.40 / 0.0001  # 戻す

    # --- E. 2022年除外 sum ----------------------------------------------
    r = tt_base["return_pct"] / 100.0
    out["E_ex2022"] = {
        "sum_ex2022_exit_year": round(float(r[tt_base["exit"].dt.year != 2022].sum()), 4),
        "sum_ex2022_entry_year": round(float(r[tt_base["entry"].dt.year != 2022].sum()), 4),
        "sum_2022_exit_year": round(float(r[tt_base["exit"].dt.year == 2022].sum()), 4),
    }

    # --- F. ストップ約定モデル診断 --------------------------------------
    out["F_stopexit_close"] = stats(run_my(data, my_tsmom, 0.01, stop_exit_price="close"))
    out["F_intrabar_hl"] = stats(run_my(data, my_tsmom, 0.01, use_hl=True))
    out["F_intrabar_hl_exitclose"] = stats(
        run_my(data, my_tsmom, 0.01, use_hl=True, stop_exit_price="close"))

    # delay1 × cost1.5x(複合・参考)
    config.SPREADS_PIPS["XAUUSD"] = 0.60 / 0.0001
    pf_dc = run("XAUUSD", "D1", my_tsmom_delay1, {"lookback": 60}, data=data,
                size_mode="value", side="both", tsl_stop=0.01)
    out["G_delay1_cost15"] = stats(trade_table(pf_dc, data))
    config.SPREADS_PIPS["XAUUSD"] = 0.40 / 0.0001

    for k, v in out.items():
        print(k, "=>", json.dumps(v, ensure_ascii=False))

    pd.DataFrame(
        [{"test": k, **(v if isinstance(v, dict) else {"value": v})} for k, v in out.items()]
    ).to_csv(ROOT + "/research/outputs/trend2_verify_xau_tsmom_lb60_tsl1.csv", index=False)
    print("saved: research/outputs/trend2_verify_xau_tsmom_lb60_tsl1.csv")


if __name__ == "__main__":
    main()
