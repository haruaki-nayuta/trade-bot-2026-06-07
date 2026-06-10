"""敵対検証: xau_tsmom_lb60_D1_doten_tsl4 (tsmom lb60 D1 doten + tsl_stop=0.04, XAUUSD).

報告値: n=142, sum=0.6192, PF=1.733, IS=1.18, OOS=2.29 (exp_asym_exit.py)

検証項目:
  (0) vectorbt の tsl ストップのフィル価格セマンティクス監査(合成ギャップテスト)
  (1) 自前コードで再現(エンジン経由 + 完全独立の手書きシミュレーション)
  (2) 先読み監査 + エントリー1バー遅延(全シグナル shift(1))
  (3) コスト1.5倍 ($0.40 → $0.60 フルスプレッド)
  (4) 2022年除外 sum(exit年/entry年の両方)

実行: PYTHONPATH=. uv run python research/experiments/trend2/verify_xau_tsmom_lb60_D1_doten_tsl4.py
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

ROOT = "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/research/lab")

import trend_lab as tl  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

LB = 60
TSL = 0.04
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
GOLD_FULL_SPREAD = 0.40  # USD


# --- シグナル(自前実装・strategies.tsmom と独立に書く) -------------------
def my_tsmom_doten(data: pd.DataFrame, lookback: int = LB):
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0  # 確定 close のみ・先読みなし
    ls = mom > 0.0
    ss = mom < 0.0
    le = ls & ~ls.shift(fill_value=False)
    se = ss & ~ss.shift(fill_value=False)
    return le, se, se, le  # long_exits=short_entries(ドテン)


def my_tsmom_doten_delay1(data: pd.DataFrame, lookback: int = LB):
    le, lx, se, sx = my_tsmom_doten(data, lookback)

    def sh(s):
        return s.shift(1, fill_value=False)

    return sh(le), sh(lx), sh(se), sh(sx)


# --- エンジン経由プール(自前ラッパー) ------------------------------------
def set_gold_spread(full_spread_usd: float) -> None:
    tl.register_spreads()
    config.SPREADS_PIPS["XAUUSD"] = full_spread_usd / 0.0001


def build_pool_engine(gen, data, tsl, full_spread_usd=GOLD_FULL_SPREAD) -> pd.DataFrame:
    set_gold_spread(full_spread_usd)
    pf = run("XAUUSD", "D1", gen, {"lookback": LB}, data=data,
             size_mode="value", side="both", tsl_stop=tsl)
    tt = trade_table(pf, data)
    return pd.DataFrame({
        "entry": tt["entry"].to_numpy(),
        "exit": tt["exit"].to_numpy(),
        "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
        "ret": tt["return_pct"].to_numpy() / 100.0,
        "entry_price": tt["entry_price"].to_numpy(),
        "exit_price": tt["exit_price"].to_numpy(),
    })


def stats(pool: pd.DataFrame) -> dict:
    r = pool["ret"]

    def pf(x):
        g = x[x > 0].sum()
        l = -x[x < 0].sum()
        return float(g / l) if l > 0 else float("inf")

    ent = pd.to_datetime(pool["entry"], utc=True)
    is_r = r[ent < OOS_START]
    oos_r = r[ent >= OOS_START]
    return {
        "n": int(len(pool)),
        "sum_ret": round(float(r.sum()), 4),
        "pool_pf": round(pf(r), 3),
        "is_pf": round(pf(is_r), 3),
        "oos_pf": round(pf(oos_r), 3),
        "is_sum": round(float(is_r.sum()), 4),
        "oos_sum": round(float(oos_r.sum()), 4),
        "n_is": int(len(is_r)), "n_oos": int(len(oos_r)),
    }


# --- 完全独立の手書きシミュレーション --------------------------------------
def manual_sim(data: pd.DataFrame, lookback=LB, tsl=TSL, full_spread=GOLD_FULL_SPREAD,
               delay=0, stop_fill="stop") -> pd.DataFrame:
    """close 約定・状態機械で再現。stop_fill='stop'(ストップ価格で約定=vbt想定)
    / 'close'(ブリーチしたバーの close で約定=ギャップ保守的)。
    delay=1 で全シグナル1バー遅延。先読みなし: mom は確定 close のみ。"""
    close = data["close"].to_numpy(float)
    idx = data.index
    n = len(close)
    half = full_spread / 2.0

    mom = np.full(n, np.nan)
    mom[lookback:] = close[lookback:] / close[:-lookback] - 1.0
    ls = mom > 0
    ss = mom < 0
    le = ls & ~np.roll(ls, 1); le[0] = False
    se = ss & ~np.roll(ss, 1); se[0] = False
    if delay:
        le = np.roll(le, delay); le[:delay] = False
        se = np.roll(se, delay); se[:delay] = False

    trades = []
    pos = 0
    entry_i = -1
    entry_eff = np.nan
    peak = np.nan

    def slip(i):
        return half / close[i]

    def close_trade(i, fill_px, d):
        if d == 1:
            exit_eff = fill_px * (1 - slip(i))
            ret = exit_eff / entry_eff - 1.0
        else:
            exit_eff = fill_px * (1 + slip(i))
            ret = (entry_eff - exit_eff) / entry_eff
        trades.append({"entry": idx[entry_i], "exit": idx[i], "dir": d, "ret": ret})

    def open_trade(i, d):
        nonlocal pos, entry_i, entry_eff, peak
        pos = d
        entry_i = i
        entry_eff = close[i] * (1 + slip(i)) if d == 1 else close[i] * (1 - slip(i))
        peak = close[i]

    for i in range(n):
        if pos != 0 and i > entry_i:
            # 1) トレーリングストップ(レベルは前バーまでの peak から既知 → 当バーで判定)
            level = peak * (1 - tsl) if pos == 1 else peak * (1 + tsl)
            hit = close[i] <= level if pos == 1 else close[i] >= level
            if hit:
                # 'stop' = レベルで約定(ギャップでも。vbt 同等=楽観)
                # 'close' = ブリーチしたバーの close で約定(ギャップ保守的)
                fill = level if stop_fill == "stop" else close[i]
                close_trade(i, fill, pos)
                pos = 0
            else:
                peak = max(peak, close[i]) if pos == 1 else min(peak, close[i])
        # 2) ドテン/新規(close 約定)
        if pos == 1 and se[i]:
            close_trade(i, close[i], 1)
            pos = 0
            open_trade(i, -1)
        elif pos == -1 and le[i]:
            close_trade(i, close[i], -1)
            pos = 0
            open_trade(i, 1)
        elif pos == 0:
            if le[i]:
                open_trade(i, 1)
            elif se[i]:
                open_trade(i, -1)
    if pos != 0:
        close_trade(n - 1, close[-1], pos)
    return pd.DataFrame(trades)


# --- (0) 合成ギャップテスト: vbt の tsl フィル価格を実測 --------------------
def gap_audit() -> dict:
    m = 40
    px = np.full(m, 100.0)
    px[10:20] = np.linspace(100, 110, 10)   # ロング中に 110 へ
    px[20] = 90.0                            # トレイル 105.6 を大きくギャップ割れ
    px[21:] = 90.0
    di = pd.date_range("2020-01-01", periods=m, freq="D", tz="UTC")
    data = pd.DataFrame({"open": px, "high": px, "low": px, "close": px,
                         "volume": 1.0}, index=di)

    def gen(d, lookback=0):
        le = pd.Series(False, index=d.index); le.iloc[5] = True
        empty = le & False
        return le, empty, empty, empty

    config.SPREADS_PIPS["SYNTH"] = 0.0
    pf = run("SYNTH", "D1", gen, {}, data=data, size_mode="value",
             side="both", tsl_stop=TSL)
    tt = trade_table(pf, data)
    exit_px = float(tt["exit_price"].iloc[0])
    # peak=110 → トレイルレベル 105.6。close は 90 にギャップ。
    return {"trail_level": 110 * (1 - TSL), "gap_close": 90.0,
            "vbt_exit_price": exit_px,
            "fills_at_stop_despite_gap": bool(abs(exit_px - 105.6) < 1e-6)}


def main() -> None:
    out = {}
    data = tl.load_tf("XAUUSD", "D1")
    print(f"data: {data.index[0]} .. {data.index[-1]}  rows={len(data)}")

    # (0) ストップフィル監査
    out["gap_audit"] = gap_audit()
    print("\n[0] gap_audit:", json.dumps(out["gap_audit"]))

    # (1a) エンジン経由再現
    pool = build_pool_engine(my_tsmom_doten, data, TSL)
    out["repro_engine"] = stats(pool)
    print("\n[1a] repro_engine:", json.dumps(out["repro_engine"]))

    # (1b) 完全独立シミュレーション(stopフィル=vbt同等 / closeフィル=保守的)
    for tag, sf in (("manual_stopfill", "stop"), ("manual_closefill", "close")):
        mp = manual_sim(data, stop_fill=sf)
        out[f"repro_{tag}"] = stats(mp)
        print(f"[1b] repro_{tag}:", json.dumps(out[f"repro_{tag}"]))

    # (2) エントリー1バー遅延(エンジン)
    pool_d1 = build_pool_engine(my_tsmom_doten_delay1, data, TSL)
    out["delay1_engine"] = stats(pool_d1)
    print("\n[2] delay1_engine:", json.dumps(out["delay1_engine"]))
    mp_d1 = manual_sim(data, delay=1, stop_fill="close")
    out["delay1_manual_closefill"] = stats(mp_d1)
    print("[2] delay1_manual_closefill:", json.dumps(out["delay1_manual_closefill"]))

    # (3) コスト1.5倍
    pool_c = build_pool_engine(my_tsmom_doten, data, TSL, full_spread_usd=GOLD_FULL_SPREAD * 1.5)
    out["cost15_engine"] = stats(pool_c)
    print("\n[3] cost15_engine:", json.dumps(out["cost15_engine"]))

    # (4) 2022年除外 sum(基準=再現プール)
    ex = pd.to_datetime(pool["exit"], utc=True).dt.year != 2022
    en = pd.to_datetime(pool["entry"], utc=True).dt.year != 2022
    out["ex2022"] = {
        "ex2022_sum_by_exit_year": round(float(pool.loc[ex, "ret"].sum()), 4),
        "ex2022_sum_by_entry_year": round(float(pool.loc[en, "ret"].sum()), 4),
        "sum_2022_only_by_exit_year": round(float(pool.loc[~ex, "ret"].sum()), 4),
    }
    print("\n[4] ex2022:", json.dumps(out["ex2022"]))

    # 年次内訳(集中度の確認)
    yearly = pool.groupby(pd.to_datetime(pool["exit"], utc=True).dt.year)["ret"].sum()
    out["yearly"] = {int(k): round(float(v), 4) for k, v in yearly.items()}
    print("\nyearly:", json.dumps(out["yearly"]))

    with open(ROOT + "/research/outputs/verify_xau_tsmom_lb60_D1_doten_tsl4.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved: research/outputs/verify_xau_tsmom_lb60_D1_doten_tsl4.json")


if __name__ == "__main__":
    main()
