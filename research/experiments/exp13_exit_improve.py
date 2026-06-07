"""チャンピオンの「塩漬け失血」を断つ出口改善を実測する。

診断(exp12)結論: ワースト=長期保有。corr(保有,損益)=-0.85、保有31本以上は総崩れ。
→ 打ち手を実機/自作シミュで横並び比較:
   A. カタストロフ損切り sl_stop(エンジン純正・厳密)
   B. 時間ストップ max_bars(自作シミュ。"戻らないなら切る")
   C. A+B 併用
自作シミュは無ストップ時にエンジン値を再現できるか検証して信頼性を担保する。

実行: uv run python exp13_exit_improve.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config, universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table
from strategies.confluence_meanrev import _zscore, generate_signals

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
SIZE = 10_000.0
PARAMS = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35,
          "rsi_high": 65, "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}

uni.register_cross_spreads(3.0)
INSTRUMENTS = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]


# ---------- 自作シミュレータ(時間ストップ等のため) ----------
def simulate(close, le, lx, se, sx, z, half_frac, *, max_bars=None, z_stop=None,
             sl=None, size=SIZE):
    """エンジンと同じ約定規約(終値約定・往復1スプレッド)でトレードを再現。

    half_frac: 片側スリッページ(価格比) = (spread_pips*pip/2)/close[i]
    max_bars : 保有がこれ以上なら時間で手仕舞い
    z_stop   : ロングで z<=-z_stop / ショートで z>=z_stop なら損切り(行き過ぎ加速)
    sl       : 含み損が -sl(割合)に達したら損切り
    返り値: trades DataFrame(dir, entry_i, exit_i, bars_held, pnl, ret)
    """
    n = len(close)
    pos = 0  # 0 flat, 1 long, -1 short
    e_i = 0
    e_fill = 0.0
    rows = []
    le = le.values; lx = lx.values; se = se.values; sx = sx.values
    c = close.values; hf = half_frac.values; zz = z.values
    for i in range(n):
        if pos == 0:
            if le[i]:
                pos, e_i, e_fill = 1, i, c[i] * (1 + hf[i])
            elif se[i]:
                pos, e_i, e_fill = -1, i, c[i] * (1 - hf[i])
            continue
        held = i - e_i
        # 現在の含み損益(割合, コスト前)
        cur = (c[i] - e_fill) / e_fill if pos == 1 else (e_fill - c[i]) / e_fill
        exit_now = False
        if pos == 1:
            if lx[i]:
                exit_now = True
            elif max_bars is not None and held >= max_bars:
                exit_now = True
            elif z_stop is not None and zz[i] <= -z_stop:
                exit_now = True
            elif sl is not None and cur <= -sl:
                exit_now = True
        else:
            if sx[i]:
                exit_now = True
            elif max_bars is not None and held >= max_bars:
                exit_now = True
            elif z_stop is not None and zz[i] >= z_stop:
                exit_now = True
            elif sl is not None and cur <= -sl:
                exit_now = True
        if exit_now or i == n - 1:
            x_fill = c[i] * (1 - hf[i]) if pos == 1 else c[i] * (1 + hf[i])
            units = size / e_fill
            pnl = units * (x_fill - e_fill) if pos == 1 else units * (e_fill - x_fill)
            rows.append({"dir": "Long" if pos == 1 else "Short", "entry_i": e_i,
                         "exit_i": i, "bars_held": held, "pnl": pnl,
                         "ret": pnl / size * 100, "exit_ts": close.index[i]})
            pos = 0
    return pd.DataFrame(rows)


def sim_universe(**sim_kw) -> pd.DataFrame:
    frames = []
    for name in INSTRUMENTS:
        data = uni.instrument_data(name, TF)
        le, lx, se, sx = generate_signals(data, **PARAMS)
        z = _zscore(data["close"], PARAMS["window"])
        half = (config.spread_pips(name) * config.pip_size(name) / 2.0) / data["close"]
        tr = simulate(data["close"], le, lx, se, sx, z, half, **sim_kw)
        if not tr.empty:
            tr["instrument"] = name
            frames.append(tr)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.DatetimeIndex(df["exit_ts"]).year
    return df


def engine_universe(**run_kw) -> pd.DataFrame:
    frames = []
    for name in INSTRUMENTS:
        data = uni.instrument_data(name, TF)
        pf = run(name, TF, generate_signals, PARAMS, data=data, size_mode="value", **run_kw)
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        tt = tt.rename(columns={"return_pct": "ret"})
        tt["instrument"] = name
        frames.append(tt)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.DatetimeIndex(df["exit"]).year
    return df


def stats(df, label):
    pnl = df["pnl"]
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    yr = df.groupby("year")["pnl"].sum()
    return {
        "variant": label,
        "trades": len(df),
        "total_pnl": round(float(pnl.sum())),
        "PF": round(gp / gl, 3) if gl else np.inf,
        "win%": round(float((pnl > 0).mean()) * 100, 1),
        "worst%": round(float(df["ret"].min()), 2),
        "avg_hold": round(float(df["bars_held"].mean()), 1),
        "yrs+": f"{int((yr > 0).sum())}/{len(yr)}",
        "min_yr_pnl": round(float(yr.min())),
    }


def main():
    rows = []
    # 0) 妥当性検証: 自作シミュ vs エンジン(無ストップ)
    eng = engine_universe()
    sim = sim_universe()
    rows.append(stats(eng, "engine 無ストップ(基準)"))
    rows.append(stats(sim, "sim   無ストップ(検証)"))

    # A) エンジン純正 カタストロフ損切り
    for sl in (0.03, 0.02, 0.015, 0.01):
        rows.append(stats(engine_universe(sl_stop=sl), f"A. engine sl={sl:.1%}"))

    # B) 時間ストップ(自作シミュ)
    for mb in (50, 40, 30, 25, 20, 15):
        rows.append(stats(sim_universe(max_bars=mb), f"B. time max_bars={mb}"))

    # C) 併用(時間 + 緩い損切り)
    rows.append(stats(sim_universe(max_bars=30, sl=0.02), "C. max_bars=30 + sl=2%"))
    rows.append(stats(sim_universe(max_bars=25, sl=0.015), "C. max_bars=25 + sl=1.5%"))

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(config.RESULTS_DIR / "exp13_exit_improve.csv", index=False)
    print(f"\n保存: {config.RESULTS_DIR}/exp13_exit_improve.csv")


if __name__ == "__main__":
    main()
