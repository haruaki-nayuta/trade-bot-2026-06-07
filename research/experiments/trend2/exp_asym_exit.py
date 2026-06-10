"""非対称出口(トレーリングストップ)でのトレンドフォロー再測定(デバイアス・レビュー)。

これまでのトレンド検証(research/experiments/trend/)は「反対シグナルでドテン」という
対称出口だった。本物のトレンドフォローの損小利大(トレーリングで利を伸ばす)を
fxlab.backtest.run の tsl_stop で直接与え、H4/D1 を再測定する。

ファミリ(固定グリッド・追い込み禁止):
  (a) ドンチャン close 版: entry {55,100} × {H4,D1}。出口=反対シグナル(ドテン)+ tsl
  (b) tsmom: lb60(D1), lb360(H4)。出口=反対シグナル(ドテン)+ tsl
  tsl_stop ∈ {0.01, 0.02, 0.04}
対象:
  FX19(XAUUSD除外)= 18構成(exits併用版のみ)
  XAUUSD 単独: tsmom D1 lb60 × 3 tsl × {ドテン併用, エントリーのみ(出口tsl任せ)} = 6構成
計24構成。

先読みなし: rolling 極値は .shift(1)、tsmom は確定 close のみ。
コスト: メジャー0.6-1.4pips / クロス3pips / XAUUSD $0.40(tl.register_spreads)。
gross 診断は省略(gross_mean_bps = net と同値で報告)。

実行: PYTHONPATH=. uv run python research/experiments/trend2/exp_asym_exit.py
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
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from strategies.tsmom import generate_signals as tsmom_signals  # noqa: E402


# --- シグナル生成(全て確定バーのみ・先読みなし) -------------------------
def donchian_doten(data: pd.DataFrame, entry_window: int = 55):
    """close ベース・ドンチャン。出口=反対側ブレイク(ドテン)。tsl は run() 側で付与。"""
    close = data["close"]
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(entry_window).min().shift(1)
    le = close > upper
    se = close < lower
    return le, se, se, le  # long_exits=short_entries, short_exits=long_entries


def donchian_entry_only(data: pd.DataFrame, entry_window: int = 55):
    """エントリーのみ(出口はトレーリング任せ)。"""
    close = data["close"]
    upper = close.rolling(entry_window).max().shift(1)
    lower = close.rolling(entry_window).min().shift(1)
    le = close > upper
    se = close < lower
    empty = le & False
    return le, empty, se, empty


def tsmom_doten(data: pd.DataFrame, lookback: int = 60):
    """tsmom(ドテン)。strategies.tsmom と同一、band=0。"""
    return tsmom_signals(data, lookback=lookback, band=0.0)


def tsmom_entry_only(data: pd.DataFrame, lookback: int = 60):
    """tsmom エントリーのみ(符号転換時のみ建玉、出口はトレーリング任せ)。"""
    le, lx, se, sx = tsmom_signals(data, lookback=lookback, band=0.0)
    empty = le & False
    return le, empty, se, empty


# --- プール構築(tl.build_pool 同形・tsl_stop 対応) -----------------------
def build_pool_tsl(gen, params: dict, tf: str, tsl: float,
                   instruments: list[str]) -> pd.DataFrame:
    tl.register_spreads()
    frames = []
    for nm in instruments:
        data = tl.load_tf(nm, tf)
        try:
            pf = run(nm, tf, gen, params, data=data, size_mode="value",
                     side="both", tsl_stop=tsl)
            tt = trade_table(pf, data)
        except Exception as e:  # noqa: BLE001
            print(f"  !! {nm} failed: {e}", file=sys.stderr)
            continue
        if tt.empty:
            continue
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
        }))
    if not frames:
        return pd.DataFrame(columns=["instr", "entry", "exit", "dir",
                                     "entry_price", "ret", "bars_held"])
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


# --- 構成(固定・全件報告) -------------------------------------------------
FX19 = [i for i in tl.default_instruments() if i != "XAUUSD"]
TSLS = [0.01, 0.02, 0.04]


def tsl_tag(ts: float) -> str:
    return f"tsl{ts * 100:g}"


CONFIGS: list[tuple[str, object, dict, str, list[str]]] = []
# (a) ドンチャン doten+tsl, FX19
for tf in ("H4", "D1"):
    for e in (55, 100):
        for ts in TSLS:
            CONFIGS.append((
                f"donch_e{e}_{tf}_doten_{tsl_tag(ts)}_fx19",
                donchian_doten, {"entry_window": e}, tf, FX19, ts,
            ))
# (b) tsmom doten+tsl, FX19
for tf, lb in (("D1", 60), ("H4", 360)):
    for ts in TSLS:
        CONFIGS.append((
            f"tsmom_lb{lb}_{tf}_doten_{tsl_tag(ts)}_fx19",
            tsmom_doten, {"lookback": lb}, tf, FX19, ts,
        ))
# XAUUSD 単独: tsmom D1 lb60, 2出口 × 3 tsl
for ts in TSLS:
    CONFIGS.append((
        f"xau_tsmom_lb60_D1_doten_{tsl_tag(ts)}",
        tsmom_doten, {"lookback": 60}, "D1", ["XAUUSD"], ts,
    ))
for ts in TSLS:
    CONFIGS.append((
        f"xau_tsmom_lb60_D1_tslonly_{tsl_tag(ts)}",
        tsmom_entry_only, {"lookback": 60}, "D1", ["XAUUSD"], ts,
    ))


def main() -> None:
    rows = []
    for label, gen, params, tf, instrs, ts in CONFIGS:
        pool = build_pool_tsl(gen, params, tf, ts, instrs)
        st = tl.pool_stats(pool)
        row = {"label": label, "tf": tf, "side": "both",
               "params": ",".join(f"{k}={v}" for k, v in params.items()) + f",tsl_stop={ts}",
               **st}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    df = pd.DataFrame(rows)
    out = ROOT + "/research/outputs/trend2_asym_exit.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
