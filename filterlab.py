"""エントリーフィルタ研究の共有基盤(診断データセット + 変種評価ハーネス + CLI)。

Workflow のサブエージェントはこの CLI/関数を使って決定論的に検証する。

  # 1) エントリー時点特徴量つき全トレード(先読みなし)を CSV 化
  uv run python filterlab.py dataset

  # 2) フィルタ変種をユニバース評価(JSON 出力)。IS/OOS と月次Sharpe/maxDD込み
  uv run python filterlab.py eval --params er_max=0.4,slope_max=0.6

  # 3) しきい値グリッドを一括評価(高原性チェック)
  uv run python filterlab.py grid --param er_max --values 0.3,0.4,0.5,0.6,1.01
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from fxlab import config, universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table
import strategies.confluence_meanrev_filtered as strat

TF = "H4"
SIZE = 10_000.0
CHAMP = dict(strat.PARAMS)  # フィルタ既定 OFF = チャンピオン
IS_END = "2021-12-31"       # IS/OOS 分割(OOS=2022-2026)

uni.register_cross_spreads(3.0)
INSTRUMENTS = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]


# ---------------- 診断データセット(エントリー時点特徴量) ----------------
def _entry_features(data, p):
    """各バーの因果特徴量(エントリー時点で読める値のみ)。"""
    close = data["close"]
    f = pd.DataFrame(index=close.index)
    f["z"] = strat._zscore(close, p["window"])
    f["slow_z"] = strat._zscore(close, p["slow_win"])
    import vectorbt as vbt
    f["rsi"] = vbt.RSI.run(close, p["rsi_p"]).rsi
    f["er10"] = strat._efficiency_ratio(close, 10)
    f["er20"] = strat._efficiency_ratio(close, 20)
    f["er40"] = strat._efficiency_ratio(close, 40)
    f["adx14"] = strat._adx(data, 14)
    f["atr_pct"] = strat._atr_pct(data, 14)
    f["slope10"] = strat._slope_pct_per_bar(close, 10)
    f["slope20"] = strat._slope_pct_per_bar(close, 20)
    vol = close.pct_change().rolling(20).std()
    f["vol_rank"] = vol.rolling(p["vol_win"]).rank(pct=True)
    # 直前のレンジ位置
    f["dist_high20"] = (close / data["high"].rolling(20).max() - 1) * 100
    f["dist_low20"] = (close / data["low"].rolling(20).min() - 1) * 100
    f["ret20"] = (close / close.shift(20) - 1) * 100
    return f


def build_dataset(save=True) -> pd.DataFrame:
    """全対象でチャンピオンを回し、各トレードにエントリー時点特徴量と結末を付与。"""
    frames = []
    for name in INSTRUMENTS:
        data = uni.instrument_data(name, TF)
        pf = run(name, TF, strat.generate_signals, CHAMP, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        feats = _entry_features(data, CHAMP)
        idx = data.index.get_indexer(tt["entry"])
        fe = feats.iloc[idx].reset_index(drop=True)
        tt = tt.reset_index(drop=True)
        rec = pd.concat([tt[["dir", "entry", "exit", "pnl", "return_pct", "bars_held"]], fe], axis=1)
        rec["instrument"] = name
        rec["year"] = pd.DatetimeIndex(rec["exit"]).year
        # 結末ラベル
        rec["is_win"] = rec["pnl"] > 0
        frames.append(rec)
    df = pd.concat(frames, ignore_index=True)
    # 全体での分位ラベル(下位10%=カタストロフ)
    cut10 = df["return_pct"].quantile(0.10)
    df["is_cata"] = df["return_pct"] <= cut10           # 下位10%
    df["is_bleed"] = df["bars_held"] > 30               # 失血ゾーン(診断より)
    df["is_bigloss"] = df["return_pct"] <= -1.0         # 実害大の損
    if save:
        p = config.RESULTS_DIR / "entry_features.csv"
        df.to_csv(p, index=False)
        print(f"saved {p}  ({len(df)} trades, cata_cut={cut10:.3f}%)")
    return df


# ---------------- 変種評価ハーネス ----------------
def _all_trades(params, data_slice=None):
    frames = []
    for name in INSTRUMENTS:
        data = uni.instrument_data(name, TF)
        if data_slice is not None:
            lo, hi = data_slice
            data = data.loc[lo:hi]
        pf = run(name, TF, strat.generate_signals, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        tt["instrument"] = name
        frames.append(tt)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.DatetimeIndex(df["exit"]).year
    return df


def _metrics(df) -> dict:
    if df.empty:
        return {"trades": 0}
    pnl = df["pnl"]
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    yr = df.groupby("year")["pnl"].sum()
    ts = pd.DatetimeIndex(df["exit"])
    monthly = df.assign(m=ts.to_period("M")).groupby("m")["pnl"].sum().asfreq("M", fill_value=0.0)
    sharpe = float(monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() else float("nan")
    cap = SIZE * len(INSTRUMENTS)
    eq = cap + monthly.cumsum(); peak = eq.cummax()
    maxdd = float(((eq - peak) / peak).min())
    return {
        "trades": int(len(df)),
        "total_pnl": round(float(pnl.sum())),
        "PF": round(gp / gl, 3) if gl else float("inf"),
        "win_pct": round(float((pnl > 0).mean()) * 100, 1),
        "sharpe_m": round(sharpe, 3),
        "maxdd_pct": round(maxdd * 100, 3),
        "worst_pct": round(float(df["return_pct"].min()), 2),
        "yrs_pos": int((yr > 0).sum()),
        "yrs_total": int(len(yr)),
        "min_year_pnl": round(float(yr.min())),
    }


def eval_variant(params: dict) -> dict:
    """フル期間 + IS + OOS のユニバース評価をまとめて返す。"""
    full = _metrics(_all_trades(params))
    is_m = _metrics(_all_trades(params, data_slice=(None, IS_END)))
    oos = _metrics(_all_trades(params, data_slice=("2022-01-01", None)))
    return {"full": full, "is": is_m, "oos": oos}


def _parse(s):
    out = {}
    if not s:
        return out
    for kv in s.split(","):
        k, v = kv.split("=")
        k = k.strip(); v = v.strip()
        if v.lower() in ("none", "null"):
            out[k] = None
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("dataset")
    pe = sub.add_parser("eval"); pe.add_argument("--params", default="")
    pg = sub.add_parser("grid")
    pg.add_argument("--param", required=True)
    pg.add_argument("--values", required=True)
    pg.add_argument("--base", default="")
    args = ap.parse_args()

    if args.cmd == "dataset":
        build_dataset()
    elif args.cmd == "eval":
        params = dict(CHAMP); params.update(_parse(args.params))
        print(json.dumps({"params": _parse(args.params), **eval_variant(params)}, default=str))
    elif args.cmd == "grid":
        base = dict(CHAMP); base.update(_parse(args.base))
        rows = []
        for raw in args.values.split(","):
            raw = raw.strip()
            try:
                val = int(raw)
            except ValueError:
                val = None if raw.lower() == "none" else float(raw)
            params = dict(base); params[args.param] = val
            m = eval_variant(params)
            rows.append({args.param: val, **{f"full_{k}": v for k, v in m["full"].items()},
                         "oos_pnl": m["oos"]["total_pnl"], "oos_sharpe": m["oos"]["sharpe_m"],
                         "oos_yrs": f'{m["oos"]["yrs_pos"]}/{m["oos"]["yrs_total"]}'})
        print(json.dumps(rows, default=str))


if __name__ == "__main__":
    main()
