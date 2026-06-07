"""イテレーション19c: クロスセクション平均回帰 vs チャンピオン — 検証&統合価値。

(1) IS/OOS: 前半(2016-2020)で方向・利益が立つか、後半(2021-2026)でも持続するか
(2) 相関  : チャンピオン月次PnL と xsec-MR 月次PnL の相関(低いほど別エッジの証拠)
(3) 統合  : 等リスクで 50/50 ブレンドすると Sharpe / 最大DD が改善するか

各脚 $10k。実行: uv run python exp19c.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import config
from fxlab import universe as uni
from fxlab.trades import trade_table

pd.set_option("display.width", 220)
NOTIONAL = 10_000.0
INSTR = None  # set in main


# ---------- xsec mean-reversion: トレード単位の (timestamp, pnl) ----------
def xs_meanrev_trades(close, lookback, hold, score_z=0.0, vol_win=50):
    names = list(close.columns)
    mom = close.pct_change(lookback)
    vol = close.pct_change().rolling(vol_win).std()
    mean_price = close.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mean_price[p] for p in names}
    recs = []
    for t in range(max(lookback, vol_win) + 1, len(close) - hold, hold):
        score = mom.iloc[t] / vol.iloc[t]
        if score.isna().any():
            continue
        score = score - score.mean()
        s = score.sort_values()
        longs = s[s < -score_z].index[:4]
        shorts = s[s > score_z].index[-4:]
        fwd = close.iloc[t + hold] / close.iloc[t] - 1.0
        ts = close.index[t + hold]
        for p in longs:
            recs.append((ts, (fwd[p] - 2 * hs[p]) * NOTIONAL))
        for p in shorts:
            recs.append((ts, (-fwd[p] - 2 * hs[p]) * NOTIONAL))
    df = pd.DataFrame(recs, columns=["exit", "pnl"])
    return df


# ---------- champion: トレード単位の (timestamp, pnl) ----------
def champion_trades(tf, instruments, params):
    from strategies.confluence_meanrev import generate_signals
    from fxlab.backtest import run
    frames = []
    for name in instruments:
        data = uni.instrument_data(name, tf)
        pf = run(name, tf, generate_signals, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if not tt.empty:
            frames.append(tt[["exit", "pnl"]])
    return pd.concat(frames, ignore_index=True)


def monthly(df):
    s = df.copy()
    s["m"] = pd.PeriodIndex(pd.DatetimeIndex(s["exit"]), freq="M")
    return s.groupby("m")["pnl"].sum()


def yearly_pnl(df):
    s = df.copy()
    s["y"] = pd.DatetimeIndex(s["exit"]).year
    return s.groupby("y")["pnl"].sum()


def stats(m):  # m: 月次PnL series
    ann = m.mean() * 12
    vol = m.std() * np.sqrt(12)
    sharpe = ann / vol if vol > 0 else float("nan")
    eq = m.cumsum()
    dd = (eq - eq.cummax()).min()
    return ann, sharpe, dd


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    tf = "H4"
    close = pd.DataFrame({n: uni.instrument_close(n, tf) for n in instruments}).dropna()
    params = dict(window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                  vol_win=100, vol_pct=0.70, slow_win=250, slow_z=1.75)

    print(">> トレード生成中...")
    xs = xs_meanrev_trades(close, lookback=9, hold=24, score_z=0.0)
    ch = champion_trades(tf, instruments, params)

    # --- (1) IS/OOS ---
    print("\n=== (1) IS / OOS(xsec-MR, lb=9 hold=24)===")
    for label, lo, hi in [("IS  2016-2020", 2016, 2020), ("OOS 2021-2026", 2021, 2026)]:
        sub = xs[(pd.DatetimeIndex(xs["exit"]).year >= lo) & (pd.DatetimeIndex(xs["exit"]).year <= hi)]
        yp = yearly_pnl(sub)
        pf = sub["pnl"][sub["pnl"] > 0].sum() / -sub["pnl"][sub["pnl"] < 0].sum()
        print(f"  {label}: total={sub['pnl'].sum():8.0f}  PF={pf:.2f}  "
              f"プラス年率={(yp>0).mean():.0%}  trades={len(sub)}")

    # --- 年次対比 ---
    print("\n=== 年次PnL 対比 ===")
    cy, xy = yearly_pnl(ch), yearly_pnl(xs)
    comp = pd.DataFrame({"champion": cy, "xsec_MR": xy}).round(0)
    comp["both_pos"] = (comp["champion"] > 0) & (comp["xsec_MR"] > 0)
    print(comp.to_string())
    print(f"  champion total={cy.sum():.0f}  xsec_MR total={xy.sum():.0f}")

    # --- (2) 相関 ---
    cm, xm = monthly(ch), monthly(xs)
    idx = cm.index.intersection(xm.index)
    cm, xm = cm.reindex(idx).fillna(0), xm.reindex(idx).fillna(0)
    corr_m = cm.corr(xm)
    corr_y = comp["champion"].corr(comp["xsec_MR"])
    print(f"\n=== (2) 相関 ===\n  月次相関={corr_m:.3f}   年次相関={corr_y:.3f}")

    # --- (3) 等リスク 50/50 ブレンド ---
    xm_scaled = xm * (cm.std() / xm.std())          # xsec を champion と同じ月次ボラに
    blend = 0.5 * cm + 0.5 * xm_scaled              # 同じリスク予算で混合
    print("\n=== (3) 等リスク統合(月次)===")
    for label, series in [("champion 単独", cm), ("xsec_MR(等リスク化)", xm_scaled), ("50/50 ブレンド", blend)]:
        ann, sh, dd = stats(series)
        print(f"  {label:22s}: 年率PnL={ann:8.0f}  Sharpe(月次)={sh:.2f}  最大DD={dd:8.0f}")
    print("  ※ ブレンドの Sharpe がチャンピオン単独を上回れば、別アプローチの分散価値あり")


if __name__ == "__main__":
    main()
