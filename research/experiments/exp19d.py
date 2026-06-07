"""イテレーション19d: 資本マッチ + 最適ブレンド配分。

(A) 資本マッチ: チャンピオンは同時建玉~6。xsec も片側3脚(計6)に絞って公平に利益比較。
(B) 配分スキャン: champion 重み w を振り、混合 Sharpe を最大化する点を探す(相関0.13活用)。
実行: uv run python exp19d.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from fxlab import config, universe as uni
from fxlab.trades import trade_table
from exp19c import xs_meanrev_trades, champion_trades, monthly, yearly_pnl, stats

pd.set_option("display.width", 220)


def xs_capmatched(close, lookback, hold, max_legs, score_z=0.0, vol_win=50):
    names = list(close.columns)
    mom = close.pct_change(lookback); vol = close.pct_change().rolling(vol_win).std()
    mp = close.mean()
    hs = {p: config.spread_pips(p)*config.pip_size(p)/2.0/mp[p] for p in names}
    recs = []
    for t in range(max(lookback, vol_win)+1, len(close)-hold, hold):
        score = mom.iloc[t]/vol.iloc[t]
        if score.isna().any():
            continue
        score = score - score.mean(); s = score.sort_values()
        longs = s[s < -score_z].index[:max_legs]; shorts = s[s > score_z].index[-max_legs:]
        fwd = close.iloc[t+hold]/close.iloc[t]-1.0; ts = close.index[t+hold]
        for p in longs:
            recs.append((ts, (fwd[p]-2*hs[p])*10_000.0))
        for p in shorts:
            recs.append((ts, (-fwd[p]-2*hs[p])*10_000.0))
    return pd.DataFrame(recs, columns=["exit", "pnl"])


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    close = pd.DataFrame({n: uni.instrument_close(n, "H4") for n in instruments}).dropna()
    params = dict(window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                  vol_win=100, vol_pct=0.70, slow_win=250, slow_z=1.75)

    print("=== (A) 資本マッチ(片側脚数を絞る)xsec-MR lb=9 hold=24 ===")
    for ml in (4, 3, 2):
        xs = xs_capmatched(close, 9, 24, ml)
        yp = yearly_pnl(xs)
        pf = xs["pnl"][xs["pnl"]>0].sum()/-xs["pnl"][xs["pnl"]<0].sum()
        print(f"  片側{ml}脚(同時{ml*2}): total={xs['pnl'].sum():8.0f}  PF={pf:.2f}  "
              f"プラス年率={(yp>0).mean():.0%}  trades/yr={int(len(xs)/10.6)}")

    print("\n=== (B) 配分スキャン(champion重み w, 等リスク基準で混合)===")
    ch = champion_trades("H4", instruments, params)
    xs = xs_meanrev_trades(close, 9, 24, 0.0)
    cm, xm = monthly(ch), monthly(xs)
    idx = cm.index.intersection(xm.index); cm = cm.reindex(idx).fillna(0); xm = xm.reindex(idx).fillna(0)
    xm_s = xm * (cm.std()/xm.std())  # 等リスク化
    print(f"  月次相関={cm.corr(xm):.3f}")
    print(f"  {'w_champ':>8} {'年率PnL':>10} {'Sharpe':>8} {'最大DD':>10}")
    best = (None, -1)
    for w in [1.0, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]:
        series = w*cm + (1-w)*xm_s
        ann, sh, dd = stats(series)
        flag = ""
        if sh > best[1]:
            best = (w, sh);
        print(f"  {w:>8.2f} {ann:>10.0f} {sh:>8.3f} {dd:>10.0f}")
    print(f"  → 最大Sharpe は w={best[0]} (Sharpe {best[1]:.3f})  [champion単独=w1.0]")


if __name__ == "__main__":
    main()
