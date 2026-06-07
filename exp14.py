"""イテレーション14: キャリー(金利差)エッジ源の投入 — フックが示す唯一の残路。

(1) キャリー収益加味: 各トレードに保有ぶんの金利差受取/支払を加える(=スワップ込みの現実)。
(2) キャリー・フィルタ: 大きな逆キャリー(金利差が強く逆方向)を払う向きの逆張りを除外
    (金利差で構造的にトレンドする局面=平均回帰が外れやすい、を回避)。

19対象ポートで、ベース/(1)/(2) を比較し PF・毎年プラス・取引 を見る。実行: uv run python exp14.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import universe as uni
from fxlab import carry as cy
from fxlab.backtest import run
from fxlab.trades import trade_table
from strategies.confluence_meanrev import generate_signals as g

pd.set_option("display.width", 200)
TF = "H4"
P = {"window": 50, "entry_z": 2.0, "exit_z": 0.5, "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
     "vol_win": 100, "vol_pct": 0.70, "slow_win": 250, "slow_z": 1.75}


def collect(instruments, carry_filter=0.0):
    """全対象トレードを集め、キャリー込み return を付与。carry_filter>0 で逆キャリー回避。"""
    frames = []
    for nm in instruments:
        data = uni.instrument_data(nm, TF)
        pf = run(nm, TF, g, P, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        tt = cy.apply_carry(tt, nm)
        tt["instrument"] = nm
        frames.append(tt)
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.DatetimeIndex(df["exit"]).year
    if carry_filter > 0:
        # ロングで強い逆キャリー(carry_ann<-th)/ショートで強い順キャリーを払う逆張りを除外
        years = pd.DatetimeIndex(df["exit"]).year
        car = pd.Series([cy.carry_annual(r.instrument, int(y))
                         for (_, r), y in zip(df.iterrows(), years)], index=df.index)
        dirs = df["dir"].map(lambda d: 1.0 if str(d).lower().startswith("l") else -1.0)
        signed_carry = dirs = dirs if False else dirs  # noqa
        keep = (dirs * car) >= -carry_filter
        df = df[keep]
    return df


def yearly_pf(df, ret_col):
    rows = {}
    for y, gdf in df.groupby("year"):
        r = gdf[ret_col]
        gp = r[r > 0].sum(); gl = -r[r < 0].sum()
        rows[int(y)] = {"trades": len(gdf), "pf": gp/gl if gl > 0 else np.inf, "ret": r.sum()}
    t = pd.DataFrame(rows).T
    pf = t["pf"].replace(np.inf, np.nan)
    return (t["ret"] > 0).mean(), pf.median(), pf.min(), int(t["trades"].mean()), t


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe() if x != "AUDJPY"]

    base = collect(instruments)
    print(f"対象 {len(instruments)} / {TF}\n")
    print(f"{'構成':28} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7}")
    # ベース(価格のみ, return_pctベース)
    pos, med, mn, tr, _ = yearly_pf(base.assign(r=base['return_pct']/100.0), 'r')
    print(f"{'ベース(価格のみ)':28} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}")
    # (1) キャリー収益込み
    pos, med, mn, tr, _ = yearly_pf(base, 'return_carry')
    print(f"{'(1)キャリー収益込み':28} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}")
    # (2) キャリー・フィルタ(複数閾値)
    for th in [3.0, 2.0, 1.0]:
        f = collect(instruments, carry_filter=th)
        pos, med, mn, tr, _ = yearly_pf(f.assign(r=f['return_pct']/100.0), 'r')
        print(f"{'(2)逆キャリー回避 th='+str(th):28} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}")
    # (1)+(2)
    f = collect(instruments, carry_filter=2.0)
    pos, med, mn, tr, _ = yearly_pf(f, 'return_carry')
    print(f"{'(1)+(2) th=2.0':28} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}")


if __name__ == "__main__":
    main()
