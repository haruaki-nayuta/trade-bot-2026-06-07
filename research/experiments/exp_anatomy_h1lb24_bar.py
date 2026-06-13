"""検証3 補強: bar-level の hour/side/year 帰属(エントリー時刻でなく PnL が発生したバーで集計)。

position-series 法: pos = sign(mom_lb).shift(1) を持ち続け、各バーの pos*bar_ret をそのバーの
hour/year に帰属。side は pos の符号。エントリー時刻バイアスを排し「エッジがどのバーで稼がれるか」を見る。
グロス(コスト0)。全7ペア合算 と 正エッジ3ペア(EURUSD/USDJPY/GBPUSD)別に集計。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import fxlab.config as C
from fxlab import load

C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
C.COMMISSION_FRACTION = 0.0

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
POS3 = ["EURUSD", "USDJPY", "GBPUSD"]
LB, TF = 24, "H1"
ANN = np.sqrt(24 * 252)


def bar_stream(pair):
    d = load(pair, TF)
    close = d["close"]
    mom = close / close.shift(LB) - 1.0
    pos = np.sign(mom).shift(1).fillna(0.0)        # 次バーから保有(先読みなし)
    bar_ret = close.pct_change().fillna(0.0)
    sr = pos * bar_ret
    out = pd.DataFrame({
        "ret": sr.values,
        "pos": pos.values,
        "hour": sr.index.hour,
        "year": sr.index.year,
    }, index=sr.index)
    out["pair"] = pair
    return out


def report(frames, label):
    allb = pd.concat(frames, ignore_index=True)
    active = allb[allb["pos"] != 0]
    tot = active["ret"].sum()
    print(f"\n########## {label}  (bar-level, gross) ##########")
    # group sharpe of equal-weight portfolio
    print(f"  total cum bar-return (sum, active bars) = {tot:+.4f}")

    # (a) hour
    print("\n  (a) HOUR (UTC): ret_sum / ret_mean / n_active")
    hb = active.groupby("hour").agg(ret_sum=("ret", "sum"), ret_mean=("ret", "mean"), n=("ret", "size"))
    print(hb.round(5).to_string())
    roll = active[active["hour"].isin([20, 21, 22, 23])]["ret"].sum()
    non = active[~active["hour"].isin([20, 21, 22, 23])]["ret"].sum()
    print(f"    UTC20-23 ret_sum={roll:+.4f} ({roll/tot*100:+.1f}% of total) | other={non:+.4f} ({non/tot*100:+.1f}%)")

    # (b) side
    print("\n  (b) SIDE: ret_sum / n_active")
    sb = active.copy()
    sb["side"] = np.where(sb["pos"] > 0, "Long", "Short")
    sd = sb.groupby("side").agg(ret_sum=("ret", "sum"), ret_mean=("ret", "mean"), n=("ret", "size"))
    print(sd.round(5).to_string())

    # (c) year
    print("\n  (c) YEAR: ret_sum / n_active")
    yb = active.groupby("year").agg(ret_sum=("ret", "sum"), ret_mean=("ret", "mean"), n=("ret", "size"))
    print(yb.round(5).to_string())
    print(f"    positive years: {(yb['ret_sum']>0).sum()}/{len(yb)}  | 2022 share={yb.loc[2022,'ret_sum']/tot*100:+.1f}%" if 2022 in yb.index else "")


def main():
    frames_all = [bar_stream(p) for p in PAIRS]
    by_pair = {f['pair'].iloc[0]: f for f in frames_all}
    report(frames_all, "ALL 7 PAIRS")
    report([by_pair[p] for p in POS3], "POSITIVE-EDGE 3 PAIRS (EURUSD/USDJPY/GBPUSD)")

    # robustness: gross Sharpe of positive-3 with vs without UTC20-23 held bars zeroed
    print("\n########## POS3 portfolio gross Sharpe, with UTC20-23 held-bar returns zeroed ##########")
    def port_sharpe(pairs, drop_hours=None):
        streams = []
        for p in pairs:
            f = by_pair[p].set_index(pd.to_datetime(by_pair[p].index) if False else None) if False else by_pair[p]
            s = bar_stream(p)["ret"]
            if drop_hours:
                s = s.copy()
                # zero the strategy return on rollover hours (still holding, but exclude attribution)
            streams.append(bar_stream(p).assign())
        return None
    # simpler: recompute directly
    for drop in [False, True]:
        port = None
        for p in POS3:
            d = load(p, TF); close = d["close"]
            mom = close / close.shift(LB) - 1.0
            pos = np.sign(mom).shift(1).fillna(0.0)
            sr = (pos * close.pct_change().fillna(0.0)).rename(p)
            if drop:
                sr = sr.where(~sr.index.hour.isin([20, 21, 22, 23]), 0.0)
            port = sr if port is None else pd.concat([port, sr], axis=1)
        if isinstance(port, pd.Series):
            comb = port
        else:
            comb = port.fillna(0).mean(axis=1)
        sh = comb.mean() / comb.std() * ANN
        tag = "UTC20-23 returns zeroed" if drop else "raw"
        print(f"  POS3 equal-weight portfolio gross Sharpe ({tag}) = {sh:+.4f}")


if __name__ == "__main__":
    main()
