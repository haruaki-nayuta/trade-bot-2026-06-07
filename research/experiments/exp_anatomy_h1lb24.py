"""検証3: H1 lb24 tsmom グロスエッジの hour/side/year 解剖。

問い:
  (a) 損益は特定時間帯(UTC hour)に偏るか全時間か。
  (b) ロング/ショート対称か片側か。
  (c) 特定年集中か全年か。
さらにロールオーバーBIDアーティファクト署名(UTC20-23集中・USD両方向同符号・ロング偏重)を点検。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "H1"
LB = 24
BAND = 0.0

# GROSS: コストを完全にゼロ化
C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
C.COMMISSION_FRACTION = 0.0


def per_trade_table(pair):
    """1ペアのトレードを DataFrame で返す(entry時刻・side・pnl・return)。"""
    data = load(pair, TF)
    pf = run(pair, TF, generate_signals, {"lookback": LB, "band": BAND},
             data=data, size_mode="value", side="both")
    tr = pf.trades.records_readable
    if len(tr) == 0:
        return None
    df = pd.DataFrame({
        "pair": pair,
        "entry_idx": tr["Entry Timestamp"].values,
        "side": tr["Direction"].values,          # 'Long'/'Short'
        "pnl": tr["PnL"].values.astype(float),
        "ret": tr["Return"].values.astype(float),
    })
    df["entry_idx"] = pd.to_datetime(df["entry_idx"])
    df["hour"] = df["entry_idx"].dt.hour
    df["year"] = df["entry_idx"].dt.year
    return df


def gross_sharpe(pair, drop_rollover=False):
    """ペア単位のグロス Sharpe。drop_rolloverでUTC20-23のエントリーバーを除外して再シミュレート。"""
    data = load(pair, TF)
    if drop_rollover:
        # エントリー抑制: 20-23時にシグナルが立っても入らないようマスク
        def gs(d, lookback=LB, band=BAND):
            le, lx, se, sx = generate_signals(d, lookback, band)
            mask = ~d.index.hour.isin([20, 21, 22, 23])
            mask = pd.Series(mask, index=d.index)
            le = le & mask
            se = se & mask
            return le, lx, se, sx
        pf = run(pair, TF, gs, {}, data=data, size_mode="value", side="both")
    else:
        pf = run(pair, TF, generate_signals, {"lookback": LB, "band": BAND},
                 data=data, size_mode="value", side="both")
    m = metrics(pf)
    return float(m["sharpe"]), float(m["total_return"]), int(m["num_trades"])


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # --- per-pair gross Sharpe (raw) ---
    print("=== per-pair gross Sharpe (H1 lb24, raw) ===")
    sharpes_raw, rets_raw = {}, {}
    for p in PAIRS:
        s, r, n = gross_sharpe(p, drop_rollover=False)
        sharpes_raw[p] = s
        rets_raw[p] = r
        print(f"  {p}: sharpe={s:+.3f}  total_ret={r:+.4f}  trades={n}")
    avg_raw = np.mean(list(sharpes_raw.values()))
    pos_raw = sum(1 for v in sharpes_raw.values() if v > 0)
    print(f"  --> mean gross Sharpe = {avg_raw:+.4f}  | positive pairs = {pos_raw}/7")

    # --- per-pair gross Sharpe (UTC20-23 entries dropped) ---
    print("\n=== per-pair gross Sharpe (H1 lb24, UTC20-23 entries dropped) ===")
    sharpes_clean = {}
    for p in PAIRS:
        s, r, n = gross_sharpe(p, drop_rollover=True)
        sharpes_clean[p] = s
        print(f"  {p}: sharpe={s:+.3f}  total_ret={r:+.4f}  trades={n}")
    avg_clean = np.mean(list(sharpes_clean.values()))
    pos_clean = sum(1 for v in sharpes_clean.values() if v > 0)
    print(f"  --> mean gross Sharpe = {avg_clean:+.4f}  | positive pairs = {pos_clean}/7")

    # --- collect all trades ---
    frames = [per_trade_table(p) for p in PAIRS]
    allt = pd.concat([f for f in frames if f is not None], ignore_index=True)
    print(f"\n=== total trades across 7 pairs: {len(allt)} ===")

    # (a) HOUR breakdown ---------------------------------------------------
    print("\n=== (a) entry-hour breakdown (UTC) — sum of PnL & mean return & count ===")
    hb = allt.groupby("hour").agg(
        pnl_sum=("pnl", "sum"),
        ret_mean=("ret", "mean"),
        ret_sum=("ret", "sum"),
        n=("pnl", "size"),
    )
    total_pnl = allt["pnl"].sum()
    hb["pnl_share_%"] = hb["pnl_sum"] / total_pnl * 100
    print(hb.round(4).to_string())
    # rollover window aggregate
    roll = allt[allt["hour"].isin([20, 21, 22, 23])]
    nonroll = allt[~allt["hour"].isin([20, 21, 22, 23])]
    print(f"\n  UTC20-23 entries: n={len(roll)} ({len(roll)/len(allt)*100:.1f}%)  "
          f"pnl_sum={roll['pnl'].sum():+.2f} ({roll['pnl'].sum()/total_pnl*100:+.1f}% of total)  "
          f"ret_mean={roll['ret'].mean():+.5f}")
    print(f"  other hours     : n={len(nonroll)} ({len(nonroll)/len(allt)*100:.1f}%)  "
          f"pnl_sum={nonroll['pnl'].sum():+.2f} ({nonroll['pnl'].sum()/total_pnl*100:+.1f}% of total)  "
          f"ret_mean={nonroll['ret'].mean():+.5f}")
    # session grouping
    def session(h):
        if 0 <= h < 7:   return "Asia(0-6)"
        if 7 <= h < 12:  return "London(7-11)"
        if 12 <= h < 16: return "LDN/NY ovlp(12-15)"
        if 16 <= h < 20: return "NY(16-19)"
        return "Rollover(20-23)"
    allt["session"] = allt["hour"].apply(session)
    sb = allt.groupby("session").agg(pnl_sum=("pnl", "sum"), ret_mean=("ret", "mean"), n=("pnl", "size"))
    sb["pnl_share_%"] = sb["pnl_sum"] / total_pnl * 100
    print("\n  -- session grouping --")
    print(sb.round(4).to_string())

    # (b) SIDE breakdown ---------------------------------------------------
    print("\n=== (b) side breakdown (Long vs Short) ===")
    sd = allt.groupby("side").agg(
        pnl_sum=("pnl", "sum"),
        ret_mean=("ret", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        n=("pnl", "size"),
    )
    sd["pnl_share_%"] = sd["pnl_sum"] / total_pnl * 100
    print(sd.round(4).to_string())

    # USD両方向同符号チェック: USDが quote のペア(EURUSD等)とUSDが base のペア(USDJPY等)
    usd_base = ["USDJPY", "USDCHF", "USDCAD"]   # USD買い=Long
    usd_quote = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]  # USD買い=Short
    print("\n  -- USD directional sign check (rollover artifact署名) --")
    print("     USD-quote pairs (EURUSD etc): Long=USD売り / Short=USD買い")
    print("     USD-base  pairs (USDJPY etc): Long=USD買い / Short=USD売り")
    for grp, name in [(usd_quote, "USD-quote"), (usd_base, "USD-base")]:
        sub = allt[allt["pair"].isin(grp)]
        ls = sub[sub["side"] == "Long"]["pnl"].sum()
        ss = sub[sub["side"] == "Short"]["pnl"].sum()
        print(f"     {name}: Long pnl={ls:+.2f}  Short pnl={ss:+.2f}")

    # (c) YEAR breakdown ---------------------------------------------------
    print("\n=== (c) year breakdown ===")
    yb = allt.groupby("year").agg(
        pnl_sum=("pnl", "sum"),
        ret_mean=("ret", "mean"),
        n=("pnl", "size"),
    )
    yb["pnl_share_%"] = yb["pnl_sum"] / total_pnl * 100
    print(yb.round(4).to_string())
    pos_years = (yb["pnl_sum"] > 0).sum()
    print(f"  positive years: {pos_years}/{len(yb)}")
    if 2022 in yb.index:
        print(f"  2022 share of total pnl = {yb.loc[2022, 'pnl_share_%']:+.1f}%")

    # year x side cross
    print("\n  -- year x side pnl_sum --")
    yx = allt.pivot_table(index="year", columns="side", values="pnl", aggfunc="sum").fillna(0)
    print(yx.round(2).to_string())


if __name__ == "__main__":
    main()
