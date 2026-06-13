"""検証2: tsmom USDJPY H1 の plateau 高原 と 単一通貨依存の精査。

- lookback∈{6,12,18,24,36,48,72} × band∈{0,0.001,0.002} の GROSS/NET Sharpe 高原 (USDJPY H1)
- TF頑健性: M30/H4 でも USDJPY が正か (lb24 中心 + グリッド)
- GBPJPY 同グリッド & USDJPY+GBPJPY 2ペア合成が単独より頑健か
- 単一通貨集中の深刻度: USDJPY 抜いたら何が残るか

GROSS = config.SPREADS_PIPS を全て 0 にして再計算。NET = 通常スプレッド。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import metrics, run
from fxlab import universe as uni
from strategies.tsmom import generate_signals

LOOKBACKS = [6, 12, 18, 24, 36, 48, 72]
BANDS = [0.0, 0.001, 0.002]


def _f(x):
    """metrics() の Series/スカラ混在を float に強制。"""
    if isinstance(x, pd.Series):
        return float(x.iloc[0])
    return float(x)


def sharpe_of(pair, tf, lb, band, data, gross=False, side="both"):
    """1セルの Sharpe (value サイジング, 1バー遅延なし=素のシグナル)。

    gross=True のときだけ一時的にスプレッドを 0 にする。
    """
    if gross:
        saved = dict(C.SPREADS_PIPS)
        for k in list(C.SPREADS_PIPS):
            C.SPREADS_PIPS[k] = 0.0
        C.SPREADS_PIPS[pair] = 0.0
    try:
        pf = run(pair, tf, generate_signals, {"lookback": lb, "band": band},
                 data=data, size_mode="value", side=side)
        m = metrics(pf)
        sh = _f(m["sharpe"])
        nt = int(_f(m["num_trades"]))
        tr = _f(m["total_return"])
    finally:
        if gross:
            C.SPREADS_PIPS.clear()
            C.SPREADS_PIPS.update(saved)
    return sh, nt, tr


def grid(pair, tf, data, gross):
    rows = {}
    for lb in LOOKBACKS:
        for band in BANDS:
            sh, nt, tr = sharpe_of(pair, tf, lb, band, data, gross=gross)
            rows[(lb, band)] = sh
    s = pd.Series(rows)
    s.index = pd.MultiIndex.from_tuples(s.index, names=["lookback", "band"])
    return s.unstack("band")


def print_grid(title, df):
    print(f"\n=== {title} ===")
    with pd.option_context("display.float_format", lambda x: f"{x:+.3f}"):
        print(df.to_string())
    arr = df.values
    print(f"  range: min={arr.min():+.3f} max={arr.max():+.3f} "
          f"frac_positive={(arr > 0).mean():.2f} median={np.median(arr):+.3f}")


# ---- 日次リターン系列(vol正規化 two-book robust 合成用) ----
def daily_returns(pair, tf, lb, band, data):
    """value サイジングの日次リターン系列を返す(vol正規化前)。"""
    pf = run(pair, tf, generate_signals, {"lookback": lb, "band": band},
             data=data, size_mode="value", side="both")
    eq = pf.value()
    if isinstance(eq, pd.DataFrame):
        eq = eq.iloc[:, 0]
    ret = eq.pct_change().fillna(0.0)
    # 日次に集約
    daily = (1.0 + ret).resample("1D").prod() - 1.0
    return daily[daily != 0.0]  # 取引のない日を落とす(連結のため)


def vol_normalize(daily, target_ann_vol=0.10):
    """年率ボラを target に正規化したスケール係数を返す。"""
    realized = daily.std() * np.sqrt(252)
    if realized <= 0:
        return daily * 0.0, 0.0
    scale = target_ann_vol / realized
    return daily * scale, scale


def ann_sharpe(daily):
    if daily.std() <= 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))


def maxdd(daily):
    eq = (1.0 + daily).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def two_book(pairs_data, tf, lb, band):
    """vol正規化した各ペアの日次リターンを等加重合成 -> Sharpe/maxDD。"""
    sleeves = []
    for pair, data in pairs_data:
        d = daily_returns(pair, tf, lb, band, data)
        dn, _ = vol_normalize(d, 0.10)
        sleeves.append(dn.rename(pair))
    book = pd.concat(sleeves, axis=1).fillna(0.0)
    combined = book.mean(axis=1)  # 等加重
    return ann_sharpe(combined), maxdd(combined), combined


def main():
    usdjpy_h1 = uni.instrument_data("USDJPY", "H1")
    gbpjpy_h1 = uni.instrument_data("GBPJPY", "H1")

    # ---------- 1. USDJPY H1 plateau ----------
    print("############ 1. USDJPY H1 PLATEAU ############")
    g_gross = grid("USDJPY", "H1", usdjpy_h1, gross=True)
    g_net = grid("USDJPY", "H1", usdjpy_h1, gross=False)
    print_grid("USDJPY H1 GROSS Sharpe", g_gross)
    print_grid("USDJPY H1 NET Sharpe", g_net)

    # ---------- 2. TF頑健性 (USDJPY M30 / H4) ----------
    print("\n############ 2. USDJPY TF ROBUSTNESS ############")
    for tf in ["M30", "H4"]:
        d = uni.instrument_data("USDJPY", tf)
        gg = grid("USDJPY", tf, d, gross=True)
        gn = grid("USDJPY", tf, d, gross=False)
        print_grid(f"USDJPY {tf} GROSS Sharpe", gg)
        print_grid(f"USDJPY {tf} NET Sharpe", gn)

    # ---------- 3. GBPJPY 同グリッド ----------
    print("\n############ 3. GBPJPY H1 PLATEAU ############")
    gb_gross = grid("GBPJPY", "H1", gbpjpy_h1, gross=True)
    gb_net = grid("GBPJPY", "H1", gbpjpy_h1, gross=False)
    print_grid("GBPJPY H1 GROSS Sharpe", gb_gross)
    print_grid("GBPJPY H1 NET Sharpe", gb_net)

    # ---------- 4. two-book robust 合成 (lb24, band0) ----------
    print("\n############ 4. TWO-BOOK ROBUST (lb24 band0, H1, vol-norm 10%) ############")
    lb, band = 24, 0.0
    # 単独
    d_u = daily_returns("USDJPY", "H1", lb, band, usdjpy_h1)
    du_n, _ = vol_normalize(d_u, 0.10)
    sh_u, dd_u = ann_sharpe(du_n), maxdd(du_n)
    d_g = daily_returns("GBPJPY", "H1", lb, band, gbpjpy_h1)
    dg_n, _ = vol_normalize(d_g, 0.10)
    sh_g, dd_g = ann_sharpe(dg_n), maxdd(dg_n)
    # 2ペア合成
    sh2, dd2, combined = two_book(
        [("USDJPY", usdjpy_h1), ("GBPJPY", gbpjpy_h1)], "H1", lb, band)
    # 相関
    corr = pd.concat([du_n.rename("U"), dg_n.rename("G")], axis=1).fillna(0.0).corr().iloc[0, 1]
    print(f"USDJPY solo  : daily-Sharpe={sh_u:+.3f}  maxDD={dd_u:+.3f}")
    print(f"GBPJPY solo  : daily-Sharpe={sh_g:+.3f}  maxDD={dd_g:+.3f}")
    print(f"2-pair equal : daily-Sharpe={sh2:+.3f}  maxDD={dd2:+.3f}")
    print(f"daily-return corr(USDJPY,GBPJPY) = {corr:+.3f}")
    print(f"diversification: Sharpe {sh_u:+.3f}(solo U) -> {sh2:+.3f}(2pair); "
          f"maxDD {dd_u:+.3f} -> {dd2:+.3f}")

    # ---------- 5. 単一通貨集中の深刻度 ----------
    print("\n############ 5. CONCENTRATION: USDJPY抜いたら? ############")
    # 全JPYクロス + 主要メジャーで lb24 band0 H1 NET Sharpe を測る
    universe_test = ["USDJPY", "GBPJPY", "EURJPY", "AUDJPY",
                     "EURUSD", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
    print(f"{'instr':8s} {'NET Sharpe':>11s} {'GROSS Sh':>9s} {'trades':>7s} {'tot_ret%':>9s}")
    net_shs = {}
    for instr in universe_test:
        try:
            d = uni.instrument_data(instr, "H1")
            sh_n, nt, tr = sharpe_of(instr, "H1", lb, band, d, gross=False)
            sh_gr, _, _ = sharpe_of(instr, "H1", lb, band, d, gross=True)
            net_shs[instr] = sh_n
            print(f"{instr:8s} {sh_n:>+11.3f} {sh_gr:>+9.3f} {nt:>7d} {tr*100:>+9.2f}")
        except Exception as e:  # noqa: BLE001
            print(f"{instr:8s}  ERROR {e}")
    # USDJPY抜いた等加重ブックの daily-Sharpe (vol-norm)
    print("\n-- USDJPY除外の等加重ブック (vol-norm 10%) --")
    others = [i for i in universe_test if i != "USDJPY" and net_shs.get(i, -99) > -90]
    sleeves = []
    for instr in others:
        d = uni.instrument_data(instr, "H1")
        dd = daily_returns(instr, "H1", lb, band, d)
        dn, _ = vol_normalize(dd, 0.10)
        sleeves.append(dn.rename(instr))
    book_no_u = pd.concat(sleeves, axis=1).fillna(0.0).mean(axis=1)
    print(f"WITHOUT USDJPY ({len(others)} instr equal-wt): "
          f"daily-Sharpe={ann_sharpe(book_no_u):+.3f}  maxDD={maxdd(book_no_u):+.3f}")
    # 全部入り
    sleeves_all = []
    for instr in universe_test:
        if net_shs.get(instr, -99) < -90:
            continue
        d = uni.instrument_data(instr, "H1")
        dd = daily_returns(instr, "H1", lb, band, d)
        dn, _ = vol_normalize(dd, 0.10)
        sleeves_all.append(dn.rename(instr))
    book_all = pd.concat(sleeves_all, axis=1).fillna(0.0).mean(axis=1)
    print(f"WITH USDJPY    ({len(sleeves_all)} instr equal-wt): "
          f"daily-Sharpe={ann_sharpe(book_all):+.3f}  maxDD={maxdd(book_all):+.3f}")


if __name__ == "__main__":
    main()
