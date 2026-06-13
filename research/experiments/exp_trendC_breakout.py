"""再構築C: バッファ付きブレイク + 利伸ばしトレーリング(D1, 7メジャー)。

仮説:
  素のbreakout_trend(net -14%)が負けるのは「ダマシ(buffer無しでチャネル僅差抜け→即反転)」と
  「勝ちを十分伸ばせていない」可能性がある。そこで
    1) 確認バッファ: 終値 > channel高値 + k×ATR (ノイズブレイクを弾く)
    2) 利伸ばし: exit本(<entry本)の反対側ドンチャンで広めにトレーリングし勝ちを伸ばす
    3) 長期SMA方向フィルタ(順張りのみ)
  を入れる。素のbreakout_trendとの差分でバッファ&利伸ばしの効果を明示する。

判定軸:
  GROSS Sharpe ~0以下 = エッジ自体が無い(実装では救えない)
  GROSS明確に正 & NET負 = エッジはあるがコスト/回転で消える
GROSS と NET を別々のプロセス相当(設定退避→復元)で測る。

先読み防止: rolling極値は .shift()(自バー除外)。ATR/SMAは確定バーまで。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load, run, metrics

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "D1"
IS_END = "2021-12-31"
OOS_START = "2022-01-01"

# 元のスプレッド退避(GROSS/NET切替で復元するため)
_ORIG_SPREADS = dict(C.SPREADS_PIPS)
_ORIG_COMM = C.COMMISSION_FRACTION


def set_gross(on: bool):
    if on:
        C.SPREADS_PIPS = {k: 0.0 for k in _ORIG_SPREADS}
        C.COMMISSION_FRACTION = 0.0
    else:
        C.SPREADS_PIPS = dict(_ORIG_SPREADS)
        C.COMMISSION_FRACTION = _ORIG_COMM


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = data["high"], data["low"], data["close"]
    pc = c.shift()
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def make_signals(entry: int = 40, exit: int = 20, k: float = 0.5, trend: int = 200,
                 atr_period: int = 14):
    """バッファ付きブレイク + 利伸ばしトレーリング のシグナル生成関数を返す。"""
    def generate_signals(data: pd.DataFrame, **_):
        high, low, close = data["high"], data["low"], data["close"]
        # 直前バーまでの極値(自バー除外=先読み回避)
        upper = high.rolling(entry).max().shift()
        lower = low.rolling(entry).min().shift()
        exit_lower = low.rolling(exit).min().shift()
        exit_upper = high.rolling(exit).max().shift()
        atr = _atr(data, atr_period)  # 確定バーまでのATR(自バー終値時点で利用可な近似)
        sma = close.rolling(trend).mean()

        uptrend = close > sma
        downtrend = close < sma

        # 確認バッファ: 終値が channel高値 + k×ATR を超えたら本物のブレイク
        long_entries = (close > (upper + k * atr)) & uptrend
        short_entries = (close < (lower - k * atr)) & downtrend
        # 利伸ばし: 反対側の exit本ドンチャン(exit<entryで広め=勝ちを伸ばす)
        long_exits = close < exit_lower
        short_exits = close > exit_upper
        return long_entries, long_exits, short_entries, short_exits
    return generate_signals


def payoff_ratio(pf) -> float:
    """平均利益 / 平均損失(の絶対値)。>1で損小利大。"""
    try:
        tr = pf.trades.records_readable
        pnl = tr["PnL"] if "PnL" in tr.columns else tr["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        if len(wins) == 0 or len(losses) == 0:
            return float("nan")
        return float(wins.mean() / abs(losses.mean()))
    except Exception:
        return float("nan")


def eval_basket(gen, side="both", data_slice=None):
    """7ペア平均のメトリクスを GROSS/NET 両方で返す。"""
    out = {}
    for gross in (True, False):
        set_gross(gross)
        rows = []
        for p in PAIRS:
            d = load(p, TF)
            if data_slice is not None:
                d = d.loc[data_slice]
            pf = run(p, TF, gen, {}, data=d, size_mode="value", side=side)
            m = metrics(pf).iloc[0].to_dict()
            m["pair"] = p
            m["payoff"] = payoff_ratio(pf)
            rows.append(m)
        df = pd.DataFrame(rows)
        tag = "gross" if gross else "net"
        out[tag] = df
    set_gross(False)
    return out


def summarize(out, label):
    g, n = out["gross"], out["net"]
    print(f"\n=== {label} ===")
    print(f"{'pair':8} {'g_ret':>8} {'n_ret':>8} {'g_shp':>7} {'n_shp':>7} "
          f"{'n_pf':>6} {'n_pay':>6} {'n_tr':>5}")
    for i in range(len(g)):
        gr, nr = g.iloc[i], n.iloc[i]
        print(f"{gr['pair']:8} {gr['total_return']:8.3f} {nr['total_return']:8.3f} "
              f"{gr['sharpe']:7.3f} {nr['sharpe']:7.3f} {nr['profit_factor']:6.2f} "
              f"{nr['payoff']:6.2f} {int(nr['num_trades']):5d}")
    res = {
        "g_ret": float(g["total_return"].mean()),
        "n_ret": float(n["total_return"].mean()),
        "g_shp": float(g["sharpe"].mean()),
        "n_shp": float(n["sharpe"].mean()),
        "n_pf": float(n["profit_factor"].replace([np.inf, -np.inf], np.nan).mean()),
        "n_pay": float(n["payoff"].mean()),
        "n_tr": float(n["num_trades"].mean()),
        "n_pos_pairs": int((n["total_return"] > 0).sum()),
    }
    print(f"AVG      g_ret={res['g_ret']:+.3f} n_ret={res['n_ret']:+.3f} "
          f"g_shp={res['g_shp']:+.3f} n_shp={res['n_shp']:+.3f} "
          f"n_pf={res['n_pf']:.2f} n_pay={res['n_pay']:.2f} "
          f"n_tr={res['n_tr']:.0f} pos={res['n_pos_pairs']}/7")
    return res


def main():
    print("再構築C: バッファ付きブレイク + 利伸ばし (D1, 7メジャー)")
    print(f"IS<= {IS_END}  OOS>= {OOS_START}")

    # --- ベースライン: 素のbreakout_trend(差分の基準) ----------------
    from strategies.breakout_trend import generate_signals as bt_gen
    print("\n##### ベースライン: 素のbreakout_trend (entry40/exit20/trend200) #####")
    base = eval_basket(lambda d, **kw: bt_gen(d, entry=40, exit=20, trend=200),
                       side="both")
    base_full = summarize(base, "breakout_trend baseline FULL 2016-2026")

    # --- 本命: バッファ付き(デフォルト entry40/exit20/k0.5/trend200) ---
    print("\n##### 本命C: バッファ付きブレイク+利伸ばし (entry40/exit20/k0.5/trend200) #####")
    genC = make_signals(entry=40, exit=20, k=0.5, trend=200)
    C_both = eval_basket(genC, side="both")
    C_full = summarize(C_both, "C buffered both FULL 2016-2026")

    # long-only / short-only 分離
    C_long = eval_basket(genC, side="long")
    summarize(C_long, "C buffered LONG-only FULL")
    C_short = eval_basket(genC, side="short")
    summarize(C_short, "C buffered SHORT-only FULL")

    # --- IS / OOS ----------------------------------------------------------
    C_is = eval_basket(genC, side="both", data_slice=slice(None, IS_END))
    is_res = summarize(C_is, "C buffered both IS 2016-2021")
    C_oos = eval_basket(genC, side="both", data_slice=slice(OOS_START, None))
    oos_res = summarize(C_oos, "C buffered both OOS 2022-2026")

    # --- plateau: entry∈{20,40,55}/exit∈{10,20}/k∈{0,0.5,1.0} -------------
    print("\n##### PLATEAU: 7ペア平均 NET total_return / NET sharpe #####")
    print(f"{'entry':>5} {'exit':>5} {'k':>5} {'g_ret':>8} {'n_ret':>8} "
          f"{'g_shp':>7} {'n_shp':>7} {'pos/7':>6}")
    plateau_rows = []
    for entry in (20, 40, 55):
        for exit in (10, 20):
            for k in (0.0, 0.5, 1.0):
                gen = make_signals(entry=entry, exit=exit, k=k, trend=200)
                o = eval_basket(gen, side="both")
                g, n = o["gross"], o["net"]
                row = {
                    "entry": entry, "exit": exit, "k": k,
                    "g_ret": float(g["total_return"].mean()),
                    "n_ret": float(n["total_return"].mean()),
                    "g_shp": float(g["sharpe"].mean()),
                    "n_shp": float(n["sharpe"].mean()),
                    "pos": int((n["total_return"] > 0).sum()),
                }
                plateau_rows.append(row)
                print(f"{entry:5d} {exit:5d} {k:5.1f} {row['g_ret']:8.3f} "
                      f"{row['n_ret']:8.3f} {row['g_shp']:7.3f} {row['n_shp']:7.3f} "
                      f"{row['pos']:4d}/7")
    pdf = pd.DataFrame(plateau_rows)
    n_net_pos = int((pdf["n_ret"] > 0).sum())
    n_gross_pos = int((pdf["g_ret"] > 0).sum())
    print(f"\nplateau: NET total_return>0 の組合せ {n_net_pos}/{len(pdf)}, "
          f"GROSS total_return>0 {n_gross_pos}/{len(pdf)}")

    # plateau頑健性: GROSS Sharpe / NET total_return の符号がパラメータ近傍で一貫するか
    gross_shp_all_pos = bool((pdf["g_shp"] > 0).all())
    gross_shp_all_neg = bool((pdf["g_shp"] < 0).all())
    net_ret_all_pos = bool((pdf["n_ret"] > 0).all())
    net_ret_all_neg = bool((pdf["n_ret"] < 0).all())
    print(f"GROSS sharpe 全組合せ正={gross_shp_all_pos} 全負={gross_shp_all_neg}")
    print(f"NET total_return 全組合せ正={net_ret_all_pos} 全負={net_ret_all_neg}")

    # --- 差分: 素 vs バッファ(コストドラッグと効果) ---------------------
    print("\n##### 差分: 素breakout_trend vs バッファC (FULL, 7ペア平均) #####")
    print(f"baseline  net_ret={base_full['n_ret']:+.3f} net_shp={base_full['n_shp']:+.3f} "
          f"gross_shp={base_full['g_shp']:+.3f} pos={base_full['n_pos_pairs']}/7")
    print(f"buffered  net_ret={C_full['n_ret']:+.3f} net_shp={C_full['n_shp']:+.3f} "
          f"gross_shp={C_full['g_shp']:+.3f} pos={C_full['n_pos_pairs']}/7")
    print(f"delta(C-base) net_ret={C_full['n_ret']-base_full['n_ret']:+.3f} "
          f"net_shp={C_full['n_shp']-base_full['n_shp']:+.3f} "
          f"gross_shp={C_full['g_shp']-base_full['g_shp']:+.3f}")
    cost_drag = C_full['g_ret'] - C_full['n_ret']
    print(f"buffered cost_drag (g_ret-n_ret) = {cost_drag:+.3f}")

    # --- 最終サマリ(StructuredOutput用に拾う値) -------------------------
    print("\n##### FINAL (StructuredOutput) #####")
    print(f"variant: buffered_breakout_let_profits_run D1")
    print(f"gross_sharpe_avg (FULL both) = {C_full['g_shp']:.4f}")
    print(f"net_sharpe_avg   (FULL both) = {C_full['n_shp']:.4f}")
    print(f"net_total_return_avg (FULL)  = {C_full['n_ret']:.4f}")
    print(f"net_positive = {C_full['n_ret'] > 0}")
    print(f"pos_pairs (NET) = {C_full['n_pos_pairs']}/7")
    print(f"IS net total_return = {is_res['n_ret']:.4f}")
    print(f"OOS net total_return = {oos_res['n_ret']:.4f}")
    print(f"plateau gross_shp_all_neg={gross_shp_all_neg} net_ret_all_neg={net_ret_all_neg} "
          f"net_pos_combos={n_net_pos}/{len(pdf)}")


if __name__ == "__main__":
    main()
