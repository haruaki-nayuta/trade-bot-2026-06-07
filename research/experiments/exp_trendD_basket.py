"""再構築D: 分散バスケット順張り(ポートフォリオ vol-target)。

7メジャーを1つのトレンド・バスケットとして合算する。各ペアに
アンサンブルTSMOM(複数lookbackの符号平均)× vol-target サイズを与え、
バスケット全体の日次P&L系列を集計して「合算系列のポートフォリオSharpe」を出す。
(個別ペアの平均Sharpeではなく、合算P&L系列そのもののSharpe。)

狙い: 個別のidiosyncraticノイズを分散で相殺すると、集計レベルで
gross/net Sharpe が正に届くか。

2バージョン:
  A. plain  — 7ペアの自前トレンドポジションを単純合算
  B. usdfac — 共通ドルファクター: USDxxx(JPY/CHF/CAD)は+1, xxxUSD(EUR/GBP/AUD/NZD)は-1
              で「対USD方向」に符号を揃えてから合算(共通ドルトレンドを取りにいく)

GROSS(コスト0)/ NET(half-spread×turnover を計上) を両方測る。
IS = 2016..2021, OOS = 2022.. 。先読みなし(ポジションは確定バーのリターン符号、
リターンへの適用は position.shift(1))。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load
import fxlab.config as C

PAIRS = list(C.PAIRS.keys())  # EURUSD USDJPY GBPUSD AUDUSD USDCHF USDCAD NZDUSD
TF = "D1"
LOOKBACKS = (50, 100, 200)   # アンサンブルTSMOMのlookback
TARGET_VOL_DAILY = 0.0066    # ペアあたり日次目標vol (~10.5%/yr). 全ペア共通なのでSharpeには中立
VOL_WIN = 60                 # realized vol 推定窓(日)
VOL_CAP = 3.0                # vol-target スカラーの上限(過小vol時の暴走防止)

# IS/OOS 境界
IS_START = pd.Timestamp("2016-01-01", tz="UTC")
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")

# 対USD符号: 値はそのペアのclose上昇=USDがどっちか。
# xxxUSD (EURUSD等) は close↑ = USD安 → ドル方向は -1
# USDxxx (USDJPY等) は close↑ = USD高 → ドル方向は +1
USD_SIGN = {
    "EURUSD": -1, "GBPUSD": -1, "AUDUSD": -1, "NZDUSD": -1,
    "USDJPY": +1, "USDCHF": +1, "USDCAD": +1,
}


def ensemble_tsmom_position(close: pd.Series, lookbacks=None) -> pd.Series:
    """複数lookbackのモメンタム符号を平均 → [-1,1] のターゲットポジション。
    確定バーのみ使用(close.shift(lb) は過去確定値)。
    lookbacks=None のときはモジュール globals の LOOKBACKS を毎回参照
    (plateau_scan で global を差し替えても反映されるようにするため)。
    """
    if lookbacks is None:
        lookbacks = LOOKBACKS
    sigs = []
    for lb in lookbacks:
        mom = close / close.shift(lb) - 1.0
        sigs.append(np.sign(mom))
    pos = pd.concat(sigs, axis=1).mean(axis=1)  # -1..1
    return pos


def build_pair_pnl(pair: str, gross: bool, usd_factor: bool):
    """1ペアの日次P&L系列(価格比リターン単位, vol-target後)を返す。

    pnl_t = w_{t-1} * ret_t  -  cost_t
      w = (target_vol / realized_vol) * tsmom_position   (vol-target済みポジション)
      ret_t = close_t/close_{t-1} - 1
      cost_t = |Δw_t| * half_spread_frac   (turnoverに比例。NETのみ)
    先読み防止: ポジションは t-1 までの情報で決まり、t のリターンに適用。
    """
    d = load(pair, TF)
    close = d["close"]
    ret = close.pct_change()

    pos = ensemble_tsmom_position(close)  # -1..1, 確定バー
    if usd_factor:
        pos = pos * USD_SIGN[pair]  # 対USD方向へ符号統一

    # realized vol (確定バーのリターンから)。shift で先読み回避。
    rv = ret.rolling(VOL_WIN).std()
    scalar = (TARGET_VOL_DAILY / rv).clip(upper=VOL_CAP)
    w = (pos * scalar)  # vol-target済みターゲットウェイト

    w_lag = w.shift(1)  # t-1 のウェイトを t のリターンに適用 = 先読みなし
    gross_pnl = w_lag * ret

    # コスト: half-spread を価格比で。turnover = |Δw|。
    half_spread_frac = (C.spread_pips(pair) * C.pip_size(pair)) / close / 2.0
    turnover = w.diff().abs()
    cost = turnover * half_spread_frac
    cost = cost.shift(1)  # コストは建玉変更時(t-1→t)に発生、ret_tと整合

    net_pnl = gross_pnl - cost

    out = gross_pnl if gross else net_pnl
    return out.rename(pair)


def basket_series(gross: bool, usd_factor: bool) -> pd.Series:
    """7ペアを合算したバスケット日次P&L系列。"""
    cols = [build_pair_pnl(p, gross=gross, usd_factor=usd_factor) for p in PAIRS]
    df = pd.concat(cols, axis=1)
    # 共通index・NaNは未確定(lookback/vol窓不足) → 0扱いせず dropna前にfill0で合算
    basket = df.fillna(0.0).sum(axis=1)
    # 全ペアNaNだった行(初期 warmup)を除く
    valid = df.notna().any(axis=1)
    return basket[valid]


def sharpe_ann(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if pnl.std() == 0 or len(pnl) < 30:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(252))


def total_return(pnl: pd.Series) -> float:
    """加法リターン系列の単純累積(各日リターンの和=近似累積)。"""
    return float(pnl.sum())


def slice_period(s: pd.Series, start=None, end=None) -> pd.Series:
    idx = s.index
    m = pd.Series(True, index=idx)
    if start is not None:
        m &= idx >= start
    if end is not None:
        m &= idx < end
    return s[m]


def evaluate_variant(usd_factor: bool, label: str):
    g = basket_series(gross=True, usd_factor=usd_factor)
    n = basket_series(gross=False, usd_factor=usd_factor)

    g_full, n_full = g, n
    g_is = slice_period(g, IS_START, OOS_START)
    n_is = slice_period(n, IS_START, OOS_START)
    g_oos = slice_period(g, OOS_START, None)
    n_oos = slice_period(n, OOS_START, None)

    res = {
        "label": label,
        "gross_sharpe_full": sharpe_ann(g_full),
        "net_sharpe_full": sharpe_ann(n_full),
        "gross_total_full": total_return(g_full),
        "net_total_full": total_return(n_full),
        "is_net_total": total_return(n_is),
        "is_net_sharpe": sharpe_ann(n_is),
        "oos_net_total": total_return(n_oos),
        "oos_net_sharpe": sharpe_ann(n_oos),
        "is_gross_total": total_return(g_is),
        "oos_gross_total": total_return(g_oos),
        "n_days": int(len(n_full)),
    }
    return res


def main():
    print(f"=== 再構築D: 分散バスケット順張り (vol-target) ===")
    print(f"TF={TF}  pairs={PAIRS}")
    print(f"lookbacks={LOOKBACKS}  target_vol_daily={TARGET_VOL_DAILY}  vol_win={VOL_WIN}")
    print(f"IS=2016..2021  OOS=2022..\n")

    variants = [
        evaluate_variant(usd_factor=False, label="A_plain"),
        evaluate_variant(usd_factor=True, label="B_usdfactor"),
    ]
    df = pd.DataFrame(variants).set_index("label")
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}")
    pd.set_option("display.width", 200)
    print(df.T.to_string())
    print()

    for v in variants:
        gp = v["gross_sharpe_full"]
        np_ = v["net_sharpe_full"]
        verdict = ("net_positive" if v["net_total_full"] > 0 and np_ > 0 else
                   "net_negative")
        edge = ("GROSS+ but NET- = cost/turnover kills" if gp > 0.05 and np_ <= 0 else
                "no_edge_even_gross" if gp <= 0.05 else
                "edge_survives_net")
        print(f"[{v['label']}] GROSS Sharpe {gp:+.3f} / NET Sharpe {np_:+.3f} "
              f"/ NET total {v['net_total_full']:+.4f} → {verdict} | {edge}")
        print(f"    IS net total {v['is_net_total']:+.4f} (Sh {v['is_net_sharpe']:+.3f}) "
              f"| OOS net total {v['oos_net_total']:+.4f} (Sh {v['oos_net_sharpe']:+.3f})")

    # CSVに保存(任意)
    out = C.ROOT / "research" / "outputs" / "exp_trendD_basket.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()


def plateau_scan():
    """パラメータ近傍スキャン: lookbackセット×vol窓 で GROSS/NET full Sharpe の符号が
    維持されるか(頑健性=plateau)。両variant。"""
    import itertools
    global LOOKBACKS, VOL_WIN
    orig_lb, orig_vw = LOOKBACKS, VOL_WIN
    lb_sets = [(25,50,100), (50,100,200), (100,200,300), (20,60,120,250)]
    vws = [40, 60, 90]
    print("\n=== plateau scan (GROSS full Sharpe) ===")
    print(f"{'lb_set':<20}{'vw':>4}  {'A_gross':>9}{'A_net':>9}  {'B_gross':>9}{'B_net':>9}")
    rows = []
    for lbs, vw in itertools.product(lb_sets, vws):
        LOOKBACKS = lbs
        VOL_WIN = vw
        ag = basket_series(gross=True, usd_factor=False)
        an = basket_series(gross=False, usd_factor=False)
        bg = basket_series(gross=True, usd_factor=True)
        bn = basket_series(gross=False, usd_factor=True)
        a_g, a_n = sharpe_ann(ag), sharpe_ann(an)
        b_g, b_n = sharpe_ann(bg), sharpe_ann(bn)
        rows.append((a_g, a_n, b_g, b_n))
        print(f"{str(lbs):<20}{vw:>4}  {a_g:>+9.3f}{a_n:>+9.3f}  {b_g:>+9.3f}{b_n:>+9.3f}")
    LOOKBACKS, VOL_WIN = orig_lb, orig_vw
    import numpy as _np
    arr = _np.array(rows)
    print(f"\nA gross: all<=0.05? {(arr[:,0]<=0.05).all()}  max={arr[:,0].max():+.3f}")
    print(f"B gross: all<=0.05? {(arr[:,2]<=0.05).all()}  max={arr[:,2].max():+.3f}")
    print(f"A net positive count: {(arr[:,1]>0).sum()}/{len(arr)}")
    print(f"B net positive count: {(arr[:,3]>0).sum()}/{len(arr)}")


if __name__ == "__main__":
    main()
    plateau_scan()
