"""再構築A: vol-targetアンサンブルTSMOM(CTA標準実装)。

仕様(D1・各ペア):
  シグナル S_t = Σ_{L∈[21,63,126,252]} sign(close_t/close_{t-L} - 1) ∈ [-4,+4]
  no-trade band: |S_t| >= band の時だけ建て、それ未満はフラット(弱/対立トレンドは休む)
                 建玉方向 dir_t = sign(S_t) if |S_t|>=band else 0
  vol-target サイズ: 各ペアの直近60日リターン標準偏差(年率)に反比例して建玉を決定
                     w_t = clip(target_ann_vol / realized_ann_vol_t, 0, cap) * dir_t
  全て因果: シグナル・vol は t-1 まで(close.shift で確定済みのみ)、ポジションは翌バー適用。

P&L:
  daily strat return r_t = w_{t-1} * asset_ret_t            (w_{t-1}=前日終値時点で決めた建玉)
  GROSS = コスト0。NET = ポジション変更 |Δw| に往復スプレッド(price比)を課金。
  vectorbt の run() は per-bar 可変サイズ+no-tradeフラットを素直に表現できないため、
  ここでは仕様に忠実な自前ベクトル化バックテストで GROSS/NET を別計算で算出する。
  (コスト換算は config.SPREADS_PIPS と同じ pip/half-spread 規約に合わせる)

出力: 7ペア平均(等加重バスケット)と各ペアの GROSS/NET total_return・sharpe、
      NET黒字ペア数、IS(2016-2021)/OOS(2022-)、plateau(L集合ずらし・band∈{1,2,3})。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import load

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
LOOKBACKS = [21, 63, 126, 252]
VOL_WIN = 60
TARGET_ANN_VOL = 0.10      # 目標年率ボラ 10%(ペア横断で同一リスク予算)
WEIGHT_CAP = 2.0           # vol-targetレバ上限(暴走防止)
ANN = 252.0                # D1 年率化係数


def build_position(close: pd.Series, lookbacks, band: int) -> pd.Series:
    """因果な vol-target ポジション weight 系列(翌バー適用済み=実効建玉)を返す。

    w_eff_t = (前バーまでで決めた建玉) を asset_ret_t に掛ける形に整列して返す。
    """
    # アンサンブル signal(各 L の符号和)。close は確定値なので shift 不要だが
    # 「現バー終値」で判断 → 翌バーから建てる、を最後にまとめて shift(1) で表現。
    sig = pd.Series(0.0, index=close.index)
    for L in lookbacks:
        mom = close / close.shift(L) - 1.0
        sig = sig + np.sign(mom).fillna(0.0)

    direction = np.where(np.abs(sig) >= band, np.sign(sig), 0.0)
    direction = pd.Series(direction, index=close.index)

    # realized vol(日次log近似ではなく単純リターンの std を年率化)
    ret = close.pct_change()
    realized = ret.rolling(VOL_WIN).std() * np.sqrt(ANN)
    vol_scalar = (TARGET_ANN_VOL / realized).clip(upper=WEIGHT_CAP)
    vol_scalar = vol_scalar.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    w_decided = direction * vol_scalar      # 現バー終値時点で決める目標建玉
    w_eff = w_decided.shift(1).fillna(0.0)  # 翌バーから適用(先読み防止)
    return w_eff


def backtest_pair(pair: str, lookbacks, band: int):
    """1ペアの GROSS/NET 日次戦略リターン系列を返す(dict)。"""
    data = load(pair, "D1")
    close = data["close"]
    asset_ret = close.pct_change().fillna(0.0)

    w = build_position(close, lookbacks, band)

    gross_ret = w * asset_ret

    # コスト: ポジション変更 |Δw| に往復スプレッド(価格比)を課金。
    # half_spread_price/close * 2(往復) を 1 単位ノーショナル当たりコストとし、
    # |Δw| ノーショナル変化分に課金する(vectorbt 規約に整合)。
    half_spread_price = C.spread_pips(pair) * C.pip_size(pair) / 2.0
    roundtrip_cost_frac = (half_spread_price / close) * 2.0
    dw = w.diff().abs().fillna(w.abs())     # 初回建玉も建てコスト
    cost = dw * roundtrip_cost_frac
    net_ret = gross_ret - cost

    return {
        "pair": pair,
        "index": close.index,
        "gross_ret": gross_ret,
        "net_ret": net_ret,
        "w": w,
    }


def perf(ret: pd.Series) -> dict:
    """日次リターン系列から total_return / sharpe を算出(複利)。"""
    ret = ret.fillna(0.0)
    total = (1.0 + ret).prod() - 1.0
    mu = ret.mean()
    sd = ret.std()
    sharpe = (mu / sd * np.sqrt(ANN)) if sd > 0 else 0.0
    return {"total_return": float(total), "sharpe": float(sharpe), "n": int((ret != 0).sum())}


def run_config(lookbacks, band: int, label: str, verbose: bool = True):
    """指定設定で7ペア+バスケットの GROSS/NET を計算して返す。"""
    per_pair = {}
    gross_rets = []
    net_rets = []
    for p in PAIRS:
        r = backtest_pair(p, lookbacks, band)
        per_pair[p] = r
        gross_rets.append(r["gross_ret"].rename(p))
        net_rets.append(r["net_ret"].rename(p))

    # 等加重バスケット(各ペア同一リスク予算なので単純平均)
    G = pd.concat(gross_rets, axis=1).fillna(0.0)
    N = pd.concat(net_rets, axis=1).fillna(0.0)
    basket_gross = G.mean(axis=1)
    basket_net = N.mean(axis=1)

    # 各ペア成績
    pair_rows = []
    for p in PAIRS:
        g = perf(per_pair[p]["gross_ret"])
        n = perf(per_pair[p]["net_ret"])
        pair_rows.append({
            "pair": p,
            "gross_total": g["total_return"], "gross_sharpe": g["sharpe"],
            "net_total": n["total_return"], "net_sharpe": n["sharpe"],
            "active_days": n["n"],
        })
    pdf = pd.DataFrame(pair_rows)

    # 7ペア平均(各ペア単独運用の平均)
    avg_gross_total = pdf["gross_total"].mean()
    avg_gross_sharpe = pdf["gross_sharpe"].mean()
    avg_net_total = pdf["net_total"].mean()
    avg_net_sharpe = pdf["net_sharpe"].mean()
    net_pos_pairs = int((pdf["net_total"] > 0).sum())

    # バスケット
    bg = perf(basket_gross)
    bn = perf(basket_net)

    # IS/OOS(NET, バスケットおよび7ペア平均)
    def slice_perf(series_df, start, end):
        sub = series_df.loc[(series_df.index >= start) & (series_df.index < end)]
        # 各ペアの total を平均
        totals = [(1.0 + sub[c].fillna(0.0)).prod() - 1.0 for c in sub.columns]
        return float(np.mean(totals))

    is_net_avg = slice_perf(N, "2016-01-01", "2022-01-01")
    oos_net_avg = slice_perf(N, "2022-01-01", "2100-01-01")
    is_net_basket = (1.0 + basket_net.loc[(basket_net.index >= "2016-01-01") & (basket_net.index < "2022-01-01")].fillna(0.0)).prod() - 1.0
    oos_net_basket = (1.0 + basket_net.loc[basket_net.index >= "2022-01-01"].fillna(0.0)).prod() - 1.0

    if verbose:
        print(f"\n===== {label}  L={lookbacks} band={band} =====")
        print(pdf.round(4).to_string(index=False))
        print(f"-- 7ペア平均: GROSS tot {avg_gross_total:+.4f} sharpe {avg_gross_sharpe:+.3f} | "
              f"NET tot {avg_net_total:+.4f} sharpe {avg_net_sharpe:+.3f} | NET黒字 {net_pos_pairs}/7")
        print(f"-- バスケット: GROSS tot {bg['total_return']:+.4f} sharpe {bg['sharpe']:+.3f} | "
              f"NET tot {bn['total_return']:+.4f} sharpe {bn['sharpe']:+.3f}")
        print(f"-- IS/OOS (NET 7ペア平均): IS {is_net_avg:+.4f}  OOS {oos_net_avg:+.4f}")
        print(f"-- IS/OOS (NET バスケット): IS {is_net_basket:+.4f}  OOS {oos_net_basket:+.4f}")

    return {
        "label": label,
        "avg_gross_total": avg_gross_total, "avg_gross_sharpe": avg_gross_sharpe,
        "avg_net_total": avg_net_total, "avg_net_sharpe": avg_net_sharpe,
        "net_pos_pairs": net_pos_pairs,
        "basket_gross": bg, "basket_net": bn,
        "is_net_avg": is_net_avg, "oos_net_avg": oos_net_avg,
        "is_net_basket": float(is_net_basket), "oos_net_basket": float(oos_net_basket),
        "pdf": pdf,
    }


def main():
    print("データ範囲確認:")
    d = load("EURUSD", "D1")
    print(f"  EURUSD D1: {d.index.min()} .. {d.index.max()}  rows={len(d)}")

    # --- 基準設定 ---
    base = run_config(LOOKBACKS, band=2, label="BASE")

    # --- plateau: band ∈ {1,2,3} ---
    print("\n##### PLATEAU: band sweep #####")
    band_res = {}
    for b in [1, 2, 3]:
        r = run_config(LOOKBACKS, band=b, label=f"band={b}", verbose=False)
        band_res[b] = r
        print(f"band={b}: NET avg tot {r['avg_net_total']:+.4f} sharpe {r['avg_net_sharpe']:+.3f} "
              f"黒字{r['net_pos_pairs']}/7 | basket NET tot {r['basket_net']['total_return']:+.4f} "
              f"sharpe {r['basket_net']['sharpe']:+.3f}")

    # --- plateau: L集合を1段ずらす ---
    print("\n##### PLATEAU: lookback-set shift #####")
    L_variants = {
        "base[21,63,126,252]": [21, 63, 126, 252],
        "shorter[15,45,90,180]": [15, 45, 90, 180],
        "longer[30,90,180,360]": [30, 90, 180, 360],
    }
    L_res = {}
    for name, Ls in L_variants.items():
        r = run_config(Ls, band=2, label=name, verbose=False)
        L_res[name] = r
        print(f"{name}: NET avg tot {r['avg_net_total']:+.4f} sharpe {r['avg_net_sharpe']:+.3f} "
              f"黒字{r['net_pos_pairs']}/7 | basket NET tot {r['basket_net']['total_return']:+.4f} "
              f"sharpe {r['basket_net']['sharpe']:+.3f}")

    # --- plateau 判定: 符号維持か ---
    print("\n##### PLATEAU 判定 #####")
    band_signs = [np.sign(band_res[b]["basket_net"]["sharpe"]) for b in [1, 2, 3]]
    L_signs = [np.sign(L_res[n]["basket_net"]["sharpe"]) for n in L_variants]
    base_sign = np.sign(base["basket_net"]["sharpe"])
    all_signs = band_signs + L_signs
    plateau_hold = all(s == base_sign for s in all_signs) and base_sign != 0
    print(f"  base basket NET sharpe sign = {base_sign}")
    print(f"  band signs = {band_signs}")
    print(f"  L signs    = {L_signs}")
    print(f"  PLATEAU(近傍で符号維持) = {plateau_hold}")

    # --- 素のtsmom(net-12%参照)との対比のためのサマリ ---
    print("\n##### SUMMARY (machine) #####")
    import json
    summ = {
        "base_avg_gross_sharpe": round(base["avg_gross_sharpe"], 4),
        "base_avg_net_sharpe": round(base["avg_net_sharpe"], 4),
        "base_avg_gross_total": round(base["avg_gross_total"], 4),
        "base_avg_net_total": round(base["avg_net_total"], 4),
        "base_net_pos_pairs": base["net_pos_pairs"],
        "basket_gross_sharpe": round(base["basket_gross"]["sharpe"], 4),
        "basket_net_sharpe": round(base["basket_net"]["sharpe"], 4),
        "basket_gross_total": round(base["basket_gross"]["total_return"], 4),
        "basket_net_total": round(base["basket_net"]["total_return"], 4),
        "is_net_avg": round(base["is_net_avg"], 4),
        "oos_net_avg": round(base["oos_net_avg"], 4),
        "is_net_basket": round(base["is_net_basket"], 4),
        "oos_net_basket": round(base["oos_net_basket"], 4),
        "plateau_hold": bool(plateau_hold),
    }
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
