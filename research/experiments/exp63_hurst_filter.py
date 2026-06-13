"""exp63: Hurst指数 / 分散比 をエントリーフィルタとして検証(チャンピオン自体の改善, web由来)。

reports/08 のフィルタ・ベイクオフは ER / ADX / slope / ATR / slow_z_cap を比較し、ER(40)≤0.55 だけが
非カーブフィットで効いた。**Hurst指数・分散比(VRT)はその比較集合に入っていなかった**=未検証の
literature-standard レジーム検出器。Hurst<0.5=平均回帰的、>0.5=トレンド的。

プロトコル([[28]] wickbias / [[08]]):
  チャンピオン d1 プール(既に ER≤0.55 通過済み)各トレードのエントリー時点で Hurst と VRT を計算し、
  ① ER との相関(冗長か直交か) ② 勝敗の AUC ③ **除去コホートは純損か**(高Hurst=トレンド寄りを
  切ると、切られる側が純損失なら有効。儲かるトレードを捨てるだけなら無効)。
  ER で既に取れているものの焼き直しでないか(ER で層別した上で Hurst が追加識別するか)を見る。

Hurst は構造関数法(log std(Δτ) vs log τ の傾き, τ=2..max_lag)。因果(エントリー時点までの close)。
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1  # noqa: E402
from fxlab import universe as uni  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)


def hurst_struct(logp: np.ndarray, max_lag=20) -> float:
    """構造関数法 Hurst: slope of log(std(p[t+τ]-p[t])) vs log(τ)。logp は対数価格配列。"""
    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        d = logp[lag:] - logp[:-lag]
        tau.append(np.std(d))
    tau = np.array(tau)
    if np.any(tau <= 0) or len(tau) < 3:
        return np.nan
    return float(np.polyfit(np.log(lags), np.log(tau), 1)[0])


def variance_ratio(ret: np.ndarray, k=4) -> float:
    """分散比 VR(k) = Var(k期間和)/(k*Var(1期間))。<1=平均回帰, >1=トレンド。"""
    if len(ret) < k * 3:
        return np.nan
    v1 = np.var(ret)
    agg = np.add.reduceat(ret, np.arange(0, len(ret) - len(ret) % k, k))
    vk = np.var(agg)
    return float(vk / (k * v1)) if v1 > 0 else np.nan


def er(logp_or_px: np.ndarray, w=40) -> float:
    x = logp_or_px
    if len(x) < w + 1:
        return np.nan
    direction = abs(x[-1] - x[-1 - w])
    vol = np.sum(np.abs(np.diff(x[-1 - w:])))
    return float(direction / vol) if vol > 0 else np.nan


def main():
    print("=" * 72)
    print("  exp63: Hurst / 分散比 エントリーフィルタ検証(チャンピオン改善)")
    print("=" * 72)
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1(tf="H4")
    print(f"  champion d1 pool: {len(pool)} trades")

    # 各銘柄の close をキャッシュ
    closes = {}
    for nm in pool["instr"].unique():
        closes[nm] = uni.instrument_close(nm, "H4")

    L = 100  # Hurst/VRT のローリング窓
    rows = []
    for _, tr in pool.iterrows():
        nm = tr["instr"]; t = tr["entry"]
        c = closes[nm]
        seg = c.loc[:t]
        if len(seg) < L + 5:
            continue
        seg = seg.iloc[-L:].to_numpy()
        lp = np.log(seg)
        ret = np.diff(lp)
        rows.append({
            "ret": tr["ret"], "win": 1 if tr["ret"] > 0 else 0,
            "hurst": hurst_struct(lp), "vr4": variance_ratio(ret, 4),
            "er40": er(seg, 40),
        })
    d = pd.DataFrame(rows).dropna()
    print(f"  特徴量計算済み: {len(d)} trades  (勝率 {d['win'].mean():.1%}, 平均ret {d['ret'].mean()*100:+.3f}%)")

    # 相関(冗長性)
    print("\n[①] 特徴量間の相関(ER との冗長性チェック)")
    print(d[["hurst", "vr4", "er40"]].corr().round(3).to_string())

    # AUC
    from itertools import product
    def auc(feat, label):
        x = d[feat].to_numpy(); y = d[label].to_numpy()
        order = np.argsort(x)
        xr = np.empty_like(x); xr[order] = np.arange(len(x))
        n1 = y.sum(); n0 = len(y) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        return float((xr[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
    print("\n[②] 勝敗 AUC(0.5=無情報。Hurst低=平均回帰=勝ち寄りなら <0.5 期待)")
    for f in ["hurst", "vr4", "er40"]:
        print(f"     AUC({f} → win) = {auc(f, 'win'):.3f}")

    # 除去コホート純損テスト: 高Hurst(トレンド寄り)を切る
    print("\n[③] 除去コホート純損テスト(高Hurst/高VR を切ると、切られる側は純損か?)")
    for f in ["hurst", "vr4"]:
        for q in [0.7, 0.8, 0.9]:
            thr = d[f].quantile(q)
            removed = d[d[f] > thr]
            kept = d[d[f] <= thr]
            print(f"     {f}>{thr:.3f}(上位{1-q:.0%}): 除去群 n={len(removed)} 平均ret {removed['ret'].mean()*100:+.3f}% 合計 {removed['ret'].sum()*100:+.2f}% | "
                  f"残存 合計 {kept['ret'].sum()*100:+.2f}%(全体{d['ret'].sum()*100:+.2f}%)")
    print("     → 除去群の合計が明確にマイナスなら Hurst/VR フィルタは純益を上げる(切る価値あり)")

    # ER で層別した上で Hurst が追加識別するか(直交性)
    print("\n[④] ER≤中央 / >中央 で層別し、各層内で Hurst 高低の純益差(ER の焼き直しでないか)")
    erm = d["er40"].median()
    for lab, sub in [("ER低(回帰寄り)", d[d["er40"] <= erm]), ("ER高(トレンド寄り)", d[d["er40"] > erm])]:
        hm = sub["hurst"].median()
        lo = sub[sub["hurst"] <= hm]["ret"].mean() * 100
        hi = sub[sub["hurst"] > hm]["ret"].mean() * 100
        print(f"     {lab}: Hurst低 平均ret {lo:+.3f}% vs Hurst高 {hi:+.3f}%  (差 {lo-hi:+.3f}pp)")

    print("\n" + "=" * 72)
    print("  判定: 除去コホートが純損 かつ ER層別後も Hurst が識別 → エントリーフィルタ化して DD検証へ。")
    print("=" * 72)


if __name__ == "__main__":
    main()
