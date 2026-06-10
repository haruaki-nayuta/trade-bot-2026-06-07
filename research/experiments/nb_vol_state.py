"""nb_vol_state — 次足予測ファミリー: ボラティリティ状態と相互作用 (EURUSD M5/M1).

実行: uv run python -m research.experiments.nb_vol_state

検証内容:
1. ボラ系特徴量バッテリー(方向予測 + |次足| ボラ予測)
   - volratio_5_50 / volratio_10_100 = std(close.diff(), k) / std(close.diff(), K)
   - atr14_pct100 = ATR(14) の trailing 100 本ランク (rolling rank pct)
   - rangecomp_k = (max(high,k)-min(low,k)) / 同・直前 k 本, k ∈ {5,10,20}
   - bbw_pct100 / bbw_raw = 4*std20/SMA20 とその trailing 100 本ランク
2. 最重要: z20 = (close-SMA20)/std20 の極値ビンをボラ状態で条件付け
   (within-state 閾値评価 + 固定 z ビンのセル分割)
3. スクイーズ→ブレイク方向 (レンジ圧縮 × 直近リターン)
4. 上位候補の減衰カーブ (h1..h20) とセッションマスク [7..16]/[0..6]
5. M1 での再評価

結論(2026-06-11 実測):
- ボラ特徴量単体の方向 IC はゼロ(|IC|<0.005)。ボラ予測は非常に強い
  (atr14_pct100 IC_te≈0.30, bbw_raw≈0.36)= 状態変数としてのみ有効。
- z20 平均回帰はボラ拡大状態で増幅される。最良: z20|vr10_100>=train q90 の
  q=0.02 ビンで test 次足 +0.76p (t=2.6) / -0.61p (t=-1.8)、約 0.36 sig/day/側。
  買い側は次足だけで往復コスト 0.6p を超える(ただし n=471 で t は中程度)。
- vr>=q75 × q=0.05 は +0.30 (t=3.4) / -0.25 (t=-2.8)、2.5 sig/day、h20 まで持続。
- 低ボラ状態では z20 エッジはほぼ消滅(セルで +0.01〜0.12p)。
- スクイーズ→ブレイク継続は不成立(むしろ弱い逆張りで極値ビンは無意味)。
- M1 では相互作用の構造は再現(IC -0.015→-0.035)するが絶対値が小さく
  (極値ビン 0.05〜0.18p)コスト 0.6p に遠く及ばない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import (
    SPLIT,
    eval_horizons,
    eval_signal,
    fmt_row,
    hour_mask,
    load_xy,
)

ACTIVE_HOURS = list(range(7, 17))
ASIA_HOURS = list(range(0, 7))


def build_feats(df: pd.DataFrame) -> dict[str, pd.Series]:
    c, h, l = df["close"], df["high"], df["low"]
    ret = c.diff()
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    prev_c = c.shift()
    tr_ = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    feats = {
        "z20": (c - sma20) / std20,
        "volratio_5_50": ret.rolling(5).std() / ret.rolling(50).std(),
        "volratio_10_100": ret.rolling(10).std() / ret.rolling(100).std(),
        "atr14_pct100": tr_.rolling(14).mean().rolling(100).rank(pct=True),
        "bbw_raw": 4 * std20 / sma20,
    }
    feats["bbw_pct100"] = feats["bbw_raw"].rolling(100).rank(pct=True)
    for k in (5, 10, 20):
        rng = h.rolling(k).max() - l.rolling(k).min()
        feats[f"rangecomp_{k}"] = rng / rng.shift(k)
    return feats


def workup(name: str, feat: pd.Series, tgt: pd.Series, df: pd.DataFrame, q: float):
    """eval + 減衰カーブ + セッション別."""
    r = eval_signal(feat, tgt, q, name)
    print(fmt_row(r), f"| sig/day lo {r.get('lo_sig_per_day', 0):.2f} hi {r.get('hi_sig_per_day', 0):.2f}")
    hz = eval_horizons(feat, df, "EURUSD", q)
    print("   horizons lo: " + ", ".join(f"h{k[1:]}={v:+.2f}" for k, v in hz["lo"].items()))
    print("   horizons hi: " + ", ".join(f"h{k[1:]}={v:+.2f}" for k, v in hz["hi"].items()))
    act = hour_mask(df.index, ACTIVE_HOURS)
    asia = hour_mask(df.index, ASIA_HOURS)
    print("   " + fmt_row(eval_signal(feat[act], tgt[act], q, name + " [7-16]")))
    print("   " + fmt_row(eval_signal(feat[asia], tgt[asia], q, name + " [0-6]")))
    return r


def main():
    # ---------------- M5 ----------------
    df, tgt, pip = load_xy("EURUSD", "M5")
    print(f"M5 rows={len(df)}  {df.index[0]} -> {df.index[-1]}")
    F = build_feats(df)
    tr_m = df.index < SPLIT
    abs_tgt = tgt.abs()

    print("\n=== 1a. M5 battery: DIRECTION (next-bar pips), q=0.02 ===")
    for nm in ["volratio_5_50", "volratio_10_100", "atr14_pct100",
               "rangecomp_5", "rangecomp_10", "rangecomp_20", "bbw_pct100", "bbw_raw"]:
        print(fmt_row(eval_signal(F[nm], tgt, 0.02, nm)))

    print("\n=== 1b. M5 battery: VOL PREDICTION (|next-bar pips|), q=0.02 ===")
    for nm in ["volratio_5_50", "volratio_10_100", "atr14_pct100",
               "rangecomp_5", "rangecomp_10", "rangecomp_20", "bbw_pct100", "bbw_raw"]:
        print(fmt_row(eval_signal(F[nm], abs_tgt, 0.02, nm)))

    act = hour_mask(df.index, ACTIVE_HOURS)
    asia = hour_mask(df.index, ASIA_HOURS)
    print("\n--- |tgt| IC_test by session (vol edge timing) ---")
    for nm in ["atr14_pct100", "volratio_10_100", "bbw_pct100"]:
        ra = eval_signal(F[nm][act], abs_tgt[act], 0.02, nm)
        rs = eval_signal(F[nm][asia], abs_tgt[asia], 0.02, nm)
        print(f"{nm:<16} [7-16] {ra['ic_test']:+.3f} | [0-6] {rs['ic_test']:+.3f}")

    # ---------------- 2. interaction: z20 x vol state ----------------
    z20, vr, atrp, bbwp = F["z20"], F["volratio_10_100"], F["atr14_pct100"], F["bbw_pct100"]

    print("\n=== 2a. fixed z20 2% bin, test cell means by vol-state median split ===")
    zlo, zhi = z20[tr_m].quantile(0.02), z20[tr_m].quantile(0.98)
    te_m = ~tr_m
    for snm, f in [("atrp", atrp), ("vr10_100", vr), ("bbwp", bbwp)]:
        med = f[tr_m].median()
        for tag, zbin in [("z_lo(buy) ", z20 <= zlo), ("z_hi(sell)", z20 >= zhi)]:
            for st, sm in [("lovol", f < med), ("hivol", f >= med)]:
                yy = tgt[zbin & sm & te_m].dropna()
                n = len(yy)
                mu = yy.mean()
                t = mu / (yy.std() / np.sqrt(n)) if n > 2 else np.nan
                print(f"  {tag} & {snm:<9} {st:<6} test {mu:+.3f}p t={t:+.2f} n={n}")

    print("\n=== 2b. z20 within hi-vol states (within-state train thresholds) ===")
    vr75 = vr[tr_m].quantile(0.75)
    vr90 = vr[tr_m].quantile(0.90)
    bbw23 = bbwp[tr_m].quantile(2 / 3)
    grid = [
        ("z20|vr>=q90", z20.where(vr >= vr90), (0.02, 0.05)),
        ("z20|vr>=q75", z20.where(vr >= vr75), (0.05, 0.10)),
        ("z20|atrp>=0.8", z20.where(atrp >= 0.8), (0.02, 0.05)),
        ("z20|bbwp_hi(t3)", z20.where(bbwp >= bbw23), (0.02,)),
        ("z20*bbwp", z20 * bbwp, (0.02,)),
    ]
    for nm, f, qs in grid:
        for q in qs:
            print(fmt_row(eval_signal(f, tgt, q, f"{nm} q={q}")))

    # ---------------- 3. squeeze -> break ----------------
    print("\n=== 3. squeeze x recent-return (continuation test) ===")
    ret5p = df["close"].diff(5) / pip
    ret10p = df["close"].diff(10) / pip
    for sq_nm, sq_f, sq_q in [("rcomp20", F["rangecomp_20"], 0.25), ("bbwp", bbwp, 0.25), ("bbwp", bbwp, 0.10)]:
        thr = sq_f[tr_m].quantile(sq_q)
        mask = sq_f <= thr
        for rnm, rf in [("ret5", ret5p), ("ret10", ret10p)]:
            print(fmt_row(eval_signal(rf[mask], tgt[mask], 0.05, f"{rnm}|{sq_nm}<q{sq_q}")))

    # ---------------- 4. top candidates: full workup ----------------
    print("\n=== 4. M5 top candidates: horizons + sessions ===")
    tops = [
        ("z20_all(bench)", z20, 0.02),
        ("z20|vr>=q90", z20.where(vr >= vr90), 0.02),
        ("z20|vr>=q75", z20.where(vr >= vr75), 0.05),
        ("z20|bbwp_hi", z20.where(bbwp >= bbw23), 0.02),
    ]
    for nm, f, q in tops:
        workup(f"M5 {nm} q={q}", f, tgt, df, q)
        print()

    # ---------------- 5. M1 re-eval ----------------
    df1, tgt1, _ = load_xy("EURUSD", "M1")
    print(f"\n=== 5. M1 re-eval (rows={len(df1)}) ===")
    F1 = build_feats(df1)
    tr1 = df1.index < SPLIT
    z1, vr1, bbwp1 = F1["z20"], F1["volratio_10_100"], F1["bbw_pct100"]
    vr1_90 = vr1[tr1].quantile(0.90)
    vr1_75 = vr1[tr1].quantile(0.75)
    bbw1_23 = bbwp1[tr1].quantile(2 / 3)
    for nm, f, q in [
        ("M1 z20_all(bench)", z1, 0.02),
        ("M1 z20|vr>=q90", z1.where(vr1 >= vr1_90), 0.02),
        ("M1 z20|vr>=q75", z1.where(vr1 >= vr1_75), 0.05),
        ("M1 z20|bbwp_hi", z1.where(bbwp1 >= bbw1_23), 0.02),
    ]:
        workup(f"{nm} q={q}", f, tgt1, df1, q)
        print()


if __name__ == "__main__":
    main()
