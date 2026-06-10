"""nb_range_levels — 次足エッジ探索: レンジ内位置と価格レベル ファミリー (EURUSD M1/M5)。

共通プロトコル research/lab/nextbar_common.py に準拠。
実行: uv run python -m research.experiments.nb_range_levels

特徴量 (rolling W=100 本):
- rngpos_prev   : (close - min(low,100).shift(1)) / (range.shift(1))。<0/>1 はブレイク
- brk_signed    : ブレイク強度 (close と前100本高値/安値の差を ret-std で正規化、符号付き)
- since_hi/lo   : 100本内の高値/安値からの経過本数
- touch_hi/lo   : レンジ上端/下端 5% バンドへのタッチ回数
- d50/d100      : 0.00500 / 0.01000 グリッドへの符号付き距離 (pips)
- rngw_pips     : レンジ幅 (pips) — 条件付け用
- edge_pips     : (rngpos_prev - 0.5) * rngw_pips (位置×幅の連続交互作用)

結論 (2026-06-11 実測):
- ファミリー全体が「次足は平均回帰 (fade)」。ブレイク追随の次足エッジは存在しない
  (100本新高値クローズの次足 test -0.10p t=-2.2 / 新安値 +0.11p t=+2.1)。
- 最有力: M5 ワイドレンジ (幅 train 上位1/3) でのレンジ下端 fade ロング。
  q=0.02 test +0.43p (t=+2.9)、h20 累積 +1.55p (t=+3.0)。train/test 全年プラス。
- M1 ワイドは両側で h20 まで単調増加 (lo +0.87p t=+8.2 / hi -0.70p t=-7.0) だが次足単体は小さい。
- z20 ベンチとの直交性は低い (rank corr 0.57、シグナルバーの52%が z20<-2)。
- ラウンドナンバー・タッチ回数・経過本数は dead end。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from research.lab.nextbar_common import (
    SPLIT,
    eval_horizons,
    eval_signal,
    fmt_row,
    horizon_targets,
    hour_mask,
    load_xy,
)

W = 100
PAIR = "EURUSD"


def build_feats(tf: str, with_heavy: bool = True):
    df, tgt, pip = load_xy(PAIR, tf)
    c, h, l = df["close"], df["high"], df["low"]
    n = len(df)
    rmax_p = h.rolling(W).max().shift(1)
    rmin_p = l.rolling(W).min().shift(1)
    rng_p = rmax_p - rmin_p
    feats: dict[str, pd.Series] = {}
    feats["rngpos_prev"] = (c - rmin_p) / rng_p
    retstd = c.diff().rolling(W).std()
    feats["brk_signed"] = ((c - rmax_p) / retstd).clip(lower=0) - (
        (rmin_p - c) / retstd
    ).clip(lower=0)
    rngw = rng_p / pip
    feats["rngw_pips"] = rngw
    feats["edge_pips"] = (feats["rngpos_prev"] - 0.5) * rngw
    g50, g100 = 0.005, 0.01
    feats["d50_signed"] = ((c / g50) - (c / g50).round()) * g50 / pip
    feats["d100_signed"] = ((c / g100) - (c / g100).round()) * g100 / pip
    if with_heavy:  # sliding-window 系 (M5 のみ。M1 でも動くがメモリ/時間を食う)
        wh = sliding_window_view(h.to_numpy(), W)
        wl = sliding_window_view(l.to_numpy(), W)
        since_hi = np.full(n, np.nan)
        since_lo = np.full(n, np.nan)
        since_hi[W - 1 :] = (W - 1) - wh.argmax(axis=1)
        since_lo[W - 1 :] = (W - 1) - wl.argmin(axis=1)
        feats["since_hi"] = pd.Series(since_hi, index=df.index)
        feats["since_lo"] = pd.Series(since_lo, index=df.index)
        feats["tsince_diff"] = (feats["since_lo"] - feats["since_hi"]) / W
        wmax = wh.max(axis=1)
        wmin = wl.min(axis=1)
        wrng = wmax - wmin
        th = np.full(n, np.nan)
        tl = np.full(n, np.nan)
        th[W - 1 :] = (wh >= (wmax - 0.05 * wrng)[:, None]).sum(axis=1)
        tl[W - 1 :] = (wl <= (wmin + 0.05 * wrng)[:, None]).sum(axis=1)
        feats["touch_hi"] = pd.Series(th, index=df.index)
        feats["touch_lo"] = pd.Series(tl, index=df.index)
        feats["touch_diff"] = feats["touch_hi"] - feats["touch_lo"]
    return df, tgt, pip, feats


def show(r: dict) -> None:
    print(
        fmt_row(r),
        f"| lo tr {r['lo_train_mean_pips']:+.2f}p(t={r['lo_train_t']:+.1f})"
        f" hi tr {r['hi_train_mean_pips']:+.2f}p(t={r['hi_train_t']:+.1f})"
        f" | s/d {r.get('lo_sig_per_day', 0):.2f}/{r.get('hi_sig_per_day', 0):.2f}",
    )


def hz(feat: pd.Series, df: pd.DataFrame, name: str, q: float = 0.02) -> None:
    o = eval_horizons(feat, df, PAIR, q=q)
    print(f"{name:<34} lo: " + " ".join(f"h{k[1:]}={v:+.2f}" for k, v in o["lo"].items()))
    print(f"{'':<34} hi: " + " ".join(f"h{k[1:]}={v:+.2f}" for k, v in o["hi"].items()))


def event_mean(tgt: pd.Series, ev: pd.Series, tr: np.ndarray, name: str) -> None:
    for st, ms in [("train", tr), ("test", ~tr)]:
        yy = tgt[ev.fillna(False) & ms & tgt.notna()]
        t = yy.mean() / (yy.std() / np.sqrt(len(yy))) if len(yy) > 2 else np.nan
        print(f"  {name} {st}: mean {yy.mean():+.3f}p n={len(yy)} t={t:+.1f}")


def main() -> None:
    # ---------- M5 バッテリー ----------
    print("########## M5 battery (q=0.02) ##########")
    df, tgt, pip, F = build_feats("M5")
    tr = df.index < SPLIT
    for nm in [
        "rngpos_prev", "brk_signed", "tsince_diff", "since_hi", "since_lo",
        "touch_diff", "touch_hi", "touch_lo", "d50_signed", "d100_signed",
        "rngw_pips", "edge_pips",
    ]:
        show(eval_signal(F[nm], tgt, q=0.02, name=nm))

    # ---------- ブレイクアウトはイベントとしても確認 ----------
    print("\n########## breakout event next-bar (continuation vs fade) ##########")
    c = df["close"]
    rmax_p = df["high"].rolling(W).max().shift(1)
    rmin_p = df["low"].rolling(W).min().shift(1)
    event_mean(tgt, c > rmax_p, tr, "new 100-bar high close")
    event_mean(tgt, c < rmin_p, tr, "new 100-bar low close")

    # ---------- ラウンドナンバー: クロスイベント ----------
    print("\n########## round-number cross events ##########")
    g50 = 0.005
    lvl = np.floor(c / g50)
    lvl_p = np.floor(c.shift(1) / g50)
    event_mean(tgt, lvl > lvl_p, tr, "crossed 50-pip level up")
    event_mean(tgt, lvl < lvl_p, tr, "crossed 50-pip level dn")

    # ---------- q スイープ + セッション ----------
    print("\n########## rngpos_prev: q sweep & session masks ##########")
    rp = F["rngpos_prev"]
    for q in (0.05, 0.10):
        show(eval_signal(rp, tgt, q=q, name=f"rngpos_prev q={q}"))
    act = hour_mask(df.index, list(range(7, 17)))
    asia = hour_mask(df.index, list(range(0, 7)))
    for nm, mask in [("act7-16", act), ("asia0-6", asia)]:
        show(eval_signal(rp.where(mask), tgt.where(mask), q=0.02, name=f"rngpos {nm} q02"))
        show(eval_signal(rp.where(mask), tgt.where(mask), q=0.05, name=f"rngpos {nm} q05"))

    # ---------- レンジ幅条件付け (本命) ----------
    print("\n########## width-conditioned rngpos (top signal) ##########")
    rngw = F["rngw_pips"]
    w_hi = rngw[tr].quantile(2 / 3)
    wide = rngw >= w_hi
    print(f"wide threshold (train 2/3 quantile): {w_hi:.1f} pips")
    for q in (0.02, 0.05):
        show(eval_signal(rp.where(wide), tgt.where(wide), q=q, name=f"rngpos wide q={q}"))
    show(eval_signal(rp.where(wide & act), tgt.where(wide & act), q=0.05, name="rngpos wide&act q05"))
    show(eval_signal(rp.where(wide & asia), tgt.where(wide & asia), q=0.05, name="rngpos wide&asia q05"))

    print("\n--- horizons (test cumulative pips) ---")
    hz(rp, df, "rngpos_prev q02")
    hz(rp.where(wide), df, "rngpos wide q02")
    hz(rp.where(asia), df, "rngpos asia q02")
    hz(F["edge_pips"], df, "edge_pips q02")

    print("\n--- wide-lo yearly stability (next-bar mean pips) ---")
    f = rp.where(wide)
    lo_thr = f[f.notna() & tr].quantile(0.02)
    sig = (f <= lo_thr).fillna(False)
    y1 = tgt[sig & tgt.notna()]
    print(y1.groupby(y1.index.year).agg(["mean", "count"]))

    print("\n--- wide-lo per-horizon t (test) ---")
    hts = horizon_targets(df, PAIR)
    idx = f.index[sig & ~tr]
    for hh, ht in hts.items():
        yy = ht.reindex(idx).dropna()
        t = yy.mean() / (yy.std() / np.sqrt(len(yy)))
        print(f"  h{hh}: {yy.mean():+.2f}p t={t:+.1f} n={len(yy)}")

    # ---------- z20 ベンチマークとの直交性 ----------
    print("\n########## orthogonality vs z20 benchmark ##########")
    sma20 = c.rolling(20).mean()
    z20 = (c - sma20) / c.rolling(20).std()
    m = rp.notna() & z20.notna()
    print(
        "Spearman(rngpos_prev, z20) test:",
        round(rp[m & ~tr].rank().corr(z20[m & ~tr].rank()), 3),
    )
    zs = z20[sig & ~tr]
    print(f"z20 on wide-lo signal bars (test): mean {zs.mean():+.2f} | frac z20<-2: {(zs < -2).mean():.3f}")
    calm = z20.abs() < 1.0
    show(eval_signal(f.where(calm), tgt.where(calm), q=0.05, name="wide-lo&|z20|<1 q05"))
    show(eval_signal(z20, tgt, q=0.02, name="z20 (benchmark) q02"))

    # ---------- M1 再評価 (有望シグナルのみ) ----------
    print("\n########## M1 re-evaluation (top signals only) ##########")
    df1, tgt1, pip1, F1 = build_feats("M1", with_heavy=False)
    tr1 = df1.index < SPLIT
    rp1 = F1["rngpos_prev"]
    rngw1 = F1["rngw_pips"]
    wide1 = rngw1 >= rngw1[tr1].quantile(2 / 3)
    asia1 = hour_mask(df1.index, list(range(0, 7)))
    for q in (0.02, 0.05):
        show(eval_signal(rp1, tgt1, q=q, name=f"M1 rngpos_prev q={q}"))
    show(eval_signal(F1["brk_signed"], tgt1, q=0.02, name="M1 brk_signed q02"))
    show(eval_signal(rp1.where(wide1), tgt1.where(wide1), q=0.02, name="M1 rngpos wide q02"))
    show(eval_signal(rp1.where(asia1), tgt1.where(asia1), q=0.02, name="M1 rngpos asia q02"))
    hz(rp1.where(wide1), df1, "M1 rngpos wide q02")
    print("\n--- M1 wide per-horizon t (test) ---")
    f1 = rp1.where(wide1)
    lo1 = f1[f1.notna() & tr1].quantile(0.02)
    hi1 = f1[f1.notna() & tr1].quantile(0.98)
    hts1 = horizon_targets(df1, PAIR)
    for tag, sel in [("lo", (f1 <= lo1) & ~tr1), ("hi", (f1 >= hi1) & ~tr1)]:
        idx1 = f1.index[sel.fillna(False)]
        row = []
        for hh, ht in hts1.items():
            yy = ht.reindex(idx1).dropna()
            row.append(f"h{hh} {yy.mean():+.2f}p(t={yy.mean() / (yy.std() / np.sqrt(len(yy))):+.1f})")
        print(f"  M1 wide {tag}: " + " ".join(row) + f" n={len(idx1)}")


if __name__ == "__main__":
    main()
