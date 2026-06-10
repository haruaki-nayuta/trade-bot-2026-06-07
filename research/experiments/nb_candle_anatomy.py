"""nb_candle_anatomy — ローソク足の解剖学ファミリー(次足エッジ探索)。

EURUSD M5/M1 で「直近1〜5本の足の形」から次足 close-to-close (pips) を予測する
特徴量バッテリーを research/lab/nextbar_common.py の標準評価器で検証する。

実行: uv run python -m research.experiments.nb_candle_anatomy

主な結論(2026-06 時点の実測):
1. comb25_long: z20<=train q02 (-2.41) かつ CLV<=0.038(ゾーン内 train q25)
   → test +0.40p (t=3.6, ~1.0/day)。ロールオーバー(21-23UTC)除外でも +0.30p (t=2.6)。
   h20 まで伸びる(+1.31p)。2016-2026 で毎年プラス(2024 のみ弱い)。
2. bear_marubozu_rng2: 実体比率>=train q98 の大陰線 かつ レンジ>2x avg20
   → test +0.81p (t=3.4) だが n=263 (0.21/day)、ロールオーバー除外で +0.38p (t=1.5)。
   強気側ミラーは死んでいる(非対称)。
3. clv3(3本平均CLV)lo ビン: 次足 +0.16p (t=3.7)、h20 +0.92p (t=5.1) と持続。
   活発時間帯(7-16UTC)に集中(+0.24 vs アジア -0.03)。z20 とは r_s=0.46。
4. gap_pips のエッジはほぼ全てロールオーバー時間帯のアーティファクト(死に筋)。
5. M1 は IC こそ残るが条件付き平均が +0.01〜0.04p でコスト 0.6p に遠く及ばない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import (
    SPLIT,
    eval_horizons,
    eval_signal,
    fmt_row,
    horizon_targets,
    hour_mask,
    load_xy,
)

PAIR = "EURUSD"
ROLLOVER_HOURS = [21, 22, 23]  # NY17時前後: 実スプレッドが数倍に開く


def candle_features(df: pd.DataFrame, pip: float) -> dict[str, pd.Series]:
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    rng_ok = rng.where(rng > 0)  # ゼロレンジ足は NaN
    clv = (c - l) / rng_ok
    up_w = (h - np.maximum(o, c)) / rng_ok
    lo_w = (np.minimum(o, c) - l) / rng_ok
    body = (c - o) / rng_ok
    body_hi, body_lo = np.maximum(o, c), np.minimum(o, c)
    inside = ((h < h.shift()) & (l > l.shift())).astype(float)
    outside = ((h > h.shift()) & (l < l.shift())).astype(float)
    bull_eng = (
        (body_hi > body_hi.shift()) & (body_lo < body_lo.shift())
        & (c > o) & (c.shift() < o.shift())
    )
    bear_eng = (
        (body_hi > body_hi.shift()) & (body_lo < body_lo.shift())
        & (c < o) & (c.shift() > o.shift())
    )
    gap = (o - c.shift()) / pip
    hh5, ll5 = h.rolling(5).max(), l.rolling(5).min()
    rng_avg20 = rng.rolling(20).mean()
    return {
        "clv1": clv,
        "clv3": clv.rolling(3).mean(),
        "clv5_window": (c - ll5) / (hh5 - ll5).where((hh5 - ll5) > 0),
        "upper_wick1": up_w,
        "lower_wick1": lo_w,
        "wick_diff1": lo_w - up_w,
        "wick_diff3": (lo_w - up_w).rolling(3).mean(),
        "body_signed1": body,
        "body_abs1": body.abs(),
        "body_signed3": body.rolling(3).mean(),
        "inside_bar": inside,
        "outside_bar": outside,
        "outside_dir": outside * np.sign(c - o),
        "engulf_dir": bull_eng.astype(float) - bear_eng.astype(float),
        "gap_pips": gap,
        "lower_highs3": (h < h.shift()).astype(float).rolling(3).sum(),
        "range_rel20": rng / rng_avg20.where(rng_avg20 > 0),
    }


def binary_report(name: str, sig: pd.Series, tgt: pd.Series, df: pd.DataFrame) -> None:
    """二値シグナルの train/test 条件付き平均・t・件数・時間帯・ホライズン。"""
    tr = df.index < SPLIT
    te = ~tr
    norol = pd.Series(~np.isin(df.index.hour, ROLLOVER_HOURS), index=df.index)
    hts = horizon_targets(df, PAIR)

    def line(tag: str, mk: pd.Series) -> None:
        yy = tgt[mk].dropna()
        if len(yy) < 10:
            print(f"  {tag}: n={len(yy)} (too few)")
            return
        t = yy.mean() / (yy.std() / np.sqrt(len(yy)))
        print(f"  {tag}: n={len(yy)} mean={yy.mean():+.3f}p t={t:+.2f}")

    print(f"[{name}]")
    line("train", sig & tr)
    line("test ", sig & te)
    line("test ex-rollover", sig & te & norol)
    line("test act7-16", sig & te & hour_mask(df.index, list(range(7, 17))))
    line("test asia0-6", sig & te & hour_mask(df.index, list(range(0, 7))))
    idx = df.index[sig & te]
    hz = {f"h{k}": round(float(v.reindex(idx).mean()), 3) for k, v in hts.items()}
    print(f"  test horizons: {hz}")
    yy_all = tgt[sig & te].dropna()
    yearly = {
        int(yr): f"{g.mean():+.2f}p(n={len(g)})"
        for yr, g in yy_all.groupby(yy_all.index.year)
    }
    print(f"  test yearly: {yearly}")
    days = max((df.index[te][-1] - df.index[te][0]).days, 1)
    print(f"  test sig/day: {(sig & te).sum() / days:.2f}")


def main() -> None:
    # ---------- M5 ----------
    df, tgt, pip = load_xy(PAIR, "M5")
    feats = candle_features(df, pip)
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    z20 = (c - sma20) / std20.where(std20 > 0)
    tr = df.index < SPLIT
    clv = feats["clv1"]
    body = feats["body_signed1"]
    rng_rel = feats["range_rel20"]

    print("=" * 88)
    print("M5 battery (q=0.02)")
    print("=" * 88)
    for name, f in feats.items():
        print(fmt_row(eval_signal(f, tgt, q=0.02, name=name)))
    print(fmt_row(eval_signal(z20, tgt, q=0.02, name="z20 (benchmark)")))

    print("\n--- q sensitivity (clv3 / clv5_window) ---")
    for nm in ("clv3", "clv5_window"):
        for q in (0.02, 0.05, 0.10):
            r = eval_signal(feats[nm], tgt, q=q, name=f"{nm} q={q}")
            print(
                fmt_row(r),
                f"| sig/day {r.get('lo_sig_per_day', 0):.1f}/{r.get('hi_sig_per_day', 0):.1f}",
            )

    print("\n--- z20 orthogonality ---")
    sub = (
        pd.DataFrame(
            {"z20": z20, "clv1": clv, "clv3": feats["clv3"],
             "clv5": feats["clv5_window"], "gap": feats["gap_pips"],
             "wickd": feats["wick_diff1"]}
        )
        .loc[tr]
        .dropna()
        .sample(200_000, random_state=0)
    )
    print("Spearman corr with z20 (train 200k sample):")
    print(sub.corr(method="spearman")["z20"].round(3).to_dict())
    neutral = z20.abs() < 1.0
    for nm in ("clv3", "clv5_window", "wick_diff1"):
        print(fmt_row(eval_signal(feats[nm][neutral], tgt[neutral], q=0.02,
                                  name=f"{nm} | z20-neutral")))

    print("\n--- hour interaction (clv3, q=0.02) ---")
    for sess, hrs in [("act7-16", range(7, 17)), ("asia0-6", range(0, 7))]:
        mk = hour_mask(df.index, list(hrs))
        print(fmt_row(eval_signal(feats["clv3"][mk], tgt[mk], q=0.02,
                                  name=f"clv3 {sess}")))
    print("clv3 horizons:", eval_horizons(feats["clv3"], df, PAIR, q=0.02))

    print("\n" + "=" * 88)
    print("confluence: z20 extreme x CLV (comb25)")
    print("=" * 88)
    ztr = z20[tr].dropna()
    z_lo_thr = ztr.quantile(0.02)
    zone_lo = z20 <= z_lo_thr
    clv_thr25 = clv[zone_lo & tr].dropna().quantile(0.25)
    print(f"thresholds (train only): z20<={z_lo_thr:.3f}, clv<={clv_thr25:.3f}")
    binary_report("comb25_long = z20<=q02 & clv1<=zone-q25",
                  zone_lo & (clv <= clv_thr25), tgt, df)
    z_hi_thr = ztr.quantile(0.98)
    zone_hi = z20 >= z_hi_thr
    clv_thr75 = clv[zone_hi & tr].dropna().quantile(0.75)
    binary_report("comb75_short = z20>=q98 & clv1>=zone-q75",
                  zone_hi & (clv >= clv_thr75), tgt, df)

    print("\n" + "=" * 88)
    print("bear marubozu x range expansion")
    print("=" * 88)
    b98 = body.abs()[tr].dropna().quantile(0.98)
    for rmult in (1.5, 2.0, 2.5):
        binary_report(
            f"bear_maru rng>{rmult} = |body|/rng>={b98:.3f} & close<open & rng>{rmult}x avg20",
            (body.abs() >= b98) & (c < o) & (rng_rel > rmult), tgt, df,
        )
    binary_report("bull_maru rng>2 (mirror)",
                  (body.abs() >= b98) & (c > o) & (rng_rel > 2), tgt, df)

    print("\n--- gap_pips: rollover decomposition (dead end) ---")
    norol = pd.Series(~np.isin(df.index.hour, ROLLOVER_HOURS), index=df.index)
    print(fmt_row(eval_signal(feats["gap_pips"][norol], tgt[norol], q=0.02,
                              name="gap_pips ex-rollover")))
    print(fmt_row(eval_signal(feats["gap_pips"][~norol], tgt[~norol], q=0.02,
                              name="gap_pips rollover-only")))

    # ---------- M1 ----------
    print("\n" + "=" * 88)
    print("M1 re-evaluation of top candidates")
    print("=" * 88)
    df1, tgt1, pip1 = load_xy(PAIR, "M1")
    f1 = candle_features(df1, pip1)
    c1, o1, l1, h1 = df1["close"], df1["open"], df1["low"], df1["high"]
    sma, std = c1.rolling(20).mean(), c1.rolling(20).std()
    z20m1 = (c1 - sma) / std.where(std > 0)
    tr1 = df1.index < SPLIT
    for nm in ("clv1", "clv3", "clv5_window", "wick_diff1"):
        print(fmt_row(eval_signal(f1[nm], tgt1, q=0.02, name=f"M1 {nm}")))
    print(fmt_row(eval_signal(z20m1, tgt1, q=0.02, name="M1 z20 (benchmark)")))
    ztr1 = z20m1[tr1].dropna()
    zone1 = z20m1 <= ztr1.quantile(0.02)
    clv1_thr = f1["clv1"][zone1 & tr1].dropna().quantile(0.25)
    binary_report("M1 comb25_long", zone1 & (f1["clv1"] <= clv1_thr), tgt1, df1)
    b98_1 = f1["body_abs1"][tr1].dropna().quantile(0.98)
    binary_report(
        "M1 bear_maru rng>2",
        (f1["body_abs1"] >= b98_1) & (c1 < o1) & (f1["range_rel20"] > 2), tgt1, df1,
    )


if __name__ == "__main__":
    main()
