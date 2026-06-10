"""nb_volume_activity — 次足予測ファミリー「ティックボリュームと活動量」(EURUSD M5/M1)。

結論(2026-06 実測):
1. ボリューム系の見かけ上強いエッジ(閑散×z20極値ロング +1.5pips@h21、効率リバーサル等)は
   ほぼ全て UTC 21-22時(NYクローズ/ロールオーバー)に集中する。bid建てデータでスプレッド拡大が
   bid を機械的に押し下げて「反発」に見えるだけの、約定不能アーティファクトの可能性が濃厚。
   署名: h21 でロング側 +1.49p(median +0.90)に対しショート側 -0.23p と強い非対称。
2. ロールオーバー(20-22時)除外後に生き残るのは唯一:
   「z20 極値の平均回帰はボリューム高位(vrank100>=0.8)のときだけ働く」(クライマックス反転)。
   midvol の z20 極値は test でほぼゼロ。volume は『どの極値をフェードすべきか』のフィルタ。
   単体エッジは hi(ショート) -0.24p/t≈-3、h5 累積 -0.57p 止まりでコスト 0.6p には届かない。
3. M1 では同方向だが 1 桁小さく実用外。

実行: uv run python -m research.experiments.nb_volume_activity
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

ROLL_HOURS = [20, 21, 22]  # NYクローズ/ロールオーバー窓(UTC)


def tstat(yy: pd.Series) -> float:
    return float(yy.mean() / (yy.std() / np.sqrt(len(yy)))) if len(yy) > 2 else float("nan")


def zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_feats(df: pd.DataFrame, pip: float) -> dict[str, pd.Series]:
    c, v = df["close"], df["volume"]
    ret1p = c.diff() / pip
    ret3p = c.diff(3) / pip
    ret5p = c.diff(5) / pip
    vz100 = zscore(v, 100)
    vrank100 = v.rolling(100).rank(pct=True)
    vtrend = v.rolling(10).mean() / v.rolling(100).mean()
    z20 = zscore(c, 20)

    eff3 = ret3p.abs() / (v.rolling(3).sum() + 1)
    eff3z = zscore(eff3, 100)
    rng = (df["high"] - df["low"]) / pip
    vpr = v / (rng + 0.5)  # volume per range = absorption
    vprz = zscore(vpr, 100)

    return {
        "vz100 (directionless)": vz100,
        "vrank100 (directionless)": vrank100,
        "vtrend10/100 (directionless)": vtrend,
        "sgn1_x_vz100 (spike x sign)": np.sign(ret1p) * vz100,
        "sgn5_x_vz100": np.sign(ret5p) * vz100,
        "effrev_k3 = sgn3 x z(|r3|/vol3)": np.sign(ret3p) * eff3z,
        "sgn1_x_vprz (absorption)": np.sign(ret1p) * vprz,
        "z20 (benchmark)": z20,
        "quiet_z = z20*(1.2-vrank)": z20 * (1.2 - vrank100),
        "climax_z = z20*vrank100": z20 * vrank100,
    }


def main() -> None:
    # ---------------- M5 ----------------
    df, tgt, pip = load_xy("EURUSD", "M5")
    c, v = df["close"], df["volume"]
    te = c.index >= SPLIT
    tr = ~te
    no_roll = ~hour_mask(c.index, ROLL_HOURS)
    asia = hour_mask(c.index, list(range(0, 7)))
    act = hour_mask(c.index, list(range(7, 17)))

    feats = build_feats(df, pip)
    z20 = feats["z20 (benchmark)"]
    climax = feats["climax_z = z20*vrank100"]
    vrank100 = v.rolling(100).rank(pct=True)

    print("=" * 90)
    print("[1] M5 battery — full sample (rollover込み), q=0.02")
    for name, f in feats.items():
        print(fmt_row(eval_signal(f, tgt, q=0.02, name=name)))

    print()
    print("[2] ロールオーバー・アーティファクトの証明")
    zlo = z20[tr].quantile(0.05)
    zhi = z20[tr].quantile(0.95)
    vlo = vrank100 <= 0.3
    mb_long = (z20 <= zlo) & vlo & te
    for h in range(17, 24):
        yy = tgt[mb_long & (c.index.hour == h)].dropna()
        if len(yy) > 30:
            print(f"  z20<q05&lowvol long h{h:02d}: {yy.mean():+.3f}p (med {yy.median():+.2f}) t={tstat(yy):+.1f} n={len(yy)}")
    yy = tgt[(z20 >= zhi) & vlo & (c.index.hour == 21) & te].dropna()
    print(f"  対照: short側 h21       : {yy.mean():+.3f}p (med {yy.median():+.2f}) t={tstat(yy):+.1f} n={len(yy)}  <- 非対称=bidアーティファクト署名")
    yy = tgt[mb_long & no_roll].dropna()
    print(f"  20-22時除外後の同シグナル: {yy.mean():+.3f}p t={tstat(yy):+.1f} n={len(yy)}  <- 消滅")

    print()
    print("[3] M5 battery — 20-22時除外, q=0.02 (生き残り判定)")
    for name, f in feats.items():
        print(fmt_row(eval_signal(f[no_roll], tgt[no_roll], q=0.02, name=name)))

    print()
    print("[4] クライマックス反転の核心: z20>q95(train) を vol 三分位で分解 (no_roll)")
    vmid = (vrank100 > 0.3) & (vrank100 < 0.8)
    vhi = vrank100 >= 0.8
    for side, base0 in [("short z20>q95", z20 >= zhi), ("long  z20<q05", z20 <= zlo)]:
        for vn, vm in [("lowvol", vlo), ("midvol", vmid), ("highvol", vhi)]:
            ytr = tgt[base0 & vm & tr & no_roll].dropna()
            yte = tgt[base0 & vm & te & no_roll].dropna()
            print(
                f"  {side} & {vn:<7} train {ytr.mean():+.3f}p (n={len(ytr)}) | "
                f"test {yte.mean():+.3f}p t={tstat(yte):+.1f} n={len(yte)}"
            )

    print()
    print("[5] climax_z q sweep (no_roll)")
    for q in [0.02, 0.05, 0.10]:
        r = eval_signal(climax[no_roll], tgt[no_roll], q=q, name=f"climax_z q={q}")
        print(fmt_row(r), f"| sig/day lo={r['lo_sig_per_day']:.1f} hi={r['hi_sig_per_day']:.1f}")

    print()
    print("[6] horizons (test累積pips, no_roll, q=0.02)")
    print("  climax_z:", eval_horizons(climax[no_roll], df, "EURUSD", q=0.02))
    print("  z20 bench:", eval_horizons(z20[no_roll], df, "EURUSD", q=0.02))

    print()
    print("[7] セッション別 (q=0.05 within mask, no_roll)")
    for sname, sm in [("asia[0-6]", asia & no_roll), ("active[7-16]", act & no_roll)]:
        print(fmt_row(eval_signal(climax[sm], tgt[sm], q=0.05, name=f"climax | {sname}")))

    print()
    print("[8] 年別安定性 (climax, no_roll, 閾値はtrain q=0.02)")
    fnr, tnr = climax[no_roll], tgt[no_roll]
    trn = fnr.index < SPLIT
    lo, hi = fnr[trn].quantile(0.02), fnr[trn].quantile(0.98)
    for yr in [2023, 2024, 2025, 2026]:
        ym = fnr.index.year == yr
        for tag, mb in [("lo(long)", fnr <= lo), ("hi(short)", fnr >= hi)]:
            yy = tnr[mb & ym].dropna()
            if len(yy) > 30:
                print(f"  {yr} {tag:<9}: {yy.mean():+.3f}p t={tstat(yy):+.1f} n={len(yy)}")

    # ---------------- M1 ----------------
    print()
    print("=" * 90)
    print("[9] M1 再評価 (no_roll, q=0.02) — 生き残り上位のみ")
    df1, tgt1, pip1 = load_xy("EURUSD", "M1")
    c1, v1 = df1["close"], df1["volume"]
    nr1 = ~hour_mask(c1.index, ROLL_HOURS)
    z20_1 = zscore(c1, 20)
    climax1 = z20_1 * v1.rolling(100).rank(pct=True)
    for name, f in [("z20 bench M1", z20_1), ("climax_z M1", climax1)]:
        print(fmt_row(eval_signal(f[nr1], tgt1[nr1], q=0.02, name=name)))
    print("  M1 climax horizons:", eval_horizons(climax1[nr1], df1, "EURUSD", q=0.02))


if __name__ == "__main__":
    main()
