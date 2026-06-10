"""nb_momrev — 次足予測: モメンタム/リバーサル(系列依存)ファミリーの探索ログ。

EURUSD M5/M1。直近100本以内の系列依存特徴量(過去リターン/z-score/ラン長/
Kaufman ER/加速度/ショック足)で「次足 close-to-close 変化(pips)」を予測する。
評価は research/lab/nextbar_common.py の標準プロトコル(train<2023-01-01<test、
極値ビン閾値は train 分位、IC は Spearman)。

主な結論(実測):
- 全特徴量で方向は「反転(平均回帰)」。継続(モメンタム)方向のエッジは無い。
- ER 仮説は逆: レンジ(低ER)でなく「高ER=効率的な一方向の動き」の直後ほど反転が強い。
- 最強: z10 の下側 × ER10 上位1/3 (= 効率的な急落でz10極小) → 次足 +0.54p (t=5.0)。
  h20 まで伸び続け +1.13p。全11年プラス。活発時間帯(7-16 UTC)で +0.55p。
- M1 は同じ構造だが大きさが 1/5 以下(コスト 0.6p に遠く届かない)。M5 が主戦場。

実行: uv run python -m research.experiments.nb_momrev
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


def build_feats(df: pd.DataFrame, pip: float) -> dict[str, pd.Series]:
    """直近100本以内の系列依存特徴量(先読みなし: 全て t 時点までの確定値)。"""
    c = df["close"]
    dpips = c.diff() / pip
    std100 = dpips.rolling(100).std()
    feats: dict[str, pd.Series] = {}
    # 過去リターン(素のpips / rolling std 正規化)
    for k in [1, 3, 5, 10, 30, 100]:
        r = c.diff(k) / pip
        feats[f"ret{k}_pips"] = r
        feats[f"ret{k}_norm"] = r / (std100 * np.sqrt(k))
    # z-score
    for k in [10, 20, 50, 100]:
        feats[f"z{k}"] = (c - c.rolling(k).mean()) / c.rolling(k).std()
    # 連続陽線/陰線ラン長(符号付き)
    sgn = np.sign(dpips).fillna(0)
    grp = (sgn != sgn.shift()).cumsum()
    run = sgn.groupby(grp).cumcount() + 1
    feats["runlen_signed"] = (run * sgn).where(sgn != 0)
    # Kaufman 効率比と「ER×直近10本リターン符号」相互作用
    for k in [10, 30, 100]:
        er = c.diff(k).abs() / (dpips.abs().rolling(k).sum() * pip)
        feats[f"er{k}"] = er
        feats[f"er{k}_x_sign10"] = er * np.sign(c.diff(10))
    # 加速度(直近5本 − その前5本)
    r5 = c.diff(5) / pip
    feats["accel_5m5"] = r5 - r5.shift(5)
    feats["accel_norm"] = (r5 - r5.shift(5)) / (std100 * np.sqrt(5))
    # 大きな1本足(std100 正規化ショック)
    feats["shock1"] = dpips / std100
    return feats


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    # ---------- M5: フルバッテリー ----------
    df, tgt, pip = load_xy("EURUSD", "M5")
    feats = build_feats(df, pip)
    tr = df.index < SPLIT

    section("M5 battery q=0.02")
    for name, f in feats.items():
        print(fmt_row(eval_signal(f, tgt, q=0.02, name=name)))

    # ---------- ER レジーム条件付け(閾値は train 三分位) ----------
    section("M5 ER-regime conditioning (z/ret within ER terciles)")
    for ername in ["er10", "er30"]:
        er = feats[ername]
        q1, q2 = er[tr].quantile(1 / 3), er[tr].quantile(2 / 3)
        for regime, mask in [
            ("loER", er <= q1),
            ("midER", (er > q1) & (er <= q2)),
            ("hiER", er > q2),
        ]:
            for fname in ["z10", "z20", "ret3_norm"]:
                r = eval_signal(
                    feats[fname].where(mask), tgt, q=0.02,
                    name=f"{fname}|{ername}:{regime}",
                )
                print(fmt_row(r))

    hiER10 = feats["er10"] > feats["er10"][tr].quantile(2 / 3)
    z10_er = feats["z10"].where(hiER10)

    # ---------- q 感度(リーダーのみ) ----------
    section("M5 q sensitivity (leaders)")
    leaders = [("z10", feats["z10"]), ("z20", feats["z20"]), ("z50", feats["z50"]),
               ("ret3_norm", feats["ret3_norm"]), ("shock1", feats["shock1"]),
               ("z10|hiER10", z10_er)]
    for q in [0.005, 0.01, 0.02, 0.05, 0.10]:
        for fname, f in leaders:
            print(fmt_row(eval_signal(f, tgt, q=q, name=f"{fname} q={q}")))

    # ---------- 減衰カーブ ----------
    section("M5 horizons h=1..20 (test cumulative pips)")
    for fname, f, q in [("z10|hiER10", z10_er, 0.02), ("z10", feats["z10"], 0.02),
                        ("z20", feats["z20"], 0.02), ("z50", feats["z50"], 0.02),
                        ("ret3_norm", feats["ret3_norm"], 0.02),
                        ("shock1", feats["shock1"], 0.005)]:
        h = eval_horizons(f, df, "EURUSD", q=q)
        lo = {k: round(v, 2) for k, v in h["lo"].items()}
        hi = {k: round(v, 2) for k, v in h["hi"].items()}
        print(f"{fname} q={q}: lo {lo} | hi {hi}")

    # ---------- セッションマスク ----------
    section("M5 session masks q=0.02 (act=7-16 UTC, asia=0-6 UTC)")
    act = hour_mask(df.index, list(range(7, 17)))
    asia = hour_mask(df.index, list(range(0, 7)))
    for fname, f in leaders:
        for sname, sm in [("act7-16", act), ("asia0-6", asia)]:
            print(fmt_row(eval_signal(f.where(sm), tgt, q=0.02, name=f"{fname}|{sname}")))

    # ---------- z20 ベンチマークとの直交性 ----------
    section("M5 orthogonality vs z20 (z20-extreme bars excluded)")
    z20 = feats["z20"]
    print("rank corr z10 vs z20:", round(z10_er.rank().corr(z20.rank()), 3))
    z20lo, z20hi = z20[tr].quantile(0.02), z20[tr].quantile(0.98)
    not_ext = (z20 > z20lo) & (z20 < z20hi)
    for fname, f in [("z10", feats["z10"]), ("z10|hiER10", z10_er),
                     ("ret3_norm", feats["ret3_norm"]), ("z50", feats["z50"])]:
        print(fmt_row(eval_signal(f.where(not_ext), tgt, q=0.02,
                                  name=f"{fname}|z20-not-ext")))

    # ---------- 年次安定性(トップシグナル) ----------
    section("M5 z10|hiER10 lo q=0.02 yearly")
    f = z10_er
    m = f.notna() & tgt.notna() & np.isfinite(f)
    ff, yy = f[m], tgt[m]
    lo_thr = ff[ff.index < SPLIT].quantile(0.02)
    sel = ff <= lo_thr
    for yr in range(2016, 2027):
        y = yy[sel & (ff.index.year == yr)]
        if len(y) > 5:
            t = y.mean() / (y.std() / np.sqrt(len(y)))
            print(f"  {yr}: mean {y.mean():+.2f}p t={t:+.1f} n={len(y)}")

    # ---------- M1: 有望上位のみ再評価 ----------
    df1, tgt1, pip1 = load_xy("EURUSD", "M1")
    c1 = df1["close"]
    dp1 = c1.diff() / pip1
    std100_1 = dp1.rolling(100).std()
    z10_1 = (c1 - c1.rolling(10).mean()) / c1.rolling(10).std()
    z20_1 = (c1 - c1.rolling(20).mean()) / c1.rolling(20).std()
    z50_1 = (c1 - c1.rolling(50).mean()) / c1.rolling(50).std()
    ret3n_1 = (c1.diff(3) / pip1) / (std100_1 * np.sqrt(3))
    shock1_1 = dp1 / std100_1
    er10_1 = c1.diff(10).abs() / (dp1.abs().rolling(10).sum() * pip1)
    tr1 = df1.index < SPLIT
    z10er_1 = z10_1.where(er10_1 > er10_1[tr1].quantile(2 / 3))

    section("M1 promotion (top features)")
    for q in [0.005, 0.02]:
        for fname, f in [("z10", z10_1), ("z20", z20_1), ("z50", z50_1),
                         ("ret3_norm", ret3n_1), ("shock1", shock1_1),
                         ("z10|hiER10", z10er_1)]:
            print(fmt_row(eval_signal(f, tgt1, q=q, name=f"M1 {fname} q={q}")))

    section("M1 horizons (test)")
    for fname, f, q in [("z10|hiER10", z10er_1, 0.02), ("ret3_norm", ret3n_1, 0.02),
                        ("shock1", shock1_1, 0.005)]:
        h = eval_horizons(f, df1, "EURUSD", q=q)
        lo = {k: round(v, 2) for k, v in h["lo"].items()}
        hi = {k: round(v, 2) for k, v in h["hi"].items()}
        print(f"M1 {fname} q={q}: lo {lo} | hi {hi}")


if __name__ == "__main__":
    main()
