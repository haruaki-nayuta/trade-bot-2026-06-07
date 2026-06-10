"""nb_seqpattern — 系列パターンマイニング(データドリブン)による次足エッジ探索。

EURUSD M5/M1。直近 k 本の足を記号化し、次足 close-to-close 変化(pips)への
エッジを train(<2023)で網羅探索 → test(>=2023)で検証する。

記号化:
  A: 2値(上昇=1 / 下降・同値=0)、k=4,5,6
  B: train リターン三分位 {0,1,2}、k=3,4
  C: train リターン四分位 {大陰0,小陰1,小陽2,大陽3}、k=3
評価:
  - 素のパターン表: train n>=5000 のパターンを |train t| でランクし上位10だけ test 検証
    (多重検定対策。test の符号残存率を報告)
  - score 特徴量: 各パターンの train 平均pips を全期間にマップ(train-only fit)し、
    research.lab.nextbar_common.eval_signal で標準評価(IC / 極値ビン / horizons / 時間帯)
  - 明示的アーキタイプ: スパイク&ストール / V字反転 / 三段下げ
  - z20 ベンチマークとの相関・z20中立帯での残存(直交性)

実行: uv run python -m research.experiments.nb_seqpattern
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

PAIR = "EURUSD"
ACTIVE = list(range(7, 17))  # UTC 7-16 = London/NY 活発時間帯
ASIA = list(range(0, 7))


# ---------------------------------------------------------------- symbolize
def symbolize(r: pd.Series, mode: str):
    """足のリターン(pips)を記号化。閾値は train のみから。返り値 (symbols, base)。"""
    rt = r[r.index < SPLIT]
    if mode == "A":  # 2値: 上昇=1 / 下降・同値=0
        return (r > 0).astype(float).where(r.notna()), 2
    if mode == "B":  # train 三分位
        q1, q2 = rt.quantile(1 / 3), rt.quantile(2 / 3)
        s = pd.Series(np.select([r <= q1, r <= q2], [0, 1], 2), index=r.index, dtype=float)
        return s.where(r.notna()), 3
    if mode == "C":  # train 四分位 {大陰, 小陰, 小陽, 大陽}
        q1, q2, q3 = rt.quantile(0.25), rt.quantile(0.5), rt.quantile(0.75)
        s = pd.Series(
            np.select([r <= q1, r <= q2, r <= q3], [0, 1, 2], 3), index=r.index, dtype=float
        )
        return s.where(r.notna()), 4
    raise ValueError(mode)


def pattern_code(s: pd.Series, base: int, k: int) -> pd.Series:
    """直近 k 本(t-k+1..t)の記号列を整数コード化。最新足が最下位桁。"""
    code = pd.Series(0.0, index=s.index)
    ok = pd.Series(True, index=s.index)
    for i in range(k):
        si = s.shift(i)
        code += si * (base**i)
        ok &= si.notna()
    return code.where(ok)


def decode(pat: int, base: int, k: int) -> str:
    digs = []
    for _ in range(k):
        digs.append(pat % base)
        pat //= base
    return "".join(str(d) for d in reversed(digs))  # 左=古い足, 右=最新足


# ---------------------------------------------------------------- battery
def battery(r, tgt, mode, k, min_support=5000) -> pd.DataFrame:
    """全パターンの train/test 条件付き平均pips・t値の表(train n>=min_support のみ)。"""
    s, base = symbolize(r, mode)
    code = pattern_code(s, base, k)
    m = code.notna() & tgt.notna()
    c, y = code[m], tgt[m]
    trm = c.index < SPLIT
    st_tr = y[trm].groupby(c[trm].astype(int)).agg(["count", "mean", "std"])
    st_te = y[~trm].groupby(c[~trm].astype(int)).agg(["count", "mean", "std"])
    rows = []
    for pat, row in st_tr.iterrows():
        n, mu, sd = row["count"], row["mean"], row["std"]
        if n < min_support:
            continue
        te = st_te.loc[pat] if pat in st_te.index else None
        rows.append(
            dict(
                fam=f"{mode}k{k}", pat=int(pat), seq=decode(int(pat), base, k),
                tr_n=int(n), tr_mean=float(mu), tr_t=float(mu / (sd / np.sqrt(n))),
                te_n=int(te["count"]) if te is not None else 0,
                te_mean=float(te["mean"]) if te is not None else np.nan,
                te_t=float(te["mean"] / (te["std"] / np.sqrt(te["count"])))
                if te is not None and te["count"] > 2 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def score_feature(r, tgt, mode, k, min_support=1000) -> pd.Series:
    """train の各パターン平均pipsを score として全期間にマップ(train-only fit)。"""
    s, base = symbolize(r, mode)
    code = pattern_code(s, base, k)
    m = code.notna() & tgt.notna() & (code.index < SPLIT)
    mp = tgt[m].groupby(code[m].astype(int)).agg(["count", "mean"])
    return code.map(mp[mp["count"] >= min_support]["mean"])


def top10_report(alldf: pd.DataFrame, label: str):
    top = alldf.reindex(alldf.tr_t.abs().sort_values(ascending=False).index).head(10)
    print(f"\n=== {label}: TOP10 by |train t| (pooled) ===")
    print(top[["fam", "seq", "tr_n", "tr_mean", "tr_t", "te_n", "te_mean", "te_t"]]
          .to_string(index=False, float_format=lambda x: f"{x:+.3f}"))
    surv = float((np.sign(top.tr_mean) == np.sign(top.te_mean)).mean())
    print(f"test 符号残存率: {surv:.0%}")


# ---------------------------------------------------------------- main
def run_m5():
    df, tgt, pip = load_xy(PAIR, "M5")
    r = df["close"].diff() / pip
    combos = [("A", 4), ("A", 5), ("A", 6), ("B", 3), ("B", 4), ("C", 3)]

    # 1) 素のパターン表 + 多重検定規律(top10のみ test 検証)
    alldf = pd.concat([battery(r, tgt, m, k) for m, k in combos], ignore_index=True)
    print(f"[M5] qualifying patterns (train n>=5000): {len(alldf)}")
    top10_report(alldf, "M5")

    # 2) score 特徴量を標準評価
    print("\n=== M5 score features (train-mean map) ===")
    for mode, k in combos:
        feat = score_feature(r, tgt, mode, k)
        for q in (0.02, 0.05, 0.10):
            rr = eval_signal(feat, tgt, q, f"M5_score_{mode}k{k}_q{q}")
            print(fmt_row(rr), f"| sig/day lo {rr['lo_sig_per_day']:.1f} hi {rr['hi_sig_per_day']:.1f}")

    # 3) z20 ベンチ・直交性
    sma = df["close"].rolling(20).mean()
    sd = df["close"].rolling(20).std()
    z20 = (df["close"] - sma) / sd
    print("\n=== z20 benchmark & orthogonality ===")
    print(fmt_row(eval_signal(z20, tgt, 0.02, "z20_bench")))
    tr_idx = df.index < SPLIT
    z_lo, z_hi = z20[tr_idx].quantile(0.25), z20[tr_idx].quantile(0.75)
    neutral = (z20 >= z_lo) & (z20 <= z_hi)
    for mode, k in [("A", 5), ("C", 3)]:
        feat = score_feature(r, tgt, mode, k)
        m = feat.notna() & z20.notna()
        print(f"rank-corr(score_{mode}k{k}, z20) = {feat[m].rank().corr(z20[m].rank()):+.3f}")
        rr = eval_signal(feat.where(neutral), tgt, 0.02, f"score_{mode}k{k}|z20neutral")
        print(fmt_row(rr))

    # 4) アーキタイプ
    print("\n=== archetypes (M5) ===")
    rt = r[tr_idx]
    med_abs = rt.abs().median()
    spike = r.shift(2)
    stall = (r.shift(1).abs() <= med_abs) & (r.abs() <= med_abs)
    feat_ss = spike.where(stall & (spike.abs() >= rt.abs().quantile(0.90)))
    vbot = pd.concat([-r.shift(1), r], axis=1).min(axis=1)
    vtop = pd.concat([r.shift(1), -r], axis=1).min(axis=1)
    feat_v = (vbot.where(vbot > 0, 0.0) - vtop.where(vtop > 0, 0.0)).where(
        r.notna() & r.shift(1).notna()
    )
    sg = [r.shift(i) for i in range(5)]
    down3 = (sg[4] < 0) & (sg[3] > 0) & (sg[2] < 0) & (sg[1] > 0) & (sg[0] < 0)
    up3 = (sg[4] > 0) & (sg[3] < 0) & (sg[2] > 0) & (sg[1] < 0) & (sg[0] > 0)
    depth = -df["close"].diff(5) / pip
    feat_3p = pd.Series(np.nan, index=df.index)
    feat_3p[down3] = depth[down3]
    feat_3p[up3] = depth[up3]
    for nm, ft in [("spike_stall", feat_ss), ("v_reversal", feat_v), ("three_push", feat_3p)]:
        rr = eval_signal(ft, tgt, 0.05, f"{nm}_q.05")
        print(fmt_row(rr), f"| sig/day lo {rr.get('lo_sig_per_day', 0):.2f} hi {rr.get('hi_sig_per_day', 0):.2f}")

    # 5) 上位シグナルの horizons + 時間帯
    print("\n=== top signal deep-dive: score_Ak5 (run-reversal) ===")
    feat = score_feature(r, tgt, "A", 5)
    print("horizons all q.02:", eval_horizons(feat, df, PAIR, 0.02))
    act = hour_mask(df.index, ACTIVE)
    asia = hour_mask(df.index, ASIA)
    for label, mk in [("active7-16", act), ("asia0-6", asia)]:
        rr = eval_signal(feat.where(mk), tgt, 0.02, f"score_Ak5|{label} q.02")
        print(fmt_row(rr))
    rr = eval_signal(feat.where(act), tgt, 0.05, "score_Ak5|active7-16 q.05")
    print(fmt_row(rr), f"| sig/day lo {rr['lo_sig_per_day']:.2f} hi {rr['hi_sig_per_day']:.2f}")
    print("horizons active q.02:", eval_horizons(feat.where(act), df, PAIR, 0.02))
    print("horizons active q.05:", eval_horizons(feat.where(act), df, PAIR, 0.05))


def run_m1():
    df, tgt, pip = load_xy(PAIR, "M1")
    r = df["close"].diff() / pip
    print(f"\n[M1] rows={len(df)}")
    al = pd.concat(
        [battery(r, tgt, m, k) for m, k in [("A", 5), ("A", 6), ("C", 3)]], ignore_index=True
    )
    top10_report(al, "M1")
    print("\n=== M1 score features ===")
    act = hour_mask(df.index, ACTIVE)
    asia = hour_mask(df.index, ASIA)
    for mode, k in [("A", 5), ("A", 6), ("C", 3)]:
        feat = score_feature(r, tgt, mode, k)
        nm = f"M1_score_{mode}k{k}"
        rr = eval_signal(feat, tgt, 0.02, f"{nm}_q.02")
        print(fmt_row(rr), f"| sig/day lo {rr['lo_sig_per_day']:.1f} hi {rr['hi_sig_per_day']:.1f}")
        print("  horizons q.02:", eval_horizons(feat, df, PAIR, 0.02))
        print(" ", fmt_row(eval_signal(feat.where(act), tgt, 0.02, f"{nm}|act")))
        print(" ", fmt_row(eval_signal(feat.where(asia), tgt, 0.02, f"{nm}|asia")))


if __name__ == "__main__":
    run_m5()
    run_m1()
