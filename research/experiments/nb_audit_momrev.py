"""nb_audit_momrev — チャンピオン(z10_lo×hiER10)のリーク・プロトコル監査。

敵対的検証: nb_momrev.py のチャンピオンを定義から独立再実装し、
プロトコル(research/lab/nextbar_common.py)のリーク経路と取引現実性を監査する。

監査結論(実測、詳細は各セクションの出力):
1. 再実装一致性: 「z10 < train q02」の素直な実装は test +0.32p/2.44回日で不一致。
   オリジナルは z10.where(hiER10) を eval_signal に渡すため、q02 閾値が
   「hiERバー内の条件付き分位」(-2.30) になる。この構成で +0.537p/t=4.96/1.30回日
   を完全再現(レポートの +0.54p/t=5.0/1.3回日 と一致)。リークではないが
   公称定義と実装が乖離(条件付き分位と明記すべき)。
2. ギャップマスク(step=index.diff().shift(-1)): 形式上は未来情報だが、除外548本中
   520本は金曜20時以降(カレンダーで事前に分かる)。マスク無しでも +0.516p で
   実質影響なし。本物のリークではない。ただしライブ運用ではカレンダー規則に置換要。
3. ロールオーバー監査(主軸): UTC21時のシグナルは件数2倍・平均+2.13pで
   BIDスプレッド拡大アーティファクトの形。UTC20-23除外で h1 は +0.537→+0.473p
   (t=3.85) と統計的には生存。だが 0.6p 往復コストを下回る。
   さらに h3/h5/h20 の伸び(公称 h20=+1.13p)は除外後 t=1.2〜2.0 に崩壊
   (h20=+0.71p, t=1.5)。「h20まで伸びる」はロールオーバー時間帯依存。
4. 実行現実性: シグナルはバーt close確定後に判明。open[t+1]-close[t] は
   シグナルバーで平均+0.07p(全バー+0.01p)。t+1 open建てで測ると +0.468p。
   close建ての評価は約0.07p楽観的。
5. クラスタリング: 前バー重複5.4%、平均クラスタ長1.06、独立イベント1.23回/日。
   クラスタ先頭のみでも +0.518p(t=4.7)→ 重複は結論を変えない。
6. 年次: 2021年は除外後 -0.03p、2024年は +0.07p。「全11年プラス」は train 8年を
   含む上に脆い。OOS(2023-)除外後平均 ≈ +0.47p < 0.6p コスト。

総合: リーク・バグは無し(再現は条件付き分位の注記が必要)。ただし
「次足+0.54p」はコスト0.6p未満であり、コスト超えの根拠だった h20 への伸びは
ロールオーバー除外で有意性を失う。現状の z10_lo×hiER10 はネットで取引可能な
エッジとは言えない。

実行: uv run python -m research.experiments.nb_audit_momrev
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import (
    HORIZONS,
    SPLIT,
    eval_horizons,
    eval_signal,
    load_xy,
)


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def tstat(y: pd.Series) -> float:
    n = y.notna().sum()
    return float(y.mean() / (y.std() / np.sqrt(n))) if n > 2 else np.nan


def describe(y: pd.Series, label: str, days: int) -> None:
    n = int(y.notna().sum())
    print(
        f"  {label:<28} mean={y.mean():+.3f}p t={tstat(y):+.2f} "
        f"n={n} sig/day={n / days:.2f}"
    )


def main() -> None:
    df, tgt, pip = load_xy("EURUSD", "M5")
    close, open_ = df["close"], df["open"]
    tr_idx = df.index < SPLIT
    te_idx = ~tr_idx
    days_te = max((df.index[te_idx][-1] - df.index[te_idx][0]).days, 1)

    # ---------- 特徴量(定義から独立再実装) ----------
    z10 = (close - close.rolling(10).mean()) / close.rolling(10).std()
    dpips = close.diff() / pip
    er10 = close.diff(10).abs() / (dpips.abs().rolling(10).sum() * pip)

    # ---------- 1. 再実装一致性 ----------
    section("1. reimplementation: literal definition vs original construction")
    thr_er = er10[tr_idx].quantile(2 / 3)
    hiER10 = er10 > thr_er

    # (a) 公称定義の素直な読み: z10 q02 は無条件 train 分位
    m = z10.notna() & er10.notna() & tgt.notna() & np.isfinite(z10)
    thr_z_uncond = z10[m][df.index[m] < SPLIT].quantile(0.02)
    sig_lit = (z10 <= thr_z_uncond) & hiER10 & tgt.notna()
    describe(tgt[sig_lit & te_idx], f"literal (z thr {thr_z_uncond:.3f})", days_te)

    # (b) オリジナル構成: z10.where(hiER10) → q02 は hiER 内の条件付き分位
    z10_er = z10.where(hiER10)
    r = eval_signal(z10_er, tgt, q=0.02, name="champion(original)")
    print(
        f"  original (cond thr {r['thr_lo']:.3f})    "
        f"mean={r['lo_test_mean_pips']:+.3f}p t={r['lo_test_t']:+.2f} "
        f"n={r['lo_test_n']} sig/day={r['lo_sig_per_day']:.2f}"
    )
    print("  -> claimed +0.54p / t=5.0 / 1.3/day matches (b) only.")

    mc = z10_er.notna() & tgt.notna() & np.isfinite(z10_er)
    thr = z10_er[mc][df.index[mc] < SPLIT].quantile(0.02)
    sig = (z10_er <= thr) & tgt.notna()

    # ---------- 2. ギャップマスクの未来情報監査 ----------
    section("2. gap mask (step=index.diff().shift(-1)) audit")
    step_next = df.index.to_series().diff().shift(-1)
    med = step_next.median()
    excluded = ~(step_next <= med * 3)
    exc = df.index[excluded.values]
    fri = (exc.dayofweek == 4) & (exc.hour >= 20)
    print(
        f"  excluded bars: {int(excluded.sum())} "
        f"({excluded.sum() / len(df) * 100:.3f}%) | Fri>=20UTC(knowable): {int(fri.sum())} "
        f"| mid-week(not knowable): {int((~fri).sum())}"
    )
    tgt_raw = close.diff().shift(-1) / pip
    sig_raw = (z10_er <= thr) & tgt_raw.notna()
    describe(tgt[sig & te_idx], "masked tgt (protocol)", days_te)
    describe(tgt_raw[sig_raw & te_idx], "unmasked tgt", days_te)
    on_gap = sig_raw & te_idx & excluded.values
    print(
        f"  signals on gap bars (test): n={int(on_gap.sum())} "
        f"mean={tgt_raw[on_gap].mean():+.3f}p"
    )

    # ---------- 3. ロールオーバー/日曜オープン除外(判定の主軸) ----------
    section("3. rollover & Sunday-open exclusions (h=1)")
    hrs = df.index.hour
    nox = ~np.isin(hrs, [20, 21, 22, 23])
    step_prev = df.index.to_series().diff()
    post_gap = step_prev > med * 3
    after = post_gap.copy()
    for k in range(1, 6):
        after = after | post_gap.shift(k, fill_value=False)
    describe(tgt[sig & te_idx], "full", days_te)
    describe(tgt[sig & te_idx & nox], "excl UTC20-23", days_te)
    describe(tgt[sig & te_idx & ~after.values], "excl 30min post-gap", days_te)
    describe(tgt[sig & te_idx & nox & ~after.values], "excl both", days_te)
    sel = sig & te_idx
    by_hr = tgt[sel].groupby(df.index[sel].hour).agg(["mean", "size"])
    h21 = by_hr.loc[21]
    print(
        f"  hour-21 alone: mean={h21['mean']:+.3f}p n={int(h21['size'])} "
        f"(~{h21['mean'] * h21['size'] / (tgt[sel].sum()) * 100:.0f}% of total test pips)"
    )

    # ---------- 4. h-カーブ: eval_horizons整合性とロールオーバー依存 ----------
    section("4. horizon curve: published vs gap-free vs excl UTC20-23")
    h_pub = eval_horizons(z10_er, df, "EURUSD", q=0.02)["lo"]
    print(f"  eval_horizons(published): { {k: round(v, 3) for k, v in h_pub.items()} }")
    gap_before = excluded
    for label, smask in [("full", sig & te_idx), ("excl UTC20-23", sig & te_idx & nox)]:
        row = []
        for hh in HORIZONS:
            ht = close.diff(hh).shift(-hh) / pip
            crosses = pd.concat(
                [gap_before.shift(-k) for k in range(hh)], axis=1
            ).any(axis=1)
            y = ht[smask & ~crosses.values]
            row.append(f"h{hh}={y.mean():+.3f}p(t={tstat(y):+.1f})")
        print(f"  {label:<14} " + " ".join(row))

    # ---------- 5. 実行現実性: t close vs t+1 open ----------
    section("5. execution: open[t+1]-close[t] and next-open entry")
    gap_open = (open_.shift(-1) - close) / pip
    for label, g in [("all test bars", gap_open[te_idx & tgt.notna()]),
                     ("signal bars", gap_open[sig & te_idx])]:
        print(
            f"  {label:<14} mean={g.mean():+.4f}p median={g.median():+.4f} "
            f"std={g.std():.3f} p5/p95={g.quantile(.05):+.2f}/{g.quantile(.95):+.2f} "
            f"|g|>0.5p: {(g.abs() > 0.5).mean() * 100:.1f}%"
        )
    alt = (close.shift(-1) - open_.shift(-1)) / pip
    describe(tgt[sig & te_idx], "entry at t close", days_te)
    describe(alt[sig & te_idx], "entry at t+1 open", days_te)

    # ---------- 6. クラスタリング(独立イベント換算) ----------
    section("6. signal clustering")
    s = sig & pd.Series(te_idx, index=df.index)
    n_sig = int(s.sum())
    prev1 = s & s.shift(1, fill_value=False)
    starts = s & ~s.shift(1, fill_value=False)
    print(
        f"  overlap w/ prev bar: {int(prev1.sum())}/{n_sig} = "
        f"{prev1.sum() / n_sig * 100:.1f}% | independent clusters: "
        f"{int(starts.sum())} -> {starts.sum() / days_te:.2f}/day"
    )
    describe(tgt[starts], "first-bar-of-cluster only", days_te)

    # ---------- 7. 年次(OOSは2023-) ----------
    section("7. yearly h=1 (full | excl UTC20-23) — 2023+ is OOS")
    for yr in range(2016, 2027):
        ym = df.index.year == yr
        a, b = tgt[sig & ym], tgt[sig & ym & nox]
        if a.notna().sum() > 5:
            print(
                f"  {yr}: full {a.mean():+.3f}p (n={int(a.notna().sum())}) | "
                f"excl {b.mean():+.3f}p (n={int(b.notna().sum())})"
            )


if __name__ == "__main__":
    main()
