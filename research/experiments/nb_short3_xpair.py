"""nb_short3_xpair — ショート側候補3本の7ペアM5クロス再現(敵対的検証)。

検証対象(いずれも train 98% 分位以上 = 上昇行き過ぎ → 次足ショート):
  (1) z50_hi      : z50 = (close - SMA50) / std50
  (2) ret3_norm_hi: (close.diff(3)/pip) / (rolling_std(close.diff()/pip,100)*sqrt(3))
  (3) bbwp_z20_hi : bbwp = (4*std20/SMA20) の trailing100 rolling rank(pct) が
                    train 2/3 分位以上のバーに限定した z20 の train 98% 分位以上

プロトコル(research/lab/nextbar_common.py 準拠):
- train < 2023-01-01 <= test。閾値(98%分位・bbwp 2/3分位)は各ペアの train のみ。
- ターゲット=次足 close-to-close 変化(pips)。週末等ギャップ直前バーは除外。
- 全評価を「フル」と「UTC 20-23時エントリー除外」(ロールオーバー・アーティファクト
  監査)の両方で出す。ショートは BID スプレッド拡大で逆に不利に出るはずなので、
  除外で改善するかも見る。
- 日曜オープン直後(ギャップ後 6 本)除外の感度も EURUSD で出す。
- 減衰: h1/h3/h5/h10/h20 の test 累積平均 pips。h3-h5 の |累積| がペア別往復コスト
  を超えるペア数を数える。
- EURUSD test での 3 シグナルの Jaccard 重複率(独立部品か言い換えか)。

実行: uv run python -m research.experiments.nb_short3_xpair
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import SPLIT, load_xy

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
COSTS = {  # 往復コスト pips
    "EURUSD": 0.6, "USDJPY": 0.7, "GBPUSD": 0.9, "AUDUSD": 0.8,
    "USDCHF": 1.0, "USDCAD": 1.2, "NZDUSD": 1.4,
}
ROLLOVER_HOURS = {20, 21, 22, 23}
HORIZONS = (1, 3, 5, 10, 20)
Q = 0.98


def build_signals(df: pd.DataFrame, pip: float, tr: pd.Series) -> dict[str, pd.Series]:
    """3 候補の特徴量(全て t 時点までの確定値のみ)。bbwp 分位は train のみ。"""
    c = df["close"]
    dpips = c.diff() / pip
    std100 = dpips.rolling(100).std()
    z50 = (c - c.rolling(50).mean()) / c.rolling(50).std()
    ret3n = (c.diff(3) / pip) / (std100 * np.sqrt(3))
    sma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    z20 = (c - sma20) / sd20
    bbwp = (4 * sd20 / sma20).rolling(100).rank(pct=True)
    bb23 = bbwp[tr].quantile(2 / 3)
    return {"z50_hi": z50, "ret3_norm_hi": ret3n, "bbwp_z20_hi": z20.where(bbwp >= bb23)}


def tstat(y: pd.Series) -> tuple[float, float, int]:
    y = y.dropna()
    n = len(y)
    if n < 3 or y.std() == 0:
        return (float(y.mean()) if n else np.nan, np.nan, n)
    return float(y.mean()), float(y.mean() / (y.std() / np.sqrt(n))), n


def sunday_open_mask(index: pd.DatetimeIndex, nbars: int = 6) -> np.ndarray:
    """週明け/ギャップ直後 nbars 本を True(除外対象)に。"""
    step = index.to_series().diff()
    gap = (step > step.median() * 3).to_numpy()
    bad = np.zeros(len(index), dtype=bool)
    for off in range(nbars):
        bad[off:] |= gap[: len(index) - off] if off else gap
    return bad


def eval_pair(pair: str) -> dict[str, dict]:
    df, tgt, pip = load_xy(pair, "M5")
    c = df["close"]
    tr = pd.Series(df.index < SPLIT, index=df.index)
    te = ~tr
    sigs = build_signals(df, pip, tr)
    hts = {h: c.diff(h).shift(-h) / pip for h in HORIZONS}
    days = max((df.index[te][-1] - df.index[te][0]).days, 1)
    roll = pd.Series(np.isin(df.index.hour, list(ROLLOVER_HOURS)), index=df.index)

    out: dict[str, dict] = {}
    for name, f in sigs.items():
        thr = f[tr].quantile(Q)
        sel = (f >= thr) & tgt.notna()  # ギャップ直前バーは除外
        sel_te = sel & te
        sel_x = sel_te & ~roll
        r: dict = {"pair": pair, "thr": float(thr), "cost": COSTS[pair]}
        r["full_mean"], r["full_t"], r["full_n"] = tstat(tgt[sel_te])
        r["x_mean"], r["x_t"], r["x_n"] = tstat(tgt[sel_x])
        r["per_day"] = r["full_n"] / days
        for tag, s in [("full", sel_te), ("x", sel_x)]:
            idx = s[s].index
            r[f"{tag}_h"] = {h: float(ht.reindex(idx).mean()) for h, ht in hts.items()}
        out[name] = r
    return out


def main() -> None:
    results: dict[str, dict[str, dict]] = {p: eval_pair(p) for p in PAIRS}

    for sig in ["z50_hi", "ret3_norm_hi", "bbwp_z20_hi"]:
        print(f"\n=== {sig} (short @ train q={Q}) — M5 test (>=2023) ===")
        print(f"{'pair':<8} {'full mean(t,n)':>26} {'/day':>5} "
              f"{'excl20-23 mean(t,n)':>26} {'cost':>5}  decay_full h1/h3/h5/h10/h20 | excl")
        beat_full = beat_x = 0
        for p in PAIRS:
            r = results[p][sig]
            dh = "/".join(f"{r['full_h'][h]:+.2f}" for h in HORIZONS)
            dx = "/".join(f"{r['x_h'][h]:+.2f}" for h in HORIZONS)
            print(f"{p:<8} {r['full_mean']:+7.2f}p (t={r['full_t']:+5.1f}, n={r['full_n']:>5}) "
                  f"{r['per_day']:5.1f} {r['x_mean']:+7.2f}p (t={r['x_t']:+5.1f}, n={r['x_n']:>5}) "
                  f"{r['cost']:5.1f}  {dh} | {dx}")
            # ショートなので h3/h5 の負の累積の絶対値がコスト超えなら「コスト超え」
            if max(-r["full_h"][3], -r["full_h"][5]) > r["cost"]:
                beat_full += 1
            if max(-r["x_h"][3], -r["x_h"][5]) > r["cost"]:
                beat_x += 1
        print(f"-> h3-h5 cumulative beats round-trip cost: full {beat_full}/7, "
              f"excl20-23 {beat_x}/7")

    # ---- ロールオーバー監査: full vs excl の差(ショートは除外で改善するはず) ----
    print("\n=== rollover audit (excl20-23 minus full, next-bar mean pips) ===")
    for sig in ["z50_hi", "ret3_norm_hi", "bbwp_z20_hi"]:
        diffs = {p: results[p][sig]["x_mean"] - results[p][sig]["full_mean"] for p in PAIRS}
        s = " ".join(f"{p[:6]}:{d:+.2f}" for p, d in diffs.items())
        n_improve = sum(d < 0 for d in diffs.values())  # ショート: より負=改善
        print(f"{sig:<14} {s}  (improved-for-short {n_improve}/7)")

    # ---- EURUSD: Jaccard 重複率 ----
    print("\n=== EURUSD test signal overlap (Jaccard) ===")
    df, tgt, pip = load_xy("EURUSD", "M5")
    tr = pd.Series(df.index < SPLIT, index=df.index)
    sigs = build_signals(df, pip, tr)
    masks = {}
    for name, f in sigs.items():
        thr = f[tr].quantile(Q)
        masks[name] = ((f >= thr) & tgt.notna() & ~tr).fillna(False)
    names = list(masks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = masks[names[i]], masks[names[j]]
            inter, union = int((a & b).sum()), int((a | b).sum())
            print(f"{names[i]} vs {names[j]}: J={inter/union:.3f} "
                  f"(inter={inter}, |A|={int(a.sum())}, |B|={int(b.sum())})")

    # ---- EURUSD: 日曜オープン直後(ギャップ後6本)除外の感度 ----
    print("\n=== EURUSD sunday-open sensitivity (also excl 6 bars after gap) ===")
    sun = pd.Series(sunday_open_mask(df.index), index=df.index)
    roll = pd.Series(np.isin(df.index.hour, list(ROLLOVER_HOURS)), index=df.index)
    for name, f in sigs.items():
        thr = f[tr].quantile(Q)
        sel = (f >= thr) & tgt.notna() & ~tr & ~roll
        m0, t0, n0 = tstat(tgt[sel])
        m1, t1, n1 = tstat(tgt[sel & ~sun])
        print(f"{name:<14} excl20-23 {m0:+.2f}p(t={t0:+.1f},n={n0}) "
              f"-> +sunopen-excl {m1:+.2f}p(t={t1:+.1f},n={n1})")


if __name__ == "__main__":
    main()
