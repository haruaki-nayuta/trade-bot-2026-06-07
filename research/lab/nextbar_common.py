"""次足エッジ探索の共通評価プロトコル(M1/M5 直近100本 → 次の足)。

全エージェント共通のルール:
- 特徴量はバー t までの確定値のみ(rolling は現在足を含んでよいが shift(-) は禁止)。
- ターゲットは「次の足の close-to-close 変化(pips)」。週末等のギャップ直前バーは除外。
- train = 2023-01-01 より前 / test = それ以降。極値ビンの閾値は train 分位で固定し
  test に適用する(test 分位を使うとリーク)。
- 判断基準: test での極値ビン条件付き平均(pips)を往復コスト(EURUSD=0.6pips)と比較する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxlab import load
from fxlab.config import pip_size, spread_pips

SPLIT = pd.Timestamp("2023-01-01", tz="UTC")
HORIZONS = (1, 3, 5, 10, 20)


def load_xy(pair: str = "EURUSD", tf: str = "M5"):
    """価格 df・次足ターゲット(pips)・pip幅 を返す。ギャップ直前バーの tgt は NaN。"""
    df = load(pair, tf)
    pip = pip_size(pair)
    tgt = df["close"].diff().shift(-1) / pip
    step = df.index.to_series().diff().shift(-1)
    ok = step <= step.median() * 3
    return df, tgt.where(ok), pip


def horizon_targets(df: pd.DataFrame, pair: str, horizons=HORIZONS) -> dict[int, pd.Series]:
    """h 本先までの累積変化(pips)。エッジの持続/減衰を見る用。"""
    pip = pip_size(pair)
    c = df["close"]
    return {h: c.diff(h).shift(-h) / pip for h in horizons}


def eval_signal(feat: pd.Series, tgt: pd.Series, q: float = 0.02, name: str = "") -> dict:
    """特徴量 1 本の標準評価。

    返り値: Spearman IC (train/test) と、train 分位 q で切った上下極値ビンの
    train/test 条件付き平均 pips・t値・件数・日あたりシグナル数。
    """
    m = feat.notna() & tgt.notna() & np.isfinite(feat) & np.isfinite(tgt)
    f, y = feat[m], tgt[m]
    tr = f.index < SPLIT
    te = ~tr
    out: dict = {"name": name, "q": q, "n_train": int(tr.sum()), "n_test": int(te.sum())}
    out["ic_train"] = float(f[tr].rank().corr(y[tr].rank()))
    out["ic_test"] = float(f[te].rank().corr(y[te].rank()))
    lo, hi = f[tr].quantile(q), f[tr].quantile(1 - q)
    out["thr_lo"], out["thr_hi"] = float(lo), float(hi)
    for tag, mb in [("lo", f <= lo), ("hi", f >= hi)]:
        for st, ms in [("train", tr), ("test", te)]:
            yy = y[mb & ms]
            n = len(yy)
            out[f"{tag}_{st}_mean_pips"] = float(yy.mean()) if n else np.nan
            out[f"{tag}_{st}_t"] = (
                float(yy.mean() / (yy.std() / np.sqrt(n))) if n > 2 and yy.std() > 0 else np.nan
            )
            out[f"{tag}_{st}_n"] = n
    if te.sum():
        days = max((f[te].index[-1] - f[te].index[0]).days, 1)
        out["lo_sig_per_day"] = out["lo_test_n"] / days
        out["hi_sig_per_day"] = out["hi_test_n"] / days
    return out


def eval_horizons(
    feat: pd.Series, df: pd.DataFrame, pair: str, q: float = 0.02
) -> dict:
    """train 分位の極値ビンについて、test での h 本先累積平均 pips(減衰カーブ)。"""
    hts = horizon_targets(df, pair)
    m = feat.notna() & np.isfinite(feat)
    f = feat[m]
    tr = f.index < SPLIT
    lo, hi = f[tr].quantile(q), f[tr].quantile(1 - q)
    te_idx = f.index[(~tr)]
    out: dict = {}
    for tag, sel in [("lo", f[te_idx] <= lo), ("hi", f[te_idx] >= hi)]:
        idx = sel[sel].index
        out[tag] = {
            f"h{h}": float(ht.reindex(idx).mean()) for h, ht in hts.items()
        }
    return out


def hour_mask(index: pd.DatetimeIndex, hours: list[int]) -> pd.Series:
    """UTC 時間帯マスク(セッション相互作用の検証用)。"""
    return pd.Series(np.isin(index.hour, hours), index=index)


def fmt_row(r: dict) -> str:
    return (
        f"{r['name']:<36} IC tr/te {r['ic_train']:+.4f}/{r['ic_test']:+.4f} | "
        f"lo te {r['lo_test_mean_pips']:+.2f}p (t={r['lo_test_t']:+.1f}, n={r['lo_test_n']}) | "
        f"hi te {r['hi_test_mean_pips']:+.2f}p (t={r['hi_test_t']:+.1f}, n={r['hi_test_n']})"
    )
