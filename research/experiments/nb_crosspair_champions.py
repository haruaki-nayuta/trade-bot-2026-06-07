"""nb_crosspair_champions — ロング側チャンピオン2本の7ペア・クロスペア敵対的検証。

検証対象(いずれも EURUSD M5 で発見されたロング側の反転エッジ):
(1) z10lo_x_hiER10: z10=(close-SMA10)/std10。ER10=|close-close.shift(10)|/Σ|diff|(10)。
    ER10 がそのペアの train 上位三分位(>q2/3)のバーに限定し、限定後 z10 が
    train 2%分位以下で買い(次足ロング)。
(2) comb25_long: z20=(close-SMA20)/std20 <= train q0.02 かつ
    CLV=(close-low)/(high-low) <= (z20極値ゾーン内 train q0.25) で買い。

敵対的プロトコル:
- 閾値は全てペアごとに「そのペアの train(<2023-01-01)」から取得(リーク防止)。
- 全評価を「フル」と「UTC 20-23 時エントリー除外」の両方で出す
  (NYクローズ/ロールオーバーの BID スプレッド拡大はロング側エッジを水増しする)。
- EURUSD の (1) は日曜オープン直後(週明け最初の6本)除外感度も確認。
- 減衰 h1/h3/h5/h10/h20 の累積を、ペア別往復コストと比較する。

実行: uv run python -m research.experiments.nb_crosspair_champions
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import SPLIT, load_xy

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
# タスク指定のペア別往復コスト(pips)
COSTS = {
    "EURUSD": 0.6, "USDJPY": 0.7, "GBPUSD": 0.9, "AUDUSD": 0.8,
    "USDCHF": 1.0, "USDCAD": 1.2, "NZDUSD": 1.4,
}
ROLLOVER_HOURS = (20, 21, 22, 23)
HORIZONS = (1, 3, 5, 10, 20)


def stats(y: pd.Series) -> tuple[float, float, int]:
    y = y.dropna()
    n = len(y)
    if n < 3 or y.std() == 0:
        return (float("nan"), float("nan"), n)
    return (float(y.mean()), float(y.mean() / (y.std() / np.sqrt(n))), n)


def build_signals(df: pd.DataFrame, pip: float) -> dict[str, pd.Series]:
    """両シグナルの bool Series(閾値はそのペアの train のみから)。"""
    c, h, l = df["close"], df["high"], df["low"]
    tr = df.index < SPLIT

    # --- (1) z10 lo x hiER10 ---
    z10 = (c - c.rolling(10).mean()) / c.rolling(10).std()
    dabs = c.diff().abs()
    er10 = c.diff(10).abs() / dabs.rolling(10).sum()
    er_thr = er10[tr].quantile(2 / 3)
    hiER = er10 > er_thr
    z_masked = z10.where(hiER)
    z_thr = z_masked[tr].quantile(0.02)
    sig1 = (z_masked <= z_thr).fillna(False)

    # --- (2) comb25_long ---
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    z20 = (c - sma20) / std20.where(std20 > 0)
    rng = (h - l).where((h - l) > 0)
    clv = (c - l) / rng
    z20_thr = z20[tr].dropna().quantile(0.02)
    zone = z20 <= z20_thr
    clv_thr = clv[zone & tr].dropna().quantile(0.25)
    sig2 = (zone & (clv <= clv_thr)).fillna(False)

    return {
        "sig1_z10lo_x_hiER10": sig1,
        "sig2_comb25_long": sig2,
        "_thr": pd.Series(
            {"er_thr": er_thr, "z10_thr": z_thr, "z20_thr": z20_thr, "clv_thr": clv_thr}
        ),
    }


def horizon_cum(df: pd.DataFrame, pip: float, idx: pd.DatetimeIndex) -> dict[int, float]:
    c = df["close"]
    return {
        h: float((c.diff(h).shift(-h) / pip).reindex(idx).mean()) for h in HORIZONS
    }


def eval_pair(pair: str) -> dict:
    df, tgt, pip = load_xy(pair, "M5")
    sigs = build_signals(df, pip)
    te = df.index >= SPLIT
    norol = ~np.isin(df.index.hour, ROLLOVER_HOURS)
    days = max((df.index[te][-1] - df.index[te][0]).days, 1)
    out: dict = {"pair": pair, "cost": COSTS[pair], "thr": sigs["_thr"].to_dict()}

    for key in ("sig1_z10lo_x_hiER10", "sig2_comb25_long"):
        sig = sigs[key]
        res: dict = {}
        # フル(test)
        m_full = sig & te
        res["full_mean"], res["full_t"], res["full_n"] = stats(tgt[m_full])
        res["per_day"] = float(m_full.sum() / days)
        # ロールオーバー除外(test)
        m_ex = m_full & norol
        res["ex_mean"], res["ex_t"], res["ex_n"] = stats(tgt[m_ex])
        # ロールオーバー窓のみ(監査用)
        m_ro = m_full & ~norol
        res["ro_mean"], res["ro_t"], res["ro_n"] = stats(tgt[m_ro])
        # 減衰(test, フル / 除外)
        res["hz_full"] = horizon_cum(df, pip, df.index[m_full])
        res["hz_ex"] = horizon_cum(df, pip, df.index[m_ex])
        # 年次(test以前も含む全期間, 参考)
        yy = tgt[sig].dropna()
        res["yearly"] = {
            int(yr): round(float(g.mean()), 2) for yr, g in yy.groupby(yy.index.year)
        }
        out[key] = res
    return out


def sunday_open_sensitivity(pair: str = "EURUSD") -> dict:
    """(1) の日曜オープン直後(週明け最初の6本=30分)除外感度。"""
    df, tgt, pip = load_xy(pair, "M5")
    sigs = build_signals(df, pip)
    sig = sigs["sig1_z10lo_x_hiER10"]
    te = df.index >= SPLIT
    gap = df.index.to_series().diff() > pd.Timedelta(hours=4)  # 週末ギャップ直後バー
    after_open = gap.rolling(6, min_periods=1).max().astype(bool)  # 直後6本
    res = {}
    res["full"] = stats(tgt[sig & te])
    res["ex_sunday_open"] = stats(tgt[sig & te & ~after_open])
    res["sunday_open_only"] = stats(tgt[sig & te & after_open])
    norol = ~np.isin(df.index.hour, ROLLOVER_HOURS)
    res["ex_both"] = stats(tgt[sig & te & ~after_open & norol])
    return res


def fmt_hz(hz: dict[int, float]) -> str:
    return " ".join(f"h{h}:{v:+.2f}" for h, v in hz.items())


def main() -> None:
    results = []
    for pair in PAIRS:
        r = eval_pair(pair)
        results.append(r)
        print(f"\n{'=' * 100}\n{pair}  (cost {r['cost']}p)  thr={ {k: round(v, 3) for k, v in r['thr'].items()} }")
        for key in ("sig1_z10lo_x_hiER10", "sig2_comb25_long"):
            s = r[key]
            print(
                f" {key}: test {s['full_mean']:+.3f}p t={s['full_t']:+.2f} n={s['full_n']}"
                f" ({s['per_day']:.2f}/day) | ex20-23 {s['ex_mean']:+.3f}p t={s['ex_t']:+.2f}"
                f" n={s['ex_n']} | rollover-only {s['ro_mean']:+.3f}p (n={s['ro_n']})"
            )
            print(f"   hz full : {fmt_hz(s['hz_full'])}")
            print(f"   hz ex   : {fmt_hz(s['hz_ex'])}")
            print(f"   yearly  : {s['yearly']}")

    # コスト超えカウント(h10/h20 累積 vs 往復コスト)
    print(f"\n{'=' * 100}\ncost-beating count (test cumulative vs round-trip cost)")
    for key in ("sig1_z10lo_x_hiER10", "sig2_comb25_long"):
        for tag, hzkey in [("full", "hz_full"), ("ex20-23", "hz_ex")]:
            beat10 = [r["pair"] for r in results if r[key][hzkey][10] > r["cost"]]
            beat20 = [r["pair"] for r in results if r[key][hzkey][20] > r["cost"]]
            print(f" {key} [{tag}]  h10>cost: {len(beat10)} {beat10} | h20>cost: {len(beat20)} {beat20}")

    print(f"\n{'=' * 100}\nEURUSD sig1 Sunday-open sensitivity (first 6 bars after weekend gap)")
    for k, (m, t, n) in sunday_open_sensitivity().items():
        print(f" {k:<18}: {m:+.3f}p t={t:+.2f} n={n}")


if __name__ == "__main__":
    main()
