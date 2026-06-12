"""edge02: ギャップ起因シグナルの監査(週末/休日/セッションギャップが z を突破させたケース)。

問い: z突破が「連続的な値動き」でなく「ギャップ(取引不能な時間の価格ジャンプ)」で
成立したトレードはどれだけあり、その利益は本物か?
  (a) ギャップ埋め習性で勝ちやすい本物のエッジ
  (b) 金曜の stale な価格 vs 月曜の真の価格の差を「利益」計上しているだけの幻影
のどちらかをデータで判定する。**監査のみ・採用判断なし**(reports/19 exp56 の
週末衛生レバーは全滅済み。本実験はレバー再提案ではなく利益の実在性監査)。

計測(プール results/mm_pool_v2d1_H4_19.parquet, n=1207, sum=+1.9622):
  1. gap = first_M1_close(bar) - H4_close(prev bar) をシグナルバー(=entry の1グリッド前)
     とエントリーバーで計測。ATR20(close-diff プロキシ)比 + gap_z_share
     (z クロスに必要だった距離のうちギャップが占めた率、反実仮想 z で算出)。
     クロスは脚 M1 close の inner-join 合成(exp52 方式)で「バー最初の M1 close」を取る。
  2. 週末跨ぎ直後フラグ(前グリッドバーとのラベル差 >= 40h)・セッションギャップ
     (4h超のグリッド断絶)をシグナル/エントリー両バーで。
  3. 休日カレンダー(12/24-1/2, 聖金曜+復活祭月曜, 感謝祭+翌金, 7/4)。
  4. 反実仮想: シグナルバー close をギャップ除去値(close_{t-1} + バー内変化)に
     差し替えて rolling(50) z を再計算 → クロス不成立なら gap-made シグナル。
     エントリーバーでも同様に d1 z ゲート(|z|>0.5)の反実仮想(gap-saved)。
  5. gap-made コホートのその後: エントリー後 1/2/3 バーの方向調整リターン
     (=ギャップ埋め方向の初期回帰)と 18 バー以内のギャップ埋め完了率。

実行: uv run python research/experiments/edge02_gap_trigger.py
出力: research/outputs/edge02_trades.csv / edge02_summary.json

結論(2026-06-13 実行):
  ・プール検算 OK(n=1207, sum=+1.9622)。シグナルバー z クロスは全 1207 件で再現、
    |z_sig| は pool.z_entry と最大差 0。クロス合成の M1/H4 整合も max 4.7e-5。
  ・gap-made(ギャップ無しでは z クロス不成立)= 70件(5.8%)。平均 +12.1bps・
    勝率 64.3%・合計 +0.0845(全体 PnL の +4.3%)。残り(+16.5bps)より僅かに弱いが
    diff CI95 [-22.8,+14.4]bps は 0 を跨ぐ=区別不能。IS +0.059 / OOS +0.026 と両期間
    プラス、単年集中なし(最大 2024 の 32%)。幻影でも負けコホートでもない。
  ・gap-made の正体は「z がぎりぎり 2.0 を超えた限界クロスを 4h バー境界の小ジャンプが
    押し込んだ」ケース(|z_sig|-2.0 中央値 0.016、gap/ATR20 中央値 0.164)。
    週末ギャップ起因はわずか 9/70 件(コホート計 -0.015、n<30)で、「金曜 stale 価格
    vs 月曜真値」の幻影構図は主役ではない。週内セッション断絶起因は 0 件。
  ・(a) ギャップ埋めエッジ説も棄却: エントリー後 3 バーの方向調整リターンは
    gap-made -3.4bps vs その他 +0.6bps(diff CI [-12.0,+4.7] 跨ぐ)。gap-made も
    他トレードと同じ「第1波逆行→保有期間全体で回収」の収束プレミアム挙動。
  ・gap-saved(エントリーバーのギャップが d1 z ゲートを生かした)= 0 件。
    週明け初バー執行(wknd_entry)は 12件 +0.036(n<30)。phantom 約定チャネルは細い。
  ・holiday(薄商い暦)= 49件 計 -0.015(-0.8%)・勝率 71%。マイナスは 2022 の
    2 トレード(USDCHF -3.0% / EURCAD -3.9%)が全て=単年・単発依存、CI も 0 を跨ぐ。
  → 判定: ギャップ起因シグナルは幻影でも特別なエッジでもなく、限界クロスの
    タイミングをずらしただけの正常なプール構成要素。veto 候補の指名なし。
    正直な注意: 週末・休日バーの実効スプレッドは固定 pips モデルより広い可能性が
    あるが、該当エントリー(12+49件)は PnL 寄与が小さく口座への影響は限定的。
    また反実仮想は「そのバーでのクロス不成立」までしか言えず、ギャップ無しでも
    後続バーでクロスした可能性(機会の遅延)は経路依存のため測っていない。
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 220)

POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
OUT_DIR = ROOT / "research" / "outputs"
OUT_CSV = OUT_DIR / "edge02_trades.csv"
OUT_JSON = OUT_DIR / "edge02_summary.json"

W = 50          # 短期 z 窓
ENTRY_Z = 2.0   # クロス閾値
EXIT_Z = 0.5    # 出口 / d1 ゲート閾値
WEEKEND_H = 40.0  # 前バーとのラベル差がこれ以上 = 週末跨ぎ
BAR_H = 4.0
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
TOTAL_SUM = 1.9622


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# M1 close(メジャー=実データ, クロス=脚 inner-join 合成)— exp52 方式
# ---------------------------------------------------------------------------
_M1_MAJ: dict[str, pd.Series] = {}


def m1_close(name: str) -> pd.Series:
    """tz-naive index の M1 close。メジャーはキャッシュ、クロスは都度合成(メモリ節約)。"""
    if name in CROSS_DEFS:
        a, op, b = CROSS_DEFS[name]
        ca, cb = m1_close(a), m1_close(b)
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").dropna()
        return df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
    if name not in _M1_MAJ:
        c = load_m1(name)["close"]
        _M1_MAJ[name] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
    return _M1_MAJ[name]


def holiday_dates(y0: int, y1: int) -> set:
    """薄商いカレンダー: 12/24-1/2, 聖金曜+復活祭月曜, 感謝祭+翌金, 7/4。"""
    from dateutil.easter import easter
    days: set = set()
    for y in range(y0, y1 + 1):
        for d in pd.date_range(f"{y}-12-24", f"{y + 1}-01-02"):
            days.add(d.date())
        e = easter(y)
        days.add(e - timedelta(days=2))   # Good Friday
        days.add(e + timedelta(days=1))   # Easter Monday
        nov1 = date(y, 11, 1)
        first_thu = nov1 + timedelta(days=(3 - nov1.weekday()) % 7)
        tg = first_thu + timedelta(weeks=3)
        days.add(tg)
        days.add(tg + timedelta(days=1))  # 感謝祭翌金曜
        days.add(date(y, 7, 4))
    return days


# ---------------------------------------------------------------------------
# 特徴量計算(銘柄ごと)
# ---------------------------------------------------------------------------
FEAT_COLS = [
    "pos_e", "gap_sig", "gap_entry", "gap_sig_atr", "gap_entry_atr",
    "gap_sig_bd_bps", "gap_entry_bd_bps", "delta_sig_h", "delta_entry_h",
    "z_sig", "z_sigprev", "z_cf_sig", "z_entry_raw", "z_cf_entry",
    "gap_z_share", "r1", "r2", "r3", "ret_path", "filled_18",
    "m1_consistency",
]


def instrument_features(instr: str, g: pd.DataFrame) -> pd.DataFrame:
    d = uni.instrument_data(instr, "H4")
    c = d["close"]
    carr = c.to_numpy()
    idx = c.index
    z = ((c - c.rolling(W).mean()) / c.rolling(W).std()).to_numpy()
    atr20 = c.diff().abs().rolling(20).mean().to_numpy()
    delta_h = np.r_[np.nan, np.diff(idx.to_numpy()).astype("timedelta64[m]").astype(float) / 60.0]

    pos_of = pd.Series(np.arange(len(idx)), index=idx)
    e_pos = pos_of.reindex(g["entry"]).to_numpy()
    x_pos = pos_of.reindex(g["exit"]).to_numpy()
    assert np.isfinite(e_pos).all() and np.isfinite(x_pos).all(), f"{instr}: timestamp miss"
    e_pos = e_pos.astype(int)
    x_pos = x_pos.astype(int)
    s_pos = e_pos - 1  # シグナルバー = エントリーバーの 1 グリッド前

    # 必要バー(シグナル+エントリー)の「バー最初の M1 close」
    m1 = m1_close(instr)
    m1_idx = m1.index.to_numpy()
    m1_arr = m1.to_numpy()
    need = np.unique(np.r_[s_pos, e_pos])
    first_m1 = {}
    last_m1_prev = {}
    idx_naive = idx.tz_localize(None).to_numpy()
    for p in need:
        L = idx_naive[p]
        a = int(np.searchsorted(m1_idx, L, side="left"))
        b = int(np.searchsorted(m1_idx, L + np.timedelta64(4, "h"), side="left"))
        first_m1[p] = m1_arr[a] if a < b else np.nan
        last_m1_prev[p] = m1_arr[a - 1] if a > 0 else np.nan

    n = len(g)
    out = {k: np.full(n, np.nan) for k in FEAT_COLS}
    dirs = g["dir"].to_numpy().astype(float)

    for i in range(n):
        p_s, p_e, p_x = s_pos[i], e_pos[i], x_pos[i]
        dirv = dirs[i]
        dir_z = -dirv  # ブレイク方向(long=z下抜け= -1, short=z上抜け= +1)
        out["pos_e"][i] = p_e
        out["delta_sig_h"][i] = delta_h[p_s]
        out["delta_entry_h"][i] = delta_h[p_e]
        out["z_sig"][i] = z[p_s]
        out["z_sigprev"][i] = z[p_s - 1]
        out["z_entry_raw"][i] = z[p_e]

        # --- ギャップ(価格) -------------------------------------------------
        gs = first_m1[p_s] - carr[p_s - 1]   # シグナルバー: 前バーH4 close → 当バー最初のM1
        ge = first_m1[p_e] - carr[p_e - 1]
        out["gap_sig"][i] = gs
        out["gap_entry"][i] = ge
        # H4 close と「前バー最後の M1 close」の整合(クロス合成の検算)
        out["m1_consistency"][i] = abs(last_m1_prev[p_s] / carr[p_s - 1] - 1.0)
        if np.isfinite(atr20[p_s]) and atr20[p_s] > 0:
            out["gap_sig_atr"][i] = abs(gs) / atr20[p_s]
        if np.isfinite(atr20[p_e]) and atr20[p_e] > 0:
            out["gap_entry_atr"][i] = abs(ge) / atr20[p_e]
        out["gap_sig_bd_bps"][i] = dir_z * gs / carr[p_s - 1] * 1e4   # +=ブレイク方向のギャップ
        out["gap_entry_bd_bps"][i] = dir_z * ge / carr[p_e - 1] * 1e4

        # --- 反実仮想 z(ギャップ成分除去) ----------------------------------
        if p_s - W + 1 >= 0 and np.isfinite(first_m1[p_s]):
            seg = carr[p_s - W + 1: p_s + 1].copy()
            cf = carr[p_s - 1] + (carr[p_s] - first_m1[p_s])  # close_{t-1} + バー内変化
            seg[-1] = cf
            out["z_cf_sig"][i] = (cf - seg.mean()) / seg.std(ddof=1)
        if p_e - W + 1 >= 0 and np.isfinite(first_m1[p_e]):
            seg = carr[p_e - W + 1: p_e + 1].copy()
            cf = carr[p_e - 1] + (carr[p_e] - first_m1[p_e])
            seg[-1] = cf
            out["z_cf_entry"][i] = (cf - seg.mean()) / seg.std(ddof=1)

        # gap_z_share: クロスに必要だった z 距離のうちギャップ寄与率
        y_sig = dir_z * z[p_s]
        y_prev = dir_z * z[p_s - 1]
        y_cf = dir_z * out["z_cf_sig"][i]
        needed = ENTRY_Z - y_prev
        if np.isfinite(y_cf) and needed > 0:
            out["gap_z_share"][i] = (y_sig - y_cf) / needed

        # --- エントリー後の初期回帰(ギャップ埋め方向 = 建玉方向) ----------
        ec = carr[p_e]
        for k, key in ((1, "r1"), (2, "r2"), (3, "r3")):
            out[key][i] = dirv * (carr[min(p_e + k, p_x)] / ec - 1.0)
        out["ret_path"][i] = dirv * (carr[p_x] / ec - 1.0)
        # 18 バー以内にシグナル前 close(ギャップ前価格)まで戻ったか
        fill_level = carr[p_s - 1]
        seg_c = carr[p_e: min(p_e + 19, p_x + 1)]
        out["filled_18"][i] = float(np.any(dirv * (seg_c - fill_level) >= 0))

    return pd.DataFrame(out, index=g.index)


# ---------------------------------------------------------------------------
# コホート統計(規約: n/平均bps/PnLシェア/勝率/IS-OOS/年次/L-S + ブートCI)
# ---------------------------------------------------------------------------
def boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot=1000, seed=0):
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ai = rng.integers(0, len(a), size=(n_boot, len(a)))
    bi = rng.integers(0, len(b), size=(n_boot, len(b)))
    dd = a[ai].mean(axis=1) - b[bi].mean(axis=1)
    return (float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))


def cohort_report(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    c, r = df[mask], df[~mask]
    res = {"name": name, "n": int(len(c))}
    if len(c) == 0:
        print(f"\n[{name}] n=0 — 該当なし")
        return res
    is_c = c[c["entry"] < OOS_START]
    oos_c = c[c["entry"] >= OOS_START]
    yr = c.groupby(c["entry"].dt.year)["ret"].agg(["size", "sum"])
    ymax = yr["sum"].abs().idxmax()
    ysum = c["ret"].sum()
    yshare = yr.loc[ymax, "sum"] / ysum if ysum != 0 else np.nan
    lo, hi = boot_diff_ci(c["ret"].to_numpy(), r["ret"].to_numpy())
    res.update({
        "mean_bps": float(c["ret"].mean() * 1e4),
        "sum_pnl": float(ysum),
        "pnl_share_pct": float(ysum / TOTAL_SUM * 100),
        "win_rate_pct": float((c["ret"] > 0).mean() * 100),
        "is": {"n": int(len(is_c)), "sum": float(is_c["ret"].sum()),
               "mean_bps": float(is_c["ret"].mean() * 1e4) if len(is_c) else np.nan,
               "win_pct": float((is_c["ret"] > 0).mean() * 100) if len(is_c) else np.nan},
        "oos": {"n": int(len(oos_c)), "sum": float(oos_c["ret"].sum()),
                "mean_bps": float(oos_c["ret"].mean() * 1e4) if len(oos_c) else np.nan,
                "win_pct": float((oos_c["ret"] > 0).mean() * 100) if len(oos_c) else np.nan},
        "yearly": {int(y): {"n": int(v["size"]), "sum": float(v["sum"])}
                   for y, v in yr.iterrows()},
        "max_year": int(ymax), "max_year_share": float(yshare) if np.isfinite(yshare) else None,
        "long": {"n": int((c["dir"] > 0).sum()), "sum": float(c.loc[c["dir"] > 0, "ret"].sum()),
                 "mean_bps": float(c.loc[c["dir"] > 0, "ret"].mean() * 1e4) if (c["dir"] > 0).any() else np.nan},
        "short": {"n": int((c["dir"] < 0).sum()), "sum": float(c.loc[c["dir"] < 0, "ret"].sum()),
                  "mean_bps": float(c.loc[c["dir"] < 0, "ret"].mean() * 1e4) if (c["dir"] < 0).any() else np.nan},
        "rest_mean_bps": float(r["ret"].mean() * 1e4),
        "diff_ci_bps": [lo * 1e4, hi * 1e4],
        "unreliable": bool(len(c) < 30),
    })
    print(f"\n[{name}] n={res['n']}{' (n<30=統計的に当てにならない)' if res['unreliable'] else ''}")
    print(f"  mean {res['mean_bps']:+.1f}bps  sum {ysum:+.4f} ({res['pnl_share_pct']:+.1f}% of pool)  "
          f"win {res['win_rate_pct']:.1f}%")
    print(f"  IS: n={res['is']['n']} sum={res['is']['sum']:+.4f} ({res['is']['mean_bps']:+.1f}bps, "
          f"win {res['is']['win_pct']:.0f}%) | OOS: n={res['oos']['n']} sum={res['oos']['sum']:+.4f} "
          f"({res['oos']['mean_bps']:+.1f}bps, win {res['oos']['win_pct']:.0f}%)")
    print("  年次: " + " ".join(f"{y}:{v['n']}件/{v['sum']:+.3f}" for y, v in res["yearly"].items()))
    print(f"  単年集中: max={res['max_year']} share={yshare:+.1%}" if np.isfinite(yshare) else "  単年集中: n/a")
    print(f"  L/S: Long n={res['long']['n']} sum={res['long']['sum']:+.4f} "
          f"({res['long']['mean_bps']:+.1f}bps) / Short n={res['short']['n']} "
          f"sum={res['short']['sum']:+.4f} ({res['short']['mean_bps']:+.1f}bps)")
    print(f"  vs 残り: cohort {res['mean_bps']:+.1f} vs rest {res['rest_mean_bps']:+.1f}bps  "
          f"diff CI95 [{lo * 1e4:+.1f}, {hi * 1e4:+.1f}]bps "
          f"{'(0を跨ぐ=有意でない)' if lo < 0 < hi else '(0を跨がない)'}")
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)

    sec("0. プール読込と検算")
    pool = pd.read_parquet(POOL_PATH).reset_index(drop=True)
    print(f"n={len(pool)}  sum(ret)={pool['ret'].sum():+.4f}  (基準 n=1207 / +1.9622)")
    ok = len(pool) == 1207 and abs(pool["ret"].sum() - TOTAL_SUM) < 1e-3
    if not ok:
        print("!! プール検算が不一致 — 中断")
        return 1

    sec("1. 特徴量計算(19銘柄: H4ギャップ + M1合成バー頭 + 反実仮想 z)")
    feats = []
    for instr, g in pool.groupby("instr"):
        f = instrument_features(instr, g)
        feats.append(f)
        print(f"  {instr}: n={len(g)}  m1整合(前バーclose vs 直前M1)max={f['m1_consistency'].max():.2e}"
              f"  [{time.time() - t0:.0f}s]")
    df = pd.concat([pool, pd.concat(feats).sort_index()], axis=1)
    df["year"] = df["entry"].dt.year

    # 検算: シグナルバーのクロス再現 + z_entry 一致
    dir_z = -df["dir"].astype(float)
    y_sig = dir_z * df["z_sig"]
    y_prev = dir_z * df["z_sigprev"]
    cross_ok = ((y_sig > ENTRY_Z) & (y_prev <= ENTRY_Z)).mean()
    z_match = np.abs(df["z_sig"].abs() - df["z_entry"]).max()
    print(f"\n検算: シグナルバー z クロス再現率 = {cross_ok:.1%} / |z_sig| vs pool.z_entry 最大差 = {z_match:.1e}")
    assert cross_ok == 1.0 and z_match < 1e-9

    # フラグ定義
    df["wknd_sig"] = df["delta_sig_h"] >= WEEKEND_H
    df["wknd_entry"] = df["delta_entry_h"] >= WEEKEND_H
    df["sessgap_sig"] = (df["delta_sig_h"] > BAR_H) & ~df["wknd_sig"]
    df["sessgap_entry"] = (df["delta_entry_h"] > BAR_H) & ~df["wknd_entry"]
    hd = holiday_dates(2015, 2026)
    df["holiday"] = df["entry"].dt.date.isin(hd) | (df["entry"] - pd.Timedelta(hours=4)).dt.date.isin(hd)

    y_cf = dir_z * df["z_cf_sig"]
    df["gap_made"] = y_cf <= ENTRY_Z          # ギャップ無しではクロス不成立
    df["gap_alone"] = df["gap_z_share"] >= 1.0  # ギャップ単独でクロス到達
    y_cf_e = dir_z * df["z_cf_entry"]
    df["gap_saved"] = y_cf_e < EXIT_Z          # ギャップ無しでは d1 ゲートで消滅

    sec("2. ギャップの分布(シグナルバー)")
    gd = df["gap_sig_atr"].describe(percentiles=[0.5, 0.9, 0.99])
    print(f"gap_sig/ATR20: median {gd['50%']:.3f}  p90 {gd['90%']:.3f}  p99 {gd['99%']:.3f}  max {gd['max']:.2f}")
    print(f"gap_sig_atr>=0.5: {(df['gap_sig_atr'] >= 0.5).sum()}件 / >=1.0: {(df['gap_sig_atr'] >= 1.0).sum()}件")
    print(f"gap_z_share: median {df['gap_z_share'].median():+.3f}  "
          f">=0.5: {(df['gap_z_share'] >= 0.5).sum()}件  >=1.0: {(df['gap_z_share'] >= 1.0).sum()}件  "
          f"<=0(ギャップ逆向き): {(df['gap_z_share'] <= 0).sum()}件")
    print(f"フラグ: wknd_sig={df['wknd_sig'].sum()}  wknd_entry={df['wknd_entry'].sum()}  "
          f"sessgap_sig={df['sessgap_sig'].sum()}  holiday={df['holiday'].sum()}  "
          f"gap_made={df['gap_made'].sum()}  gap_alone={df['gap_alone'].sum()}  "
          f"gap_saved={df['gap_saved'].sum()}")
    gm = df[df["gap_made"]]
    print(f"\ngap_made の内訳: 週末シグナル {int(gm['wknd_sig'].sum())} / セッションギャップ "
          f"{int(gm['sessgap_sig'].sum())} / 通常バー {int((~gm['wknd_sig'] & ~gm['sessgap_sig']).sum())}")
    print("gap_made 銘柄別: " + " ".join(f"{i}:{c}" for i, c in
                                       gm["instr"].value_counts().head(8).items()))
    cross_share = gm["instr"].isin(CROSS_DEFS).mean() if len(gm) else np.nan
    print(f"gap_made のクロス比率: {cross_share:.0%} (プール全体 {df['instr'].isin(CROSS_DEFS).mean():.0%})")

    sec("3. コホート分析(規約準拠)")
    cohorts = {}
    for name, mask in [
        ("gap_made(ギャップ無しでクロス不成立)", df["gap_made"]),
        ("gap_alone(ギャップ単独でクロス到達)", df["gap_alone"]),
        ("gap_made & 週末シグナル", df["gap_made"] & df["wknd_sig"]),
        ("gap_made & 非週末", df["gap_made"] & ~df["wknd_sig"]),
        ("wknd_sig(シグナル=週明け初バー)", df["wknd_sig"]),
        ("wknd_entry(エントリー=週明け初バー)", df["wknd_entry"]),
        ("holiday(薄商いカレンダー)", df["holiday"]),
        ("gap_saved(ギャップが d1 ゲートを通した)", df["gap_saved"]),
        ("big_gap(gap_sig_atr>=1)", df["gap_sig_atr"] >= 1.0),
    ]:
        cohorts[name] = cohort_report(df, mask, name)

    sec("4. gap-made のその後(ギャップ埋め方向への初期回帰)")
    for label, sub in [("gap_made", df[df["gap_made"]]), ("その他", df[~df["gap_made"]])]:
        print(f"  {label}: n={len(sub)}  r1 {sub['r1'].mean() * 1e4:+.1f}  r2 {sub['r2'].mean() * 1e4:+.1f}  "
              f"r3 {sub['r3'].mean() * 1e4:+.1f}bps  ret_path {sub['ret_path'].mean() * 1e4:+.1f}bps  "
              f"18バー内ギャップ前価格到達率 {sub['filled_18'].mean():.0%}")
    r3_ci = boot_diff_ci(df.loc[df["gap_made"], "r3"].to_numpy(),
                         df.loc[~df["gap_made"], "r3"].to_numpy())
    print(f"  r3 差の CI95: [{r3_ci[0] * 1e4:+.1f}, {r3_ci[1] * 1e4:+.1f}]bps")
    gm = df[df["gap_made"]]
    early = gm["r3"].sum()
    print(f"  gap_made 合計 PnL(path) {gm['ret_path'].sum():+.4f} のうち最初3バー {early:+.4f} "
          f"({early / gm['ret_path'].sum():.0%})" if gm["ret_path"].sum() != 0 else "")

    sec("5. 保存")
    keep = ["instr", "entry", "exit", "dir", "ret", "bars_held", "year",
            "z_sig", "z_sigprev", "z_cf_sig", "z_entry_raw", "z_cf_entry",
            "gap_sig", "gap_entry", "gap_sig_atr", "gap_entry_atr",
            "gap_sig_bd_bps", "gap_entry_bd_bps", "gap_z_share",
            "delta_sig_h", "delta_entry_h", "wknd_sig", "wknd_entry",
            "sessgap_sig", "sessgap_entry", "holiday",
            "gap_made", "gap_alone", "gap_saved", "r1", "r2", "r3",
            "ret_path", "filled_18"]
    df[keep].to_csv(OUT_CSV, index=False)
    summary = {
        "pool_check": {"n": int(len(pool)), "sum_ret": float(pool["ret"].sum()),
                       "cross_repro": float(cross_ok), "z_match_max": float(z_match)},
        "gap_dist": {"median_atr": float(gd["50%"]), "p90_atr": float(gd["90%"]),
                     "p99_atr": float(gd["99%"]),
                     "n_ge_05atr": int((df["gap_sig_atr"] >= 0.5).sum()),
                     "n_ge_10atr": int((df["gap_sig_atr"] >= 1.0).sum())},
        "flags": {k: int(df[k].sum()) for k in
                  ["wknd_sig", "wknd_entry", "sessgap_sig", "holiday",
                   "gap_made", "gap_alone", "gap_saved"]},
        "cohorts": cohorts,
        "gap_fill": {"r3_gap_made_bps": float(df.loc[df["gap_made"], "r3"].mean() * 1e4),
                     "r3_rest_bps": float(df.loc[~df["gap_made"], "r3"].mean() * 1e4),
                     "r3_diff_ci_bps": [r3_ci[0] * 1e4, r3_ci[1] * 1e4],
                     "filled18_gap_made": float(df.loc[df["gap_made"], "filled_18"].mean()),
                     "filled18_rest": float(df.loc[~df["gap_made"], "filled_18"].mean())},
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, default=float))
    print(f"saved -> {OUT_CSV}\n      -> {OUT_JSON}")
    print(f"総経過 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
