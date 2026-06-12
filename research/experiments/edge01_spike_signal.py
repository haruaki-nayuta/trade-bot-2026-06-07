"""edge01: シグナルバー/エントリーバーの M1 内部解剖 — スパイク起因シグナルの監査。

問い: チャンピオン(confluence_meanrev_v2_d1, H4, 19銘柄)の z 突破シグナルのうち、
「H4 バー内の数分間のスパイク(フラッシュ的な動き・データグリッチ)が作ったもの」は
どれだけあり、そのトレードの利益は本物か?
スパイクで close が押し下げ/押し上げられた瞬間に逆張りで乗る形は、実運用では
(a) その価格に流動性がない (b) スプレッドが爆発している ため約定現実性が疑わしい
= バックテスト幻影の典型源。本実験は 3 レンズ監査のうち (a)幻影 (c)データ品質を担当。

計測(各トレード × シグナルバー[=エントリーの1本前, 位置ベース]/エントリーバー, M1 closes):
  1. minutes_present : バー内 M1 本数(240-n = 流動性の穴)。クロスは脚 inner-join 後。
  2. max1m / spike5(最大の連続5分窓 |move|)/ spike_share_net(=spike5/|netΔclose|)
     / spike_share_hl(=spike5/(max-min))
  3. last30_share : 最後30分の動き / バー net move(引け間際の急変か)
  4. stale_run : 同一 close の最大連続分数(データ凍結)
  5. エントリーバーは + close_drift: (H4 close − 最後15分の中央値) / ATR20(H4, M1由来)
     = 執行価格がスパイクの先端かの検出。drift_fav = dir×(med15−close)/ATR(正=有利方向に乖離)
  6. 反実仮想: シグナルバー close を置換して z 再計算 → クロス不成立 = spike-made。
     spec 版 = 「最後5分を除いた中央値」置換(バー後半の動き全体が必要だったか=広義)
     tight 版 = 「最後5分開始直前の close」置換(末尾5分スパイクだけが作ったか=狭義)

クロスの M1 は脚 close の inner-join 合成(exp44/52 と同一方式)。合成 M1 の「スパイク」は
脚のスパイクの合成であり、脚のティック非同期(片脚だけ動いた分)も1分足では正味の動きに
含まれる点に注意して解釈する。XAUUSD は対象外(プールに元々含まれない)。

実行: uv run python research/experiments/edge01_spike_signal.py(リポジトリ直下から)
出力: research/outputs/edge01_trades.csv / edge01_summary.json

結論(2026-06-13 実行, n=1207 / sum=+1.9622 / z 再計算 maxdiff=0 検算一致):
  ・幻影の署名(スパイク起因コホートの利益が本物より高い+執行価格がスパイク先端)は
    どこにも無い。むしろスパイク依存シグナルほど利益は低い。
  ・spike-made 広義(close→バー中央値置換でクロス不成立)= n=725(60%)。これは
    「z クロスの限界性」の検出器であり(クロスの大半は z≈2.0 ぎりぎり)、mean +12.4bps
    < rest +22.0bps(CI [-19.0,-1.2])。利益が低い側 = 幻影でない。IS/OOS 両プラス。
  ・spike-made 狭義(末尾5分の動きだけがクロスを作った)= n=98(8.1%)。実体はフラッシュ
    スパイクではなく閾値限界(|z_sig|-2.0 中央値 0.036 vs rest 0.209、83% が反実 z で
    閾値±0.1)。mean +1.8bps ≈ エッジゼロ(CI diff [-30.1,-2.9])、OOS -2.7bps。
    PnL シェア +0.9% に過ぎず幻影源ではないが、唯一の veto 候補として口座検証に指名
    (除外で残存 mean 16.3→17.5bps、OOS +0.011 改善 / IS -0.029)。
  ・spike_share_hl>0.6(バーのレンジの6割超が単一5分窓)= n=75 は mean +39.7bps と
    本物より「高い」(CI [+6.8,+44.0]、PnL15.2%)が、幻影条件を満たさない: d1 遅延で
    建玉はスパイクの 4-8h 後、エントリーバー |close_drift| は中央値 0.07 ATR・最大
    0.33 ATR(>0.5 はゼロ)、drift_gt05 コホートとの重なり 0 件。= 急変後の歪み拡大に
    平均回帰プレミアムが乗る正常な機構(IS/OOS 両側・年次分散・L/S 両プラス)。
  ・データ品質: stale_run 最大 19 分(60 分超ゼロ=凍結なし)。欠損>25% バーは 33 件で
    全て週開閉境界(Sun20/Fri12)+大晦日1件の構造的短バー、成績は rest と区別不能。
    引け間際急変(last30>60%)も rest と同成績。close_drift>0.5ATR は n=11(mean -7.9bps)
    = 執行価格がスパイク先端のケースはむしろ僅かに損 → バックテストが幻影利益を
    計上している証拠なし。
  → レンズ (a) 幻影・(c) データ品質とも「チャンピオンの数字は防衛された」が主結論。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

OUT_DIR = ROOT / "research" / "outputs"
OUT_CSV = OUT_DIR / "edge01_trades.csv"
OUT_JSON = OUT_DIR / "edge01_summary.json"

POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
EXPECT_N = 1207
EXPECT_SUM = 1.9622

WINDOW = 50       # 短期 z の窓(チャンピオン本番値)
ENTRY_Z = 2.0     # z 突破閾値
H4 = np.timedelta64(4, "h").astype("timedelta64[ns]").astype(np.int64)
MIN5 = np.timedelta64(5, "m").astype("timedelta64[ns]").astype(np.int64)
MIN15 = np.timedelta64(15, "m").astype("timedelta64[ns]").astype(np.int64)
MIN30 = np.timedelta64(30, "m").astype("timedelta64[ns]").astype(np.int64)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# M1 close ロード(メジャー=実データ、クロス=脚 inner-join 合成。exp44/52 方式)
# ---------------------------------------------------------------------------
def load_m1_closes(instruments: list[str]) -> dict[str, pd.Series]:
    majors = sorted({leg for nm in instruments if nm in CROSS_DEFS
                     for leg in (CROSS_DEFS[nm][0], CROSS_DEFS[nm][2])}
                    | {nm for nm in instruments if nm not in CROSS_DEFS})
    raw: dict[str, pd.Series] = {}
    for p in majors:
        c = load_m1(p)["close"]
        raw[p] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))  # 非破壊 tz-naive
        print(f"  M1 {p}: {len(raw[p]):,} rows")
    out: dict[str, pd.Series] = {}
    for nm in instruments:
        if nm in CROSS_DEFS:
            a, op, b = CROSS_DEFS[nm]
            df = pd.concat([raw[a].rename("a"), raw[b].rename("b")],
                           axis=1, join="inner").dropna()
            out[nm] = df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
            print(f"  M1 synth {nm} = {a}{op}{b}: {len(out[nm]):,} rows")
        else:
            out[nm] = raw[nm]
    return out


# ---------------------------------------------------------------------------
# H4 バー1本の M1 内部メトリクス
# ---------------------------------------------------------------------------
def bar_metrics(times: np.ndarray, vals: np.ndarray, t0: int) -> dict:
    """times: int64(ns) 昇順, vals: float。バー = [t0, t0+4h)。"""
    t1 = t0 + H4
    a = int(np.searchsorted(times, t0, "left"))
    b = int(np.searchsorted(times, t1, "left"))
    n = b - a
    r = {"minutes": n, "max1m": np.nan, "spike5": np.nan, "spike_share_net": np.nan,
         "spike_share_hl": np.nan, "last30_share": np.nan, "stale_run": np.nan,
         "net_move": np.nan, "hl_range": np.nan, "med_ex5": np.nan, "close_at_5m": np.nan,
         "med15": np.nan, "last_close": np.nan}
    if n < 5:
        return r
    v = vals[a:b]
    t = times[a:b]
    prevc = vals[a - 1] if a > 0 else v[0]
    net = v[-1] - prevc
    hl = float(v.max() - v.min())
    lr = np.abs(np.diff(np.log(v)))
    # 最大の連続5分窓 |move|(時刻ベース)
    j = np.searchsorted(t, t + MIN5, "right") - 1
    spike5 = float(np.abs(v[j] - v).max())
    # 最後30分の動き(基準 = 30分前以前の最終 close)
    k30 = int(np.searchsorted(t, t1 - MIN30, "left"))
    base30 = v[k30 - 1] if k30 > 0 else prevc
    # 同一 close の最大連続(分; 壁時計時間)
    chg = np.flatnonzero(np.diff(v) != 0)
    starts = np.concatenate([[0], chg + 1])
    ends = np.concatenate([chg, [n - 1]])
    stale = float(((t[ends] - t[starts]).max()) / 60e9) + 1.0
    # 反実仮想用の置換値
    k5 = int(np.searchsorted(t, t1 - MIN5, "left"))
    k15 = int(np.searchsorted(t, t1 - MIN15, "left"))
    r.update({
        "max1m": float(lr.max()) if len(lr) else np.nan,
        "spike5": spike5,
        "spike_share_net": float(spike5 / abs(net)) if net != 0 else np.nan,
        "spike_share_hl": float(spike5 / hl) if hl > 0 else np.nan,
        "last30_share": float((v[-1] - base30) / net) if net != 0 else np.nan,
        "stale_run": stale,
        "net_move": float(net), "hl_range": hl,
        "med_ex5": float(np.median(v[:k5])) if k5 > 0 else np.nan,
        "close_at_5m": float(v[k5 - 1]) if k5 > 0 else np.nan,
        "med15": float(np.median(v[k15:])) if k15 < n else np.nan,
        "last_close": float(v[-1]),
    })
    return r


# ---------------------------------------------------------------------------
# コホート統計(規約: n/平均bps/PnLシェア/勝率/IS・OOS/年次/L-S/ブートCI)
# ---------------------------------------------------------------------------
def boot_ci_diff(a: np.ndarray, b: np.ndarray, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(a) < 2 or len(b) < 2:
        return (np.nan, np.nan)
    ma = rng.choice(a, size=(n_boot, len(a)), replace=True).mean(axis=1)
    mb = rng.choice(b, size=(n_boot, len(b)), replace=True).mean(axis=1)
    lo, hi = np.percentile(ma - mb, [2.5, 97.5])
    return float(lo * 1e4), float(hi * 1e4)


def cohort_stats(df: pd.DataFrame, member: pd.Series, name: str, definition: str) -> dict:
    valid = member.notna()
    sub = df[valid & (member == True)]   # noqa: E712
    rest = df[valid & (member == False)]  # noqa: E712
    r = sub["ret"]
    res = {"name": name, "definition": definition, "n": int(len(sub)),
           "n_excluded_nan": int((~valid).sum())}
    if len(sub) == 0:
        print(f"\n[{name}] n=0 (def: {definition})")
        return res
    res.update({
        "mean_bps": float(r.mean() * 1e4),
        "sum_pnl": float(r.sum()),
        "pnl_share_pct": float(r.sum() / EXPECT_SUM * 100),
        "win_rate_pct": float((r > 0).mean() * 100),
        "rest_mean_bps": float(rest["ret"].mean() * 1e4) if len(rest) else np.nan,
        "ci_diff_bps": boot_ci_diff(r.to_numpy(), rest["ret"].to_numpy()),
        "small_n": bool(len(sub) < 30),
    })
    # IS / OOS
    for tag, m in [("IS", sub["entry"] < OOS_START), ("OOS", sub["entry"] >= OOS_START)]:
        s = sub[m]["ret"]
        res[tag] = {"n": int(len(s)), "mean_bps": float(s.mean() * 1e4) if len(s) else np.nan,
                    "sum": float(s.sum()), "win_pct": float((s > 0).mean() * 100) if len(s) else np.nan}
    # 年次(エントリー年)
    ys = sub.groupby(sub["entry"].dt.year)["ret"].sum()
    res["yearly_sum"] = {int(k): round(float(v), 4) for k, v in ys.items()}
    tot = r.sum()
    if len(ys) and tot != 0:
        ymax = ys.abs().idxmax()
        share = float(ys[ymax] / tot)
        res["max_year"] = int(ymax)
        res["max_year_share"] = share
        res["single_year_flag"] = bool(abs(share) > 0.5)
    # Long / Short
    for tag, d in [("long", 1), ("short", -1)]:
        s = sub[sub["dir"] == d]["ret"]
        res[tag] = {"n": int(len(s)), "mean_bps": float(s.mean() * 1e4) if len(s) else np.nan,
                    "sum": float(s.sum()), "win_pct": float((s > 0).mean() * 100) if len(s) else np.nan}
    ci = res["ci_diff_bps"]
    print(f"\n[{name}] def: {definition}")
    print(f"  n={res['n']}{' (n<30=統計的に当てにならない)' if res['small_n'] else ''} "
          f"(NaN除外 {res['n_excluded_nan']})  mean={res['mean_bps']:+.1f}bps  "
          f"sum={res['sum_pnl']:+.4f} ({res['pnl_share_pct']:+.1f}% of pool)  "
          f"win={res['win_rate_pct']:.1f}%")
    print(f"  vs rest mean={res['rest_mean_bps']:+.1f}bps  diff CI95=[{ci[0]:+.1f}, {ci[1]:+.1f}]bps")
    print(f"  IS : n={res['IS']['n']} mean={res['IS']['mean_bps']:+.1f}bps sum={res['IS']['sum']:+.4f} "
          f"win={res['IS']['win_pct']:.0f}%")
    print(f"  OOS: n={res['OOS']['n']} mean={res['OOS']['mean_bps']:+.1f}bps sum={res['OOS']['sum']:+.4f} "
          f"win={res['OOS']['win_pct']:.0f}%")
    print(f"  年次: {res['yearly_sum']}" +
          (f"  最大寄与 {res.get('max_year')} ({res.get('max_year_share', 0):+.0%})"
           f"{' ← 単年依存フラグ' if res.get('single_year_flag') else ''}" if "max_year" in res else ""))
    print(f"  Long : n={res['long']['n']} mean={res['long']['mean_bps']:+.1f}bps sum={res['long']['sum']:+.4f}")
    print(f"  Short: n={res['short']['n']} mean={res['short']['mean_bps']:+.1f}bps sum={res['short']['sum']:+.4f}")
    return res


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    pool = pd.read_parquet(POOL_PATH)
    ok_pool = len(pool) == EXPECT_N and abs(pool["ret"].sum() - EXPECT_SUM) < 1e-3
    print(f"=== edge01: スパイク起因シグナル監査 ===")
    print(f"pool: n={len(pool)} sum={pool['ret'].sum():+.4f}  検算一致: {ok_pool}")
    if not ok_pool:
        print("!! プール検算不一致 — 中断")
        return 1
    instruments = sorted(pool["instr"].unique())
    n_cross = sum(nm in CROSS_DEFS for nm in instruments)
    print(f"instruments: {len(instruments)} ({n_cross} crosses = 脚M1合成)")

    sec("0. M1 ロード / H4 系列準備")
    m1 = load_m1_closes(instruments)
    m1_arr = {nm: (s.index.to_numpy().astype(np.int64), s.to_numpy()) for nm, s in m1.items()}

    h4 = {}
    for nm in instruments:
        d = uni.instrument_data(nm, "H4")
        close = d["close"]
        z = (close - close.rolling(WINDOW).mean()) / close.rolling(WINDOW).std()
        # ATR20 は全銘柄 M1 由来 H4 OHLC で統一(クロスの H4 high/low=close 問題を回避)
        s = m1[nm]
        rs = s.resample("4h", label="left", closed="left")
        oh = pd.DataFrame({"h": rs.max(), "l": rs.min(), "c": rs.last()}).dropna()
        pc = oh["c"].shift()
        tr = pd.concat([oh["h"] - oh["l"], (oh["h"] - pc).abs(), (oh["l"] - pc).abs()],
                       axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        atr_on_idx = atr20.reindex(close.index.tz_localize(None)).to_numpy()
        h4[nm] = {"idx": close.index, "close": close.to_numpy(), "z": z.to_numpy(),
                  "atr": atr_on_idx}
    del m1
    print(f"H4 prepared ({time.time()-t0:.0f}s)")

    sec("1. 各トレードの M1 内部メトリクス計測")
    n = len(pool)
    rows = []
    z_check_max = 0.0
    closediff_bps = {"major": [], "cross": []}
    for instr, g in pool.groupby("instr"):
        times, vals = m1_arr[instr]
        H = h4[instr]
        e_pos = H["idx"].get_indexer(g["entry"])
        assert (e_pos > 0).all(), f"{instr}: entry not found / first bar"
        for ti, ep in zip(g.index, e_pos):
            dirv = int(pool.at[ti, "dir"])
            sig_pos = ep - 1
            sig_label = H["idx"][sig_pos]
            ent_label = H["idx"][ep]
            contig = (ent_label - sig_label) == pd.Timedelta(hours=4)
            sig_t0 = sig_label.tz_localize(None).value
            ent_t0 = ent_label.tz_localize(None).value
            ms = bar_metrics(times, vals, sig_t0)
            me = bar_metrics(times, vals, ent_t0)

            # 検算: 最終 M1 close と H4 close の乖離(bps)
            if np.isfinite(ms["last_close"]):
                d_bps = abs(ms["last_close"] / H["close"][sig_pos] - 1) * 1e4
                closediff_bps["cross" if instr in CROSS_DEFS else "major"].append(d_bps)

            # z 検算(プール z_entry はシグナルバー |z|)
            z_check_max = max(z_check_max,
                              abs(abs(H["z"][sig_pos]) - pool.at[ti, "z_entry"]))

            # 反実仮想: シグナルバー close 置換 → z 再計算 → クロス成立判定
            sm_med, sm_l5, z_cf_med, z_cf_l5 = np.nan, np.nan, np.nan, np.nan
            if sig_pos - WINDOW + 1 >= 0:
                seg = H["close"][sig_pos - WINDOW + 1: sig_pos + 1].astype(float)
                if np.isfinite(seg).all():
                    for key, surro in [("med", ms["med_ex5"]), ("l5", ms["close_at_5m"])]:
                        if not np.isfinite(surro):
                            continue
                        s2 = seg.copy()
                        s2[-1] = surro
                        mu, sd = s2.mean(), s2.std(ddof=1)
                        z_cf = (surro - mu) / sd if sd > 0 else np.nan
                        crossed = (z_cf < -ENTRY_Z) if dirv == 1 else (z_cf > ENTRY_Z)
                        if key == "med":
                            z_cf_med, sm_med = z_cf, (not crossed)
                        else:
                            z_cf_l5, sm_l5 = z_cf, (not crossed)

            # close_drift(エントリーバー): 執行価格(H4 close)と最後15分中央値の乖離
            atr = H["atr"][ep]
            drift, drift_fav = np.nan, np.nan
            if np.isfinite(me["med15"]) and np.isfinite(atr) and atr > 0:
                ec = H["close"][ep]
                drift = (ec - me["med15"]) / atr
                drift_fav = dirv * (me["med15"] - ec) / atr  # 正 = 有利方向の先端で約定(幻影方向)

            rows.append({
                "ti": ti, "instr": instr, "is_cross": instr in CROSS_DEFS,
                "entry": pool.at[ti, "entry"], "exit": pool.at[ti, "exit"],
                "dir": dirv, "ret": pool.at[ti, "ret"],
                "bars_held": pool.at[ti, "bars_held"], "z_entry": pool.at[ti, "z_entry"],
                "sig_label": sig_label, "sig_contig": contig,
                "z_sig": H["z"][sig_pos], "z_cf_med": z_cf_med, "z_cf_l5": z_cf_l5,
                "spike_made_med": sm_med, "spike_made_l5": sm_l5,
                **{f"sig_{k}": v for k, v in ms.items() if k not in
                   ("med_ex5", "close_at_5m", "med15", "last_close")},
                **{f"ent_{k}": v for k, v in me.items() if k not in
                   ("med_ex5", "close_at_5m", "med15", "last_close")},
                "ent_close_drift": drift, "ent_drift_fav": drift_fav,
            })
    df = pd.DataFrame(rows).set_index("ti").sort_index()
    assert len(df) == n
    print(f"computed {len(df)} trades  ({time.time()-t0:.0f}s)")
    print(f"z 検算(プール z_entry vs 再計算): max diff = {z_check_max:.2e}")
    for k, v in closediff_bps.items():
        v = np.array(v)
        print(f"最終M1close vs H4close 乖離 [{k}]: median {np.median(v):.2f}bps / "
              f"p99 {np.percentile(v, 99):.2f}bps / max {v.max():.2f}bps")
    print(f"シグナルバーが entry-4h でない(週末跨ぎ): {(~df['sig_contig']).sum()}件")
    print(f"M1 メトリクス計算不能(分足<5本): sig {df['sig_max1m'].isna().sum()}件 / "
          f"ent {df['ent_max1m'].isna().sum()}件")

    sec("2. メトリクス分布(コホート閾値の足場)")
    cols = ["sig_minutes", "sig_max1m", "sig_spike_share_net", "sig_spike_share_hl",
            "sig_last30_share", "sig_stale_run", "ent_minutes", "ent_close_drift",
            "ent_drift_fav"]
    qs = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).T
    print(qs.to_string(float_format=lambda x: f"{x:.3f}"))
    lowmin = df[df["sig_minutes"] < 180]
    if len(lowmin):
        dows = pd.DatetimeIndex(lowmin["sig_label"]).day_name().value_counts()
        print(f"\n欠損>25%のシグナルバー: 曜日分布 {dows.to_dict()} "
              f"(Fri/Sun = 週の開閉境界の構造的短バー)")

    sec("3. コホート分析(規約: n/bps/PnLシェア/勝率/IS-OOS/年次/L-S/ブートCI)")
    summary = {}
    cohorts = [
        ("spike_made_med", df["spike_made_med"],
         "反実仮想(広義): sig close→最後5分除く中央値 置換で z クロス不成立"),
        ("spike_made_l5", df["spike_made_l5"],
         "反実仮想(狭義): sig close→最後5分直前 close 置換で z クロス不成立 = 末尾5分スパイクが作った"),
        ("spike_share_hi", (df["sig_spike_share_net"] > 0.6).where(df["sig_spike_share_net"].notna()),
         "sig バーの 5分窓最大|move| > 0.6×|netΔclose|"),
        ("spike_share_hl_hi", (df["sig_spike_share_hl"] > 0.6).where(df["sig_spike_share_hl"].notna()),
         "sig バーの 5分窓最大|move| > 0.6×(max-min)"),
        ("missing_gt25", (df["sig_minutes"] < 180).where(df["sig_minutes"].notna()),
         "sig バーの M1 欠損 > 25%(minutes<180)"),
        ("stale_gt60", (df["sig_stale_run"] > 60).where(df["sig_stale_run"].notna()),
         "sig バーの同一 close 連続 > 60分(データ凍結)"),
        ("late_move", (df["sig_last30_share"].abs() > 0.6).where(df["sig_last30_share"].notna()),
         "sig バーの net move の60%超が最後30分(引け間際の急変)"),
        ("drift_gt05", (df["ent_close_drift"].abs() > 0.5).where(df["ent_close_drift"].notna()),
         "エントリーバー |close − 最後15分中央値| > 0.5 ATR20(執行価格がスパイク先端)"),
        ("drift_fav_gt05", (df["ent_drift_fav"] > 0.5).where(df["ent_drift_fav"].notna()),
         "エントリーバー drift_fav > 0.5 ATR(有利方向の先端で約定=幻影方向)"),
    ]
    for name, member, definition in cohorts:
        summary[name] = cohort_stats(df, member, name, definition)

    sec("4. コホート重なり(件数行列)")
    masks = {name: (member == True) for name, member, _ in cohorts}  # noqa: E712
    names = list(masks)
    ov = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for i in names:
        for j in names:
            ov.loc[i, j] = int((masks[i] & masks[j]).sum())
    print(ov.to_string())

    sec("5. 補助: spike-made(狭義)の個票")
    sm = df[df["spike_made_l5"] == True]  # noqa: E712
    if len(sm):
        show = sm[["instr", "dir", "entry", "ret", "z_sig", "z_cf_l5", "sig_spike5",
                   "sig_max1m", "sig_minutes", "ent_close_drift"]].copy()
        show["entry"] = show["entry"].dt.strftime("%Y-%m-%d %H:%M")
        show["ret_bps"] = show.pop("ret") * 1e4
        print(show.to_string(float_format=lambda x: f"{x:.3f}"))

    # 保存 --------------------------------------------------------------------
    out = df.copy()
    out["sig_label"] = out["sig_label"].astype(str)
    out.to_csv(OUT_CSV, index=False)
    payload = {
        "pool_check": {"n": int(len(pool)), "sum_ret": float(pool["ret"].sum()),
                       "match": bool(ok_pool)},
        "z_check_max_diff": float(z_check_max),
        "m1_h4_close_diff_bps": {k: {"median": float(np.median(v)), "max": float(np.max(v))}
                                 for k, v in ((k, np.array(v)) for k, v in closediff_bps.items())},
        "cohorts": summary,
        "overlap": ov.to_dict(),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n      -> {OUT_JSON}")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
