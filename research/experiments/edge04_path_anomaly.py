"""edge04: 直前経路の異常監査(d1 プール) — 不自然な値動きパターンの3レンズ解剖。

問い: シグナル成立までの「経路」が異常なケース — 一方通行の暴走・1バーでの暴力的な
z突破・データ凍結 — でトレードの質が構造的に変わる帯はないか?
レンズ: (a) バックテスト幻影(スパイク戻りの見かけ利益) (b) 構造的コスト支配
(c) データ品質異常(凍結/フラッシュ)がシグナルを立てたケース。

計測(すべてシグナルバー=エントリーラベルの1本前、確定値のみ・因果):
  dz_jump          |z_t - z_{t-1}|(暴力的クロス)
  bar_range_ratio  シグナルバーの|Δclose| / catr20(前バーまで)。catr20=|Δclose|の20本平均
                   (クロスは H4 high/low=close 代用のため close ベースで統一)
  is_max50         シグナルバーの|Δclose|が直近50本で最大(anomaly-is-the-signal)
  oneway_run       シグナル方向(=建玉と逆方向)への連続同方向バー数(直前10本中)
  counter_share10  直前10本のうち建玉方向(=戻し)バーの比率。低=一方通行
  accel            直近3本リターン/直近10本リターン(方向調整、10本が順方向の時のみ)
  flat_bars20      直前20本中の「ほぼゼロ値幅」バー数(|Δclose| ≤ 0.05×catr20)
  zero_bars20      直前20本中の完全ゼロ値幅バー数(データ凍結)
  recross_n        エピソード(zが exit 域 ∓0.5 を最後に離れてから)内での
                   同方向 ±2 突破の再突入回数(0=初回タッチ)
  bars_stretch     exit 域を離れてからシグナルまでの経過バー数
  sig_hour         シグナルバーの UTC 時(ロールオーバー帯 20:00 の品質タグ)
  gap_capture      待機バー(シグナル→エントリー)で建玉方向に戻った幅 / シグナルバー値幅
                   ※ルール化禁止(exp55)。幻影監査の記述用のみ。

コホート: 各特徴の極端尾(95/99パーセンタイル or 自然な離散カット)のみ。
規約: n/平均bps/合計PnLシェア/勝率/IS(-2021)・OOS(2022-)/年次(単年>50%フラグ)/
L-S別/ブートストラップCI(1000回, コホート平均-残り平均)。n<30 は統計的に当てにならない。
ER(40)との重なり(相関・コホート内ER分布・ER≤0.45条件付き残存効果)を必ず併記。

実行: uv run python research/experiments/edge04_path_anomaly.py
出力: research/outputs/edge04_trades.csv / edge04_summary.json

結論(実行済み・確定):
  プール検算 n=1207 / sum=+1.9622 一致(z 再計算も 1207/1207 一致)。
  極端尾コホートで「平均が有意に負」または「利益が幻影的」と判定できる帯は **皆無**。
  ・暴力的クロス(dz_jump P95 n=61 +30.5bps 勝率89% / bar_range P95 n=61 +30.5bps /
    is_max50 n=137 +24.2bps)は むしろプール平均(+16.3bps)以上。IS/OOS 両正・
    スプレッド2倍でも正・gap_capture≈0(利益はスパイク戻りでなくエントリー後の回帰)。
    d1 の z ゲートが 1 バー型フラッシュを建玉前に既に間引いている。
  ・一方通行(oneway_run≥6 / counter_share10≤0.1)・accel 尾は平均がやや低い
    (+5〜+17bps)が CI が 0 を跨ぎ、符号は正のまま=拒否しても損なだけ。
  ・データ凍結(zero_bars20≥1)は n=25(<30=当てにならない)・平均+3.1bps・
    シェア+0.4%。中身は年末年始/休日週の閑散バーで、規模・有意性とも無視できる。
  ・唯一 CI が 0 を跨がないのは recross_n≥2(再突入, n=281, +26.4bps,
    diff CI[+2.4,+24.4]、ER≤0.45 内残存 CI[+0.4,+20.7]、IS+26.3/OOS+26.4)だが、
    これは**正のコホート**(z が exit 域に戻れず ±2 を何度も叩く「持続的伸長」ほど
    回帰が効く)。問題帯ではなく、補集合 first_touch も +12.9bps と正なので
    拒否・押し出しのどちらにも使えない(プール段の品質勾配の記述に留まる)。
  ・ロールオーバーBIDアーティファクト署名は無し: シグナルバー 20:00(n=87)も
    約定 20:00 UTC close(sig_hour=12, n=358, L+17.0/S+15.2bps)もロング膨張なし。
  → 拒否ルール候補の指名なし。「経路の異常」レンズではチャンピオンの数字は防衛された。
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
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
TS_OOS = pd.Timestamp("2022-01-01", tz="UTC")
TOTAL_PNL = 1.9622  # 規約のプール検算値
N_BOOT = 1000

WINDOW, ENTRY_Z, EXIT_Z, ER_WIN = 50, 2.0, 0.5, 40


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_path_features(pool: pd.DataFrame) -> pd.DataFrame:
    """シグナルバー(=エントリーバーの位置-1)確定値のみの経路特徴量。"""
    n = len(pool)
    cols = ["dz_jump", "bar_range_ratio", "is_max50", "oneway_run", "counter_share10",
            "accel", "flat_bars20", "zero_bars20", "recross_n", "bars_stretch",
            "sig_hour", "gap_capture", "er40_sig", "z_sig_check"]
    out = {c: np.full(n, np.nan) for c in cols}

    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        idx = close.index
        pos_of = pd.Series(np.arange(len(idx)), index=idx)

        z = _zscore(close, WINDOW).to_numpy()
        carr = close.to_numpy()
        diff = np.diff(carr, prepend=np.nan)            # Δclose
        amove = np.abs(diff)
        catr20 = pd.Series(amove, index=idx).rolling(20).mean().to_numpy()
        er_dir = (close - close.shift(ER_WIN)).abs()
        er_vol = close.diff().abs().rolling(ER_WIN).sum()
        er = (er_dir / er_vol).replace([np.inf, -np.inf], np.nan).to_numpy()
        hours = idx.hour.to_numpy()

        e_pos = pos_of.reindex(g["entry"]).to_numpy()
        for ti, ep, dirv in zip(g.index.to_numpy(), e_pos, g["dir"].to_numpy()):
            if not np.isfinite(ep):
                continue
            sp = int(ep) - 1                              # シグナルバー位置
            if sp < 60:
                continue
            dvar = float(dirv)
            out["z_sig_check"][ti] = abs(z[sp])
            out["dz_jump"][ti] = abs(z[sp] - z[sp - 1])
            out["bar_range_ratio"][ti] = (amove[sp] / catr20[sp - 1]
                                          if catr20[sp - 1] > 0 else np.nan)
            out["is_max50"][ti] = float(amove[sp] >= np.nanmax(amove[sp - 49:sp + 1]))
            # 一方通行: シグナル方向 = -dir(建玉と逆へ動いて z が突破した)
            sgn = np.sign(diff[sp - 9:sp + 1])
            out["counter_share10"][ti] = float((sgn == dvar).mean())
            run = 0
            for k in range(sp, sp - 10, -1):
                if np.sign(diff[k]) == -dvar:
                    run += 1
                else:
                    break
            out["oneway_run"][ti] = run
            m3 = -dvar * (carr[sp] / carr[sp - 3] - 1.0)
            m10 = -dvar * (carr[sp] / carr[sp - 10] - 1.0)
            out["accel"][ti] = m3 / m10 if m10 > 1e-12 else np.nan
            w20 = amove[sp - 19:sp + 1]
            thr = 0.05 * catr20[sp]
            out["flat_bars20"][ti] = float((w20 <= thr).sum())
            out["zero_bars20"][ti] = float((w20 == 0.0).sum())
            # z エピソード: exit 域(|z| ≤ 0.5 側)を最後に離れてからの再突入回数
            zthr, xthr = -dvar * ENTRY_Z, -dvar * EXIT_Z   # long: -2.0 / -0.5
            ep_start = sp - 1
            while ep_start > 0:
                if (dvar > 0 and z[ep_start] > xthr) or (dvar < 0 and z[ep_start] < xthr):
                    break
                ep_start -= 1
            rec = 0
            for q in range(ep_start + 1, sp):
                if dvar > 0 and z[q] < zthr <= z[q - 1]:
                    rec += 1
                elif dvar < 0 and z[q] > zthr >= z[q - 1]:
                    rec += 1
            out["recross_n"][ti] = rec
            out["bars_stretch"][ti] = sp - ep_start
            out["sig_hour"][ti] = hours[sp]
            # 待機バーの戻し(記述用のみ。ルール化禁止 exp55)
            if amove[sp] > 0 and sp + 1 < len(carr):
                out["gap_capture"][ti] = dvar * (carr[sp + 1] - carr[sp]) / amove[sp]
            out["er40_sig"][ti] = er[sp]
    return pd.DataFrame(out, index=pool.index)


def boot_diff_ci(a: np.ndarray, b: np.ndarray, rng, n_boot=N_BOOT):
    """コホート平均 - 残り平均 の差のブートストラップ95%CI(bps)。"""
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    da = np.empty(n_boot)
    for i in range(n_boot):
        da[i] = (rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean())
    return (float(np.percentile(da, 2.5) * 1e4), float(np.percentile(da, 97.5) * 1e4))


def cohort_report(df: pd.DataFrame, mask: pd.Series, name: str, definition: str,
                  rng) -> dict:
    sub, rest = df[mask], df[~mask]
    n = len(sub)
    res = {"name": name, "def": definition, "n": n}
    if n == 0:
        print(f"\n--- {name}: n=0 (該当なし)")
        return res
    r = sub["ret"]
    res.update(
        mean_bps=float(r.mean() * 1e4), sum_pnl=float(r.sum()),
        share_pct=float(r.sum() / TOTAL_PNL * 100), win_pct=float((r > 0).mean() * 100),
        median_bps=float(r.median() * 1e4))
    isq, oos = sub[sub["entry"] < TS_OOS], sub[sub["entry"] >= TS_OOS]
    res["is"] = {"n": len(isq), "mean_bps": float(isq["ret"].mean() * 1e4) if len(isq) else np.nan,
                 "sum": float(isq["ret"].sum())}
    res["oos"] = {"n": len(oos), "mean_bps": float(oos["ret"].mean() * 1e4) if len(oos) else np.nan,
                  "sum": float(oos["ret"].sum())}
    yearly = sub.groupby(sub["entry"].dt.year)["ret"].sum()
    res["yearly"] = {int(k): float(v) for k, v in yearly.items()}
    tot = r.sum()
    ymax = yearly.abs().idxmax()
    yshare = yearly[ymax] / tot if tot != 0 else np.nan
    res["max_year"] = int(ymax)
    res["max_year_share"] = float(yshare) if np.isfinite(yshare) else None
    res["single_year_flag"] = bool(np.isfinite(yshare) and yshare > 0.5 and
                                   np.sign(yearly[ymax]) == np.sign(tot))
    ls = {}
    for dv, lab in ((1, "L"), (-1, "S")):
        s = sub[sub["dir"] == dv]["ret"]
        ls[lab] = {"n": len(s), "mean_bps": float(s.mean() * 1e4) if len(s) else np.nan,
                   "sum": float(s.sum())}
    res["long_short"] = ls
    lo, hi = boot_diff_ci(r.to_numpy(), rest["ret"].to_numpy(), rng)
    res["diff_ci_bps"] = [lo, hi]
    res["diff_bps"] = float((r.mean() - rest["ret"].mean()) * 1e4)
    # ER 重なり
    res["er40_cohort"] = float(sub["er40_sig"].mean())
    res["er40_rest"] = float(rest["er40_sig"].mean())
    strict = df["er40_sig"] <= 0.45
    sub2, rest2 = df[mask & strict], df[~mask & strict]
    if len(sub2) >= 5:
        lo2, hi2 = boot_diff_ci(sub2["ret"].to_numpy(), rest2["ret"].to_numpy(), rng)
        res["er045_resid"] = {"n": len(sub2), "mean_bps": float(sub2["ret"].mean() * 1e4),
                              "diff_ci_bps": [lo2, hi2]}
    # 幻影監査: 短期保有への利益集中・スパイク戻り・追加コスト耐性
    short_hold = sub[sub["bars_held"] <= 2]["ret"].sum()
    res["pnl_share_hold_le2"] = float(short_hold / tot) if tot != 0 else None
    res["gap_capture_mean"] = float(sub["gap_capture"].mean())
    res["mean_bps_extra_cost"] = float((r - sub["cost_ret"]).mean() * 1e4)
    res["hour20_share"] = float((sub["sig_hour"] == 20).mean())

    flag = " ★n<30=統計的に当てにならない" if n < 30 else ""
    print(f"\n--- {name}  [{definition}]{flag}")
    print(f"  n={n}  mean={res['mean_bps']:+.1f}bps (med {res['median_bps']:+.1f})  "
          f"sum={res['sum_pnl']:+.4f} ({res['share_pct']:+.1f}% of pool)  "
          f"win={res['win_pct']:.0f}%")
    print(f"  diff vs rest = {res['diff_bps']:+.1f}bps  CI95[{lo:+.1f}, {hi:+.1f}]"
          f"{'  ←CIが0を跨がない' if lo * hi > 0 else '  (CI跨ぎ=ノイズと区別不能)'}")
    print(f"  IS n={res['is']['n']} {res['is']['mean_bps']:+.1f}bps / "
          f"OOS n={res['oos']['n']} {res['oos']['mean_bps']:+.1f}bps")
    ys = " ".join(f"{k}:{v:+.3f}" for k, v in res["yearly"].items())
    print(f"  年次(entry年): {ys}" +
          (f"  ★単年依存({res['max_year']})" if res["single_year_flag"] else ""))
    print(f"  L n={ls['L']['n']} {ls['L']['mean_bps']:+.1f}bps / "
          f"S n={ls['S']['n']} {ls['S']['mean_bps']:+.1f}bps")
    print(f"  ER40: cohort {res['er40_cohort']:.3f} vs rest {res['er40_rest']:.3f}" +
          (f" / ER≤0.45内残存: n={res['er045_resid']['n']} "
           f"{res['er045_resid']['mean_bps']:+.1f}bps "
           f"CI[{res['er045_resid']['diff_ci_bps'][0]:+.1f},"
           f"{res['er045_resid']['diff_ci_bps'][1]:+.1f}]" if "er045_resid" in res else ""))
    print(f"  幻影監査: 保有≤2barのPnLシェア={res['pnl_share_hold_le2']}, "
          f"待機バー戻し(gap_capture)平均={res['gap_capture_mean']:+.2f}, "
          f"スプレッド2倍時 mean={res['mean_bps_extra_cost']:+.1f}bps, "
          f"hour20シェア={res['hour20_share']:.0%}")
    return res


def main() -> int:
    t0 = time.time()
    rng = np.random.default_rng(0)
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL_PATH)
    sec(f"edge04: 直前経路の異常監査  pool n={len(pool)} sum={pool['ret'].sum():+.4f}")
    assert len(pool) == 1207, "プール n 不一致"
    assert abs(pool["ret"].sum() - 1.9622) < 1e-3, "プール sum 不一致"

    feats = build_path_features(pool)
    df = pd.concat([pool, feats], axis=1)

    # 1往復スプレッドの ret 換算(追加1往復の感度テスト用)
    from fxlab import config
    pip = pool["instr"].map(lambda s: 0.01 if s.endswith("JPY") else 0.0001)
    spr = pool["instr"].map(lambda s: config.SPREADS_PIPS.get(s, 1.0))
    df["cost_ret"] = (spr * pip / pool["entry_price"]).astype(float)

    # サニティ: z_entry(プール格納のシグナル|z|)と再計算が一致するか
    chk = (df["z_sig_check"] - df["z_entry"]).abs()
    print(f"z再計算サニティ: max|Δz| = {chk.max():.2e} (一致={int((chk < 1e-6).sum())}/{len(df)})")
    print(f"特徴量欠損: " + " ".join(f"{c}:{int(df[c].isna().sum())}" for c in feats.columns))

    sec("特徴量の分布(パーセンタイル)")
    cont = ["dz_jump", "bar_range_ratio", "oneway_run", "counter_share10", "accel",
            "flat_bars20", "zero_bars20", "recross_n", "bars_stretch", "gap_capture"]
    qs = df[cont].quantile([0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99])
    print(qs.to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nis_max50=1: {int(df['is_max50'].sum())}件 / "
          f"accel定義可(10本順方向): {int(df['accel'].notna().sum())}件")
    print("ERとの相関(prefilter後プール内): " + "  ".join(
        f"{c}:{df[c].corr(df['er40_sig']):+.2f}" for c in cont))
    print("retとの相関(中央域含む参考値・傾斜はML掃引済みで死区): " + "  ".join(
        f"{c}:{df[c].corr(df['ret']):+.2f}" for c in cont))

    sec("コホート分析(極端尾のみ)")
    P = lambda c, q: float(df[c].quantile(q))
    findings = []

    specs = [
        ("dz_jump_p95", df["dz_jump"] >= P("dz_jump", 0.95),
         f"dz_jump >= P95 ({P('dz_jump', 0.95):.3f})"),
        ("dz_jump_p99", df["dz_jump"] >= P("dz_jump", 0.99),
         f"dz_jump >= P99 ({P('dz_jump', 0.99):.3f})"),
        ("bar_range_p95", df["bar_range_ratio"] >= P("bar_range_ratio", 0.95),
         f"bar_range_ratio >= P95 ({P('bar_range_ratio', 0.95):.2f}x catr20)"),
        ("bar_range_p99", df["bar_range_ratio"] >= P("bar_range_ratio", 0.99),
         f"bar_range_ratio >= P99 ({P('bar_range_ratio', 0.99):.2f}x catr20)"),
        ("sigbar_is_max50", df["is_max50"] == 1,
         "シグナルバーの|Δclose|が直近50本で最大"),
        ("oneway_run_tail", df["oneway_run"] >= P("oneway_run", 0.95),
         f"oneway_run >= P95 ({P('oneway_run', 0.95):.0f}本連続)"),
        ("pure_oneway", df["counter_share10"] <= 0.1,
         "counter_share10 <= 0.1 (直前10本で戻しバー1本以下)"),
        ("accel_p95", df["accel"] >= P("accel", 0.95),
         f"accel >= P95 ({P('accel', 0.95):.2f}: 末期3本に動き集中)"),
        ("accel_p99", df["accel"] >= P("accel", 0.99),
         f"accel >= P99 ({P('accel', 0.99):.2f})"),
        ("flat_tail", df["flat_bars20"] >= max(2.0, P("flat_bars20", 0.95)),
         f"flat_bars20 >= max(2, P95={P('flat_bars20', 0.95):.0f}) (超閑散)"),
        ("data_freeze", df["zero_bars20"] >= 1,
         "zero_bars20 >= 1 (完全ゼロ値幅バー=データ凍結の疑い)"),
        ("first_touch", df["recross_n"] == 0, "初回タッチ(再突入なし)"),
        ("recross_ge1", df["recross_n"] >= 1, "再突入1回以上"),
        ("recross_ge2", df["recross_n"] >= 2, "再突入2回以上"),
        ("stretch_p95", df["bars_stretch"] >= P("bars_stretch", 0.95),
         f"bars_stretch >= P95 ({P('bars_stretch', 0.95):.0f}本: 長くexit域に戻れない)"),
        ("rollover_h20", df["sig_hour"] == 20,
         "シグナルバー=20:00 UTC (ロールオーバー帯の品質タグ)"),
        ("rollover_fill20", df["sig_hour"] == 12,
         "約定=20:00 UTC close (sig_hour=12 → 16:00バーのclose=ロールオーバー境界で建玉)"),
    ]
    for name, mask, definition in specs:
        findings.append(cohort_report(df, mask.fillna(False), name, definition, rng))

    sec("約定時刻×方向の分解(ロールオーバーBIDアーティファクト署名の検出)")
    print("署名: 20:00 UTC 約定でロング平均が膨張+ショート平均が萎縮、なら偽エッジ。")
    print("(fill時刻 = sig_hour + 8h。プールは H4 close 執行)")
    rows = []
    for h in sorted(df["sig_hour"].unique()):
        s = df[df["sig_hour"] == h]
        rows.append({"sig_hour": int(h), "fill_hour": int((h + 8) % 24), "n": len(s),
                     "mean_bps": s["ret"].mean() * 1e4,
                     "L_n": int((s["dir"] > 0).sum()),
                     "L_bps": s.loc[s["dir"] > 0, "ret"].mean() * 1e4,
                     "S_n": int((s["dir"] < 0).sum()),
                     "S_bps": s.loc[s["dir"] < 0, "ret"].mean() * 1e4})
    hour_tbl = pd.DataFrame(rows)
    print(hour_tbl.to_string(index=False, float_format=lambda x: f"{x:+.1f}"))

    sec("極端コホートの重複と個票(dz_jump P99)")
    top = df[df["dz_jump"] >= P("dz_jump", 0.99)].sort_values("dz_jump", ascending=False)
    show = top[["instr", "dir", "entry", "ret", "bars_held", "dz_jump", "bar_range_ratio",
                "gap_capture", "sig_hour", "er40_sig"]].copy()
    show["entry"] = show["entry"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    sec("データ凍結コホート個票(zero_bars20 >= 1)")
    fz = df[df["zero_bars20"] >= 1].sort_values("zero_bars20", ascending=False)
    show = fz[["instr", "dir", "entry", "ret", "bars_held", "zero_bars20", "flat_bars20",
               "sig_hour"]].copy()
    show["entry"] = show["entry"].dt.strftime("%Y-%m-%d %H:%M")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    df.to_csv(OUT_DIR / "edge04_trades.csv", index=False)
    (OUT_DIR / "edge04_summary.json").write_text(
        json.dumps({"pool": {"n": len(pool), "sum": float(pool["ret"].sum())},
                    "findings": findings}, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'edge04_trades.csv'} / edge04_summary.json")
    print(f"総経過 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
