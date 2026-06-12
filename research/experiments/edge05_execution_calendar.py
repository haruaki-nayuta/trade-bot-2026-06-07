"""edge05: 執行・カレンダー現実性監査(d1 プール) — ロールオーバー/時刻/出口側。

問い: d1 チャンピオンの約定タイムスタンプ(エントリー/出口とも H4 close)が
「実際には不利・約定困難な時間帯」(UTC20-22 ロールオーバー窓・日曜オープン薄商い・
金曜引け際)にどれだけ落ちており、そこに利益がどれだけ載っているか?

背景(既知): reports/12 = UTC20-22 のスプレッド爆発が BID データに偽エッジを量産。
reports/15 = v2(d0) で「20時バー汚染 ≤ 純益の2.3%」を測定済み。だが d1 はエントリーが
全部1バー後ろにズレたので時刻分布が変わっており再計測が必要。**出口側の執行現実性は
これまで一度も監査されていない。**

新規性 3 点:
  (a) d1 プールでの時刻分布×PnL の再計測(エントリー側)
  (b) 出口側の同分解(初)
  (c) ストレス再価格: 20:00 close 約定のスプレッド×3 / 日曜オープン×2 / パラノイド
      (M1実約定が 20-22 UTC 窓内のものも×3 + 0:00 close ×2)で +1.9622 がどこまで残るか

計測:
  1. エントリー/出口の約定時刻(= ラベル+4h)の UTC 時刻分布と時刻別 PnL。
  2. ラベル曜日 × 約定時刻のヒートマップ(件数と PnL)。日曜オープンバー
     (ラベル=日曜20:00, 実約定 日曜23:5x)と金曜遅バー(ラベル=金曜20:00,
     名目 close は土曜0:00 だが実約定は金曜21:5x = ロールオーバー窓のド真ん中)を特定。
  3. M1 突合: 全約定バーの「実際の最終ティック時刻」を M1(クロスは脚合成)で求め、
     名目時刻との乖離を暴く。さらに H4 close と「close 15-5 分前の M1 中央値」の乖離
     (dev_bps)を全時刻で測り、20:00 close だけ歪んでいないか対照群つきで検定。
     方向符号付き favorable_bps(+=約定価格が当該トレードに有利)で幻影利益を定量化。
  4. ストレス再価格(片側半スプレッド slip を倍率引き上げ、Δret≈-(m-1)*hs/side):
     S1 rollover20x3 / S2 sunday x2 / S3 = S1+S2 / S4 パラノイド。年次分解で全年プラス維持か。
  5. コホート規約レポート(n/平均bps/PnLシェア/勝率/IS-OOS/年次集中/L-S/ブートCI)。

実行: uv run python research/experiments/edge05_execution_calendar.py
出力: research/outputs/edge05_trades.csv / edge05_summary.json

結論(2026-06-13 実行): **執行・カレンダー面はほぼクリーン。チャンピオン防衛。**
  検算: n=1207 / sum=+1.9622 / コスト計上 median(計上/(hs_e+hs_x))=1.000 で一致。
  ・d1 で 20:00close(16:00ラベル)エントリーは 358件=PnLの29.1%(d0 の14.4%から倍増)だが
    件数シェア29.7%と比例 = per-trade +16.0bps はプール平均(+16.3bps)と同水準。
    diff-vs-rest CI [-10.4,+9.6]bps = 時刻優位なし。出口側(初監査)も同様に異常なし。
    出口PnLの44%は 16:00close(ロンドン/NY重複=最良流動性)で立つ = 良性。
  ・幻影監査: close vs 直前15-5分M1中央値の乖離は 20:00close の方が小さい
    (|dev| median 1.60 vs 2.24bps)。方向符号付き有利バイアスの対照超過は
    エントリー +0.07bps/件(=プール比0.13%)・出口はむしろ不利側 = 幻影なし。
  ・実約定が UTC20-22 窓内に落ちるのは金曜短縮バー(ラベル金20:00, 実約定 20-21:5x)のみ:
    エントリー29件(sum -0.0154 = 負!)/ 出口11件(+0.0316)。利益は窓内に載っていない。
  ・ストレス再価格: S1(20:00close×3スプレッド)Δ-0.1038 → +1.8584(94.7%残存)、
    S2(日曜×2)Δ-0.0043、S4 パラノイド(+実約定窓内×3 / 0:00close×2)でも
    +1.8215(92.8%残存)。**全シナリオで全暦年プラス維持。**
  ・唯一の veto 候補: 金曜遅バー(ラベル金20:00)エントリー禁止。raw -5.3bps /
    再価格後 -7.3bps・diff CI [-46.9,-1.5] で rest 比有意に劣後。ただし n=29(<30)・
    OOS は +14.0bps(+0.018)・2018単年集中のため弱い候補(口座レベル検証要)。
    実運用上は金曜21:55の薄商い+直後週末という執行困難性の除去が主動機。
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
sys.path.insert(0, str(ROOT / "research" / "lab"))

from fxlab import config  # noqa: E402
from fxlab import universe as uni  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.universe import CROSS_DEFS  # noqa: E402

pd.set_option("display.width", 240)

OUT_DIR = ROOT / "research" / "outputs"
POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
TOTAL_EXPECT = 1.9622
N_EXPECT = 1207
N_BOOT = 1000
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# M1 close(メジャー=実データ / クロス=脚 inner-join 合成, exp52 方式)
# ---------------------------------------------------------------------------
_LEG: dict[str, pd.Series] = {}


def m1_close(name: str) -> pd.Series:
    """tz-naive UTC index の M1 close。脚はキャッシュ、クロス合成はその場で作る。"""
    if name in CROSS_DEFS:
        a, op, b = CROSS_DEFS[name]
        ca, cb = m1_close(a), m1_close(b)
        df = pd.concat([ca.rename("a"), cb.rename("b")], axis=1, join="inner").dropna()
        return df["a"] / df["b"] if op == "/" else df["a"] * df["b"]
    if name not in _LEG:
        c = load_m1(name)["close"]
        _LEG[name] = pd.Series(c.to_numpy(), index=c.index.tz_localize(None))
    return _LEG[name]


# ---------------------------------------------------------------------------
# H4 close 再構成(entry_close / exit_close / 半スプレッド率)
# ---------------------------------------------------------------------------
def reconstruct(pool: pd.DataFrame) -> pd.DataFrame:
    n = len(pool)
    out = pd.DataFrame(index=pool.index)
    ec = np.full(n, np.nan)
    xc = np.full(n, np.nan)
    for instr, g in pool.groupby("instr"):
        s = uni.instrument_data(instr, "H4")["close"]
        ie = s.index.get_indexer(g["entry"])
        ix = s.index.get_indexer(g["exit"])
        assert (ie >= 0).all() and (ix >= 0).all(), f"{instr}: timestamp miss"
        rows = g.index.to_numpy()
        ec[rows] = s.to_numpy()[ie]
        xc[rows] = s.to_numpy()[ix]
    out["entry_close"] = ec
    out["exit_close"] = xc
    sp = pool["instr"].map(lambda i: config.spread_pips(i) * config.pip_size(i))
    out["hs_e"] = sp.to_numpy() / 2.0 / ec  # 半スプレッド率(エントリー側 slip)
    out["hs_x"] = sp.to_numpy() / 2.0 / xc
    d = pool["dir"].to_numpy().astype(float)
    out["gross"] = d * (xc / ec - 1.0)
    out["cost_model"] = out["gross"] - pool["ret"].to_numpy()
    return out


# ---------------------------------------------------------------------------
# M1 突合: 実最終ティック時刻 + close前 15-5 分中央値との乖離
# ---------------------------------------------------------------------------
def m1_exec_features(pool: pd.DataFrame) -> pd.DataFrame:
    n = len(pool)
    out = {
        "act_e_ts": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "act_x_ts": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "m1_e_px": np.full(n, np.nan), "m1_x_px": np.full(n, np.nan),
        "dev_e_bps": np.full(n, np.nan), "dev_x_bps": np.full(n, np.nan),
    }
    for instr, g in pool.groupby("instr"):
        s = m1_close(instr)
        at = s.index.values
        ac = s.to_numpy()
        rows = g.index.to_numpy()
        for side, labcol in (("e", "entry"), ("x", "exit")):
            labs = g[labcol].dt.tz_localize(None).values
            tend = labs + np.timedelta64(4, "h")
            pos = np.searchsorted(at, tend, side="left")
            ok = pos > 0
            last_i = np.maximum(pos - 1, 0)
            last_ts = at[last_i]
            ok &= last_ts >= labs  # バー内にティックがある
            out[f"act_{side}_ts"][rows[ok]] = last_ts[ok]
            out[f"m1_{side}_px"][rows[ok]] = ac[last_i][ok]
            # 「close の 15-5 分前」窓の M1 中央値 vs H4 close
            lo = np.searchsorted(at, last_ts - np.timedelta64(15, "m"), "left")
            hi = np.searchsorted(at, last_ts - np.timedelta64(5, "m"), "right")
            for k in np.where(ok)[0]:
                a, b = lo[k], hi[k]
                if b - a >= 3:
                    med = float(np.median(ac[a:b]))
                    out[f"dev_{side}_bps"][rows[k]] = (ac[last_i[k]] / med - 1.0) * 1e4
    return pd.DataFrame(out, index=pool.index)


# ---------------------------------------------------------------------------
# コホート規約レポート
# ---------------------------------------------------------------------------
def boot_ci(x: np.ndarray, seed=0):
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def boot_diff_ci(a: np.ndarray, b: np.ndarray, seed=0):
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    ma = rng.choice(a, size=(N_BOOT, len(a)), replace=True).mean(axis=1)
    mb = rng.choice(b, size=(N_BOOT, len(b)), replace=True).mean(axis=1)
    return tuple(np.percentile(ma - mb, [2.5, 97.5]))


def cohort_report(df: pd.DataFrame, mask: pd.Series, name: str, total: float,
                  extra_cost: np.ndarray | None = None) -> dict:
    sub = df[mask]
    rest = df[~mask]
    n = len(sub)
    rec = {"name": name, "n": int(n)}
    if n == 0:
        print(f"\n--- {name}: n=0(該当なし)")
        return rec
    r = sub["ret"].to_numpy()
    rec.update({
        "mean_bps": float(r.mean() * 1e4),
        "sum": float(r.sum()),
        "share_pct": float(r.sum() / total * 100),
        "win_pct": float((r > 0).mean() * 100),
    })
    ci = boot_ci(r)
    dci = boot_diff_ci(r, rest["ret"].to_numpy())
    rec["ci_mean_bps"] = [float(ci[0] * 1e4), float(ci[1] * 1e4)]
    rec["ci_diff_bps"] = [float(dci[0] * 1e4), float(dci[1] * 1e4)]
    is_m = sub["entry"] < pd.Timestamp("2022-01-01", tz="UTC")
    rec["is"] = {"n": int(is_m.sum()), "mean_bps": float(sub.loc[is_m, "ret"].mean() * 1e4) if is_m.any() else np.nan,
                 "sum": float(sub.loc[is_m, "ret"].sum())}
    rec["oos"] = {"n": int((~is_m).sum()), "mean_bps": float(sub.loc[~is_m, "ret"].mean() * 1e4) if (~is_m).any() else np.nan,
                  "sum": float(sub.loc[~is_m, "ret"].sum())}
    yr = sub.groupby(sub["entry"].dt.year)["ret"].sum()
    rec["yearly"] = {int(k): float(v) for k, v in yr.items()}
    tot = r.sum()
    rec["max_year_share"] = float((yr.abs().max() / abs(tot)) if tot != 0 else np.nan)
    rec["single_year_flag"] = bool(rec["max_year_share"] > 0.5) if np.isfinite(rec["max_year_share"]) else False
    lm = sub["dir"] > 0
    rec["long"] = {"n": int(lm.sum()), "sum": float(sub.loc[lm, "ret"].sum()),
                   "mean_bps": float(sub.loc[lm, "ret"].mean() * 1e4) if lm.any() else np.nan}
    rec["short"] = {"n": int((~lm).sum()), "sum": float(sub.loc[~lm, "ret"].sum()),
                    "mean_bps": float(sub.loc[~lm, "ret"].mean() * 1e4) if (~lm).any() else np.nan}
    if extra_cost is not None:
        ec = extra_cost[mask.to_numpy()]
        rec["repriced_mean_bps"] = float((r - ec).mean() * 1e4)
        rec["repriced_sum"] = float((r - ec).sum())
    print(f"\n--- {name}")
    print(f"  n={n}  mean={rec['mean_bps']:+.1f}bps  sum={rec['sum']:+.4f} "
          f"(シェア {rec['share_pct']:+.1f}%)  win={rec['win_pct']:.1f}%")
    print(f"  mean 95%CI [{rec['ci_mean_bps'][0]:+.1f}, {rec['ci_mean_bps'][1]:+.1f}]bps  "
          f"diff-vs-rest CI [{rec['ci_diff_bps'][0]:+.1f}, {rec['ci_diff_bps'][1]:+.1f}]bps")
    print(f"  IS n={rec['is']['n']} {rec['is']['mean_bps']:+.1f}bps sum={rec['is']['sum']:+.3f} | "
          f"OOS n={rec['oos']['n']} {rec['oos']['mean_bps']:+.1f}bps sum={rec['oos']['sum']:+.3f}")
    print("  年次: " + " ".join(f"{k}:{v:+.3f}" for k, v in rec["yearly"].items()) +
          f"  最大単年シェア {rec['max_year_share']:.0%}" +
          ("  **単年依存**" if rec["single_year_flag"] else ""))
    print(f"  Long n={rec['long']['n']} sum={rec['long']['sum']:+.3f} ({rec['long']['mean_bps']:+.1f}bps) | "
          f"Short n={rec['short']['n']} sum={rec['short']['sum']:+.3f} ({rec['short']['mean_bps']:+.1f}bps)")
    if extra_cost is not None:
        print(f"  現実的再価格後: mean={rec['repriced_mean_bps']:+.1f}bps  sum={rec['repriced_sum']:+.4f}")
    if n < 30:
        print("  ※ n<30: 統計的に当てにならない")
        rec["small_n"] = True
    return rec


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = pd.read_parquet(POOL_PATH)
    total = float(pool["ret"].sum())
    print(f"=== edge05: 執行・カレンダー現実性監査 (d1 pool n={len(pool)}, sum={total:+.4f}) ===")
    assert len(pool) == N_EXPECT and abs(total - TOTAL_EXPECT) < 1e-3, "プール検算失敗"

    rc = reconstruct(pool)
    df = pd.concat([pool, rc], axis=1)
    # コスト検算: モデル計上コスト ≈ hs_e + hs_x(往復1スプレッド)
    ratio = (df["cost_model"] / (df["hs_e"] + df["hs_x"])).median()
    print(f"コスト検算: median(計上コスト / (hs_e+hs_x)) = {ratio:.3f}(≈1 なら整合)")

    df["e_exec"] = df["entry"] + pd.Timedelta(hours=4)   # 名目約定時刻
    df["x_exec"] = df["exit"] + pd.Timedelta(hours=4)
    df["e_hour"] = df["e_exec"].dt.hour
    df["x_hour"] = df["x_exec"].dt.hour
    df["e_lab_wd"] = df["entry"].dt.dayofweek   # バーラベルの曜日
    df["x_lab_wd"] = df["exit"].dt.dayofweek
    df["e_lab_hour"] = df["entry"].dt.hour
    df["x_lab_hour"] = df["exit"].dt.hour

    # --- 1. 約定時刻分布(エントリー/出口) -----------------------------------
    sec("1. 約定時刻(=ラベル+4h, UTC)分布 × PnL — エントリー側 / 出口側")
    for side, hcol in (("エントリー", "e_hour"), ("出口", "x_hour")):
        g = df.groupby(hcol)["ret"]
        t = pd.DataFrame({"n": g.size(), "mean_bps": g.mean() * 1e4, "sum": g.sum(),
                          "share_pct": g.sum() / total * 100,
                          "win_pct": g.apply(lambda s: (s > 0).mean() * 100)})
        print(f"\n[{side}] close 時刻別:")
        print(t.to_string(float_format=lambda x: f"{x:+.2f}"))

    # --- 2. ラベル曜日 × 約定時刻ヒートマップ --------------------------------
    sec("2. バーラベル曜日 × 約定時刻 ヒートマップ(件数 / PnL合計)")
    for side, wdc, hc in (("エントリー", "e_lab_wd", "e_hour"), ("出口", "x_lab_wd", "x_hour")):
        cnt = df.pivot_table(index=wdc, columns=hc, values="ret", aggfunc="size", fill_value=0)
        pnl = df.pivot_table(index=wdc, columns=hc, values="ret", aggfunc="sum", fill_value=0.0)
        cnt.index = [WD[i] for i in cnt.index]
        pnl.index = [WD[i] for i in pnl.index]
        print(f"\n[{side}] 件数:")
        print(cnt.to_string())
        print(f"[{side}] PnL合計:")
        print(pnl.to_string(float_format=lambda x: f"{x:+.3f}"))

    # 特殊バー: 日曜オープン(ラベル日曜)・金曜遅バー(ラベル金曜20:00)
    df["e_sunday_open"] = df["e_lab_wd"] == 6
    df["x_sunday_open"] = df["x_lab_wd"] == 6
    df["e_fri_late"] = (df["e_lab_wd"] == 4) & (df["e_lab_hour"] == 20)
    df["x_fri_late"] = (df["x_lab_wd"] == 4) & (df["x_lab_hour"] == 20)
    df["e_fri20"] = (df["e_lab_wd"] == 4) & (df["e_hour"] == 20)  # 金曜16:00ラベル=金曜20:00close
    df["x_fri20"] = (df["x_lab_wd"] == 4) & (df["x_hour"] == 20)

    # --- 3. M1 突合: 実約定時刻と close 乖離 ---------------------------------
    sec("3. M1 突合: 実最終ティック時刻 / close前15-5分中央値との乖離")
    mf = m1_exec_features(pool)
    df = pd.concat([df, mf], axis=1)
    px_chk_e = ((df["m1_e_px"] - df["entry_close"]).abs() / df["entry_close"] * 1e4)
    print(f"検算: M1最終ティック価格 vs H4 close 乖離 median={px_chk_e.median():.3f}bps "
          f"p95={px_chk_e.quantile(0.95):.3f}bps(≈0 なら M1 と H4 が同一ソース)")
    for side in ("e", "x"):
        act = pd.DatetimeIndex(df[f"act_{side}_ts"])
        nom = df[f"{side}_exec"].dt.tz_localize(None)
        lagm = (nom - act).dt.total_seconds() / 60.0
        df[f"act_{side}_hour"] = act.hour
        df[f"trunc_{side}"] = lagm > 10  # 名目より10分以上早く終わった「短縮バー」
        lbl = "エントリー" if side == "e" else "出口"
        print(f"\n[{lbl}] 名目close − 実最終ティック: median={lagm.median():.1f}分, "
              f">10分(短縮バー)= {int((lagm > 10).sum())} 件")
        tr = df[df[f"trunc_{side}"]]
        if len(tr):
            vc = tr.groupby([f"{side}_lab_wd", f"act_{side}_hour"])["ret"].agg(["size", "sum"])
            vc.index = [f"{WD[w]} act{h:02d}h" for w, h in vc.index]
            print(f"  短縮バーの内訳(ラベル曜日×実約定時, n/sum):")
            print("  " + vc.to_string().replace("\n", "\n  "))
        # 実約定が UTC20-22 窓内
        df[f"{side}_in_roll"] = pd.Series(act.hour, index=df.index).isin([20, 21]) & act.notna()

    n_e_roll = int(df["e_in_roll"].sum())
    n_x_roll = int(df["x_in_roll"].sum())
    print(f"\n実約定が UTC20-22 窓内: エントリー {n_e_roll} 件 / 出口 {n_x_roll} 件")

    # close 乖離(dev_bps)の対照比較: 20:00 close vs その他
    sec("3b. close乖離 dev_bps(close vs 直前15-5分中央値): 20:00close は歪んでいるか")
    for side, hcol in (("e", "e_hour"), ("x", "x_hour")):
        lbl = "エントリー" if side == "e" else "出口"
        dv = df[f"dev_{side}_bps"]
        d20 = dv[df[hcol] == 20]
        rest = dv[df[hcol] != 20]
        print(f"\n[{lbl}] |dev| median: 20:00close={d20.abs().median():.2f}bps "
              f"vs その他={rest.abs().median():.2f}bps | "
              f"|dev|>5bps 率: {(d20.abs() > 5).mean():.1%} vs {(rest.abs() > 5).mean():.1%}")
        # 方向符号付き: トレードに有利な向きの乖離(幻影利益の定量化)
        sign = -df["dir"] if side == "e" else df["dir"]
        fav = (sign * dv).rename(f"fav_{side}_bps")
        df[f"fav_{side}_bps"] = fav
        f20, frest = fav[df[hcol] == 20], fav[df[hcol] != 20]
        ci20 = boot_ci(f20.dropna().to_numpy())
        print(f"  favorable bias(+=有利約定): 20:00close mean={f20.mean():+.2f}bps "
              f"[CI {ci20[0]:+.2f},{ci20[1]:+.2f}] vs その他 {frest.mean():+.2f}bps")
        print(f"  20:00close 有利バイアス合計 = {f20.sum() / 1e4:+.4f}(プール比 "
              f"{f20.sum() / 1e4 / total * 100:+.2f}%)")

    # --- 4. ストレス再価格 -----------------------------------------------------
    sec("4. ストレス再価格(Δret = -(m-1)*半スプレッド/該当サイド)")
    hs_e = df["hs_e"].to_numpy()
    hs_x = df["hs_x"].to_numpy()
    e20 = (df["e_hour"] == 20).to_numpy()
    x20 = (df["x_hour"] == 20).to_numpy()
    e0 = (df["e_hour"] == 0).to_numpy()
    x0 = (df["x_hour"] == 0).to_numpy()
    e_sun = df["e_sunday_open"].to_numpy()
    x_sun = df["x_sunday_open"].to_numpy()
    e_roll = df["e_in_roll"].to_numpy()
    x_roll = df["x_in_roll"].to_numpy()

    def scenario(me: np.ndarray, mx: np.ndarray, name: str) -> dict:
        extra = (me - 1.0) * hs_e + (mx - 1.0) * hs_x
        new = df["ret"].to_numpy() - extra
        yr = pd.Series(new, index=df.index).groupby(df["entry"].dt.year).sum()
        rec = {"name": name, "delta": float(-extra.sum()), "new_total": float(new.sum()),
               "kept_pct": float(new.sum() / total * 100),
               "n_affected": int((extra > 0).sum()),
               "years_negative": [int(y) for y, v in yr.items() if v < 0],
               "yearly": {int(k): float(v) for k, v in yr.items()}}
        print(f"\n[{name}] 影響 {rec['n_affected']}件  Δ={rec['delta']:+.4f}  "
              f"新合計={rec['new_total']:+.4f}({rec['kept_pct']:.1f}% 残存)  "
              f"マイナス年={rec['years_negative'] or 'なし'}")
        print("  年次: " + " ".join(f"{k}:{v:+.3f}" for k, v in rec["yearly"].items()))
        return rec, extra

    ones = np.ones(len(df))
    s1_me = np.where(e20, 3.0, 1.0)
    s1_mx = np.where(x20, 3.0, 1.0)
    s1, extra_s1 = scenario(s1_me, s1_mx, "S1: 20:00close 約定スプレッド×3")
    s2_me = np.where(e_sun, 2.0, 1.0)
    s2_mx = np.where(x_sun, 2.0, 1.0)
    s2, _ = scenario(s2_me, s2_mx, "S2: 日曜オープンバー約定×2")
    s3, _ = scenario(np.maximum(s1_me, s2_me), np.maximum(s1_mx, s2_mx), "S3: S1+S2")
    s4_me = np.maximum.reduce([s1_me, s2_me, np.where(e_roll, 3.0, 1.0), np.where(e0, 2.0, 1.0)])
    s4_mx = np.maximum.reduce([s1_mx, s2_mx, np.where(x_roll, 3.0, 1.0), np.where(x0, 2.0, 1.0)])
    s4, extra_s4 = scenario(s4_me, s4_mx, "S4: パラノイド(+実約定20-22窓×3 / 0:00close×2)")

    # --- 5. コホート規約レポート ----------------------------------------------
    sec("5. コホート規約レポート(ブートストラップCI 1000回)")
    cohorts = []
    # 20:00 close エントリーの再価格は ×3 ストレスの自コスト分
    extra_e20 = np.where(e20, 2.0 * hs_e, 0.0)
    extra_x20 = np.where(x20, 2.0 * hs_x, 0.0)
    cohorts.append(cohort_report(df, df["e_hour"] == 20, "C1 エントリー約定@20:00close(=16:00ラベル)", total, extra_e20))
    cohorts.append(cohort_report(df, df["x_hour"] == 20, "C2 出口約定@20:00close(=16:00ラベル)", total, extra_x20))
    cohorts.append(cohort_report(df, df["e_hour"] == 0, "C3 エントリー約定@0:00close(=20:00ラベル)", total,
                                 np.where(e0, 1.0 * hs_e, 0.0)))
    cohorts.append(cohort_report(df, df["x_hour"] == 0, "C4 出口約定@0:00close(=20:00ラベル)", total,
                                 np.where(x0, 1.0 * hs_x, 0.0)))
    cohorts.append(cohort_report(df, df["e_sunday_open"], "C5 日曜オープンバーでエントリー", total,
                                 np.where(e_sun, 1.0 * hs_e, 0.0)))
    cohorts.append(cohort_report(df, df["x_sunday_open"], "C6 日曜オープンバーで出口", total,
                                 np.where(x_sun, 1.0 * hs_x, 0.0)))
    cohorts.append(cohort_report(df, df["e_fri_late"], "C7 金曜遅バー(ラベル金20:00, 実約定21:5x)エントリー", total,
                                 np.where(df["e_fri_late"].to_numpy(), 2.0 * hs_e, 0.0)))
    cohorts.append(cohort_report(df, df["x_fri_late"], "C8 金曜遅バー出口", total,
                                 np.where(df["x_fri_late"].to_numpy(), 2.0 * hs_x, 0.0)))
    cohorts.append(cohort_report(df, df["e_fri20"], "C9 金曜20:00close(週末直前ロールオーバー)エントリー", total,
                                 np.where(df["e_fri20"].to_numpy(), 2.0 * hs_e, 0.0)))
    cohorts.append(cohort_report(df, df["x_fri20"], "C10 金曜20:00close 出口", total,
                                 np.where(df["x_fri20"].to_numpy(), 2.0 * hs_x, 0.0)))
    cohorts.append(cohort_report(df, df["e_in_roll"], "C11 実約定がUTC20-22窓内エントリー(M1実測)", total,
                                 np.where(e_roll, 2.0 * hs_e, 0.0)))
    cohorts.append(cohort_report(df, df["x_in_roll"], "C12 実約定がUTC20-22窓内出口(M1実測)", total,
                                 np.where(x_roll, 2.0 * hs_x, 0.0)))

    # --- 保存 -------------------------------------------------------------------
    keep = [c for c in df.columns if not c.startswith("m1_")]
    df[keep].to_csv(OUT_DIR / "edge05_trades.csv", index=False)
    summary = {
        "pool": {"n": len(pool), "total": total},
        "cost_model_ratio_median": float(ratio),
        "scenarios": [s1, s2, s3, s4],
        "cohorts": cohorts,
        "fav_bias_20close": {
            "entry_sum": float(df.loc[df["e_hour"] == 20, "fav_e_bps"].sum() / 1e4),
            "exit_sum": float(df.loc[df["x_hour"] == 20, "fav_x_bps"].sum() / 1e4),
        },
        "n_in_rollover_window": {"entry": n_e_roll, "exit": n_x_roll},
    }
    (OUT_DIR / "edge05_summary.json").write_text(
        json.dumps(summary, indent=2, default=float, ensure_ascii=False))
    print(f"\nsaved -> {OUT_DIR / 'edge05_trades.csv'} / edge05_summary.json")
    print(f"総経過 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
