"""exp41: 実現ボラ推定窓 vol_sd_win 20→{15,25,30,35,40} のレバー検証。

背景: チャンピオン v2 の calm フィルタは「20本実現ボラ ≤ rolling(100)の70%分位」。
reports/15 の閾値マップ(anatomy_gap_2)で vol_sd_win=25 が唯一「露出ほぼ不変で全指標改善」
(sum +6.3% / PF1.71→1.79 / 両時代改善 / 11年プラス維持)した軸だが、断面 15→20→25 が
単調増の**グリッド端**だった。本実験は 15..40 へ断面を延長し、
  (1) プールレベルで高原か単調増(=端逃げ)かを判定
  (2) 口座レベル(mm_lab.simulate × champion_sizing P=4, mp8)で同一テール判定プロトコル
      (ペアシード robust 較正 / レバ偽装署名 / IS-argmax / 2022除外 / 全年プラス)
を実測する。シグナル再実装は anatomy_gap_2.py(参照プール 1e-6 一致検証済み)を流用。

実行: PYTHONPATH=. uv run python research/experiments/exp41_volwin.py
      PYTHONPATH=. uv run python research/experiments/exp41_volwin.py --fine
        (追補: 22/24/26/28 の細断面=スパイクか狭い高原かの判定 +
         w20/w25 の emp_k における p95 をブートシード0-2でペア測定=署名のノイズ監査)
出力: research/outputs/exp41_volwin_pool.csv / exp41_volwin_account.csv / exp41_volwin.json
      (--fine: exp41_volwin_fine.csv)
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    calibrate_empirical,
    calibrate_robust_seeded,
    cagr_of,
    max_dd,
    protocol_eval,
    yearly_returns,
)

import vectorbt as vbt  # noqa: E402
from fxlab import config, universe as uni  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
MAX_POS = 8
WINDOWS = [15, 20, 25, 30, 35, 40]
BASE_WIN = 20
SEEDS_FULL = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
POOL_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"
OUT_DIR = ROOT / "research" / "outputs"
OUT_POOL = OUT_DIR / "exp41_volwin_pool.csv"
OUT_ACC = OUT_DIR / "exp41_volwin_account.csv"
OUT_JSON = OUT_DIR / "exp41_volwin.json"

# 本番ベースライン(anatomy_gap_2.BASE と同値)
BASE = {
    "window": 50, "entry_z": 2.0, "exit_z": 0.5,
    "rsi_p": 14, "rsi_low": 35, "rsi_high": 65,
    "vol_sd_win": 20, "vol_win": 100, "vol_pct": 0.70,
    "slow_win": 250, "slow_z": 1.75,
    "er_win": 40, "er_max": 0.55,
}


# --- シグナル再実装(anatomy_gap_2 検証済みコードの流用) -----------------
def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def _er(close: pd.Series, w: int) -> pd.Series:
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


class IndicatorCache:
    def __init__(self, close: pd.Series):
        self.close = close
        self._z, self._rsi, self._vol, self._calm, self._erc = {}, {}, {}, {}, {}

    def z(self, w):
        if w not in self._z:
            self._z[w] = _zscore(self.close, w)
        return self._z[w]

    def rsi(self, p):
        if p not in self._rsi:
            self._rsi[p] = vbt.RSI.run(self.close, p).rsi
        return self._rsi[p]

    def vol(self, sd_win):
        if sd_win not in self._vol:
            self._vol[sd_win] = self.close.pct_change().rolling(sd_win).std()
        return self._vol[sd_win]

    def calm(self, sd_win, vol_win, vol_pct):
        key = (sd_win, vol_win, vol_pct)
        if key not in self._calm:
            v = self.vol(sd_win)
            self._calm[key] = v <= v.rolling(vol_win).quantile(vol_pct)
        return self._calm[key]

    def er(self, w):
        if w not in self._erc:
            self._erc[w] = _er(self.close, w)
        return self._erc[w]


def make_gen(cache: IndicatorCache):
    def gen(data: pd.DataFrame, *, window, entry_z, exit_z, rsi_p, rsi_low, rsi_high,
            vol_sd_win, vol_win, vol_pct, slow_win, slow_z, er_win, er_max):
        z = cache.z(window)
        rsi = cache.rsi(rsi_p)
        calm = cache.calm(vol_sd_win, vol_win, vol_pct)
        zs = cache.z(slow_win)
        long_ok = (zs < -slow_z).fillna(False)
        short_ok = (zs > slow_z).fillna(False)
        er_ok = (cache.er(er_win) <= er_max).fillna(False)
        le = (z < -entry_z) & (z.shift() >= -entry_z) & (rsi < rsi_low) & calm & long_ok & er_ok
        se = (z > entry_z) & (z.shift() <= entry_z) & (rsi > rsi_high) & calm & short_ok & er_ok
        lx = z > -exit_z
        sx = z < exit_z
        return le.fillna(False), lx.fillna(False), se.fillna(False), sx.fillna(False)

    return gen


# --- mm スキーマのプール構築(mm_lab.build_pool 互換列) -------------------
def build_mm_pool(datas, caches, params) -> pd.DataFrame:
    frames = []
    win = params["window"]
    for nm, data in datas.items():
        pf = run(nm, TF, make_gen(caches[nm]), params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        z = caches[nm].z(win)
        vol = caches[nm].vol(20)  # 参考列(champion_sizing は不使用)
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(z.reindex(tt["entry"]).to_numpy()),
            "vol_entry": vol.reindex(tt["entry"]).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


def pool_stats(pool: pd.DataFrame) -> dict:
    r = pool["ret"]
    pos, neg = r[r > 0].sum(), r[r < 0].sum()
    yearly = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    era1 = pool.loc[pd.to_datetime(pool["exit"]) < "2022-01-01", "ret"].sum()
    era2 = pool.loc[pd.to_datetime(pool["exit"]) >= "2022-01-01", "ret"].sum()
    return {
        "n": int(len(pool)), "sum_ret": float(r.sum()),
        "mean_bps": float(r.mean() * 1e4), "win": float((r > 0).mean()),
        "pf": float(pos / abs(neg)) if neg < 0 else np.inf,
        "pos_years": int((yearly > 0).sum()), "n_years": int(len(yearly)),
        "worst_year": float(yearly.min()),
        "era_16_21": float(era1), "era_22_26": float(era2),
        "med_hold": float(pool["bars_held"].median()),
    }


# --- 口座レベル評価 --------------------------------------------------------
def make_eq_fn(pool, closes, mk):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    return eq_of_k


def eval_window(label, pool, closes, seeds):
    """empirical + robust(seeds) + IS較正(emp/robust)→OOS素検証。"""
    mk = champion_sizing(pool, max_pos=MAX_POS)
    eq_of_k = make_eq_fn(pool, closes, mk)
    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    eq_emp = eq_of_k(res["emp_k"])
    yr_emp = yearly_returns(eq_emp)
    res["worst_year"] = float(yr_emp.min())
    res["neg_years_emp"] = int((yr_emp < 0).sum())
    res["yr_emp"] = {int(y): float(v) for y, v in yr_emp.items()}
    # robust seed0 の k での年次も保持(2022除外チェック用)
    eq_r0 = eq_of_k(res["rob"][seeds[0]]["k"])
    yr_r0 = yearly_returns(eq_r0)
    res["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())

    # IS(<2022) 較正 → OOS 素検証
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    eq_is_of_k = make_eq_fn(is_pool, is_cl, mk)
    eq_oos_of_k = make_eq_fn(oos_pool, oos_cl, mk)

    k_is_emp = calibrate_empirical(eq_is_of_k, 0.20)
    res["k_is_emp"] = k_is_emp
    res["is_emp_cagr"] = cagr_of(eq_is_of_k(k_is_emp))
    eqo = eq_oos_of_k(k_is_emp)
    res["oos_emp_cagr"] = cagr_of(eqo)
    res["oos_emp_dd"] = max_dd(eqo)

    k_is_rob = calibrate_robust_seeded(eq_is_of_k, 0.20, seed=0)
    res["k_is_rob"] = k_is_rob
    res["is_rob_cagr"] = cagr_of(eq_is_of_k(k_is_rob))
    eqor = eq_oos_of_k(k_is_rob)
    res["oos_rob_cagr"] = cagr_of(eqor)
    res["oos_rob_dd"] = max_dd(eqor)
    print(f"      IS emp k={k_is_emp:5.2f} ISC={res['is_emp_cagr']:+7.2%} -> "
          f"OOS {res['oos_emp_cagr']:+7.2%} (DD {res['oos_emp_dd']:+5.1%}) | "
          f"IS rob k={k_is_rob:5.2f} ISC={res['is_rob_cagr']:+7.2%} -> "
          f"OOS {res['oos_rob_cagr']:+7.2%} (DD {res['oos_rob_dd']:+5.1%})")
    return res


def fine_main() -> int:
    """追補: 細断面 22/24/26/28 + p95署名のブートシード監査。"""
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    instruments = mm.default_instruments()
    datas = {nm: uni.instrument_data(nm, TF) for nm in instruments}
    caches = {nm: IndicatorCache(datas[nm]["close"]) for nm in instruments}
    closes = mm.load_closes()
    from tail_protocol import boot_dd  # noqa: PLC0415

    fine_wins = [20, 22, 24, 25, 26, 28, 30]
    print(f"=== exp41 --fine: 細断面 {fine_wins} + p95署名シード監査 ===\n")
    rows = []
    pools = {}
    for w in fine_wins:
        p = dict(BASE)
        p["vol_sd_win"] = w
        pool = build_mm_pool(datas, caches, p)
        pools[w] = pool
        st = pool_stats(pool)
        mk = champion_sizing(pool, max_pos=MAX_POS)
        eq_of_k = make_eq_fn(pool, closes, mk)
        k_emp = calibrate_empirical(eq_of_k, 0.20)
        eq_e = eq_of_k(k_emp)
        bs = boot_dd(eq_e, n_boot=1500, seed=0)
        k_rob = calibrate_robust_seeded(eq_of_k, 0.20, seed=0)
        rows.append({"vol_sd_win": w, **st, "emp_k": k_emp, "emp_cagr": cagr_of(eq_e),
                     "emp_p95": bs["p95"], "rob_s0_k": k_rob,
                     "rob_s0_cagr": cagr_of(eq_of_k(k_rob))})
        print(f"  w={w:>2}: n={st['n']} sum={st['sum_ret']:+.4f} PF={st['pf']:.3f} | "
              f"emp k={k_emp:.2f} CAGR={rows[-1]['emp_cagr']:+.2%} p95={bs['p95']:+.1%} | "
              f"rob_s0 {rows[-1]['rob_s0_cagr']:+.2%}  [{time.time()-t0:.0f}s]")
    fdf = pd.DataFrame(rows)
    fdf.to_csv(OUT_DIR / "exp41_volwin_fine.csv", index=False)

    print("\n--- p95署名のブートシード監査(各 emp_k 固定, n_boot=1500, seeds 0-2) ---")
    for w in [20, 25]:
        mk = champion_sizing(pools[w], max_pos=MAX_POS)
        eq_of_k = make_eq_fn(pools[w], closes, mk)
        k_emp = float(fdf.loc[fdf["vol_sd_win"] == w, "emp_k"].iloc[0])
        eq = eq_of_k(k_emp)
        p95s = [boot_dd(eq, n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)]
        print(f"  w{w} (emp_k={k_emp:.2f}): p95 = " +
              " / ".join(f"s{sd}:{v:+.2%}" for sd, v in zip((0, 1, 2), p95s)))

    print("\n--- 細断面のロバストCAGR形状 ---")
    print(fdf[["vol_sd_win", "sum_ret", "pf", "emp_cagr", "emp_p95", "rob_s0_cagr"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved -> {OUT_DIR / 'exp41_volwin_fine.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


def main() -> int:
    if "--fine" in sys.argv:
        return fine_main()
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    instruments = mm.default_instruments()
    print(f"=== exp41: vol_sd_win 断面 {WINDOWS} ({TF}, {len(instruments)}銘柄, "
          f"mp{MAX_POS}, P=4.0) ===\n")
    datas = {nm: uni.instrument_data(nm, TF) for nm in instruments}
    caches = {nm: IndicatorCache(datas[nm]["close"]) for nm in instruments}
    closes = mm.load_closes()
    print(f"closes grid: {len(closes)} bars [{closes.index[0]} .. {closes.index[-1]}]")
    print("champion_sizing は ctx['z'] のみ参照 → per-trade 乗数辞書は不使用(キー衝突 n/a)\n")

    # --- 1. プール再生成 + ベースライン一致検算 ---------------------------
    pools, prows = {}, []
    ref = pd.read_parquet(POOL_PATH)
    for w in WINDOWS:
        p = dict(BASE)
        p["vol_sd_win"] = w
        pool = build_mm_pool(datas, caches, p)
        pools[w] = pool
        st = pool_stats(pool)
        prows.append({"vol_sd_win": w, **st})
        print(f"  pool w={w:>2}: n={st['n']} sum={st['sum_ret']:+.4f} PF={st['pf']:.3f} "
              f"[{time.time()-t0:.0f}s]")
        if w == BASE_WIN:
            n_ok = st["n"] == len(ref)
            s_ok = abs(st["sum_ret"] - ref["ret"].sum()) < 1e-6
            mg = pool.merge(ref[["instr", "entry", "ret", "z_entry"]], on=["instr", "entry"],
                            how="inner", suffixes=("", "_ref"))
            zdiff = float(np.nanmax(np.abs(mg["z_entry"] - mg["z_entry_ref"])))
            rdiff = float(np.nanmax(np.abs(mg["ret"] - mg["ret_ref"])))
            print(f"    [検算] n一致={n_ok} sum一致(<1e-6)={s_ok} "
                  f"max|Δret|={rdiff:.2e} max|Δz_entry|={zdiff:.2e}")
            if not (n_ok and s_ok and rdiff < 1e-9):
                print("    !! ベースライン再現失敗。以降の比較は無効。")
                return 1

    pdf = pd.DataFrame(prows)
    base_sum = float(pdf.loc[pdf["vol_sd_win"] == BASE_WIN, "sum_ret"].iloc[0])
    pdf["d_sum_%"] = (pdf["sum_ret"] - base_sum) / base_sum * 100
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(OUT_POOL, index=False)
    print("\n=== 1. プールレベル断面 ===")
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    seq = pdf["sum_ret"].to_numpy()
    dif = np.diff(seq)
    mono = "単調増(端逃げ疑い)" if (dif > 0).all() else (
        "単調減" if (dif < 0).all() else "山/高原")
    am = int(pdf.loc[pdf["sum_ret"].idxmax(), "vol_sd_win"])
    print(f"  断面形状: {mono} / プール最良 w={am}")

    # 年次分解(プール, 候補比較用に保持)
    pool_yearly = {w: pools[w].groupby(pd.to_datetime(pools[w]["exit"]).dt.year)["ret"].sum()
                   for w in WINDOWS}

    # --- 2. 口座レベル ステージ1(seed0) -----------------------------------
    print("\n=== 2. 口座シミュ ステージ1: empirical + robust seed0 + IS/OOS ===")
    results = {}
    for w in WINDOWS:
        results[w] = eval_window(f"w{w}", pools[w], closes, seeds=(0,))
        print(f"    [{time.time()-t0:.0f}s]")

    # --- 3. ステージ2: 上位 ≤3 + ベースラインを seeds 0-4 フル評価 --------
    cand = sorted((w for w in WINDOWS if w != BASE_WIN),
                  key=lambda w: -results[w]["rob"][0]["cagr"])[:3]
    print(f"\n=== 3. ステージ2: seeds {SEEDS_FULL} フル評価 = baseline w{BASE_WIN} + {cand} ===")
    for w in [BASE_WIN] + cand:
        res = eval_window(f"w{w}_full", pools[w], closes, seeds=SEEDS_FULL)
        results[w].update({k: res[k] for k in
                           ["rob", "rob_cagr_mean", "rob_k_mean"]})
        results[w]["full_seeds"] = True
        print(f"    [{time.time()-t0:.0f}s]")

    # ペアシード比較表
    base = results[BASE_WIN]
    print("\n--- ペアシード robust(p95=20%) CAGR 比較(同一シード集合) ---")
    hdr = "  seed | " + f"w{BASE_WIN}(base)" + " | " + " | ".join(f"w{w} (diff)" for w in cand)
    print(hdr)
    for sd in SEEDS_FULL:
        cells = []
        for w in cand:
            c = results[w]["rob"][sd]["cagr"]
            cells.append(f"{c:+.2%} ({(c - base['rob'][sd]['cagr']) * 100:+.2f}pp)")
        print(f"   s{sd}  | {base['rob'][sd]['cagr']:+.2%}  | " + " | ".join(cells))
    print(f"  mean | {base['rob_cagr_mean']:+.2%}  | " + " | ".join(
        f"{results[w]['rob_cagr_mean']:+.2%} ({(results[w]['rob_cagr_mean']-base['rob_cagr_mean'])*100:+.2f}pp)"
        for w in cand))
    print(f"  empirical 20%: base {base['emp_cagr']:+.2%} (p95 {base['emp_p95']:+.1%}) | " +
          " | ".join(f"w{w} {results[w]['emp_cagr']:+.2%} (p95 {results[w]['emp_p95']:+.1%})"
                     for w in cand))

    # --- 4. IS-argmax 監査 --------------------------------------------------
    print("\n=== 4. IS-argmax 監査(IS<2022 のみで window を選んだら?) ===")
    print("  w   IS_rob_cagr  IS_emp_cagr   OOS_rob   OOS_emp")
    for w in WINDOWS:
        r = results[w]
        print(f"  {w:>2}  {r['is_rob_cagr']:+10.2%}  {r['is_emp_cagr']:+10.2%}  "
              f"{r['oos_rob_cagr']:+8.2%}  {r['oos_emp_cagr']:+8.2%}")
    arg_rob = max(WINDOWS, key=lambda w: results[w]["is_rob_cagr"])
    arg_emp = max(WINDOWS, key=lambda w: results[w]["is_emp_cagr"])
    print(f"  IS-argmax: robust基準 w={arg_rob} / empirical基準 w={arg_emp}")
    print(f"  -> IS選択(robust) w={arg_rob} の OOS: rob較正 {results[arg_rob]['oos_rob_cagr']:+.2%} "
          f"/ emp較正 {results[arg_rob]['oos_emp_cagr']:+.2%} "
          f"(baseline w20 OOS: {results[BASE_WIN]['oos_rob_cagr']:+.2%} / "
          f"{results[BASE_WIN]['oos_emp_cagr']:+.2%})")

    # --- 5. 2022除外・年次分解・全年プラス ---------------------------------
    print("\n=== 5. 年次分解(トップ候補 vs baseline) ===")
    top = cand[0]
    for tag, key in [("empirical", "yr_emp"), ("robust_s0", "yr_rob0")]:
        yb = pd.Series(base[key])
        yc = pd.Series(results[top][key])
        d = (yc - yb).dropna()
        best_y = int(d.idxmax())
        print(f"  [{tag}] w{top} - w{BASE_WIN} 年次差分:")
        print("    " + "  ".join(f"{int(y)}:{v:+.1%}" for y, v in d.items()))
        print(f"    合計 {d.sum():+.2%} / 最良年 {best_y}({d[best_y]:+.2%}) 除外後 "
              f"{d.drop(best_y).sum():+.2%} / 2022除外後 "
              f"{d.drop(2022).sum() if 2022 in d.index else float('nan'):+.2%}")
    # プールレベルでも
    dp = (pool_yearly[top] - pool_yearly[BASE_WIN]).dropna()
    by = int(dp.idxmax())
    print(f"  [pool] w{top} - w{BASE_WIN}: 合計 {dp.sum():+.4f} / 最良年 {by}({dp[by]:+.4f}) "
          f"除外後 {dp.drop(by).sum():+.4f} / 2022除外後 "
          f"{dp.drop(2022).sum() if 2022 in dp.index else float('nan'):+.4f}")
    print("\n  全年プラス(empirical 較正の年次):")
    for w in [BASE_WIN] + cand:
        yr = pd.Series(results[w]["yr_emp"])
        neg = [f"{int(y)}({v:+.1%})" for y, v in yr.items() if v < 0]
        print(f"    w{w:>2}: 負け年 {len(neg)} {neg if neg else ''} / "
              f"robust_s0 負け年 {results[w]['neg_years_rob0']}")

    # --- 6. レバ偽装署名 ----------------------------------------------------
    print("\n=== 6. レバ偽装署名(emp CAGR↑ かつ p95悪化 → reject) ===")
    for w in cand:
        r = results[w]
        sig = (r["emp_cagr"] > base["emp_cagr"]) and (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        print(f"  w{w}: emp {r['emp_cagr']:+.2%} (base {base['emp_cagr']:+.2%}) "
              f"p95 {r['emp_p95']:+.1%} (base {base['emp_p95']:+.1%}) -> 署名 {'あり!' if sig else 'なし'}")

    # --- 保存 ---------------------------------------------------------------
    rows = []
    for w in WINDOWS:
        r = results[w]
        rows.append({
            "vol_sd_win": w, "full_seeds": r.get("full_seeds", False),
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"], "worst_year": r["worst_year"],
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            **{f"rob_s{sd}": r["rob"].get(sd, {}).get("cagr") for sd in SEEDS_FULL},
            "rob_mean": r.get("rob_cagr_mean"),
            "k_is_emp": r["k_is_emp"], "is_emp_cagr": r["is_emp_cagr"],
            "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
            "k_is_rob": r["k_is_rob"], "is_rob_cagr": r["is_rob_cagr"],
            "oos_rob_cagr": r["oos_rob_cagr"], "oos_rob_dd": r["oos_rob_dd"],
        })
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT_ACC, index=False)
    OUT_JSON.write_text(json.dumps(
        {str(w): {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                  for k, v in r.items()}
         for w, r in results.items()}, indent=2, default=float))
    print(f"\nsaved -> {OUT_POOL}\n      -> {OUT_ACC}\n      -> {OUT_JSON}")
    print("\n=== 口座レベル最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf.to_string(index=False))
    print(f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
