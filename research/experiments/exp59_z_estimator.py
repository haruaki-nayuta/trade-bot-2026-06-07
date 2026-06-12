"""exp59: z スコア推定器の置換 — 第4ラウンド(アルゴリズム改善・FX限定・リスク契約固定)。

ユーザー制約: リスク契約(robust p95 DD=20%)固定 / 金・銀・原油・株価指数の追加禁止 /
既存パラメータの再調整ではなくアルゴリズムの構造改善。

仮説: チャンピオンの z は 10 年間 rolling mean/std(window) 固定。rolling std は
  (a) ボライベントが窓に入っている間は分母が膨らみ、**機会が最も濃い直後に z が人工的に縮む**
  (b) イベントが窓から抜ける瞬間に分母が階段状に落ちて z が跳ぶ(窓落ちアーティファクト)
という構造欠陥を持つ。推定器を EWMA(滑らかな忘却)や median/MAD(外れ値頑健)に置換すれば、
同じ閾値のままで「乖離の深さ」の測定品質が上がる可能性がある。

規律(事前登録):
  - **閾値・窓・P・mp など既存パラメータは一切再調整しない**(estimator のみ差し替え。
    EWMA は span=window、MAD は定数 1.4826 の標準形のみ=推定器内チューニング禁止)
  - z は entry/exit/slow/d1ゲート/サイジング z_entry に**一貫適用**(スコープは事前定義の2種)
  - 変種(6固定): base / ewma_short / ewma_both / mad_short / mad_both / and_conf
    (and_conf = roll と ewma の両方が entry 閾値を越えたときだけ建玉=確認型合流。
     exit/サイジングは roll を使用)
  - 取引数の変化を開示(閾値は roll スケールで較正されているため、推定器によって実効的な
    選別率が変わる。これは変種の一部として受容するが、ゲート(IS/OOS・単年・署名)で検証)
  - 判定: seed0 スカウト → 生存者(>base)のみ seeds 0-4 + 6ゲート + IS-argmax 監査

実行: PYTHONPATH=. uv run python research/experiments/exp59_z_estimator.py
出力: research/outputs/exp59_result.csv / exp59_result.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import vectorbt as vbt  # noqa: E402

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from exp47_entry_delay import year_diff_audit  # noqa: E402
from fxlab import config, universe as uni  # noqa: E402
from fxlab.backtest import run as bt_run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from strategies.confluence_meanrev_v2 import PARAMS  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
MAX_POS = 8


# --- z 推定器(標準形のみ・チューニング禁止) -------------------------------
def z_roll(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def z_ewma(s: pd.Series, w: int) -> pd.Series:
    m = s.ewm(span=w, adjust=False).mean()
    sd = s.ewm(span=w, adjust=False).std()
    return (s - m) / sd


def z_mad(s: pd.Series, w: int) -> pd.Series:
    med = s.rolling(w).median()
    mad = (s - med).abs().rolling(w).median()
    return (s - med) / (1.4826 * mad)


Z_FUNCS = {"roll": z_roll, "ewma": z_ewma, "mad": z_mad}


# --- 変種シグナル生成(v2 + d1 を推定器注入で再実装) ------------------------
def gen_signals_var(data: pd.DataFrame, z_short_est="roll", z_slow_est="roll",
                    and_with=None):
    """confluence_meanrev_v2 + d1 のシグナルを z 推定器差し替えで生成。

    and_with: 追加の推定器名。指定時は entry のみ「主推定器 AND 追加推定器」の
    両方が閾値越えを要求(exit/d1ゲートは主推定器)。
    返り値: (long_entries, long_exits, short_entries, short_exits, z_short)
    """
    p = PARAMS
    close = data["close"]
    zf = Z_FUNCS[z_short_est]
    z = zf(close, p["window"])
    rsi = vbt.RSI.run(close, p["rsi_p"]).rsi

    vol = close.pct_change().rolling(20).std()
    calm = vol <= vol.rolling(p["vol_win"]).quantile(p["vol_pct"])

    zs = Z_FUNCS[z_slow_est](close, p["slow_win"])
    long_ok = (zs < -p["slow_z"]).fillna(False)
    short_ok = (zs > p["slow_z"]).fillna(False)

    ez, xz = p["entry_z"], p["exit_z"]
    base_long = (z < -ez) & (z.shift() >= -ez) & (rsi < p["rsi_low"]) & calm & long_ok
    base_short = (z > ez) & (z.shift() <= ez) & (rsi > p["rsi_high"]) & calm & short_ok

    if and_with is not None:
        z2 = Z_FUNCS[and_with](close, p["window"])
        base_long &= (z2 < -ez)
        base_short &= (z2 > ez)

    # ER フィルタ(v2)
    direction = (close - close.shift(p["er_win"])).abs()
    volat = close.diff().abs().rolling(p["er_win"]).sum()
    er = (direction / volat).replace([np.inf, -np.inf], np.nan)
    ok = (er <= p["er_max"]).fillna(False)
    le0 = (base_long & ok).fillna(False)
    se0 = (base_short & ok).fillna(False)

    # d1: 1バー遅延 + 遅延先で z(主推定器)が exit 域なら見送り
    long_entries = le0.shift(1, fill_value=False) & (z <= -xz)
    short_entries = se0.shift(1, fill_value=False) & (z >= xz)
    long_exits = (z > -xz).fillna(False)
    short_exits = (z < xz).fillna(False)
    return (long_entries.fillna(False), long_exits,
            short_entries.fillna(False), short_exits, z)


def build_pool_var(tag, z_short_est, z_slow_est, and_with=None, cache=True):
    cache_path = config.RESULTS_DIR / f"mm_pool_v2d1_{tag}_H4_19.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    instruments = mm.default_instruments()
    frames = []
    for nm in instruments:
        data = uni.instrument_data(nm, "H4")

        def gen(d, **kw):
            le, lx, se, sx, _ = gen_signals_var(d, z_short_est, z_slow_est, and_with)
            return le, lx, se, sx
        pf = bt_run(nm, "H4", gen, {}, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        # サイジング入力 = シグナルバー(執行の1本前)の主推定器 |z|(d1 規約)
        zser = Z_FUNCS[z_short_est](data["close"], PARAMS["window"]).shift(1)
        vol_sig = data["close"].pct_change().rolling(20).std().shift(1)
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(zser.reindex(tt["entry"]).to_numpy()),
            "vol_entry": vol_sig.reindex(tt["entry"]).to_numpy(),
        }))
    pool = pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)
    if cache:
        pool.to_parquet(cache_path)
    return pool


def pool_summary(tag, pool):
    r = pool["ret"]
    yr = pool.groupby(pool["exit"].dt.year)["ret"].sum()
    pos, neg = r[r > 0].sum(), r[r < 0].sum()
    return {"cfg": tag, "n": len(pool), "sum_ret": float(r.sum()),
            "pf": float(pos / abs(neg)) if neg < 0 else np.inf,
            "win": float((r > 0).mean()),
            "neg_years_pool": int((yr < 0).sum()),
            "yearly": {int(y): float(v) for y, v in yr.items()}}


def account_eval(tag, pool, closes, seeds):
    mk = champion_sizing(pool, max_pos=MAX_POS)
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    r = protocol_eval(eq_of_k, label=tag, seeds=seeds)
    eq_e = eq_of_k(r["emp_k"])
    yr_e = yearly_returns(eq_e)
    r["worst_year"] = float(yr_e.min())
    r["neg_years_emp"] = int((yr_e < 0).sum())
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    k_r0 = r["rob"][seeds[0]]["k"]
    yr_r0 = yearly_returns(eq_of_k(k_r0))
    r["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    r["neg_years_rob0"] = int((yr_r0 < 0).sum())
    # IS較正→OOS
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]

    def eq_fn(pl, cl):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(pl, cl, mk(kk), max_pos=MAX_POS)[0]
            return c[kk]
        return f
    fi, fo = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_ir = calibrate_robust_seeded(fi, 0.20, seed=0)
    r["is_rob_cagr"] = cagr_of(fi(k_ir))
    r["oos_rob_cagr"] = cagr_of(fo(k_ir))
    r["oos_rob_dd"] = max_dd(fo(k_ir))
    k_ie = calibrate_empirical(fi, 0.20)
    r["is_emp_cagr"] = cagr_of(fi(k_ie))
    r["oos_emp_cagr"] = cagr_of(fo(k_ie))
    r["oos_emp_dd"] = max_dd(fo(k_ie))
    return r


VARIANTS = {
    "base": dict(z_short_est="roll", z_slow_est="roll"),
    "ewma_short": dict(z_short_est="ewma", z_slow_est="roll"),
    "ewma_both": dict(z_short_est="ewma", z_slow_est="ewma"),
    "mad_short": dict(z_short_est="mad", z_slow_est="roll"),
    "mad_both": dict(z_short_est="mad", z_slow_est="mad"),
    "and_conf": dict(z_short_est="roll", z_slow_est="roll", and_with="ewma"),
}


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    closes = mm.load_closes()
    print(f"=== exp59: z 推定器置換 (6変種, d1+P4.0+mp8 固定) ===")

    pools, psum = {}, []
    for tag, kw in VARIANTS.items():
        pools[tag] = build_pool_var(tag, **kw)
        s = pool_summary(tag, pools[tag])
        psum.append(s)
        print(f"  {tag:12s} n={s['n']:5d}  sum={s['sum_ret']:+.4f}  PF={s['pf']:.3f}  "
              f"win={s['win']:.1%}  プール負け年={s['neg_years_pool']}  [{time.time()-t0:.0f}s]")

    # 検算: base が本番 d1 プールと一致するか
    base_ref = pd.read_parquet(config.RESULTS_DIR / "mm_pool_v2d1_H4_19.parquet")
    ok = len(pools["base"]) == len(base_ref) and abs(
        pools["base"]["ret"].sum() - base_ref["ret"].sum()) < 1e-9
    print(f"\nbase 検算: n={len(pools['base'])} vs 本番 {len(base_ref)}, 一致: {ok}")
    if not ok:
        print("!! base 再現失敗 — 中断")
        return 1

    print("\n--- 口座 seed0 スカウト ---")
    results = {}
    for tag in VARIANTS:
        results[tag] = account_eval(tag, pools[tag], closes, seeds=(0,))
        print(f"    [{time.time()-t0:.0f}s]")

    base_s0 = results["base"]["rob"][0]["cagr"]
    finalists = [t for t in VARIANTS if t != "base"
                 and results[t]["rob"][0]["cagr"] > base_s0]
    print(f"\nseed0 で base({base_s0:+.2%}) 超え: {finalists}")

    print("\n--- 生存者 + base: seeds 0-4 フル ---")
    for tag in ["base"] + finalists:
        results[tag] = account_eval(tag, pools[tag], closes, seeds=SEEDS)
        print(f"    [{time.time()-t0:.0f}s]")

    base = results["base"]
    rows = []
    for tag in VARIANTS:
        r = results[tag]
        full = len(r["rob"]) == len(SEEDS)
        row = {"cfg": tag, "n_trades": len(pools[tag]),
               "pool_sum": float(pools[tag]["ret"].sum()),
               "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_p95": r["emp_p95"],
               "rob_s0": r["rob"][0]["cagr"],
               "rob_mean": r["rob_cagr_mean"] if full else np.nan,
               "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
               "oos_emp": r["oos_emp_cagr"],
               "worst_year": r["worst_year"], "neg_emp": r["neg_years_emp"],
               "neg_rob0": r["neg_years_rob0"]}
        if tag != "base" and full:
            per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
            sig = (r["emp_cagr"] > base["emp_cagr"]) and \
                  (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
            row["gain_pp"] = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
            row["all_seeds_pos"] = all(v > 0 for v in per_seed.values())
            row["signature"] = bool(sig)
            row["g3_oos"] = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
                            (r["oos_emp_cagr"] > base["oos_emp_cagr"])
            row["g4_years"] = (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0)
            a_emp = year_diff_audit("emp", r["yr_emp"], base["yr_emp"])
            row["g5_keep_emp"] = a_emp["keep_share_excl_best"]
            print(f"  [{tag}] gain {row['gain_pp']:+.2f}pp seeds " +
                  " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
                  f" 署名={'あり' if sig else 'なし'} OOS={'+' if row['g3_oos'] else 'x'}"
                  f" G5keep={a_emp['keep_share_excl_best']:.0%}")
        rows.append(row)

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "exp59_result.csv", index=False)
    payload = {"pool_summaries": psum,
               "results": {t: {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                               for k, v in r.items() if k not in ("yr_emp", "yr_rob0")}
                           for t, r in results.items()}}
    (OUT_DIR / "exp59_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp59_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
