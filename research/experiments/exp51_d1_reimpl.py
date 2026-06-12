"""exp51: エントリー1バー遅延 d1 の独立再実装による再現監査(採用前の最終敵対検証)。

exp47 は「ベースプール(mm_pool_v2_H4_19)の entry/価格を事後シフト」する再構成方式で
d1 の全ゲート通過を確認した。本実験は **完全に別経路** で同じ d1 を実装して突合する:

  exp47 経路: ベースプール → reconstruct()(close 再構成)→ delayed_pool()(entry を
              1 本後ろへ、価格 = 遅延バー close × 元スリッページ比率、コストは元のまま)
  exp51 経路: strategies/confluence_meanrev_v2_d1.py(シグナルレベルで shift(1) + z ゲート)
              → fxlab.backtest.run(vectorbt の約定・スリッページ・コスト処理)
              → trade_table → プール化(z_entry はシグナルバー時点の |z| を再付与)

  両者の約定・コスト処理は独立(片や手計算の再構成、片や vectorbt エンジン)。
  ここで n / sum(ret) / トレード集合が一致し、口座レベル(tail_protocol)の数値が
  exp47 の ±0.5pp 以内で再現されれば、d1 の利得は「再構成方式のアーティファクト」では
  ないと言える。ズレが出たら原因(vbt の同バー entry/exit 処理・スリッページ・端点)を
  特定し、どちらが本番実装に忠実かを判定する。

検証項目:
  1. ベースプール再現: build_pool(cache=False) 再構築 vs キャッシュ parquet(n=1214, +1.9086)
  2. d1 プール構築(vbt 経路)と exp47 d1 プール(再構成経路)の突合
     (n / sum / (instr,entry,exit,dir) キー一致率 / 一致トレードの ret 差)
  3. 口座レベル: tail_protocol.protocol_eval(champion_sizing, mp8) seeds 0-4 フル
     + IS(<2022) 較正 → OOS 素検証。exp47 の base +16.41% / d1 +18.63% と ±0.5pp 比較
  4. レバレッジ監査: emp_k での 1 玉最大レバと総同時露出 max が国内規制 25x に収まるか

実行: PYTHONPATH=. uv run python research/experiments/exp51_d1_reimpl.py
出力: research/outputs/exp51_pool_compare.csv / exp51_account.csv /
      exp51_leverage.csv / exp51_result.json
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
sys.path.insert(0, str(ROOT))
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
from fxlab import universe as uni  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from strategies.confluence_meanrev_v2 import PARAMS as V2_PARAMS  # noqa: E402
import strategies.confluence_meanrev_v2_d1 as d1mod  # noqa: E402

# exp47 の再構成方式(突合の対向プール生成のためにのみ使用)
from research.experiments.exp47_entry_delay import delayed_pool, reconstruct  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"

# exp47 の確定数値(±0.5pp 一致判定の基準)
EXP47 = {
    "base_rob_mean": 0.16408835165441676, "base_emp": 0.24643242987359737,
    "d1_rob_mean": 0.18625453529025582, "d1_emp": 0.27499459104059665,
    "d1_emp_k": 8.895088391304014,
    "base_oos_rob": 0.2104332372636608, "d1_oos_rob": 0.24792161697528914,
    "d1_pool_n": 1207, "d1_pool_sum": 1.96224087528572,
    "base_pool_n": 1214, "base_pool_sum": 1.9086,
}


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


# --- 1. d1 プール構築(vectorbt 経路・独立実装) ---------------------------
def build_pool_d1(tf=TF, instruments=None, params=None, cross_spread=3.0) -> pd.DataFrame:
    """d1 戦略のトレードを口座シミュ用の最小表に収集(mm_lab.build_pool と同形式)。

    z_entry は **シグナルバー(=エントリーバーの 1 本前)時点の |z|**(exp47 と同じ
    因果定義)。vol_entry も同様にシグナルバー時点。例外は握りつぶさない(銘柄欠落の
    無言スキップを許さない)。キャッシュは使わない(独立再現のため毎回構築)。
    """
    uni.register_cross_spreads(cross_spread)
    instruments = instruments or mm.default_instruments()
    params = params or dict(V2_PARAMS)
    win = params.get("window", 50)

    frames = []
    for nm in instruments:
        data = uni.instrument_data(nm, tf)
        pf = run(nm, tf, d1mod.generate_signals, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        close = data["close"]
        z_sig = _zscore(close, win).shift(1)        # シグナルバー時点の z
        vol_sig = close.pct_change().rolling(20).std().shift(1)
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(z_sig.reindex(tt["entry"]).to_numpy()),
            "vol_entry": vol_sig.reindex(tt["entry"]).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)


# --- 2. プール突合 ----------------------------------------------------------
def compare_pools(mine: pd.DataFrame, theirs: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """(instr, entry, exit, dir) キーでトレード集合を突合。"""
    key_cols = ["instr", "entry", "exit", "dir"]
    a = mine.set_index(key_cols)
    b = theirs.set_index(key_cols)
    common = a.index.intersection(b.index)
    only_mine = a.index.difference(b.index)
    only_theirs = b.index.difference(a.index)
    dret = (a.loc[common, "ret"] - b.loc[common, "ret"])
    dz = (a.loc[common, "z_entry"] - b.loc[common, "z_entry"])
    res = {
        "n_mine": len(mine), "n_exp47": len(theirs), "n_common": len(common),
        "match_rate": len(common) / max(len(mine), len(theirs)),
        "n_only_mine": len(only_mine), "n_only_exp47": len(only_theirs),
        "sum_mine": float(mine["ret"].sum()), "sum_exp47": float(theirs["ret"].sum()),
        "sum_diff": float(mine["ret"].sum() - theirs["ret"].sum()),
        "ret_diff_mean": float(dret.mean()), "ret_diff_median": float(dret.median()),
        "ret_diff_maxabs": float(dret.abs().max()),
        "ret_diff_sum": float(dret.sum()),
        "z_entry_diff_maxabs": float(dz.abs().max()),
        "sum_only_mine": float(a.loc[only_mine, "ret"].sum()) if len(only_mine) else 0.0,
        "sum_only_exp47": float(b.loc[only_theirs, "ret"].sum()) if len(only_theirs) else 0.0,
    }
    rows = []
    for tag, idx, src in [("only_mine", only_mine, a), ("only_exp47", only_theirs, b)]:
        for k in idx:
            rows.append({"which": tag, "instr": k[0], "entry": k[1], "exit": k[2],
                         "dir": k[3], "ret": float(src.loc[k, "ret"])})
    return res, pd.DataFrame(rows)


# --- 3. 口座レベル(tail_protocol フル) ------------------------------------
def make_eq_fn(pool, closes, mk, max_pos=MAX_POS):
    cache: dict[float, pd.Series] = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=max_pos)
            cache[kk] = eqm
        return cache[kk]

    return eq_of_k


def account_eval(label, pool, closes, seeds=SEEDS) -> dict:
    mk = champion_sizing(pool, max_pos=MAX_POS)
    eq_of_k = make_eq_fn(pool, closes, mk)
    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    yr = yearly_returns(eq_of_k(res["emp_k"]))
    res["worst_year"] = float(yr.min())
    res["neg_years_emp"] = int((yr < 0).sum())
    yr0 = yearly_returns(eq_of_k(res["rob"][seeds[0]]["k"]))
    res["neg_years_rob0"] = int((yr0 < 0).sum())

    # IS(<2022) 較正 → OOS 素検証(exp47 と同条件: mk はフルプール基準)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    eq_is = make_eq_fn(is_pool, closes[closes.index < OOS_START], mk)
    eq_oos = make_eq_fn(oos_pool, closes[closes.index >= OOS_START], mk)
    k_ie = calibrate_empirical(eq_is, 0.20)
    res["k_is_emp"] = k_ie
    res["is_emp_cagr"] = cagr_of(eq_is(k_ie))
    res["oos_emp_cagr"] = cagr_of(eq_oos(k_ie))
    res["oos_emp_dd"] = max_dd(eq_oos(k_ie))
    k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
    res["k_is_rob"] = k_ir
    res["is_rob_cagr"] = cagr_of(eq_is(k_ir))
    res["oos_rob_cagr"] = cagr_of(eq_oos(k_ir))
    res["oos_rob_dd"] = max_dd(eq_oos(k_ir))
    print(f"      IS emp k={k_ie:5.2f} ISC={res['is_emp_cagr']:+7.2%} -> "
          f"OOS {res['oos_emp_cagr']:+7.2%} (DD {res['oos_emp_dd']:+5.1%}) | "
          f"IS rob k={k_ir:5.2f} ISC={res['is_rob_cagr']:+7.2%} -> "
          f"OOS {res['oos_rob_cagr']:+7.2%} (DD {res['oos_rob_dd']:+5.1%})")
    return res


# --- 4. レバレッジ監査(mm_lab.simulate のロジックを露出トラッキング付で複製) --
def leverage_audit(pool, closes, sizing, max_pos=MAX_POS, init=10_000.0):
    """各バーの 総建玉/MtM equity と 最大1玉/MtM equity を追跡。

    mm_lab.simulate と同一の決済→MtM→新規エントリー順。検算のため最終実現 equity も返す。
    """
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    entry_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    exit_pos = np.clip(np.searchsorted(gi, pool["exit"].to_numpy(), side="left"), 0, n - 1)
    by_entry: dict[int, list[int]] = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(entry_pos[ti]), []).append(ti)
    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    eprice_arr = pool["entry_price"].to_numpy()
    ret_arr = pool["ret"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()
    bars_arr = pool["bars_held"].to_numpy()

    equity = init
    peak_mtm = init
    open_pos: list[dict] = []
    lev_tot = np.zeros(n)
    lev_single = np.zeros(n)
    alloc_frac_at_entry = []   # alloc / MtM equity(発注時点)
    for b in range(n):
        still = []
        for p in open_pos:
            if p["exit_pos"] <= b:
                equity += p["alloc"] * p["ret"]
            else:
                still.append(p)
        open_pos = still
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            unreal += p["alloc"] * (p["dir"] * (px / p["eprice"] - 1.0))
        mtm = equity + unreal
        peak_mtm = max(peak_mtm, mtm)
        dd_mtm = mtm / peak_mtm - 1.0
        if b in by_entry:
            for ti in by_entry[b]:
                if len(open_pos) >= max_pos:
                    continue
                ctx = {"equity_real": equity, "equity_mtm": mtm, "peak_mtm": peak_mtm,
                       "dd_mtm": dd_mtm, "n_open": len(open_pos), "max_pos": max_pos,
                       "recent_vol": float("nan"), "z": float(z_arr[ti]),
                       "instr": instr_arr[ti], "ret": float(ret_arr[ti]),
                       "bars_held": int(bars_arr[ti])}
                alloc = float(sizing(ctx))
                if alloc <= 0:
                    continue
                open_pos.append({"col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": eprice_arr[ti], "alloc": alloc,
                                 "exit_pos": int(exit_pos[ti]), "ret": float(ret_arr[ti])})
                alloc_frac_at_entry.append(alloc / mtm if mtm > 0 else np.nan)
        if open_pos and mtm > 0:
            allocs = [p["alloc"] for p in open_pos]
            lev_tot[b] = sum(allocs) / mtm
            lev_single[b] = max(allocs) / mtm
    lt = pd.Series(lev_tot, index=grid)
    ls = pd.Series(lev_single, index=grid)
    af = np.asarray(alloc_frac_at_entry)
    return {
        "final_equity": equity,
        "max_lev_total": float(lt.max()), "t_max_lev_total": str(lt.idxmax()),
        "p99_lev_total": float(np.percentile(lt[lt > 0], 99)),
        "max_lev_single": float(ls.max()), "t_max_lev_single": str(ls.idxmax()),
        "max_alloc_frac_entry": float(np.nanmax(af)),
        "p99_alloc_frac_entry": float(np.nanpercentile(af, 99)),
        "mean_alloc_frac_entry": float(np.nanmean(af)),
    }


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    uni.register_cross_spreads(3.0)
    result: dict = {}

    sec("1. ベースプール再現監査: build_pool(cache=False) 再構築 vs キャッシュ parquet")
    pool_base_cached = mm.build_pool()  # = exp47 が使った results/mm_pool_v2_H4_19.parquet
    pool_base_fresh = mm.build_pool(cache=False)
    res_base, mism_base = compare_pools(pool_base_fresh, pool_base_cached)
    print(f"  cached : n={len(pool_base_cached)}  sum={pool_base_cached['ret'].sum():+.4f}")
    print(f"  fresh  : n={len(pool_base_fresh)}  sum={pool_base_fresh['ret'].sum():+.4f}")
    print(f"  一致率 {res_base['match_rate']:.4%} / 片側のみ mine {res_base['n_only_mine']} "
          f"exp47側 {res_base['n_only_exp47']} / ret差 maxabs {res_base['ret_diff_maxabs']:.2e}")
    result["base_rebuild"] = res_base
    print(f"  [{time.time()-t0:.0f}s]")

    sec("2. d1 プール: vbt経路(独立実装) vs exp47再構成経路 の突合")
    pool_d1_mine = build_pool_d1()
    rc = reconstruct(pool_base_cached)
    pool_d1_exp47, kept, _, _ = delayed_pool(pool_base_cached, rc, 1)
    print(f"  mine(vbt) : n={len(pool_d1_mine)}  sum={pool_d1_mine['ret'].sum():+.4f}")
    print(f"  exp47再構成: n={len(pool_d1_exp47)}  sum={pool_d1_exp47['ret'].sum():+.4f} "
          f"(消滅 {int((~kept).sum())}件)")
    res_d1, mism_d1 = compare_pools(pool_d1_mine, pool_d1_exp47)
    print(f"  一致率 {res_d1['match_rate']:.4%}  共通 {res_d1['n_common']} / "
          f"mineのみ {res_d1['n_only_mine']} ({res_d1['sum_only_mine']:+.4f}) / "
          f"exp47のみ {res_d1['n_only_exp47']} ({res_d1['sum_only_exp47']:+.4f})")
    print(f"  共通トレード ret 差: mean {res_d1['ret_diff_mean']:+.2e} / "
          f"median {res_d1['ret_diff_median']:+.2e} / maxabs {res_d1['ret_diff_maxabs']:.2e} / "
          f"sum {res_d1['ret_diff_sum']:+.5f}")
    print(f"  z_entry 差 maxabs: {res_d1['z_entry_diff_maxabs']:.2e}")
    if len(mism_d1):
        print("\n  不一致トレード明細:")
        print(mism_d1.to_string(index=False))
    result["d1_pool_compare"] = res_d1
    pd.DataFrame([{"comparison": "base_fresh_vs_cached", **res_base},
                  {"comparison": "d1_mine_vs_exp47", **res_d1}]).to_csv(
        OUT_DIR / "exp51_pool_compare.csv", index=False)
    mism_all = pd.concat([mism_base.assign(comparison="base"),
                          mism_d1.assign(comparison="d1")], ignore_index=True)
    mism_all.to_csv(OUT_DIR / "exp51_pool_mismatch.csv", index=False)
    print(f"  [{time.time()-t0:.0f}s]")

    sec(f"3. 口座レベル: protocol_eval seeds {SEEDS} (mp{MAX_POS}, P=4.0) + IS較正→OOS")
    closes = mm.load_closes()
    accounts = {}
    accounts["base_d0"] = account_eval("base_d0(cached pool)", pool_base_cached, closes)
    print(f"    [{time.time()-t0:.0f}s]")
    accounts["d1_mine"] = account_eval("d1_mine(vbt reimpl)", pool_d1_mine, closes)
    print(f"    [{time.time()-t0:.0f}s]")

    rows = []
    for label, r in accounts.items():
        rows.append({"label": label, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
                     "emp_dd": r["emp_dd"], "emp_p95": r["emp_p95"],
                     **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS},
                     **{f"rob_k_s{sd}": r["rob"][sd]["k"] for sd in SEEDS},
                     "rob_mean": r["rob_cagr_mean"],
                     "worst_year": r["worst_year"], "neg_years_emp": r["neg_years_emp"],
                     "neg_years_rob0": r["neg_years_rob0"],
                     "k_is_emp": r["k_is_emp"], "is_emp_cagr": r["is_emp_cagr"],
                     "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
                     "k_is_rob": r["k_is_rob"], "is_rob_cagr": r["is_rob_cagr"],
                     "oos_rob_cagr": r["oos_rob_cagr"], "oos_rob_dd": r["oos_rob_dd"]})
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT_DIR / "exp51_account.csv", index=False)

    b, d = accounts["base_d0"], accounts["d1_mine"]
    checks = {
        "base_rob_mean": (b["rob_cagr_mean"], EXP47["base_rob_mean"]),
        "d1_rob_mean": (d["rob_cagr_mean"], EXP47["d1_rob_mean"]),
        "base_emp": (b["emp_cagr"], EXP47["base_emp"]),
        "d1_emp": (d["emp_cagr"], EXP47["d1_emp"]),
        "base_oos_rob": (b["oos_rob_cagr"], EXP47["base_oos_rob"]),
        "d1_oos_rob": (d["oos_rob_cagr"], EXP47["d1_oos_rob"]),
    }
    print("\n  --- exp47 との ±0.5pp 一致判定 ---")
    agree = {}
    for k, (mine, ref) in checks.items():
        diff = (mine - ref) * 100
        ok = abs(diff) <= 0.5
        agree[k] = {"mine": mine, "exp47": ref, "diff_pp": diff, "within_0.5pp": ok}
        print(f"  {k:16s} mine {mine:+.2%}  exp47 {ref:+.2%}  diff {diff:+.3f}pp  "
              f"{'OK' if ok else 'NG'}")
    adv = (d["rob_cagr_mean"] - b["rob_cagr_mean"]) * 100
    print(f"\n  d1−base robust優位(本実装): {adv:+.2f}pp "
          f"(exp47: {(EXP47['d1_rob_mean']-EXP47['base_rob_mean'])*100:+.2f}pp)")
    print("  ペアシード別 d1−base: " + "  ".join(
        f"s{sd}:{(d['rob'][sd]['cagr']-b['rob'][sd]['cagr'])*100:+.2f}pp" for sd in SEEDS))
    result["account"] = {k: {kk: vv for kk, vv in v.items() if kk != "rob"} |
                         {"rob": {str(s): r for s, r in v["rob"].items()}}
                         for k, v in accounts.items()}
    result["agreement"] = agree
    result["d1_minus_base_rob_pp"] = adv

    sec("4. レバレッジ監査(emp_k / rob_k平均, 国内規制25x)")
    mk_d1 = champion_sizing(pool_d1_mine, max_pos=MAX_POS)
    lev_rows = []
    for tag, k in [("emp_k", d["emp_k"]), ("rob_k_mean", d["rob_k_mean"]),
                   ("exp47_emp_k", EXP47["d1_emp_k"])]:
        la = leverage_audit(pool_d1_mine, closes, mk_d1(k))
        # 検算: 同一 k の mm.simulate と最終実現 equity が一致するか
        _, eqr, _ = mm.simulate(pool_d1_mine, closes, mk_d1(k), max_pos=MAX_POS)
        sim_ok = abs(la["final_equity"] - eqr.iloc[-1]) < 1e-6
        lev_rows.append({"basis": tag, "k": k, **la, "matches_mm_simulate": sim_ok,
                         "within_25x_total": la["max_lev_total"] <= 25.0,
                         "within_25x_single": la["max_lev_single"] <= 25.0})
        print(f"  [{tag}] k={k:.2f}  総露出max {la['max_lev_total']:.2f}x "
              f"(p99 {la['p99_lev_total']:.2f}x, at {la['t_max_lev_total']}) | "
              f"1玉max(対MtM) {la['max_lev_single']:.2f}x | "
              f"発注時1玉 max/p99/mean {la['max_alloc_frac_entry']:.2f}/"
              f"{la['p99_alloc_frac_entry']:.2f}/{la['mean_alloc_frac_entry']:.2f}x | "
              f"25x以内: 総 {la['max_lev_total'] <= 25.0} | simulate検算 {sim_ok}")
    ldf = pd.DataFrame(lev_rows)
    ldf.to_csv(OUT_DIR / "exp51_leverage.csv", index=False)
    result["leverage"] = lev_rows

    (OUT_DIR / "exp51_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nsaved -> {OUT_DIR / 'exp51_pool_compare.csv'}")
    print(f"      -> {OUT_DIR / 'exp51_pool_mismatch.csv'}")
    print(f"      -> {OUT_DIR / 'exp51_account.csv'}")
    print(f"      -> {OUT_DIR / 'exp51_leverage.csv'}")
    print(f"      -> {OUT_DIR / 'exp51_result.json'}")
    print(f"総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
