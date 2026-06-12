"""exp46: 位相オフセット採集(phase harvest)— H4グリッド位相 0/1/2/3h の因果統合レバー検証。

仮説: チャンピオン(confluence_meanrev_v2, H4, 19銘柄)の H4 グリッド(UTC 0/4/8/12/16/20h)は
恣意的な位相であり、サンプリング間隔内に発生して戻る鋭い乖離を構造的に見逃す。
+1h/+2h/+3h オフセットグリッドで同戦略を走らせ、「先にシグナルを出したグリッドが取る」
一玉/銘柄の causal greedy union(未来情報なし)で統合すれば、ユニークな回帰エピソードを
採集して口座 CAGR を押し上げられるか。

除染規則: バーlabel T の最終M1時刻は T+3h59m。(T.hour+3)%24 ∈ {20,21,22} のバー
(=UTC20:00-22:59 のM1でプライシング=ロールオーバーBIDアーティファクト帯)は
entry/exit ともシグナルを False に強制(exit は次のクリーンなバーへ自然繰延べ)。
オフセット0h はこの規則で1本も落ちない(assert で確認)。

候補:
  V_base : results/mm_pool_v2_H4_19.parquet そのまま(ベースライン再現の検算)
  V_all  : G0+G1+G2+G3 統合(除染済み)
  V_02   : G0+G2 統合(除染済み)
  G1/G2/G3 単独(除染済み, seed0 のみ)= チャンピオンが位相運でないことの定量化
  V_all 除染なし(プールレベルのみ)= アーティファクトの水増し率の負の知見

判定: research/lab/tail_protocol.py(empirical 20% / ペアシード robust p95=20% 5シード /
レバ偽装署名 / IS<2022 較正→OOS素検証 / 全年プラス / 単年依存禁止 / スキップ率)。
口座シミュは mm_lab.simulate × mm_production.champion_sizing(P=4.0, z0=2.2)× max_pos=8。
MtM グリッドは mm.load_closes() のH4ベースグリッド(従来の全比較と同一の物差し)。

実行:
  PYTHONPATH=. uv run python research/experiments/exp46_phase_harvest.py --pools-only
  PYTHONPATH=. uv run python research/experiments/exp46_phase_harvest.py
出力:
  research/outputs/exp46_pools.csv / exp46_account.csv / exp46_result.json
  results/mm_pool_v2ph{h}_H4_19.parquet (位相プールキャッシュ, raw/除染)
  results/mm_pool_v2_phase_all_H4_19.parquet (V_all 統合プール, provenance付き)
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

from fxlab import config, universe as uni  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402
from strategies.confluence_meanrev_v2 import PARAMS as V2_PARAMS  # noqa: E402
from strategies.confluence_meanrev_v2 import generate_signals as v2_gen  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
MAX_POS = 8
OFFSETS = [0, 1, 2, 3]
SEEDS_FULL = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
CONTAM_HOURS = (20, 21, 22)  # (label.hour + 3) % 24 がここに入るバーを判定から外す
PASS_ABS = 0.1805            # 合格ライン(+10%相対, reports/16 基準)
PARTIAL_ABS = 0.172          # 部分的前進ライン

POOL_REF = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"
UNION_OUT = config.RESULTS_DIR / "mm_pool_v2_phase_all_H4_19.parquet"
OUT_DIR = ROOT / "research" / "outputs"
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

# スカウト参照値(オーケストレーター実測)
SCOUT_G2_RAW = {"n": 1244, "sum": 1.7341}


# --- オフセットグリッドのデータ生成 ---------------------------------------
_major_cache: dict = {}


def major_offset(pair: str, h: int) -> pd.DataFrame:
    key = (pair, h)
    if key not in _major_cache:
        df = load_m1(pair).resample("4h", label="left", closed="left",
                                    offset=f"{h}h").agg(AGG)
        _major_cache[key] = df.dropna(subset=["open", "high", "low", "close"])
    return _major_cache[key]


def instrument_data_offset(name: str, h: int) -> pd.DataFrame:
    """メジャーはオフセット集約、クロスはオフセット集約済み close から合成(OHLC=close)。"""
    if name not in uni.CROSS_DEFS:
        return major_offset(name, h)
    a, op, b = uni.CROSS_DEFS[name]
    ca, cb = major_offset(a, h)["close"], major_offset(b, h)["close"]
    df = pd.concat([ca, cb], axis=1).dropna()
    c = df.iloc[:, 0] / df.iloc[:, 1] if op == "/" else df.iloc[:, 0] * df.iloc[:, 1]
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0},
                        index=c.index)


# --- 除染シグナルラッパ ----------------------------------------------------
def bad_mask(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series(np.isin((index.hour + 3) % 24, CONTAM_HOURS), index=index)


def make_gen(decon: bool):
    def gen(data: pd.DataFrame, **params):
        le, lx, se, sx = v2_gen(data, **params)
        if decon:
            ok = ~bad_mask(data.index)
            le, lx, se, sx = le & ok, lx & ok, se & ok, sx & ok
        return le, lx, se, sx
    return gen


# --- プール構築(mm_lab.build_pool 互換列) ---------------------------------
def _zscore(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_phase_pool(h: int, instruments, decon=True, cache=True) -> pd.DataFrame:
    tag = f"v2ph{h}" + ("" if decon else "raw")
    path = config.RESULTS_DIR / f"mm_pool_{tag}_H4_19.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    gen = make_gen(decon)
    params = {k: v for k, v in V2_PARAMS.items()}
    win = params["window"]
    frames = []
    for nm in instruments:
        data = instrument_data_offset(nm, h)
        if h == 0:
            nb = int(bad_mask(data.index).sum())
            assert nb == 0, f"offset0 に汚染バーが {nb} 本存在 ({nm}) — 規則の前提が崩れている"
        pf = run(nm, TF, gen, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        close = data["close"]
        z = _zscore(close, win)
        vol = close.pct_change().rolling(20).std()
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
    pool = pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)
    if cache:
        pool.to_parquet(path)
    return pool


# --- 因果的な統合(causal greedy union) ------------------------------------
def causal_union(pools_by_offset: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """全グリッドの全トレードを entry 昇順に走査し、その銘柄に建玉が無いときだけ採用。

    同時刻タイはオフセット小優先(グリッドの label 集合は互いに素なので実際には発生しない)。
    採用したら exit まで busy = 一玉/銘柄。未来情報を一切使わない運用ルール。
    """
    frames = []
    for h in sorted(pools_by_offset):
        q = pools_by_offset[h].copy()
        q["provenance"] = h
        frames.append(q)
    allp = pd.concat(frames, ignore_index=True)
    allp = allp.sort_values(["entry", "provenance"], kind="mergesort").reset_index(drop=True)
    busy: dict[str, pd.Timestamp] = {}
    keep = []
    for row in allp.itertuples():
        b = busy.get(row.instr)
        if b is not None and row.entry < b:
            continue
        keep.append(row.Index)
        busy[row.instr] = row.exit
    return allp.loc[keep].reset_index(drop=True)


def displacement_report(union: pd.DataFrame, base: pd.DataFrame) -> dict:
    """ベース由来の生存/置換(displacement)/ユニーク追加の構成と損益差。"""
    ukeys0 = set(zip(union.loc[union["provenance"] == 0, "instr"],
                     union.loc[union["provenance"] == 0, "entry"]))
    bkeys = list(zip(base["instr"], base["entry"]))
    displaced = base[[k not in ukeys0 for k in bkeys]]
    nonbase = union[union["provenance"] > 0]
    blockers_idx: set = set()
    nb_by_instr = {i: g for i, g in nonbase.groupby("instr")}
    for r in displaced.itertuples():
        g = nb_by_instr.get(r.instr)
        if g is None:
            continue
        m = g[(g["entry"] <= r.entry) & (g["exit"] > r.entry)]
        blockers_idx.update(m.index.tolist())
    blockers = nonbase.loc[sorted(blockers_idx)]
    unique_add = nonbase.drop(index=sorted(blockers_idx))
    return {
        "base_n": int(len(base)),
        "survived_base_n": int(len(ukeys0)),
        "displaced_n": int(len(displaced)),
        "displaced_sum": float(displaced["ret"].sum()),
        "blockers_n": int(len(blockers)),
        "blockers_sum": float(blockers["ret"].sum()),
        "displacement_pnl_diff": float(blockers["ret"].sum() - displaced["ret"].sum()),
        "unique_add_n": int(len(unique_add)),
        "unique_add_sum": float(unique_add["ret"].sum()),
        "union_n": int(len(union)),
        "union_sum": float(union["ret"].sum()),
    }


# --- プールレベル統計(exp41 流用) ------------------------------------------
def pool_stats(pool: pd.DataFrame) -> dict:
    r = pool["ret"]
    pos, neg = r[r > 0].sum(), r[r < 0].sum()
    yearly = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    return {
        "n": int(len(pool)), "sum_ret": float(r.sum()),
        "mean_bps": float(r.mean() * 1e4), "win": float((r > 0).mean()),
        "pf": float(pos / abs(neg)) if neg < 0 else np.inf,
        "pos_years": int((yearly > 0).sum()), "n_years": int(len(yearly)),
        "worst_year": float(yearly.min()),
        "era_16_21": float(pool.loc[pd.to_datetime(pool["exit"]) < "2022-01-01", "ret"].sum()),
        "era_22_26": float(pool.loc[pd.to_datetime(pool["exit"]) >= "2022-01-01", "ret"].sum()),
        "med_hold": float(pool["bars_held"].median()),
    }


# --- 口座レベル評価(exp41 流儀) --------------------------------------------
def make_eq_fn(pool, closes, mk):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    return eq_of_k


def eval_candidate(label, pool, closes, seeds, *, with_oos=True) -> dict:
    mk = champion_sizing(pool, max_pos=MAX_POS)
    eq_of_k = make_eq_fn(pool, closes, mk)
    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    res["n_pool"] = int(len(pool))

    eq_emp = eq_of_k(res["emp_k"])
    yr_emp = yearly_returns(eq_emp)
    res["worst_year_emp"] = float(yr_emp.min())
    res["neg_years_emp"] = int((yr_emp < 0).sum())
    res["yr_emp"] = {int(y): float(v) for y, v in yr_emp.items()}
    eq_r0 = eq_of_k(res["rob"][seeds[0]]["k"])
    yr_r0 = yearly_returns(eq_r0)
    res["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())

    # 同時建玉/スキップ統計(emp_k で実測。スキップ=max_pos 満杯による見送り)
    _, _, info = mm.simulate(pool, closes, mk(res["emp_k"]), max_pos=MAX_POS)
    years = (closes.index[-1] - closes.index[0]).days / 365.25
    res["skip"] = {
        "skipped": int(info["skipped"]), "n_taken": int(info["n_taken"]),
        "skip_rate": float(info["skipped"] / max(info["skipped"] + info["n_taken"], 1)),
        "max_conc": int(info["max_conc"]), "avg_conc": float(info["avg_conc"]),
        "trades_per_year": float(info["n_taken"] / years),
    }
    print(f"      skip: {res['skip']['skipped']}/{res['skip']['skipped']+res['skip']['n_taken']} "
          f"({res['skip']['skip_rate']:.1%}) max_conc={res['skip']['max_conc']} "
          f"avg={res['skip']['avg_conc']:.2f} 年間取引 {res['skip']['trades_per_year']:.0f}")

    if with_oos:
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


def yearly_gate(cand_yr: dict, base_yr: dict) -> dict:
    """年次差分(候補−ベース): 最良年除外後の残存率 と 2022除外後の利得。"""
    yc, yb = pd.Series(cand_yr), pd.Series(base_yr)
    d = (yc - yb).dropna()
    total = float(d.sum())
    best_y = int(d.idxmax())
    remain = float(d.drop(best_y).sum())
    ex2022 = float(d.drop(2022).sum()) if 2022 in d.index else float("nan")
    ok = (total > 0) and (remain >= 0.5 * total) and (ex2022 > 0)
    return {"total": total, "best_year": best_y, "best_val": float(d[best_y]),
            "remain_after_best": remain,
            "remain_frac": remain / total if total > 0 else float("nan"),
            "ex2022": ex2022, "pass": bool(ok),
            "diffs": {int(y): float(v) for y, v in d.items()}}


def judge_gates(cand: dict, base: dict) -> dict:
    rel = cand["rob_cagr_mean"] / base["rob_cagr_mean"] - 1
    g1 = bool(cand["rob_cagr_mean"] >= max(PASS_ABS, base["rob_cagr_mean"] * 1.10))
    partial = bool(cand["rob_cagr_mean"] >= PARTIAL_ABS)
    sig = (cand["emp_cagr"] > base["emp_cagr"]) and \
          (abs(cand["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
    g2 = not sig
    g3 = bool((cand["oos_rob_cagr"] > base["oos_rob_cagr"]) and
              (cand["oos_emp_cagr"] > base["oos_emp_cagr"]))
    g4 = bool(cand["neg_years_emp"] == 0 and cand["neg_years_rob0"] == 0)
    y_emp = yearly_gate(cand["yr_emp"], base["yr_emp"])
    y_rob = yearly_gate(cand["yr_rob0"], base["yr_rob0"])
    g5 = bool(y_emp["pass"] and y_rob["pass"])
    g6_note = (f"skip_rate={cand['skip']['skip_rate']:.1%} "
               f"max_conc={cand['skip']['max_conc']}")
    all_pass = g1 and g2 and g3 and g4 and g5
    return {"rob_gain_rel": float(rel), "g1_rob10pct": g1, "g1_partial": partial,
            "g2_no_lev_disguise": g2, "lev_signature": bool(sig),
            "g3_oos_advantage": g3,
            "oos_rob_adv_pp": float((cand["oos_rob_cagr"] - base["oos_rob_cagr"]) * 100),
            "oos_emp_adv_pp": float((cand["oos_emp_cagr"] - base["oos_emp_cagr"]) * 100),
            "g4_all_years_positive": g4,
            "g5_no_single_year": g5, "g5_emp": y_emp, "g5_rob0": y_rob,
            "g6_skip_note": g6_note, "all_pass": bool(all_pass)}


# --- メイン ------------------------------------------------------------------
def main() -> int:
    pools_only = "--pools-only" in sys.argv
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    instruments = mm.default_instruments()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== exp46: phase harvest (H4 offsets {OFFSETS}, {len(instruments)}銘柄, "
          f"mp{MAX_POS}, champion_sizing P=4.0) ===\n")

    # --- 1. プール構築 + 検算 ------------------------------------------------
    ref = pd.read_parquet(POOL_REF)
    pools_d, pools_r = {}, {}
    prows = []
    for h in OFFSETS:
        pools_r[h] = build_phase_pool(h, instruments, decon=False)
        pools_d[h] = build_phase_pool(h, instruments, decon=True)
        sr, sd = pool_stats(pools_r[h]), pool_stats(pools_d[h])
        prows.append({"pool": f"G{h}_raw", **sr})
        prows.append({"pool": f"G{h}_decon", **sd})
        print(f"  G{h}: raw n={sr['n']} sum={sr['sum_ret']:+.4f} PF={sr['pf']:.3f} | "
              f"decon n={sd['n']} sum={sd['sum_ret']:+.4f} PF={sd['pf']:.3f} "
              f"[{time.time()-t0:.0f}s]")

    # 検算1: G0(raw=decon) がベース参照と一致
    g0 = pools_d[0]
    n_ok = len(g0) == len(ref)
    s_ok = abs(g0["ret"].sum() - ref["ret"].sum()) < 1e-6
    mg = g0.merge(ref[["instr", "entry", "ret", "z_entry"]], on=["instr", "entry"],
                  how="inner", suffixes=("", "_ref"))
    rdiff = float(np.nanmax(np.abs(mg["ret"] - mg["ret_ref"])))
    zdiff = float(np.nanmax(np.abs(mg["z_entry"] - mg["z_entry_ref"])))
    raw_eq_decon = len(pools_r[0]) == len(g0) and \
        abs(pools_r[0]["ret"].sum() - g0["ret"].sum()) < 1e-12
    print(f"\n  [検算1] G0 vs ベース参照: n一致={n_ok} sum一致(<1e-6)={s_ok} "
          f"max|Δret|={rdiff:.2e} max|Δz|={zdiff:.2e} raw==decon={raw_eq_decon}")
    if not (n_ok and s_ok and rdiff < 1e-9 and raw_eq_decon):
        print("  !! ベースライン再現失敗。以降の比較は無効。")
        return 1
    # 検算2: G2 raw がスカウト値と一致
    g2n, g2s = len(pools_r[2]), float(pools_r[2]["ret"].sum())
    g2_ok = (g2n == SCOUT_G2_RAW["n"]) and (abs(g2s - SCOUT_G2_RAW["sum"]) <= 0.001)
    print(f"  [検算2] G2 raw n={g2n} (期待{SCOUT_G2_RAW['n']}) sum={g2s:+.4f} "
          f"(期待{SCOUT_G2_RAW['sum']:+.4f}) -> {'OK' if g2_ok else 'NG'}")
    if not g2_ok:
        print("  !! スカウト値と不一致。以降の比較は無効。")
        return 1

    # --- 2. 統合プール --------------------------------------------------------
    v_all = causal_union(pools_d)
    v_02 = causal_union({0: pools_d[0], 2: pools_d[2]})
    v_all_nodecon = causal_union(pools_r)
    v_all.to_parquet(UNION_OUT)
    prows.append({"pool": "V_all", **pool_stats(v_all)})
    prows.append({"pool": "V_02", **pool_stats(v_02)})
    prows.append({"pool": "V_all_nodecon", **pool_stats(v_all_nodecon)})
    pdf = pd.DataFrame(prows)
    pdf.to_csv(OUT_DIR / "exp46_pools.csv", index=False)
    print(f"\n=== 2. プールレベル(統合) ===")
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    disp_all = displacement_report(v_all, ref)
    disp_02 = displacement_report(v_02, ref)
    print("\n--- displacement 構成 (V_all) ---")
    for k, v in disp_all.items():
        print(f"  {k:24s}: {v:+.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    print("--- displacement 構成 (V_02) ---")
    for k, v in disp_02.items():
        print(f"  {k:24s}: {v:+.4f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    # 除染なしの水増し率(負の知見)
    inflate = (v_all_nodecon["ret"].sum() - v_all["ret"].sum()) / abs(v_all["ret"].sum())
    print(f"\n  [負の知見] V_all 除染なし sum={v_all_nodecon['ret'].sum():+.4f} vs "
          f"除染後 {v_all['ret'].sum():+.4f} -> アーティファクト水増し {inflate:+.1%} "
          f"(n: {len(v_all_nodecon)} vs {len(v_all)})")

    provenance_mix = v_all["provenance"].value_counts().sort_index().to_dict()
    print(f"  V_all provenance 構成: {provenance_mix}")

    if pools_only:
        print(f"\n--pools-only 完了 ({time.time()-t0:.0f}s)")
        return 0

    # --- 3. 口座レベル ステージ1(seed0, 位相単独監査込み) --------------------
    closes = mm.load_closes()
    print(f"\n=== 3. 口座シミュ ステージ1 (seed0) === grid {len(closes)}本")
    results: dict[str, dict] = {}
    for label, pool in [("G1_solo", pools_d[1]), ("G2_solo", pools_d[2]),
                        ("G3_solo", pools_d[3])]:
        results[label] = eval_candidate(label, pool, closes, seeds=(0,), with_oos=False)
        print(f"    [{time.time()-t0:.0f}s]")

    # --- 4. ステージ2: V_base / V_all / V_02 を seeds 0-4 フル -----------------
    print(f"\n=== 4. ステージ2: seeds {SEEDS_FULL} フル評価 ===")
    for label, pool in [("V_base", ref), ("V_all", v_all), ("V_02", v_02)]:
        results[label] = eval_candidate(label, pool, closes, seeds=SEEDS_FULL)
        print(f"    [{time.time()-t0:.0f}s]")

    base = results["V_base"]
    print("\n--- ベースライン再現チェック (参照: rob_mean +16.41% / emp +24.64%) ---")
    print(f"  V_base rob_mean={base['rob_cagr_mean']:+.2%} emp={base['emp_cagr']:+.2%} "
          f"p95={base['emp_p95']:+.1%}")

    # ペアシード比較表
    cands = ["V_all", "V_02"]
    print("\n--- ペアシード robust(p95=20%) CAGR 比較(同一シード集合) ---")
    print("  seed | V_base | " + " | ".join(f"{c} (diff)" for c in cands))
    for sd in SEEDS_FULL:
        cells = []
        for c in cands:
            v = results[c]["rob"][sd]["cagr"]
            cells.append(f"{v:+.2%} ({(v - base['rob'][sd]['cagr'])*100:+.2f}pp)")
        print(f"   s{sd}  | {base['rob'][sd]['cagr']:+.2%} | " + " | ".join(cells))
    print(f"  mean | {base['rob_cagr_mean']:+.2%} | " + " | ".join(
        f"{results[c]['rob_cagr_mean']:+.2%} "
        f"({(results[c]['rob_cagr_mean']-base['rob_cagr_mean'])*100:+.2f}pp)" for c in cands))
    print(f"  empirical: base {base['emp_cagr']:+.2%} (p95 {base['emp_p95']:+.1%}) | " +
          " | ".join(f"{c} {results[c]['emp_cagr']:+.2%} (p95 {results[c]['emp_p95']:+.1%})"
                     for c in cands))

    # --- 5. ゲート判定 ---------------------------------------------------------
    print("\n=== 5. 判定ゲート ===")
    gates = {}
    for c in cands:
        gates[c] = judge_gates(results[c], base)
        g = gates[c]
        print(f"\n  [{c}] rob_mean {results[c]['rob_cagr_mean']:+.2%} "
              f"(rel {g['rob_gain_rel']:+.1%})")
        print(f"    G1 +10%相対(≥{max(PASS_ABS, base['rob_cagr_mean']*1.10):.2%}): "
              f"{g['g1_rob10pct']} (部分的前進≥{PARTIAL_ABS:.1%}: {g['g1_partial']})")
        print(f"    G2 レバ偽装署名なし: {g['g2_no_lev_disguise']} "
              f"(emp {results[c]['emp_cagr']:+.2%} vs base {base['emp_cagr']:+.2%}, "
              f"p95 {results[c]['emp_p95']:+.1%} vs {base['emp_p95']:+.1%})")
        print(f"    G3 OOS優位持続: {g['g3_oos_advantage']} "
              f"(rob {g['oos_rob_adv_pp']:+.2f}pp / emp {g['oos_emp_adv_pp']:+.2f}pp)")
        print(f"    G4 全年プラス: {g['g4_all_years_positive']} "
              f"(neg emp={results[c]['neg_years_emp']} rob0={results[c]['neg_years_rob0']})")
        print(f"    G5 単年依存なし: {g['g5_no_single_year']} "
              f"(emp: 最良年{g['g5_emp']['best_year']} 残存{g['g5_emp']['remain_frac']:.0%} "
              f"2022除外{g['g5_emp']['ex2022']:+.2%} | "
              f"rob0: 残存{g['g5_rob0']['remain_frac']:.0%} 2022除外{g['g5_rob0']['ex2022']:+.2%})")
        print(f"    G6 {g['g6_skip_note']}")
        print(f"    => ALL PASS: {g['all_pass']}")

    # 年次差分の詳細
    for c in cands:
        for tag in ("emp", "rob0"):
            d = gates[c][f"g5_{tag}"]["diffs"]
            print(f"\n  {c} - V_base 年次差分 [{tag}]: " +
                  "  ".join(f"{y}:{v:+.1%}" for y, v in d.items()))

    # 位相単独監査サマリ
    print("\n=== 6. 位相単独監査 (seed0, 除染済み) ===")
    print("  位相  n     emp_cagr   rob_s0    (V_base rob_s0: "
          f"{base['rob'][0]['cagr']:+.2%})")
    for label in ("G1_solo", "G2_solo", "G3_solo"):
        r = results[label]
        print(f"  {label}: n={r['n_pool']} emp={r['emp_cagr']:+.2%} "
              f"rob_s0={r['rob'][0]['cagr']:+.2%} neg_years_emp={r['neg_years_emp']}")

    # --- 保存 -------------------------------------------------------------------
    rows = []
    for label, r in results.items():
        rows.append({
            "label": label, "n_pool": r["n_pool"],
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"],
            **{f"rob_s{sd}": r["rob"].get(sd, {}).get("cagr") for sd in SEEDS_FULL},
            "rob_mean": r.get("rob_cagr_mean"),
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            "worst_year_emp": r["worst_year_emp"],
            "skip_rate": r["skip"]["skip_rate"], "max_conc": r["skip"]["max_conc"],
            "trades_per_year": r["skip"]["trades_per_year"],
            "k_is_emp": r.get("k_is_emp"), "is_emp_cagr": r.get("is_emp_cagr"),
            "oos_emp_cagr": r.get("oos_emp_cagr"), "oos_emp_dd": r.get("oos_emp_dd"),
            "k_is_rob": r.get("k_is_rob"), "is_rob_cagr": r.get("is_rob_cagr"),
            "oos_rob_cagr": r.get("oos_rob_cagr"), "oos_rob_dd": r.get("oos_rob_dd"),
        })
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT_DIR / "exp46_account.csv", index=False)

    out = {
        "pools": prows,
        "displacement": {"V_all": disp_all, "V_02": disp_02},
        "artifact_inflation_nodecon": float(inflate),
        "provenance_mix_V_all": {str(k): int(v) for k, v in provenance_mix.items()},
        "results": {lbl: {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                          for k, v in r.items()} for lbl, r in results.items()},
        "gates": gates,
    }
    (OUT_DIR / "exp46_result.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp46_pools.csv'}")
    print(f"      -> {OUT_DIR / 'exp46_account.csv'}")
    print(f"      -> {OUT_DIR / 'exp46_result.json'}")
    print(f"      -> {UNION_OUT}")
    print("\n=== 口座レベル最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf.to_string(index=False))
    print(f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
