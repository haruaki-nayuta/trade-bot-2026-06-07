"""exp47: エントリー d バー遅延レバー(d∈{1,2,3})のフル検証 — 同一テール判定プロトコル。

背景(オーケストレーターの事前スカウト):
  ベースプール(mm_pool_v2_H4_19)の entry を自グリッドで d バー遅延し、entry 価格を
  遅延バー close に置換(コストは元トレードの再構成コストを維持)すると、プールレベルで
  d=1 +2.8% / d=2 +9.0%(PF 1.707→1.858) / d=3 +4.5%。ただし d=2 の利得 +0.171 のうち
  2020年だけで +0.151(最良年除外後 +0.020)= 単年依存の強い署名。IS/OOS も
  d1 IS-0.9%/OOS+6.1%, d2 IS+13.4%/OOS+5.0%, d3 IS+16.3%/OOS-6.0% と不整合。
  本実験は口座レベル(robust 較正は DD 形状も変える)で正式に結論を出し、
  落ちるなら「どの機構で死んだか」を確定させる(負の知見の確定が目的)。

実装の規約:
  ・entry_close/exit_close/cost は anatomy_cost.py の方法で再構成
    (検算: sum(gross-cost) = +1.9086 一致が比較の前提)。
  ・遅延後 entry = 自銘柄 H4 グリッドの d 本後バー(タイムスタンプも更新)。
    entry_price = 遅延バー close × (元の片道スリッページ比率 entry_price/entry_close)。
    ret = dir × (exit_close/遅延close − 1) − cost(元トレードの往復コストを維持)。
  ・遅延後 entry ≥ exit となるトレードは除外(消滅トレード。失う ret は差分に計上)。
  ・z_entry(シグナル時点の値=因果)・vol_entry は維持。
  ・口座レベル: champion_sizing(P=4.0, z0=2.2) × max_pos=8、MtM グリッドは mm.load_closes()。
  ・判定: tail_protocol(empirical/robust ペアシード較正、seeds 0-4)+ レバ偽装署名 +
    IS(<2022)較正→OOS素検証 + IS-argmax 監査 + 年次差分(最良年除外・2022除外・全年プラス)。

追加監査(本文 6-8 節):
  6) 署名のブートシード監査(p95 はシード依存 ±0.4-0.8pp → seeds 0-2 でペア測定)
  7) ロールオーバーBIDアーティファクト監査: 遅延後エントリーバーの時刻別利得分解 +
     「遅延後 h20 バーに乗ったら 1 本送る」除染変種(プール段+名目PASS候補は口座段も)
  8) 経験的最大DDエピソードの位置(較正が単一事象の軟化に依存していないか)
ゲート5(単年依存)は口座段(emp較正/rob_s0較正)に加え**プール段**でも判定する
(レバ上振れは全年に均等に乗るため、口座段の年次差分は単年依存の判別力が弱い)。

実行: PYTHONPATH=. uv run python research/experiments/exp47_entry_delay.py
      PYTHONPATH=. uv run python research/experiments/exp47_entry_delay.py --pool-only
      PYTHONPATH=. uv run python research/experiments/exp47_entry_delay.py --ext
        (追補: d=4,5,6 断面延長=口座レベル用量反応が端で伸び続けるかの判定)
出力: research/outputs/exp47_pool.csv / exp47_account.csv / exp47_result.json
      (--ext: exp47_ext.csv)
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
from fxlab import universe as uni  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

BASE_NET = 1.9086
DELAYS = [1, 2, 3]
MAX_POS = 8
SEEDS_FULL = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
OUT_POOL = OUT_DIR / "exp47_pool.csv"
OUT_ACC = OUT_DIR / "exp47_account.csv"
OUT_JSON = OUT_DIR / "exp47_result.json"


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --- 価格再構成(anatomy_cost.py の方法) ---------------------------------
def reconstruct(pool: pd.DataFrame) -> dict:
    closes_by = {i: uni.instrument_close(i, "H4") for i in sorted(pool["instr"].unique())}
    n = len(pool)
    idx_e = np.full(n, -1)
    idx_x = np.full(n, -1)
    entry_close = np.full(n, np.nan)
    exit_close = np.full(n, np.nan)
    for instr, g in pool.groupby("instr"):
        s = closes_by[instr]
        ie = s.index.get_indexer(g["entry"])
        ix = s.index.get_indexer(g["exit"])
        assert (ie >= 0).all() and (ix >= 0).all(), f"{instr}: timestamp miss"
        rows = g.index.to_numpy()
        idx_e[rows] = ie
        idx_x[rows] = ix
        entry_close[rows] = s.to_numpy()[ie]
        exit_close[rows] = s.to_numpy()[ix]
    d = pool["dir"].to_numpy().astype(float)
    gross = d * (exit_close / entry_close - 1.0)
    cost = gross - pool["ret"].to_numpy()
    slip = pool["entry_price"].to_numpy() / entry_close  # 片道スリッページ比率
    return {"closes_by": closes_by, "idx_e": idx_e, "idx_x": idx_x,
            "entry_close": entry_close, "exit_close": exit_close,
            "gross": gross, "cost": cost, "slip": slip}


# --- 遅延プール生成 --------------------------------------------------------
def delayed_pool(pool: pd.DataFrame, rc: dict, dly: int, skip_h20: bool = False):
    """entry を自銘柄グリッドで d 本遅延。返り値: (mod_pool, kept, ret_new, extra).

    skip_h20=True: 遅延後の entry バーが UTC20:00 ラベル(=close が 23:59 の
    ロールオーバー汚染 M1 でプライシング)なら、さらに 1 本後ろ(00:00 バー)へ送る除染規則。
    """
    n = len(pool)
    dclose = np.full(n, np.nan)
    dts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    ie_d_full = np.full(n, -1)
    for instr, g in pool.groupby("instr"):
        rows = g.index.to_numpy()
        s = rc["closes_by"][instr]
        ie_d = rc["idx_e"][rows] + dly
        if skip_h20:
            hrs = pd.DatetimeIndex(s.index.values[np.minimum(ie_d, len(s) - 1)]).hour
            ie_d = ie_d + (hrs == 20).astype(int)
        ie_cl = np.minimum(ie_d, len(s) - 1)
        dclose[rows] = s.to_numpy()[ie_cl]
        dts[rows] = s.index.values[ie_cl]
        ie_d_full[rows] = ie_d
    kept = ie_d_full < rc["idx_x"]
    d = pool["dir"].to_numpy().astype(float)
    ret_new = d * (rc["exit_close"] / dclose - 1.0) - rc["cost"]
    mod = pool.copy()
    mod["entry"] = pd.DatetimeIndex(dts).tz_localize("UTC")
    mod["entry_price"] = dclose * rc["slip"]
    mod["ret"] = ret_new
    mod["bars_held"] = np.maximum(pool["bars_held"].to_numpy() - dly, 1)
    mod = mod[kept].sort_values("entry").reset_index(drop=True)
    extra = {"dts": pd.DatetimeIndex(dts).tz_localize("UTC"), "dclose": dclose}
    return mod, kept, ret_new, extra


def pool_cross_section(pool: pd.DataFrame, mod: pd.DataFrame, kept: np.ndarray,
                       ret_new: np.ndarray, dly: int) -> dict:
    """プールレベル差分の分解(年次=決済年 / IS-OOS=元シグナル年 / 方向)。"""
    ret0 = pool["ret"].to_numpy()
    r = mod["ret"]
    pos, neg = r[r > 0].sum(), r[r < 0].sum()
    # トレード単位差分(見送り=失う ret)
    diff_tr = np.where(kept, ret_new - ret0, -ret0)
    yr = pd.Series(diff_tr).groupby(pool["exit"].dt.year).sum()
    ynew = pd.Series(np.where(kept, ret_new, 0.0)).groupby(pool["exit"].dt.year).sum()
    is_mask = (pool["entry"] < OOS_START).to_numpy()
    dmask = pool["dir"].to_numpy() > 0
    best_y = int(yr.idxmax())
    total = float(diff_tr.sum())
    return {
        "dly": dly, "n": int(len(mod)), "sum_ret": float(r.sum()),
        "diff": total, "diff_pct": total / BASE_NET * 100,
        "pf": float(pos / abs(neg)) if neg < 0 else np.inf,
        "win": float((r > 0).mean()),
        "dropped": int((~kept).sum()), "dropped_ret_sum": float(ret0[~kept].sum()),
        "diff_is": float(diff_tr[is_mask].sum()), "diff_oos": float(diff_tr[~is_mask].sum()),
        "diff_long": float(diff_tr[dmask].sum()), "diff_short": float(diff_tr[~dmask].sum()),
        "best_year": best_y, "best_year_diff": float(yr[best_y]),
        "excl_best": float(yr.drop(best_y).sum()),
        "excl_2022": float(yr.drop(2022).sum()) if 2022 in yr.index else total,
        "neg_years_pool": int((ynew < 0).sum()),
        "yearly_diff": yr,
    }


# --- 口座レベル評価(exp41 踏襲・キャッシュ永続) -------------------------
class CfgEval:
    def __init__(self, label: str, pool: pd.DataFrame, closes: pd.DataFrame):
        self.label = label
        self.pool, self.closes = pool, closes
        self.mk = champion_sizing(pool, max_pos=MAX_POS)
        self._cache: dict[float, tuple] = {}

    def _sim(self, k: float):
        kk = round(float(k), 10)
        if kk not in self._cache:
            self._cache[kk] = mm.simulate(self.pool, self.closes, self.mk(kk), max_pos=MAX_POS)
        return self._cache[kk]

    def eq_of_k(self, k):
        return self._sim(k)[0]

    def info_of_k(self, k):
        return self._sim(k)[2]


def make_eq_fn(pool, closes, mk):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    return eq_of_k


def evaluate(cfg: CfgEval, seeds) -> dict:
    """empirical + robust(seeds) + 年次 + IS較正(emp/robust)→OOS素検証 + skip率。"""
    res = protocol_eval(cfg.eq_of_k, label=cfg.label, seeds=seeds)
    eq_emp = cfg.eq_of_k(res["emp_k"])
    yr_emp = yearly_returns(eq_emp)
    res["worst_year"] = float(yr_emp.min())
    res["neg_years_emp"] = int((yr_emp < 0).sum())
    res["yr_emp"] = {int(y): float(v) for y, v in yr_emp.items()}
    info_e = cfg.info_of_k(res["emp_k"])
    res["skip_emp"] = info_e["skipped"] / (info_e["skipped"] + info_e["n_taken"])
    # robust seed0 較正の年次(2022除外チェック用)+ skip率
    k_r0 = res["rob"][seeds[0]]["k"]
    yr_r0 = yearly_returns(cfg.eq_of_k(k_r0))
    res["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())
    info_r = cfg.info_of_k(k_r0)
    res["skip_rob0"] = info_r["skipped"] / (info_r["skipped"] + info_r["n_taken"])

    # IS(<2022) 較正 → OOS 素検証
    is_pool = cfg.pool[cfg.pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = cfg.pool[cfg.pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = cfg.closes[cfg.closes.index < OOS_START]
    oos_cl = cfg.closes[cfg.closes.index >= OOS_START]
    eq_is_of_k = make_eq_fn(is_pool, is_cl, cfg.mk)
    eq_oos_of_k = make_eq_fn(oos_pool, oos_cl, cfg.mk)

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
          f"OOS {res['oos_rob_cagr']:+7.2%} (DD {res['oos_rob_dd']:+5.1%}) | "
          f"skip emp/rob0 {res['skip_emp']:.1%}/{res['skip_rob0']:.1%}")
    return res


def year_diff_audit(tag: str, yc: dict, yb: dict) -> dict:
    """年次差分(候補−ベース)の単年依存監査。"""
    d = (pd.Series(yc) - pd.Series(yb)).dropna()
    total = float(d.sum())
    best_y = int(d.idxmax())
    excl_best = float(d.drop(best_y).sum())
    excl_2022 = float(d.drop(2022).sum()) if 2022 in d.index else total
    keep_share = excl_best / total if total > 0 else np.nan
    return {"basis": tag, "total": total, "best_year": best_y,
            "best_year_diff": float(d[best_y]), "excl_best": excl_best,
            "excl_2022": excl_2022, "keep_share_excl_best": keep_share,
            "yearly": {int(y): float(v) for y, v in d.items()}}


def ext_main() -> int:
    """追補 --ext: d=4,5,6 の断面延長(seed0)— 口座レベル用量反応が端で伸び続けるか。"""
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = mm.build_pool()
    closes = mm.load_closes()
    rc = reconstruct(pool)
    print(f"=== exp47 --ext: d=4,5,6 断面延長 (seed0) ===")
    ret0 = pool["ret"].to_numpy()
    rows = []
    for dly in (4, 5, 6):
        mod, kept, ret_new, _ = delayed_pool(pool, rc, dly)
        diff = float(np.where(kept, ret_new - ret0, -ret0).sum())
        cfg = CfgEval(f"delay_d{dly}", mod, closes)
        r = evaluate(cfg, seeds=(0,))
        rows.append({"dly": dly, "n": len(mod), "dropped": int((~kept).sum()),
                     "pool_diff": diff, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
                     "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"],
                     "is_rob_cagr": r["is_rob_cagr"], "oos_rob_cagr": r["oos_rob_cagr"],
                     "oos_rob_dd": r["oos_rob_dd"], "oos_emp_dd": r["oos_emp_dd"]})
        print(f"  d={dly}: n={len(mod)} (消滅{rows[-1]['dropped']}) pool_diff={diff:+.4f}  "
              f"[{time.time()-t0:.0f}s]")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp47_ext.csv", index=False)
    print("\n" + df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nsaved -> {OUT_DIR / 'exp47_ext.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


def main() -> int:
    if "--ext" in sys.argv:
        return ext_main()
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"=== exp47: エントリー d バー遅延 d={DELAYS} (H4, 19銘柄, mp{MAX_POS}, P=4.0) ===")
    print(f"pool {len(pool)} trades  sum(ret)={pool['ret'].sum():+.4f} (基準 {BASE_NET:+.4f})  "
          f"grid {len(closes)} bars")

    sec("0. 価格再構成(anatomy_cost 方式)の検算")
    rc = reconstruct(pool)
    net = (rc["gross"] - rc["cost"]).sum()
    ok = abs(net - pool["ret"].sum()) < 1e-9 and abs(net - BASE_NET) < 1e-3
    print(f"sum(gross)={rc['gross'].sum():+.4f}  sum(cost)={rc['cost'].sum():+.4f}  "
          f"sum(gross-cost)={net:+.6f}  (=sum(ret) 1e-9一致 かつ ≈{BASE_NET}): {ok}")
    d_arr = pool["dir"].to_numpy().astype(float)
    half_sp = d_arr * (rc["slip"] - 1.0)
    print(f"片道スリッページ比率: median {np.median(half_sp)*1e4:+.2f}bps "
          f"(全件 dir×(slip−1)>0: {(half_sp > 0).all()})")
    if not ok:
        print("!! ベースライン再構成失敗。以降の比較は無効。")
        return 1

    sec("1. プールレベル断面(遅延 d=1,2,3)")
    pools, prows, extras = {}, [], {}
    for dly in DELAYS:
        mod, kept, ret_new, extra = delayed_pool(pool, rc, dly)
        pools[dly] = mod
        extras[dly] = {"kept": kept, "ret_new": ret_new, **extra}
        cs = pool_cross_section(pool, mod, kept, ret_new, dly)
        prows.append(cs)
        print(f"  d={dly}: n={cs['n']} (消滅{cs['dropped']}件, 失うret {cs['dropped_ret_sum']:+.4f})"
              f"  sum={cs['sum_ret']:+.4f}  diff={cs['diff']:+.4f} ({cs['diff_pct']:+.2f}%)"
              f"  PF={cs['pf']:.3f}")
        print(f"        IS差 {cs['diff_is']:+.4f} / OOS差 {cs['diff_oos']:+.4f} | "
              f"long {cs['diff_long']:+.4f} / short {cs['diff_short']:+.4f} | "
              f"最良年 {cs['best_year']}({cs['best_year_diff']:+.4f}) 除外後 {cs['excl_best']:+.4f} "
              f"/ 2022除外 {cs['excl_2022']:+.4f} / 新プール負け年 {cs['neg_years_pool']}")
        print("        年次差分: " + "  ".join(f"{int(y)}:{v:+.3f}" for y, v in
                                               cs["yearly_diff"].items() if abs(v) > 5e-4))
    pdf = pd.DataFrame([{k: v for k, v in r.items() if k != "yearly_diff"} for r in prows])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(OUT_POOL, index=False)

    if "--pool-only" in sys.argv:
        print("\n--pool-only: 口座レベルはスキップ")
        return 0

    # --- 2. 口座レベル ステージ1(seed0 全候補 + IS/OOS) -------------------
    sec(f"2. 口座シミュ ステージ1: seed0 全候補 (empirical + robust s0 + IS/OOS)")
    cfgs = {0: CfgEval("base_d0", pool, closes)}
    for dly in DELAYS:
        cfgs[dly] = CfgEval(f"delay_d{dly}", pools[dly], closes)
    results = {}
    for dly, cfg in cfgs.items():
        results[dly] = evaluate(cfg, seeds=(0,))
        print(f"    [{time.time()-t0:.0f}s]")

    # --- 3. ステージ2: 最良候補 + ベースを seeds 0-4(他候補も安価なので併走) --
    best_s0 = max(DELAYS, key=lambda x: results[x]["rob"][0]["cagr"])
    sec(f"3. ステージ2: seeds {SEEDS_FULL} フル評価 (seed0最良 d={best_s0}; "
        f"計算が安価なため全候補+ベースを併走)")
    for dly, cfg in cfgs.items():
        res = evaluate(cfg, seeds=SEEDS_FULL)
        results[dly] = res
        print(f"    [{time.time()-t0:.0f}s]")

    base = results[0]
    print("\n--- ペアシード robust(p95=20%) CAGR 比較(同一シード集合) ---")
    print("  seed |  base(d0) | " + " | ".join(f"d{x} (diff)" for x in DELAYS))
    for sd in SEEDS_FULL:
        cells = [f"{results[x]['rob'][sd]['cagr']:+.2%} "
                 f"({(results[x]['rob'][sd]['cagr'] - base['rob'][sd]['cagr'])*100:+.2f}pp)"
                 for x in DELAYS]
        print(f"   s{sd}  | {base['rob'][sd]['cagr']:+.2%}  | " + " | ".join(cells))
    print(f"  mean | {base['rob_cagr_mean']:+.2%}  | " + " | ".join(
        f"{results[x]['rob_cagr_mean']:+.2%} "
        f"({(results[x]['rob_cagr_mean']-base['rob_cagr_mean'])*100:+.2f}pp)" for x in DELAYS))
    print(f"  empirical 20%: base {base['emp_cagr']:+.2%} (p95 {base['emp_p95']:+.1%}) | " +
          " | ".join(f"d{x} {results[x]['emp_cagr']:+.2%} (p95 {results[x]['emp_p95']:+.1%})"
                     for x in DELAYS))

    # --- 4. IS-argmax 監査 ----------------------------------------------------
    sec("4. IS-argmax 監査(IS<2022 だけ見た研究者は d をいくつ選ぶか)")
    print("   d   IS_rob_cagr  IS_emp_cagr   OOS_rob   OOS_emp")
    for dly in [0] + DELAYS:
        r = results[dly]
        print(f"  {dly:>2}  {r['is_rob_cagr']:+10.2%}  {r['is_emp_cagr']:+10.2%}  "
              f"{r['oos_rob_cagr']:+8.2%}  {r['oos_emp_cagr']:+8.2%}")
    arg_rob = max([0] + DELAYS, key=lambda x: results[x]["is_rob_cagr"])
    arg_emp = max([0] + DELAYS, key=lambda x: results[x]["is_emp_cagr"])
    print(f"  IS-argmax: robust基準 d={arg_rob} / empirical基準 d={arg_emp}")
    print(f"  -> IS選択(robust) d={arg_rob} の OOS: rob較正 {results[arg_rob]['oos_rob_cagr']:+.2%} "
          f"/ emp較正 {results[arg_rob]['oos_emp_cagr']:+.2%} "
          f"(baseline d0 OOS: {results[0]['oos_rob_cagr']:+.2%} / {results[0]['oos_emp_cagr']:+.2%})")

    # --- 5. 年次差分・最良年除外・2022除外・全年プラス・署名 -----------------
    sec("5. 年次差分監査(口座+プール, 候補−ベース)+ 全年プラス + レバ偽装署名")
    pmap = {r["dly"]: r for r in prows}
    audits, gates = {}, {}
    for dly in DELAYS:
        r = results[dly]
        a_emp = year_diff_audit("emp", r["yr_emp"], base["yr_emp"])
        a_rob = year_diff_audit("rob0", r["yr_rob0"], base["yr_rob0"])
        audits[dly] = {"emp": a_emp, "rob0": a_rob}
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        rob_gain = r["rob_cagr_mean"] - base["rob_cagr_mean"]
        rob_gain_rel = rob_gain / base["rob_cagr_mean"]
        ps = pmap[dly]
        g5_pool = (ps["diff"] > 0 and ps["excl_best"] >= 0.5 * ps["diff"]
                   and ps["excl_2022"] > 0)
        g = {
            "g1_rob_gain_rel": rob_gain_rel,
            "g1_adopt_line": rob_gain_rel >= 0.10,
            "g1_partial": r["rob_cagr_mean"] >= 0.172,
            "g2_no_signature": not sig,
            "g3_oos_rob": r["oos_rob_cagr"] > base["oos_rob_cagr"],
            "g3_oos_emp": r["oos_emp_cagr"] > base["oos_emp_cagr"],
            "g4_all_years_pos": (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0),
            "g5_emp": (a_emp["total"] > 0 and a_emp["keep_share_excl_best"] >= 0.5
                       and a_emp["excl_2022"] > 0),
            "g5_rob0": (a_rob["total"] > 0 and a_rob["keep_share_excl_best"] >= 0.5
                        and a_rob["excl_2022"] > 0),
            # プール段の年次差分(レバ上振れで年次が均される口座段より判別力が高い。
            # スカウトの単年依存署名はここで測られた)
            "g5_pool": g5_pool,
            "g6_skip_emp": r["skip_emp"], "g6_skip_rob0": r["skip_rob0"],
        }
        g["g3_oos"] = g["g3_oos_rob"] and g["g3_oos_emp"]
        g["g5_single_year"] = g["g5_emp"] and g["g5_rob0"] and g["g5_pool"]
        g["all_pass"] = (g["g1_adopt_line"] and g["g2_no_signature"] and g["g3_oos"]
                         and g["g4_all_years_pos"] and g["g5_single_year"])
        gates[dly] = g
        print(f"\n  [d={dly}] rob_mean {r['rob_cagr_mean']:+.2%} (base {base['rob_cagr_mean']:+.2%}, "
              f"{rob_gain*100:+.2f}pp, 相対 {rob_gain_rel:+.1%})")
        for tag, a in [("emp較正", a_emp), ("rob_s0較正", a_rob)]:
            print(f"    {tag}: 年次差分合計 {a['total']:+.2%} / 最良年 {a['best_year']}"
                  f"({a['best_year_diff']:+.2%}) 除外後 {a['excl_best']:+.2%} "
                  f"(残存率 {a['keep_share_excl_best']:.0%}) / 2022除外 {a['excl_2022']:+.2%}")
        print(f"    負け年: emp {r['neg_years_emp']} (worst {r['worst_year']:+.1%}) / "
              f"rob_s0 {r['neg_years_rob0']}   (base: {base['neg_years_emp']}/{base['neg_years_rob0']})")
        print(f"    レバ偽装署名: emp {r['emp_cagr']:+.2%} vs base {base['emp_cagr']:+.2%}, "
              f"p95 {r['emp_p95']:+.1%} vs base {base['emp_p95']:+.1%} -> "
              f"{'あり=reject' if sig else 'なし'}")
        print(f"    プール段G5: 利得 {ps['diff']:+.4f} / 最良年除外後 {ps['excl_best']:+.4f} "
              f"(残存率 {ps['excl_best']/ps['diff']*100 if ps['diff'] else 0:.0f}%) / "
              f"2022除外 {ps['excl_2022']:+.4f} -> {'pass' if g5_pool else 'FAIL'}")
        print(f"    ゲート: G1adopt={g['g1_adopt_line']} G2署名なし={g['g2_no_signature']} "
              f"G3oos={g['g3_oos']} (rob {g['g3_oos_rob']}/emp {g['g3_oos_emp']}) "
              f"G4全年+={g['g4_all_years_pos']} G5単年={g['g5_single_year']} "
              f"(emp {g['g5_emp']}/rob0 {g['g5_rob0']}/pool {g['g5_pool']}) -> 総合 "
              f"{'PASS' if g['all_pass'] else 'FAIL'}")

    # --- 6. レバ偽装署名のブートシード監査(p95 はシード依存 ±0.4-0.8pp) ----
    sec("6. 署名のブートシード監査(各 emp_k 固定, n_boot=1500, seeds 0-2)")
    from tail_protocol import boot_dd  # noqa: PLC0415
    sig_audit = {}
    p95_base = {sd: boot_dd(cfgs[0].eq_of_k(results[0]["emp_k"]), n_boot=1500, seed=sd)["p95"]
                for sd in (0, 1, 2)}
    print("  base_d0: " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95_base.items()))
    for dly in DELAYS:
        eq = cfgs[dly].eq_of_k(results[dly]["emp_k"])
        p95s = {sd: boot_dd(eq, n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)}
        emp_up = results[dly]["emp_cagr"] > base["emp_cagr"]
        sigs = {sd: emp_up and (abs(p95s[sd]) > abs(p95_base[sd]) + 0.005) for sd in p95s}
        n_sig = sum(sigs.values())
        sig_audit[dly] = {"p95": p95s, "sig": sigs, "n_sig": n_sig}
        print(f"  d={dly}:    " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95s.items()) +
              f"   署名(>0.5pp悪化): {n_sig}/3 シード " +
              " ".join(f"s{sd}:{'X' if v else '-'}" for sd, v in sigs.items()))

    # --- 7. ロールオーバーBIDアーティファクト監査 ---------------------------
    sec("7. ロールオーバー監査: 遅延後エントリーバーの時刻別利得 + h20除染変種")
    ret0 = pool["ret"].to_numpy()
    y_exit = pool["exit"].dt.year.to_numpy()
    h20_audit = {}
    for dly in DELAYS:
        ex = extras[dly]
        hrs = ex["dts"].hour.to_numpy()
        diff_tr = np.where(ex["kept"], ex["ret_new"] - ret0, -ret0)
        tab = pd.DataFrame({"hour": hrs, "dir": pool["dir"].to_numpy(), "diff": diff_tr,
                            "year": y_exit})
        by_h = tab.groupby("hour")["diff"].sum()
        total = float(diff_tr.sum())
        h20 = float(by_h.get(20, 0.0))
        h20_long = float(tab.loc[(tab.hour == 20) & (tab["dir"] > 0), "diff"].sum())
        n_h20 = int((hrs == 20).sum())
        # d2 単年依存の中身: 最良年(プール)内の h20 寄与
        best_y = prows[dly - 1]["best_year"]
        h20_besty = float(tab.loc[(tab.hour == 20) & (tab.year == best_y), "diff"].sum())
        print(f"\n  [d={dly}] 利得合計 {total:+.4f} | 時刻別: " +
              "  ".join(f"{int(h)}:{v:+.4f}" for h, v in by_h.items()))
        print(f"    h20 バーでの利得 {h20:+.4f} ({h20/total*100 if total else 0:.0f}%) "
              f"うちロング {h20_long:+.4f} | h20エントリー {n_h20}件 | "
              f"最良年{best_y}内のh20寄与 {h20_besty:+.4f}")
        # 除染変種(h20 に乗ったら 1 本送る)
        mod_s, kept_s, ret_new_s, _ = delayed_pool(pool, rc, dly, skip_h20=True)
        diff_s = float(np.where(kept_s, ret_new_s - ret0, -ret0).sum())
        print(f"    除染変種(d={dly}+h20スキップ): n={len(mod_s)} プール差 {diff_s:+.4f} "
              f"(無除染 {total:+.4f}, 変化 {diff_s-total:+.4f})")
        h20_audit[dly] = {"total": total, "h20": h20, "h20_long": h20_long,
                          "h20_share": h20 / total if total else np.nan, "n_h20": n_h20,
                          "h20_best_year": h20_besty, "decon_pool_diff": diff_s,
                          "by_hour": {int(h): float(v) for h, v in by_h.items()},
                          "decon_pool": mod_s}

    # 除染変種の口座レベル再判定(名目ゲートを通った候補のみ, seeds 0-4)
    decon_results = {}
    for dly in DELAYS:
        if not gates[dly]["all_pass"]:
            continue
        print(f"\n  --- 除染変種 d={dly}+h20スキップ の口座レベル(seeds {SEEDS_FULL}) ---")
        cfg_s = CfgEval(f"delay_d{dly}_decon", h20_audit[dly]["decon_pool"], closes)
        r = evaluate(cfg_s, seeds=SEEDS_FULL)
        decon_results[dly] = r
        gain = r["rob_cagr_mean"] - base["rob_cagr_mean"]
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        print(f"      rob_mean {r['rob_cagr_mean']:+.2%} (base比 {gain*100:+.2f}pp, "
              f"相対 {gain/base['rob_cagr_mean']:+.1%}) | emp {r['emp_cagr']:+.2%} "
              f"p95 {r['emp_p95']:+.1%} 署名 {'あり' if sig else 'なし'} | "
              f"負け年 emp {r['neg_years_emp']}/rob0 {r['neg_years_rob0']}")
        gates[dly]["g7_decon_rob_mean"] = r["rob_cagr_mean"]
        gates[dly]["g7_decon_gain_rel"] = gain / base["rob_cagr_mean"]
        gates[dly]["g7_decon_signature"] = bool(sig)
        gates[dly]["g7_decon_adopt_line"] = (gain / base["rob_cagr_mean"] >= 0.10) and not sig \
            and r["neg_years_emp"] == 0 and r["neg_years_rob0"] == 0

    # --- 8. 経験的最大DDエピソードの位置(単一事象依存の点検) --------------
    sec("8. 経験的最大DDエピソード(emp_k / rob_s0 k での peak→trough)")
    dd_eps = {}
    for dly in [0] + DELAYS:
        r = results[dly]
        row = {}
        for tag, k in [("emp", r["emp_k"]), ("rob0", r["rob"][0]["k"])]:
            eq = cfgs[dly].eq_of_k(k)
            dd = eq / eq.cummax() - 1.0
            t_tr = dd.idxmin()
            t_pk = eq.loc[:t_tr].idxmax()
            row[tag] = {"peak": str(t_pk.date()), "trough": str(t_tr.date()),
                        "dd": float(dd.min())}
        dd_eps[dly] = row
        print(f"  d={dly}: emp_k {row['emp']['dd']:+.1%} [{row['emp']['peak']} -> "
              f"{row['emp']['trough']}] | rob_s0 {row['rob0']['dd']:+.1%} "
              f"[{row['rob0']['peak']} -> {row['rob0']['trough']}]")

    # --- 保存 -------------------------------------------------------------
    rows = []
    for dly in [0] + DELAYS:
        r = results[dly]
        rows.append({
            "dly": dly,
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"], "worst_year": r["worst_year"],
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS_FULL},
            "rob_mean": r["rob_cagr_mean"],
            "skip_emp": r["skip_emp"], "skip_rob0": r["skip_rob0"],
            "k_is_emp": r["k_is_emp"], "is_emp_cagr": r["is_emp_cagr"],
            "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
            "k_is_rob": r["k_is_rob"], "is_rob_cagr": r["is_rob_cagr"],
            "oos_rob_cagr": r["oos_rob_cagr"], "oos_rob_dd": r["oos_rob_dd"],
            **({f"gate_{k}": v for k, v in gates[dly].items()} if dly in gates else {}),
        })
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT_ACC, index=False)
    payload = {
        "baseline": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                     for k, v in base.items()},
        "variants": {str(dly): {
            "pool": {k: ({int(y): float(x) for y, x in v.items()}
                         if k == "yearly_diff" else v) for k, v in prows[i].items()},
            "account": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                        for k, v in results[dly].items()},
            "year_audit": audits[dly],
            "gates": gates[dly],
        } for i, dly in enumerate(DELAYS)},
        "is_argmax": {"rob": arg_rob, "emp": arg_emp,
                      "oos_rob_of_argmax_rob": results[arg_rob]["oos_rob_cagr"],
                      "oos_emp_of_argmax_rob": results[arg_rob]["oos_emp_cagr"]},
        "best_seed0": best_s0,
        "sig_audit": {str(d): {"p95": {str(s): v for s, v in a["p95"].items()},
                               "sig": {str(s): bool(v) for s, v in a["sig"].items()},
                               "n_sig": a["n_sig"]} for d, a in sig_audit.items()},
        "h20_audit": {str(d): {k: v for k, v in a.items() if k != "decon_pool"}
                      for d, a in h20_audit.items()},
        "decon_account": {str(d): {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                                   for k, v in r.items()} for d, r in decon_results.items()},
        "dd_episodes": {str(d): v for d, v in dd_eps.items()},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_POOL}\n      -> {OUT_ACC}\n      -> {OUT_JSON}")
    print("\n=== 口座レベル最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf[[c for c in adf.columns if not c.startswith("gate_")]].to_string(index=False))
    print(f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
