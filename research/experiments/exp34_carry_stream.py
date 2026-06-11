"""exp34: キャリー・スリーブの ens_lab 統合 — robust 較正での再評価(本命候補)。

背景(reports/10, exp22_altpremia_bleed の実測):
  キャリー(高金利ロング/低金利ショート, fxlab/carry.py の年次近似金利)はチャンピオンの
  失血窓(2022型USDトレンド)で稼ぐ「本物の保険」(in_bleed=+420/月, IS+285/OOS+570, 単体黒字)。
  当時の共有 max_pos 統合は empirical 較正 CAGR で僅差棄却(+21.0% < 単独+21.6%)だったが、
  p95 は -28.7%→-25.0% と 3.7pp 改善していた。当時は (a) robust 較正でテール改善をレバに
  変換する再較正が未実施 (b) ens_lab(ストリーム別建玉枠)が無くキャリーが champ の枠を食っていた。
  reports/11 の金tsmom統合は、まさに「テール起源が違うスリーブを小枠で足して robust 較正」で
  +15.4→+17.4% を達成した。同じ機構がキャリーで効くかを実測する。

設計:
  - キャリープール: mm.load_closes() の H4 grid 上、t=0 から hold 本ごとに
    carry_annual(instr, entry年)で順位付け → 上位 legs ロング/下位 legs ショート、hold 本保有。
    ret = dir*price_ret + dir*(carry_annual/100)*(days/365) − 2*half_spread。
    z_entry=2.2(fz=1 の中立), stream="carry"。
  - 統合: ens.simulate_streams, budgets={"champ":11, "carry":2*legs}(キャリーは常時 2*legs 脚)。
  - スイープ: legs∈{2,5} × hold∈{42,63} × w(計12構成 + champ単独ベースライン)。
  - 判定: 同一テール判定プロトコル(2段階)。
      stage1: empirical較正 + そのkの boot p95(n_boot=600, seed0) + robust seed0
      stage2: robust seed0 ≥ +15.6% の上位≤3構成のみ seeds 1,2 を追加 → mean3
    合格 = robust mean3 ≥ +16.6%(ベースライン mp12 +15.04% の +10% 相対)。
    「empirical CAGR↑ かつ empirical-k p95 が baseline(-29.4%)より有意悪化」はレバ偽装=却下。
  - OOS: IS(<2022)プール+ISグリッドで empirical 較正 → OOS グリッド素シミュ(exp30 と同手順)。
  - リーク監査(必須): carry_annual は暦年平均金利=年初時点では未確定情報を含む。
    最良構成は金利を1年ラグ(year-1 で順位付け・受払計算)した変種を必ず実測。

実行: PYTHONPATH=. uv run python research/experiments/exp34_carry_stream.py
出力: research/outputs/exp34_carry_stage1.csv / exp34_carry_final.json
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

import mm_lab as mm  # noqa: E402
import ens_lab as ens  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd, yearly_returns,
)
from fxlab import carry, config  # noqa: E402
from fxlab import universe as uni  # noqa: E402

pd.set_option("display.width", 240)

OOS = pd.Timestamp("2022-01-01", tz="UTC")
COLS = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]
PASS_STAGE1 = 0.156   # robust seed0 がこれ未満なら seeds 1,2 不要
PASS_MEAN3 = 0.166    # 合格ライン(robust mean3)
BASE_EMP_P95 = -0.294  # mp11 empirical-k の boot p95(レバ偽装チェックの基準)
K_HI = 40.0  # 較正上限: slots_total=21 では champ 同等サイズに k≈22 が要るため既定16では不足
OUT_DIR = ROOT / "research" / "outputs"


# --- キャリープール ------------------------------------------------------
def build_carry_pool(closes: pd.DataFrame, hold: int, legs: int, lag: int = 0) -> pd.DataFrame:
    """高金利ロング/低金利ショートのトレード表(ens_lab 契約)。

    lag=1 で「前年の金利」で順位付け・受払計算(リーク監査用の因果変種)。
    """
    names = list(closes.columns)
    mp = closes.mean()
    hs = {p: config.spread_pips(p) * config.pip_size(p) / 2.0 / mp[p] for p in names}
    idx = closes.index
    recs = []
    for t in range(0, len(closes) - hold, hold):
        ts_e, ts_x = idx[t], idx[t + hold]
        year = ts_e.year - lag
        c0, c1 = closes.iloc[t], closes.iloc[t + hold]
        valid = [p for p in names if np.isfinite(c0[p]) and np.isfinite(c1[p])]
        if len(valid) < 2 * legs:
            continue
        car = pd.Series({p: carry.carry_annual(p, year) for p in valid}).sort_values()
        days = (ts_x - ts_e).total_seconds() / 86400.0
        for p, d in [(p, -1) for p in car.index[:legs]] + [(p, +1) for p in car.index[-legs:]]:
            ret = (d * (c1[p] / c0[p] - 1.0)
                   + d * (carry.carry_annual(p, year) / 100.0) * (days / 365.0)
                   - 2 * hs[p])
            recs.append((p, ts_e, ts_x, d, float(c0[p]), float(ret)))
    df = pd.DataFrame(recs, columns=["instr", "entry", "exit", "dir", "entry_price", "ret"])
    df["z_entry"] = 2.2  # fz(2.2)=1.0 の中立
    df["stream"] = "carry"
    df["w"] = 1.0
    return df.sort_values("entry").reset_index(drop=True)


def carry_diag(cp: pd.DataFrame, label: str) -> dict:
    """単体診断: 年次ΣR・失血窓(2021-2023)寄与・IS/OOS。"""
    yearly = cp.groupby(pd.DatetimeIndex(cp["exit"]).year)["ret"].sum()
    is_r = cp.loc[cp["entry"] < OOS, "ret"].sum()
    oos_r = cp.loc[cp["entry"] >= OOS, "ret"].sum()
    bleed = float(yearly.reindex([2021, 2022, 2023]).fillna(0).sum())
    d = {"label": label, "n": len(cp), "sum_ret": float(cp["ret"].sum()),
         "is_ret": float(is_r), "oos_ret": float(oos_r), "bleed_2021_23": bleed,
         "yearly": {int(y): round(float(v), 4) for y, v in yearly.items()}}
    print(f"  [{label}] n={d['n']} ΣR={d['sum_ret']:+.3f} IS={d['is_ret']:+.3f} "
          f"OOS={d['oos_ret']:+.3f} 失血窓21-23={d['bleed_2021_23']:+.3f}")
    print(f"    年次: {d['yearly']}")
    return d


# --- プロトコル(2段階) -------------------------------------------------
def stage1(eq_of_k, label: str) -> dict:
    t0 = time.time()
    k_e = calibrate_empirical(eq_of_k, 0.20, hi=K_HI)
    eq_e = eq_of_k(k_e)
    bs = boot_dd(eq_e, n_boot=600, seed=0)
    k_r0 = calibrate_robust_seeded(eq_of_k, 0.20, n_boot=600, seed=0, hi=K_HI)
    eq_r0 = eq_of_k(k_r0)
    yr = yearly_returns(eq_r0)
    out = {"label": label, "emp_k": float(k_e), "emp_cagr": float(cagr_of(eq_e)),
           "emp_dd": float(max_dd(eq_e)), "emp_p95": float(bs["p95"]),
           "rob_k0": float(k_r0), "rob_cagr0": float(cagr_of(eq_r0)),
           "rob_worst_year": float(yr.min()), "rob_pos_year": float((yr > 0).mean())}
    print(f"  {label:28s} emp k={k_e:5.2f} CAGR={out['emp_cagr']:+7.2%} p95={out['emp_p95']:+6.1%}"
          f" | rob s0 k={k_r0:5.2f} CAGR={out['rob_cagr0']:+7.2%} worst_yr={out['rob_worst_year']:+5.1%}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    return out


def stage2(eq_of_k, res: dict) -> dict:
    cagrs = {0: res["rob_cagr0"]}
    for sd in (1, 2):
        k_r = calibrate_robust_seeded(eq_of_k, 0.20, n_boot=600, seed=sd, hi=K_HI)
        cagrs[sd] = float(cagr_of(eq_of_k(k_r)))
    res["rob_cagr_seeds"] = cagrs
    res["rob_cagr_mean3"] = float(np.mean(list(cagrs.values())))
    print(f"  {res['label']:28s} rob mean3={res['rob_cagr_mean3']:+7.2%} "
          f"(s0 {cagrs[0]:+.2%} s1 {cagrs[1]:+.2%} s2 {cagrs[2]:+.2%})", flush=True)
    return res


def oos_check(pool, closes, budgets, fbars, label: str) -> dict:
    """IS(<2022)プール+ISグリッドで empirical 較正 → OOS グリッド素シミュ(exp30 同手順)。"""
    is_pool = pool[pool["entry"] < OOS].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS].reset_index(drop=True)
    is_cl = closes[closes.index < OOS]
    oos_cl = closes[closes.index >= OOS]
    k_is = calibrate_empirical(
        lambda k: ens.simulate_streams(is_pool, is_cl, k, budgets, fbars=fbars)[0], 0.20, hi=K_HI)
    eq_o = ens.simulate_streams(oos_pool, oos_cl, k_is, budgets, fbars=fbars)[0]
    out = {"label": label, "k_is": float(k_is), "oos_cagr": float(cagr_of(eq_o)),
           "oos_dd": float(max_dd(eq_o))}
    print(f"  OOS {label:24s} k_is={k_is:5.2f} CAGR={out['oos_cagr']:+7.2%} DD={out['oos_dd']:+6.1%}",
          flush=True)
    return out


def main() -> int:
    t_start = time.time()
    uni.register_cross_spreads(3.0)
    champ = mm.build_pool().copy()
    champ["stream"] = "champ"
    champ["w"] = 1.0
    closes = mm.load_closes()
    fbars = ens.stream_fbars(champ)  # champ の f̄(carry は z=2.2 で f̄=1.0)
    fbars["carry"] = 1.0
    print(f"champ pool {len(champ)} trades / grid {len(closes)} bars "
          f"{closes.index[0].date()}..{closes.index[-1].date()}", flush=True)

    # --- 1. キャリープール単体診断 -------------------------------------
    print("\n=== 1. キャリープール単体(年次ΣR / 失血窓寄与) ===", flush=True)
    carry_pools: dict[tuple, pd.DataFrame] = {}
    diags = []
    for hold in (42, 63):
        for legs in (2, 5):
            cp = build_carry_pool(closes, hold, legs)
            carry_pools[(hold, legs)] = cp
            diags.append(carry_diag(cp, f"carry h{hold} legs{legs}"))

    # --- 2. stage1: empirical + robust seed0 ---------------------------
    print("\n=== 2. stage1(empirical較正 + boot p95@emp-k + robust seed0) ===", flush=True)
    results = []

    def make_eq_of_k(pool, budgets):
        return lambda kk: ens.simulate_streams(pool, closes, kk, budgets, fbars=fbars)[0]

    # ベースライン(champ 単独, ens 経由で同一土俵)
    base_eq = make_eq_of_k(champ[COLS], {"champ": 11})
    res_base = stage1(base_eq, "baseline mp11 (ens)")
    results.append(res_base)

    cfgs = []
    for hold in (42, 63):
        ws = (0.25, 0.5, 0.75, 1.0) if hold == 42 else (0.5, 1.0)
        for legs in (2, 5):
            for w in ws:
                cfgs.append((hold, legs, w))

    pools_by_label = {}
    for hold, legs, w in cfgs:
        cp = carry_pools[(hold, legs)].copy()
        cp["w"] = w
        pool = pd.concat([champ[COLS], cp[COLS]], ignore_index=True)
        pool = pool.sort_values("entry").reset_index(drop=True)
        budgets = {"champ": 11, "carry": 2 * legs}
        label = f"h{hold} legs{legs} w{w}"
        pools_by_label[label] = (pool, budgets)
        r = stage1(make_eq_of_k(pool, budgets), label)
        r.update({"hold": hold, "legs": legs, "w": w})
        results.append(r)
        pd.DataFrame(results).to_csv(OUT_DIR / "exp34_carry_stage1.csv", index=False)

    # --- 3. stage2: 上位≤3構成に seeds 1,2 -----------------------------
    cand = [r for r in results if r.get("hold") and r["rob_cagr0"] >= PASS_STAGE1]
    cand = sorted(cand, key=lambda r: -r["rob_cagr0"])[:3]
    print(f"\n=== 3. stage2(seeds 1,2)対象: {len(cand)} 構成 "
          f"(robust s0 ≥ {PASS_STAGE1:+.1%}) ===", flush=True)
    for r in cand:
        pool, budgets = pools_by_label[r["label"]]
        stage2(make_eq_of_k(pool, budgets), r)

    final = {"baseline": res_base, "stage1": results, "diags": diags}

    # 最良構成(stage2 があれば mean3、無ければ rob_cagr0 順)
    scored = [r for r in results if r.get("hold")]
    best = max(scored, key=lambda r: r.get("rob_cagr_mean3", r["rob_cagr0"]))
    print(f"\n最良構成: {best['label']}", flush=True)

    # --- 4. OOS(IS較正→OOS素シミュ)— ベースラインと最良構成 ---------
    print("\n=== 4. OOS 検証(IS<2022 empirical較正 → OOS素シミュ) ===", flush=True)
    final["oos_base"] = oos_check(champ[COLS], closes, {"champ": 11}, fbars, "baseline mp11")
    bpool, bbudgets = pools_by_label[best["label"]]
    final["oos_best"] = oos_check(bpool, closes, bbudgets, fbars, best["label"])

    # --- 5. リーク監査: 金利1年ラグ変種(最良構成) ---------------------
    print("\n=== 5. リーク監査: carry_annual を year-1 にラグ(最良構成) ===", flush=True)
    cp_lag = build_carry_pool(closes, int(best["hold"]), int(best["legs"]), lag=1)
    final["diag_lag"] = carry_diag(cp_lag, f"carry h{best['hold']} legs{best['legs']} LAG1")
    cp_lag["w"] = best["w"]
    pool_lag = pd.concat([champ[COLS], cp_lag[COLS]], ignore_index=True)
    pool_lag = pool_lag.sort_values("entry").reset_index(drop=True)
    eq_lag = make_eq_of_k(pool_lag, {"champ": 11, "carry": 2 * int(best["legs"])})
    r_lag = stage1(eq_lag, f"LAG1 {best['label']}")
    if r_lag["rob_cagr0"] >= PASS_STAGE1:
        stage2(eq_lag, r_lag)
    final["lag_variant"] = r_lag
    final["oos_lag"] = oos_check(pool_lag, closes, {"champ": 11, "carry": 2 * int(best["legs"])},
                                 fbars, f"LAG1 {best['label']}")

    # --- 6. 最良構成の年次リターン(robust seed0 較正) ------------------
    eq_best = make_eq_of_k(bpool, bbudgets)(best["rob_k0"])
    yr = yearly_returns(eq_best)
    final["best_yearly"] = {int(y): round(float(v), 4) for y, v in yr.items()}
    print("\n最良構成 年次リターン(robust s0 較正):")
    print((yr * 100).round(1).to_string(), flush=True)

    (OUT_DIR / "exp34_carry_final.json").write_text(json.dumps(final, indent=2, default=str))
    pd.DataFrame(results).to_csv(OUT_DIR / "exp34_carry_stage1.csv", index=False)
    print(f"\nsaved -> {OUT_DIR}/exp34_carry_stage1.csv, exp34_carry_final.json "
          f"({(time.time()-t_start)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
