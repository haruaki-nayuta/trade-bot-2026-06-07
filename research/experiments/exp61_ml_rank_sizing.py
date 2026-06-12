"""exp61: WF-ML ランクサイジング傾斜 — exp60 の副産物(序列は当たるが拒否は不可)の最終活用。

exp60 の発見: WF 予測の下位コホートも実現リターンは正(拒否=取引数削減は損)だが、
採用側 +17.8〜19.4bps vs 拒否側 +2.5〜7.0bps の**序列(ランキング能力)は実在**する。
→ トレードは全部取り、**配分だけ WF 予測で傾ける**(乖離連動 f(z) の直交補完)。
   z_depth は特徴量に含まれるため、傾斜の限界価値=「z 以外の特徴量の相互作用」。

規律(事前登録):
  - 形状2種固定(チューニング禁止):
      half : m=1.25(WF学習分布の中央値以上)/ 0.75(未満)
      quint: m=0.6+0.2×五分位(学習分布基準, 0.6/0.8/1.0/1.2/1.4)
  - モデル2種固定(exp60 と同一): gbr / ridge。計4構成+base。
  - 2015-2018(burn-in)は m=1。alloc = equity*(k/mp)*(f(z)/fbar)*(m/mbar)。
    mbar=プール平均で正規化(総量は k に線形=較正の単調性維持)。
  - 判定: 口座 seed0 → base 超えのみ seeds 0-4 + 6ゲート + 配分集中チェック。

実行: PYTHONPATH=. uv run python research/experiments/exp61_ml_rank_sizing.py
出力: research/outputs/exp61_result.csv / exp61_result.json
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

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1, _fz  # noqa: E402
from tail_protocol import (  # noqa: E402
    cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from exp47_entry_delay import year_diff_audit  # noqa: E402
from exp60_wf_ml_veto import build_features  # noqa: E402
from fxlab import universe as uni  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
MAX_POS = 8


def wf_multiplier(pool, feats, model_name, shape):
    """年次 WF 予測 → 形状固定の配分乗数 m(2015-2018 は 1.0)。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = feats.to_numpy(dtype=float)
    y = pool["ret"].to_numpy()
    entry, exit_ = pool["entry"], pool["exit"]
    m = np.ones(len(pool))
    for yy in range(2019, int(entry.dt.year.max()) + 1):
        t0 = pd.Timestamp(f"{yy}-01-01", tz="UTC")
        t1 = pd.Timestamp(f"{yy + 1}-01-01", tz="UTC")
        tr = (exit_ < t0).to_numpy()
        ap = ((entry >= t0) & (entry < t1)).to_numpy()
        if tr.sum() < 100 or ap.sum() == 0:
            continue
        if model_name == "gbr":
            mdl = make_pipeline(SimpleImputer(strategy="median"),
                                HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                              random_state=0))
        else:
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        mdl.fit(X[tr], y[tr])
        pred_tr = mdl.predict(X[tr])
        pred_ap = mdl.predict(X[ap])
        idx = np.where(ap)[0]
        if shape == "half":
            med = np.median(pred_tr)
            m[idx] = np.where(pred_ap >= med, 1.25, 0.75)
        else:  # quint
            qs = np.quantile(pred_tr, [0.2, 0.4, 0.6, 0.8])
            quint = np.searchsorted(qs, pred_ap)  # 0..4
            m[idx] = 0.6 + 0.2 * quint
    return m


def make_sizing_ml(pool, mvec, max_pos=MAX_POS):
    fz = np.array([_fz(z) for z in pool["z_entry"].to_numpy()])
    w = fz * mvec
    wbar = float(w.mean()) or 1.0
    wnorm = w / wbar
    # simulate は ctx 経由で z しか渡せないため、トレード順の重みを z_entry 列に埋め込む
    # 代わりに「pool のコピーで z_entry を重みそのものに置換し、f を恒等にする」方式を取る。
    pool2 = pool.copy()
    pool2["z_entry"] = wnorm

    def make(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * ctx["z"]
    return pool2, make


def account_eval(tag, pool, closes, mk, seeds):
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
    yr0 = yearly_returns(eq_of_k(k_r0))
    r["yr_rob0"] = {int(y): float(v) for y, v in yr0.items()}
    r["neg_years_rob0"] = int((yr0 < 0).sum())
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
    r["is_rob_cagr"], r["oos_rob_cagr"] = cagr_of(fi(k_ir)), cagr_of(fo(k_ir))
    r["oos_rob_dd"] = max_dd(fo(k_ir))
    k_ie = calibrate_empirical(fi, 0.20)
    r["oos_emp_cagr"] = cagr_of(fo(k_ie))
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1()
    closes = mm.load_closes()
    feats = build_features(pool)
    print(f"=== exp61: WF-ML ランクサイジング傾斜 (n={len(pool)}) ===")

    configs = [(mname, shape) for mname in ("gbr", "ridge") for shape in ("half", "quint")]
    pools_mk = {}
    # base: m=1(f(z) のみ)— make_sizing_ml の恒等化経路で同一実装にして公平比較
    pool_b, mk_b = make_sizing_ml(pool, np.ones(len(pool)))
    pools_mk["base"] = (pool_b, mk_b)
    for mname, shape in configs:
        tag = f"{mname}_{shape}"
        mvec = wf_multiplier(pool, feats, mname, shape)
        scope = (pool["entry"] >= pd.Timestamp("2019-01-01", tz="UTC")).to_numpy()
        # 傾斜と実現リターンの整合(プール段診断): 上位半分の mean ret
        hi = scope & (mvec > 1.0)
        lo = scope & (mvec < 1.0)
        print(f"  [{tag}] m>1: {hi.sum()}件 mean={pool.loc[hi,'ret'].mean()*1e4:+.1f}bps / "
              f"m<1: {lo.sum()}件 mean={pool.loc[lo,'ret'].mean()*1e4:+.1f}bps")
        pools_mk[tag] = make_sizing_ml(pool, mvec)

    print("\n--- 口座 seed0 スカウト ---")
    results = {}
    for tag, (pl, mk) in pools_mk.items():
        results[tag] = account_eval(tag, pl, closes, mk, seeds=(0,))
        print(f"    [{time.time()-t0:.0f}s]")
    base_s0 = results["base"]["rob"][0]["cagr"]
    finalists = [t for t in pools_mk if t != "base"
                 and results[t]["rob"][0]["cagr"] > base_s0]
    print(f"\nseed0 で base({base_s0:+.2%}) 超え: {finalists or 'なし'}")

    for tag in (["base"] + finalists if finalists else []):
        pl, mk = pools_mk[tag]
        results[tag] = account_eval(tag, pl, closes, mk, seeds=SEEDS)
        print(f"    [{time.time()-t0:.0f}s]")

    base = results["base"]
    rows = []
    for tag, r in results.items():
        full = len(r["rob"]) == len(SEEDS)
        row = {"cfg": tag, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"],
               "rob_mean": r["rob_cagr_mean"] if full else np.nan,
               "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
               "oos_emp": r["oos_emp_cagr"], "worst_year": r["worst_year"]}
        if tag != "base" and full:
            per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
            sig = (r["emp_cagr"] > base["emp_cagr"]) and \
                  (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
            row["gain_pp"] = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
            row["all_seeds_pos"] = all(v > 0 for v in per_seed.values())
            row["signature"] = bool(sig)
            row["g3_oos"] = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
                            (r["oos_emp_cagr"] > base["oos_emp_cagr"])
            a_emp = year_diff_audit("emp", r["yr_emp"], base["yr_emp"])
            row["g5_keep_emp"] = a_emp["keep_share_excl_best"]
            print(f"  [{tag}] gain {row['gain_pp']:+.2f}pp seeds " +
                  " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
                  f" 署名={'あり' if sig else 'なし'} OOS={'+' if row['g3_oos'] else 'x'}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp61_result.csv", index=False)
    payload = {"results": {t: {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                               for k, v in r.items() if k not in ("yr_emp", "yr_rob0")}
                           for t, r in results.items()}}
    (OUT_DIR / "exp61_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp61_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
