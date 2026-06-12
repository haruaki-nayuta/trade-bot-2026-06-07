"""exp60: ウォークフォワード ML 拒否器 — 第4ラウンド(アルゴリズム改善・FX限定・リスク契約固定)。

仮説: reports/08 はエントリーフィルタを**単一特徴量**で網羅し ER のみ生存と結論した。
特徴量の**相互作用**(例: 「深い z × 高ボラ × 直近持続トレンド」の組合せだけが危険)は
線形しきい値では表現できず、未検証の最後のエントリー層。トレード単位の因果特徴量から
期待リターンを学習し、予測最下位コホートを建玉拒否する。

規律(事前登録):
  - **完全因果 WF**: 年初 t に「exit が t より前に確定したトレード」のみで学習し、
    その年のエントリーに適用。最初の学習は 2019-01-01(burn-in 約3.5年・約330取引)。
    2015-2018 のトレードは無加工(拒否なし)。
  - 特徴量はシグナルバー確定値のみ 12 個(z深さ/slow_z整合/ER/RSI/ボラ分位/ATR%/
    直近リターン×方向/傾き×方向/高安からの距離/時刻/曜日/クロスか否か)。
  - モデル2種固定: HistGradientBoostingRegressor(既定+max_depth=3, lr=0.05)/Ridge。
    ハイパーパラメータ探索はしない。
  - 拒否用量2種固定: 学習セット予測分布の下位 10% / 20% を閾値化。
  - 判定: プール段(拒否コホートの実現リターンが本当に悪いか=WF外挿の成否)→
    口座 seed0 → 生存者のみ seeds 0-4 + 6ゲート。
  - 露出した死角の確認: 拒否で取引数が減る分は k 較正が自然に吸収(同一リスク契約)。

実行: PYTHONPATH=. uv run python research/experiments/exp60_wf_ml_veto.py
出力: research/outputs/exp60_result.csv / exp60_result.json
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
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from fxlab import universe as uni  # noqa: E402
from strategies.confluence_meanrev_v2 import PARAMS  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
MAX_POS = 8
FIRST_TRAIN = pd.Timestamp("2019-01-01", tz="UTC")


def build_features(pool: pd.DataFrame) -> pd.DataFrame:
    """シグナルバー(=エントリーの1本前)確定値のみの因果特徴量。"""
    p = PARAMS
    feats = pd.DataFrame(index=pool.index)
    feats["z_depth"] = pool["z_entry"].to_numpy()          # シグナル時 |z|(d1規約で既に因果)
    feats["vol_entry"] = pool["vol_entry"].to_numpy()
    dirs = pool["dir"].to_numpy().astype(float)
    feats["is_cross"] = (~pool["instr"].isin(
        ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"])).astype(float)
    feats["is_jpy"] = pool["instr"].str.endswith("JPY").astype(float)

    cols = {c: np.full(len(pool), np.nan) for c in
            ["slow_z_dir", "er40", "rsi_dist", "vol_pctile", "atr_pct",
             "pre_ret_dir", "slope_dir", "dist_high", "dist_low"]}
    hour = np.full(len(pool), np.nan)
    dow = np.full(len(pool), np.nan)
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        sig_ts = pd.DatetimeIndex(g["entry"])  # 執行バー。シグナルバー値は shift(1) で参照
        zs = ((close - close.rolling(p["slow_win"]).mean())
              / close.rolling(p["slow_win"]).std()).shift(1)
        er_dir = (close - close.shift(p["er_win"])).abs()
        er_vol = close.diff().abs().rolling(p["er_win"]).sum()
        er = (er_dir / er_vol).replace([np.inf, -np.inf], np.nan).shift(1)
        delta = close.diff()
        up = delta.clip(lower=0).ewm(alpha=1 / p["rsi_p"], adjust=False).mean()
        dn = (-delta.clip(upper=0)).ewm(alpha=1 / p["rsi_p"], adjust=False).mean()
        rsi = (100 - 100 / (1 + up / dn)).shift(1)
        vol = close.pct_change().rolling(20).std()
        vpc = vol.rolling(p["vol_win"]).rank(pct=True).shift(1)
        tr = pd.concat([(d["high"] - d["low"]), (d["high"] - close.shift()).abs(),
                        (d["low"] - close.shift()).abs()], axis=1).max(axis=1)
        atrp = (tr.rolling(14).mean() / close * 100).shift(1)
        ret10 = close.pct_change(10).shift(1)
        x = np.arange(10)
        xm = x.mean()
        den = ((x - xm) ** 2).sum()
        slope = close.rolling(10).apply(
            lambda a: float(((x - xm) * (a / a[0] - (a / a[0]).mean())).sum() / den * 100),
            raw=True).shift(1)
        hi = close.rolling(50).max().shift(1)
        lo = close.rolling(50).min().shift(1)
        c1 = close.shift(1)
        rows = g.index.to_numpy()
        gd = g["dir"].to_numpy().astype(float)
        cols["slow_z_dir"][rows] = -gd * zs.reindex(sig_ts).to_numpy()   # ロングなら-zs(深いほど+)
        cols["er40"][rows] = er.reindex(sig_ts).to_numpy()
        rsiv = rsi.reindex(sig_ts).to_numpy()
        cols["rsi_dist"][rows] = np.where(gd > 0, p["rsi_low"] - rsiv, rsiv - p["rsi_high"])
        cols["vol_pctile"][rows] = vpc.reindex(sig_ts).to_numpy()
        cols["atr_pct"][rows] = atrp.reindex(sig_ts).to_numpy()
        cols["pre_ret_dir"][rows] = -gd * ret10.reindex(sig_ts).to_numpy()  # 逆行の深さ
        cols["slope_dir"][rows] = -gd * slope.reindex(sig_ts).to_numpy()
        c1v = c1.reindex(sig_ts).to_numpy()
        cols["dist_high"][rows] = (hi.reindex(sig_ts).to_numpy() - c1v) / c1v * 100
        cols["dist_low"][rows] = (c1v - lo.reindex(sig_ts).to_numpy()) / c1v * 100
        hour[rows] = sig_ts.hour.to_numpy()
        dow[rows] = sig_ts.dayofweek.to_numpy()
    for c, v in cols.items():
        feats[c] = v
    feats["hour"] = hour
    feats["dow"] = dow
    return feats


def wf_veto_mask(pool, feats, model_name, q):
    """年次 WF: 各年のエントリーを「その年初までに exit 確定したトレード」で学習したモデルで
    スコア化し、学習セット予測分布の下位 q 分位未満を拒否(True=拒否)。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = feats.to_numpy(dtype=float)
    y = pool["ret"].to_numpy()
    entry = pool["entry"]
    exit_ = pool["exit"]
    veto = np.zeros(len(pool), dtype=bool)
    years = range(2019, int(entry.dt.year.max()) + 1)
    n_train_log = {}
    for yy in years:
        t_start = pd.Timestamp(f"{yy}-01-01", tz="UTC")
        t_end = pd.Timestamp(f"{yy + 1}-01-01", tz="UTC")
        tr_mask = (exit_ < t_start).to_numpy()
        ap_mask = ((entry >= t_start) & (entry < t_end)).to_numpy()
        if tr_mask.sum() < 100 or ap_mask.sum() == 0:
            continue
        if model_name == "gbr":
            mdl = make_pipeline(SimpleImputer(strategy="median"),
                                HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                              random_state=0))
        else:
            mdl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
        mdl.fit(X[tr_mask], y[tr_mask])
        pred_tr = mdl.predict(X[tr_mask])
        th = np.quantile(pred_tr, q)
        pred_ap = mdl.predict(X[ap_mask])
        veto[np.where(ap_mask)[0][pred_ap < th]] = True
        n_train_log[yy] = int(tr_mask.sum())
    return veto, n_train_log


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
    k_r0 = r["rob"][seeds[0]]["k"]
    r["neg_years_rob0"] = int((yearly_returns(eq_of_k(k_r0)) < 0).sum())
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
    k_ie = calibrate_empirical(fi, 0.20)
    r["oos_emp_cagr"] = cagr_of(fo(k_ie))
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1()
    closes = mm.load_closes()
    print(f"=== exp60: WF-ML 拒否器 (d1 pool n={len(pool)}, burn-in 2015-2018) ===")
    feats = build_features(pool)
    nan_share = feats.isna().mean()
    print("特徴量 NaN率>5%: " +
          (", ".join(f"{c}:{v:.0%}" for c, v in nan_share[nan_share > 0.05].items()) or "なし"))
    print(f"特徴量 {feats.shape[1]} 列 / 対象トレード(2019+) "
          f"{int((pool['entry'] >= FIRST_TRAIN).sum())} 件  [{time.time()-t0:.0f}s]")

    configs = [(m, q) for m in ("gbr", "ridge") for q in (0.10, 0.20)]
    rows, vetoes = [], {}
    for m, q in configs:
        tag = f"{m}_q{int(q * 100)}"
        veto, ntr = wf_veto_mask(pool, feats, m, q)
        vetoes[tag] = veto
        scope = (pool["entry"] >= FIRST_TRAIN).to_numpy()
        v_in = veto & scope
        ret_veto = pool.loc[v_in, "ret"]
        ret_kept = pool.loc[scope & ~veto, "ret"]
        yr_v = pool.loc[v_in].groupby(pool.loc[v_in, "exit"].dt.year)["ret"].sum()
        row = {"cfg": tag, "n_veto": int(v_in.sum()),
               "veto_rate": float(v_in.sum() / scope.sum()),
               "veto_mean_ret_bps": float(ret_veto.mean() * 1e4) if v_in.any() else np.nan,
               "veto_sum_ret": float(ret_veto.sum()),
               "kept_mean_ret_bps": float(ret_kept.mean() * 1e4),
               "veto_win": float((ret_veto > 0).mean()) if v_in.any() else np.nan}
        rows.append(row)
        print(f"  [{tag}] 拒否 {row['n_veto']}件({row['veto_rate']:.1%}) "
              f"拒否コホート mean={row['veto_mean_ret_bps']:+.1f}bps "
              f"sum={row['veto_sum_ret']:+.4f} win={row['veto_win']:.1%} | "
              f"採用側 mean={row['kept_mean_ret_bps']:+.1f}bps")
        if v_in.any():
            print("      拒否コホート年次: " +
                  "  ".join(f"{int(y)}:{v:+.3f}" for y, v in yr_v.items()))

    # プール段判定: 拒否コホートの sum_ret < 0 (=本当に悪いトレードを外せている)もののみ口座へ
    pdf = pd.DataFrame(rows)
    cand = [r["cfg"] for r in rows if r["veto_sum_ret"] < 0]
    print(f"\nプール段生存(拒否コホートが純損失): {cand or 'なし'}")

    results = {"base": account_eval("base", pool, closes, seeds=(0,))}
    print(f"    [{time.time()-t0:.0f}s]")
    for tag in cand:
        mod = pool[~vetoes[tag]].reset_index(drop=True)
        results[tag] = account_eval(tag, mod, closes, seeds=(0,))
        print(f"    [{time.time()-t0:.0f}s]")
    base_s0 = results["base"]["rob"][0]["cagr"]
    finalists = [t for t in cand if results[t]["rob"][0]["cagr"] > base_s0]
    print(f"\nseed0 で base({base_s0:+.2%}) 超え: {finalists or 'なし'}")
    for tag in (["base"] + finalists if finalists else []):
        pl = pool if tag == "base" else pool[~vetoes[tag]].reset_index(drop=True)
        results[tag] = account_eval(tag, pl, closes, seeds=SEEDS)
        print(f"    [{time.time()-t0:.0f}s]")

    out_rows = []
    base = results["base"]
    for tag, r in results.items():
        full = len(r["rob"]) == len(SEEDS)
        row = {"cfg": tag, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"],
               "rob_mean": r["rob_cagr_mean"] if full else np.nan,
               "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
               "oos_emp": r["oos_emp_cagr"], "worst_year": r["worst_year"]}
        if tag != "base" and full:
            per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
            row["gain_pp"] = (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100
            row["all_seeds_pos"] = all(v > 0 for v in per_seed.values())
            sig = (r["emp_cagr"] > base["emp_cagr"]) and \
                  (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
            row["signature"] = bool(sig)
            row["g3_oos"] = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
                            (r["oos_emp_cagr"] > base["oos_emp_cagr"])
        out_rows.append(row)
    adf = pd.DataFrame(out_rows)
    adf.to_csv(OUT_DIR / "exp60_result.csv", index=False)
    payload = {"pool_stage": rows,
               "results": {t: {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                               for k, v in r.items()} for t, r in results.items()}}
    (OUT_DIR / "exp60_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(adf.to_string(index=False))
    print(pdf.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp60_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
