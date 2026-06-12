"""exp53: d1(エントリー1バー遅延)の機構実在性と近傍頑健性の監査 — 懐疑者検証。

検証対象: exp47 で全6ゲート通過した d1(robust 5シード平均 +18.63% vs base +16.41%)。
主張される機構(reports/15): エントリー直後1〜3本は平均で逆行 → 1本待つと
 (a) 僅かに有利な価格(プール+2.8%) (b) 逆行第1波を MtM パスから外す → DD形状が浅くなり
 較正 k が上がる(利得の主因)。

攻め方: 「機構の収穫」なら近傍でも・固定kでも・別部分期間でも再現するはず。
「d=1 という1点のフィット」なら近傍で消える。
  1. OAT近傍10構成(window45/55, entry_z1.9/2.1, exit_z0.4/0.6, slow_z1.6/1.9,
     er_max0.50/0.60)で d1−d0(robust seed0 較正)を全構成実測
  2. 固定k(ベース emp較正k)で d0/d1 の CAGR・年次・MtM DD・p95 を較正抜きで比較
  3. per-trade MAE(H4 close 走行最悪乖離)分布の d0/d1 直接実測(浅瀬/深手コホート)
  4. 時代4分割(2016-18/2019-21/2022-23/2024-26)での固定k equity比の伸び
  5. emp較正kでの p95 をシード0-9でペア測定(レバ偽装署名の出現率)

シグナル再実装は exp41(参照プール 1e-6 一致検証済み)、遅延プール生成は exp47
(sum=+1.9086 検算済み)の関数をそのまま流用する。

実行: PYTHONPATH=. uv run python research/experiments/exp53_d1_mechanism.py
      (--oat / --fixedk / --mae / --era / --seeds で個別実行可。無指定=全部)
出力: research/outputs/exp53_oat.csv / exp53_fixedk.csv / exp53_fixedk_yearly.csv /
      exp53_mae_cohorts.csv / exp53_mae_bins.csv / exp53_era.csv / exp53_seeds.csv /
      exp53_result.json
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
sys.path.insert(0, str(ROOT / "research" / "experiments"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    calibrate_empirical,
    calibrate_robust_seeded,
    cagr_of,
    max_dd,
    yearly_returns,
)
from exp41_volwin import (  # noqa: E402  検証済みシグナル再実装
    BASE,
    IndicatorCache,
    build_mm_pool,
    make_eq_fn,
)
from exp47_entry_delay import delayed_pool, reconstruct  # noqa: E402  検証済み遅延生成
from fxlab import universe as uni  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

MAX_POS = 8
BASE_NET = 1.9086
SHALLOW = -0.005  # reports/15 の浅瀬境界(MAE -0.5%以内)
OUT_DIR = ROOT / "research" / "outputs"
ERAS = [
    ("2016-18", None, "2019-01-01"),
    ("2019-21", "2019-01-01", "2022-01-01"),
    ("2022-23", "2022-01-01", "2024-01-01"),
    ("2024-26", "2024-01-01", None),
]

# OAT 近傍構成(チャンピオン中核 5 軸 × 上下)
OAT = [
    ("window", 45), ("window", 55),
    ("entry_z", 1.9), ("entry_z", 2.1),
    ("exit_z", 0.4), ("exit_z", 0.6),
    ("slow_z", 1.6), ("slow_z", 1.9),
    ("er_max", 0.50), ("er_max", 0.60),
]


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def d1_of(pool: pd.DataFrame):
    """exp47 の遅延生成(d=1)。返り値: (mod, kept, ret_new, rc)。"""
    rc = reconstruct(pool)
    mod, kept, ret_new, _ = delayed_pool(pool, rc, 1)
    return mod, kept, ret_new, rc


# ---------------------------------------------------------------------------
# 1. パラメータ近傍 OAT
# ---------------------------------------------------------------------------
def run_oat(closes, t0) -> pd.DataFrame:
    sec("1. パラメータ近傍 OAT(robust seed0 較正での d1−d0)")
    instruments = mm.default_instruments()
    datas = {nm: uni.instrument_data(nm, "H4") for nm in instruments}
    caches = {nm: IndicatorCache(datas[nm]["close"]) for nm in instruments}

    rows = []
    configs = [("base", None)] + OAT
    for key, val in configs:
        p = dict(BASE)
        label = "base(champion)" if key == "base" else f"{key}={val}"
        if key != "base":
            p[key] = val
        pool_c = build_mm_pool(datas, caches, p)
        if key == "base":  # 検算: 参照プールと一致するか
            ref = mm.build_pool()
            ok = len(pool_c) == len(ref) and abs(pool_c["ret"].sum() - ref["ret"].sum()) < 1e-6
            print(f"  [検算] base 再生成: n={len(pool_c)} (ref {len(ref)}) "
                  f"sum={pool_c['ret'].sum():+.4f} (ref {ref['ret'].sum():+.4f}) 一致={ok}")
            if not ok:
                raise RuntimeError("base pool 再現失敗 — OAT 比較は無効")
        mod, kept, ret_new, _ = d1_of(pool_c)
        ret0 = pool_c["ret"].to_numpy()
        pool_diff = float(np.where(kept, ret_new - ret0, -ret0).sum())

        out = {"param": key, "value": val if key != "base" else np.nan, "label": label,
               "n_d0": len(pool_c), "sum_d0": float(ret0.sum()),
               "n_d1": len(mod), "dropped": int((~kept).sum()),
               "pool_diff": pool_diff,
               "pool_diff_rel": pool_diff / ret0.sum() if ret0.sum() else np.nan}
        for tag, pl in [("d0", pool_c), ("d1", mod)]:
            mk = champion_sizing(pl, max_pos=MAX_POS)
            eq_fn = make_eq_fn(pl, closes, mk)
            k_r = calibrate_robust_seeded(eq_fn, 0.20, seed=0)
            out[f"rob0_k_{tag}"] = k_r
            out[f"rob0_cagr_{tag}"] = cagr_of(eq_fn(k_r))
        out["diff_pp"] = (out["rob0_cagr_d1"] - out["rob0_cagr_d0"]) * 100
        rows.append(out)
        print(f"  {label:16s} n {out['n_d0']:>4}->{out['n_d1']:>4} "
              f"pool_diff {pool_diff:+.4f} | rob0 d0 {out['rob0_cagr_d0']:+.2%} "
              f"(k {out['rob0_k_d0']:.2f}) -> d1 {out['rob0_cagr_d1']:+.2%} "
              f"(k {out['rob0_k_d1']:.2f}) | diff {out['diff_pp']:+.2f}pp  "
              f"[{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp53_oat.csv", index=False)
    nb = df[df["param"] != "base"]
    n_pos = int((nb["diff_pp"] > 0).sum())
    print(f"\n  近傍10構成: d1−d0 正 {n_pos}/10 | 平均 {nb['diff_pp'].mean():+.2f}pp "
          f"| 中央値 {nb['diff_pp'].median():+.2f}pp | 最小 {nb['diff_pp'].min():+.2f}pp "
          f"| 最大 {nb['diff_pp'].max():+.2f}pp")
    print(f"  プール段: 正 {int((nb['pool_diff'] > 0).sum())}/10 "
          f"| 平均 {nb['pool_diff'].mean():+.4f}")
    return df


# ---------------------------------------------------------------------------
# 2. 固定k分析(較正上振れの希釈を排除)
# ---------------------------------------------------------------------------
def run_fixedk(pool, mod, closes, eq0_fn, eq1_fn, k_emp0, t0) -> dict:
    sec(f"2. 固定k分析(両者 k={k_emp0:.2f} = ベース emp較正k に固定)")
    eq0, eq1 = eq0_fn(k_emp0), eq1_fn(k_emp0)
    c0, c1 = cagr_of(eq0), cagr_of(eq1)
    dd0, dd1 = max_dd(eq0), max_dd(eq1)
    fbar0 = float(np.mean([np.clip((z / 2.2) ** 4.0, 0.3, 3.0)
                           for z in pool["z_entry"].to_numpy() if np.isfinite(z)]))
    fbar1 = float(np.mean([np.clip((z / 2.2) ** 4.0, 0.3, 3.0)
                           for z in mod["z_entry"].to_numpy() if np.isfinite(z)]))
    print(f"  CAGR : d0 {c0:+.2%} -> d1 {c1:+.2%} (diff {(c1-c0)*100:+.2f}pp)")
    print(f"  MtMDD: d0 {dd0:+.2%} -> d1 {dd1:+.2%} (diff {(dd1-dd0)*100:+.2f}pp; "
          f"正=浅くなった)")
    print(f"  fbar : d0 {fbar0:.4f} / d1 {fbar1:.4f} (サイズ正規化の差 "
          f"{(fbar0/fbar1-1)*100:+.2f}%)")

    p95s = {}
    for sd in (0, 1, 2):
        p0 = boot_dd(eq0, n_boot=1500, seed=sd)["p95"]
        p1 = boot_dd(eq1, n_boot=1500, seed=sd)["p95"]
        p95s[sd] = (p0, p1)
        print(f"  p95 s{sd}: d0 {p0:+.2%} -> d1 {p1:+.2%} "
              f"(diff {(p1-p0)*100:+.2f}pp; 正=浅くなった)")

    yr0, yr1 = yearly_returns(eq0), yearly_returns(eq1)
    yd = (yr1 - yr0).dropna()
    n_pos_years = int((yd > 0).sum())
    best_y = int(yd.idxmax())
    print(f"\n  年次差分(d1−d0, 固定k):")
    print("    " + "  ".join(f"{int(y)}:{v:+.2%}" for y, v in yd.items()))
    print(f"    合計 {yd.sum():+.2%} / 正の年 {n_pos_years}/{len(yd)} / "
          f"最良年 {best_y}({yd[best_y]:+.2%}) 除外後 {yd.drop(best_y).sum():+.2%} "
          f"(残存率 {yd.drop(best_y).sum()/yd.sum()*100 if yd.sum() else 0:.0f}%)")

    ydf = pd.DataFrame({"year": yd.index, "d0": yr0.reindex(yd.index).to_numpy(),
                        "d1": yr1.reindex(yd.index).to_numpy(), "diff": yd.to_numpy()})
    ydf.to_csv(OUT_DIR / "exp53_fixedk_yearly.csv", index=False)
    res = {"k_fixed": k_emp0, "cagr_d0": c0, "cagr_d1": c1, "diff_pp": (c1 - c0) * 100,
           "dd_d0": dd0, "dd_d1": dd1, "dd_diff_pp": (dd1 - dd0) * 100,
           "fbar_d0": fbar0, "fbar_d1": fbar1,
           **{f"p95_s{sd}_d0": v[0] for sd, v in p95s.items()},
           **{f"p95_s{sd}_d1": v[1] for sd, v in p95s.items()},
           "yd_sum": float(yd.sum()), "yd_pos_years": n_pos_years,
           "yd_n_years": len(yd), "yd_best_year": best_y,
           "yd_excl_best": float(yd.drop(best_y).sum())}
    pd.DataFrame([res]).to_csv(OUT_DIR / "exp53_fixedk.csv", index=False)
    print(f"  [{time.time()-t0:.0f}s]")
    return res


# ---------------------------------------------------------------------------
# 3. MAE/DD形状の直接実測
# ---------------------------------------------------------------------------
def run_mae(pool, kept, ret_new, rc, t0) -> dict:
    sec("3. per-trade MAE(H4 close 走行最悪乖離)の d0/d1 分布")
    n = len(pool)
    arr_by = {i: s.to_numpy() for i, s in rc["closes_by"].items()}
    dirs = pool["dir"].to_numpy().astype(float)
    instr = pool["instr"].to_numpy()
    i0s, i1s = rc["idx_e"], rc["idx_x"]
    mae0 = np.zeros(n)
    mae1 = np.full(n, np.nan)
    for i in range(n):
        a = arr_by[instr[i]]
        seg = a[i0s[i]:i1s[i] + 1]
        body = dirs[i] * (seg[1:] / seg[0] - 1.0)
        mae0[i] = min(body.min(), 0.0) if len(body) else 0.0
        if kept[i]:
            seg1 = a[i0s[i] + 1:i1s[i] + 1]
            body1 = dirs[i] * (seg1[1:] / seg1[0] - 1.0)
            mae1[i] = min(body1.min(), 0.0) if len(body1) else 0.0

    ret0 = pool["ret"].to_numpy()
    net0 = ret0.sum()
    r1 = np.where(kept, ret_new, np.nan)
    net1 = np.nansum(r1)
    diff_tr = np.where(kept, ret_new - ret0, -ret0)

    # --- 浅瀬/深手コホート(reports/15 再現 + d1 での変化) ---
    sh0 = mae0 >= SHALLOW
    rows = []
    print(f"  [d0] 浅瀬(MAE>={SHALLOW:.1%}) {sh0.sum()}件 ({sh0.mean():.1%}) "
          f"純益寄与 {ret0[sh0].sum()/net0*100:+.1f}% | "
          f"深手 {(~sh0).sum()}件 寄与 {ret0[~sh0].sum()/net0*100:+.1f}% "
          f"(reports/15: 浅瀬713件 +207.7% / 深手501件 -107.7%)")
    rows.append({"pool": "d0", "cohort": "shallow", "n": int(sh0.sum()),
                 "sum_ret": float(ret0[sh0].sum()),
                 "share_of_net": float(ret0[sh0].sum() / net0)})
    rows.append({"pool": "d0", "cohort": "deep", "n": int((~sh0).sum()),
                 "sum_ret": float(ret0[~sh0].sum()),
                 "share_of_net": float(ret0[~sh0].sum() / net0)})
    sh1 = mae1 >= SHALLOW  # nan -> False
    k1 = kept & np.isfinite(mae1)
    for tag, m in [("shallow", sh1 & k1), ("deep", (~sh1) & k1)]:
        rows.append({"pool": "d1", "cohort": tag, "n": int(m.sum()),
                     "sum_ret": float(np.nansum(np.where(m, ret_new, 0.0))),
                     "share_of_net": float(np.nansum(np.where(m, ret_new, 0.0)) / net1)})
    print(f"  [d1] 浅瀬 {int((sh1 & k1).sum())}件 ({(sh1 & k1).sum()/k1.sum():.1%} of kept) "
          f"純益寄与 {rows[2]['share_of_net']*100:+.1f}% | "
          f"深手 {int(((~sh1) & k1).sum())}件 寄与 {rows[3]['share_of_net']*100:+.1f}%")

    # 移行行列(kept のみ)
    mig = {}
    for a, b, lab in [(sh0, sh1, "sh->sh"), (sh0, ~sh1, "sh->dp"),
                      (~sh0, sh1, "dp->sh"), (~sh0, ~sh1, "dp->dp")]:
        m = a & b & k1
        mig[lab] = {"n": int(m.sum()), "diff_sum": float(diff_tr[m].sum())}
    print("  移行(kept): " + " | ".join(
        f"{lab} n={v['n']} diff {v['diff_sum']:+.4f}" for lab, v in mig.items()))

    # MAE 分布シフト
    q = [0.05, 0.25, 0.50, 0.75]
    q0 = np.quantile(mae0, q)
    q1 = np.nanquantile(mae1[k1], q)
    print(f"  MAE 分位 d0: " + " ".join(f"p{int(p*100)}:{v:+.3%}" for p, v in zip(q, q0)))
    print(f"  MAE 分位 d1: " + " ".join(f"p{int(p*100)}:{v:+.3%}" for p, v in zip(q, q1)))

    # 利得の所在: d0-MAE ビン別の diff_tr
    bins = [-np.inf, -0.02, -0.01, -0.005, -0.0025, -1e-12, 0.0]
    labels = ["<=-2%", "-2..-1%", "-1..-0.5%", "-0.5..-0.25%", "-0.25..0%", "=0"]
    cut = pd.cut(mae0, bins=bins, labels=labels, include_lowest=True)
    bt = pd.DataFrame({"bin": cut, "diff": diff_tr, "ret0": ret0,
                       "kept": kept}).groupby("bin", observed=False).agg(
        n=("diff", "size"), diff_sum=("diff", "sum"), ret0_sum=("ret0", "sum"),
        dropped=("kept", lambda s: int((~s).sum())))
    bt["diff_share"] = bt["diff_sum"] / diff_tr.sum()
    print("\n  d1利得の所在(d0-MAEビン別):")
    print(bt.to_string(float_format=lambda x: f"{x:+.4f}"))
    bt.reset_index().to_csv(OUT_DIR / "exp53_mae_bins.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "exp53_mae_cohorts.csv", index=False)

    res = {"cohorts": rows, "migration": mig,
           "mae_q_d0": {f"p{int(p*100)}": float(v) for p, v in zip(q, q0)},
           "mae_q_d1": {f"p{int(p*100)}": float(v) for p, v in zip(q, q1)},
           "n_shallow_d0": int(sh0.sum()), "n_shallow_d1": int((sh1 & k1).sum()),
           "diff_total": float(diff_tr.sum()),
           "diff_by_bin": {str(i): float(v) for i, v in bt["diff_sum"].items()}}
    print(f"  [{time.time()-t0:.0f}s]")
    return res


# ---------------------------------------------------------------------------
# 4. 時代4分割(固定k equity 比)
# ---------------------------------------------------------------------------
def run_era(eq0_fn, eq1_fn, k_emp0, t0) -> pd.DataFrame:
    sec(f"4. 時代4分割: 固定k={k_emp0:.2f} の d1/d0 equity 比の伸び")
    eq0, eq1 = eq0_fn(k_emp0), eq1_fn(k_emp0)
    ratio = (eq1 / eq0).dropna()
    total_lg = float(np.log(ratio.iloc[-1] / ratio.iloc[0]))
    rows = []
    for label, a, b in ERAS:
        m = pd.Series(True, index=ratio.index)
        if a:
            m &= ratio.index >= pd.Timestamp(a, tz="UTC")
        if b:
            m &= ratio.index < pd.Timestamp(b, tz="UTC")
        sub = ratio[m]
        lg = float(np.log(sub.iloc[-1] / sub.iloc[0])) if len(sub) > 1 else np.nan
        years = (sub.index[-1] - sub.index[0]).days / 365.25 if len(sub) > 1 else np.nan
        rows.append({"era": label, "log_growth": lg, "share": lg / total_lg,
                     "ann_pp": (np.exp(lg / years) - 1) * 100 if years else np.nan,
                     "n_bars": len(sub)})
        print(f"  {label}: log伸び {lg:+.4f} (シェア {lg/total_lg*100:+.0f}%) "
              f"年率換算 {rows[-1]['ann_pp']:+.2f}pp")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp53_era.csv", index=False)
    n_pos = int((df["log_growth"] > 0).sum())
    mx = df.loc[df["log_growth"].idxmax()]
    print(f"  合計 log {total_lg:+.4f} | 正の時代 {n_pos}/4 | "
          f"最大時代 {mx['era']} シェア {mx['share']*100:.0f}%")
    print(f"  [{time.time()-t0:.0f}s]")
    return df


# ---------------------------------------------------------------------------
# 5. ブートシード拡張(emp較正kでの p95 ペア測定 s0-9)
# ---------------------------------------------------------------------------
def run_seeds(eq0_fn, eq1_fn, k_emp0, k_emp1, t0) -> pd.DataFrame:
    sec(f"5. ブートシード拡張: emp較正k (d0 {k_emp0:.2f} / d1 {k_emp1:.2f}) の p95 s0-9")
    eq0, eq1 = eq0_fn(k_emp0), eq1_fn(k_emp1)
    rows = []
    for sd in range(10):
        p0 = boot_dd(eq0, n_boot=1500, seed=sd)["p95"]
        p1 = boot_dd(eq1, n_boot=1500, seed=sd)["p95"]
        sig = abs(p1) > abs(p0) + 0.005
        rows.append({"seed": sd, "p95_d0": p0, "p95_d1": p1,
                     "diff_pp": (p1 - p0) * 100, "signature": sig})
        print(f"  s{sd}: d0 {p0:+.2%} / d1 {p1:+.2%} (diff {(p1-p0)*100:+.2f}pp) "
              f"署名 {'X' if sig else '-'}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp53_seeds.csv", index=False)
    print(f"  署名出現 {int(df['signature'].sum())}/10 シード | "
          f"p95差 平均 {df['diff_pp'].mean():+.2f}pp (正=d1が浅い)")
    print(f"  [{time.time()-t0:.0f}s]")
    return df


# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    flags = {f for f in sys.argv[1:] if f.startswith("--")}
    do_all = not flags
    uni.register_cross_spreads(3.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"=== exp53: d1 機構実在性と近傍頑健性の監査 (H4, 19銘柄, mp{MAX_POS}) ===")
    print(f"base pool {len(pool)} trades sum={pool['ret'].sum():+.4f} (基準 {BASE_NET:+.4f})")

    mod, kept, ret_new, rc = d1_of(pool)
    diff = float(np.where(kept, ret_new - pool['ret'].to_numpy(), -pool['ret'].to_numpy()).sum())
    print(f"d1 pool {len(mod)} trades (消滅 {int((~kept).sum())}) "
          f"pool_diff {diff:+.4f} ({diff/BASE_NET*100:+.2f}%; exp47 基準 +2.81%)")

    eq0_fn = make_eq_fn(pool, closes, champion_sizing(pool, max_pos=MAX_POS))
    eq1_fn = make_eq_fn(mod, closes, champion_sizing(mod, max_pos=MAX_POS))
    k_emp0 = calibrate_empirical(eq0_fn, 0.20)
    k_emp1 = calibrate_empirical(eq1_fn, 0.20)
    print(f"emp較正 k: d0 {k_emp0:.3f} (exp47: 8.269) / d1 {k_emp1:.3f} (exp47: 8.895)")

    res_path = OUT_DIR / "exp53_result.json"
    payload = json.loads(res_path.read_text()) if res_path.exists() else {}
    payload.update({"pool_diff": diff, "k_emp0": k_emp0, "k_emp1": k_emp1})

    if do_all or "--oat" in flags:
        oat = run_oat(closes, t0)
        nb = oat[oat["param"] != "base"]
        payload["oat"] = {"n_pos": int((nb["diff_pp"] > 0).sum()),
                          "mean_pp": float(nb["diff_pp"].mean()),
                          "median_pp": float(nb["diff_pp"].median()),
                          "min_pp": float(nb["diff_pp"].min()),
                          "max_pp": float(nb["diff_pp"].max()),
                          "base_diff_pp": float(
                              oat.loc[oat["param"] == "base", "diff_pp"].iloc[0])}
    if do_all or "--fixedk" in flags:
        payload["fixedk"] = run_fixedk(pool, mod, closes, eq0_fn, eq1_fn, k_emp0, t0)
    if do_all or "--mae" in flags:
        payload["mae"] = run_mae(pool, kept, ret_new, rc, t0)
    if do_all or "--era" in flags:
        era = run_era(eq0_fn, eq1_fn, k_emp0, t0)
        payload["era"] = era.to_dict("records")
    if do_all or "--seeds" in flags:
        sdf = run_seeds(eq0_fn, eq1_fn, k_emp0, k_emp1, t0)
        payload["seeds"] = {"n_signature": int(sdf["signature"].sum()),
                            "mean_diff_pp": float(sdf["diff_pp"].mean())}

    res_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {res_path}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
