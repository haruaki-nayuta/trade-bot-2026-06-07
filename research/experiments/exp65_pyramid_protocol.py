"""exp65: 利益確認型ピラミッド(prof+θ で1回積み増し)のフルプロトコル — exp64 の生存者。

exp64: 経路条件付き積み増し5種のうち、ナンピン系(-3.6〜-3.9pp)・収束確認系(-1.2pp)・
時間系(-0.6pp)は同一テールで死亡。唯一 prof+0.5%(含み益0.5%到達で f(|z_now|) サイズを
1回積む)が rob3 +0.38pp かつ p95 改善(-27.3→-25.3%)で生存。
機構: 積み増しの98.6%が非ワースト側(ワースト10%付着は220件中3件)=「収束が始まった」
ことの確認後にのみ露出を足すため、追加分はテールをほぼ深くしない(de-risking 方向。
emp CAGR は下がる(k 8.89→7.20)が、p95 がそれ以上に浅くなり robust が上がる)。

本実験(事前登録):
  - 用量曲線 θ ∈ {0.3%, 0.5%, 0.75%, 1.0%}(1点スパイク検査+IS-argmax 監査)
  - 各 θ: プール段(トランシェ年次G5/IS-OOS) + 口座 seeds 0-4 ペア + 6ゲート
    (署名ブートシード0-2 / IS較正→OOS rob・emp両生CAGR / 全年プラス / 年次G5)
  - 最良 θ(IS-argmax が選ぶもの)に M1 粒度監査(exp52 方式, rob_m5 k)→ 掛け目込み実効CAGR
  - 採用判定はゲート全通過+実効CAGRでベース(+17.72%)超えが条件。判定は別途(本実験は測定)。

実行: PYTHONPATH=. uv run python research/experiments/exp65_pyramid_protocol.py
出力: research/outputs/exp65_result.csv / exp65_result.json
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
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from exp47_entry_delay import reconstruct, year_diff_audit  # noqa: E402
import exp52_d1_m1audit as a52  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 8
SEEDS = (0, 1, 2, 3, 4)
THETAS = (0.003, 0.005, 0.0075, 0.010)


def build_addons(pool, rc, theta):
    """含み益 +theta 到達の最初のバー close で 1 回積み増し(出口は元トレードと同一)。"""
    zcache = {}
    recs = []
    for instr, g in pool.groupby("instr"):
        s = rc["closes_by"][instr]
        if instr not in zcache:
            zcache[instr] = ((s - s.rolling(50).mean()) / s.rolling(50).std()).to_numpy()
        zarr = zcache[instr]
        carr = s.to_numpy()
        tarr = s.index.values
        for ti in g.index.to_numpy():
            e, x = int(rc["idx_e"][ti]), int(rc["idx_x"][ti])
            if x - e < 2:
                continue
            d = float(pool.at[ti, "dir"])
            cs = carr[e:x + 1]
            path = d * (cs / cs[0] - 1.0)
            hit = np.where(path[1:-1] >= theta)[0]
            if not len(hit):
                continue
            j = int(hit[0]) + 1
            fwd = d * (rc["exit_close"][ti] / cs[j] - 1.0) - rc["cost"][ti]
            recs.append({"instr": instr,
                         "entry": pd.Timestamp(tarr[e + j]).tz_localize("UTC"),
                         "exit": pool.at[ti, "exit"], "dir": int(d),
                         "entry_price": cs[j] * rc["slip"][ti], "ret": fwd,
                         "bars_held": x - e - j,
                         "z_entry": abs(zarr[e + j]), "vol_entry": np.nan})
    return pd.DataFrame(recs)


def full_eval(tag, pl, closes, seeds=SEEDS):
    mk = champion_sizing(pl, max_pos=MAX_POS)
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pl, closes, mk(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    r = protocol_eval(eq_of_k, label=tag, seeds=seeds)
    yr_e = yearly_returns(eq_of_k(r["emp_k"]))
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    r["neg_years_emp"] = int((yr_e < 0).sum())
    r["worst_year"] = float(yr_e.min())
    yr0 = yearly_returns(eq_of_k(r["rob"][seeds[0]]["k"]))
    r["yr_rob0"] = {int(y): float(v) for y, v in yr0.items()}
    r["neg_years_rob0"] = int((yr0 < 0).sum())
    is_pool = pl[pl["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pl[pl["entry"] >= OOS_START].reset_index(drop=True)
    is_cl, oos_cl = closes[closes.index < OOS_START], closes[closes.index >= OOS_START]

    def eq_fn(p2, c2):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(p2, c2, mk(kk), max_pos=MAX_POS)[0]
            return c[kk]
        return f
    fi, fo = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_ir = calibrate_robust_seeded(fi, 0.20, seed=0)
    r["is_rob_cagr"], r["oos_rob_cagr"] = cagr_of(fi(k_ir)), cagr_of(fo(k_ir))
    r["oos_rob_dd"] = max_dd(fo(k_ir))
    k_ie = calibrate_empirical(fi, 0.20)
    r["is_emp_cagr"], r["oos_emp_cagr"] = cagr_of(fi(k_ie)), cagr_of(fo(k_ie))
    r["oos_emp_dd"] = max_dd(fo(k_ie))
    r["eq_of_k"] = eq_of_k
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy().reset_index(drop=True)
    closes = mm.load_closes()
    rc = reconstruct(pool)
    print(f"=== exp65: 利益確認型ピラミッド フルプロトコル (θ={THETAS}) ===")

    print("\n--- 0. プール段(トランシェの前向き純リターン+単年依存) ---")
    addons, prows = {}, []
    for th in THETAS:
        ad = build_addons(pool, rc, th)
        addons[th] = ad
        r = ad["ret"]
        yr = ad.groupby(ad["exit"].dt.year)["ret"].sum()
        by = int(yr.idxmax())
        excl_best = float(yr.drop(by).sum())
        is_m = ad["entry"] < OOS_START
        prows.append({"theta": th, "n": len(ad), "sum": float(r.sum()),
                      "mean_bps": r.mean() * 1e4, "win": (r > 0).mean(),
                      "is": float(r[is_m].sum()), "oos": float(r[~is_m].sum()),
                      "best_year": by, "keep_excl_best": excl_best / r.sum() if r.sum() > 0 else np.nan,
                      "neg_years_pool": int((yr < 0).sum())})
        print(f"  θ={th:.3%}: n={len(ad)} sum={r.sum():+.4f} mean={r.mean()*1e4:+.1f}bps "
              f"win={(r>0).mean():.0%} IS/OOS {r[is_m].sum():+.3f}/{r[~is_m].sum():+.3f} "
              f"最良年{by}除外後残存 {excl_best/r.sum()*100 if r.sum()>0 else 0:.0f}% "
              f"プール負け年{int((yr<0).sum())}")

    print("\n--- 1. 口座段 seeds 0-4 (base + 各θ) ---")
    base = full_eval("base", pool, closes)
    print(f"    [{time.time()-t0:.0f}s]")
    results = {"base": base}
    for th in THETAS:
        aug = pd.concat([pool, addons[th]], ignore_index=True
                        ).sort_values("entry").reset_index(drop=True)
        r = full_eval(f"pyr{th:.3%}", aug, closes)
        results[th] = r
        per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        g3 = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
             (r["oos_emp_cagr"] > base["oos_emp_cagr"])
        a_emp = year_diff_audit("emp", r["yr_emp"], base["yr_emp"])
        a_rob = year_diff_audit("rob0", r["yr_rob0"], base["yr_rob0"])
        r["gates"] = {
            "gain_pp": (r["rob_cagr_mean"] - base["rob_cagr_mean"]) * 100,
            "all_seeds_pos": all(v > 0 for v in per_seed.values()),
            "per_seed": {sd: v * 100 for sd, v in per_seed.items()},
            "signature": bool(sig), "g3_oos_raw_both": bool(g3),
            "g4_years": (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0),
            "g5_emp_keep": a_emp["keep_share_excl_best"],
            "g5_rob0_keep": a_rob["keep_share_excl_best"],
            "g5_excl2022_emp": a_emp["excl_2022"],
        }
        g = r["gates"]
        print(f"      gain {g['gain_pp']:+.2f}pp seeds " +
              " ".join(f"s{sd}:{v:+.2f}" for sd, v in g["per_seed"].items()) +
              f" 署名={'あり' if sig else 'なし'} G3raw={'+' if g3 else 'x'} "
              f"全年+={'+' if g['g4_years'] else 'x'} G5keep emp/rob0 "
              f"{g['g5_emp_keep']:.0%}/{g['g5_rob0_keep']:.0%}  [{time.time()-t0:.0f}s]")

    print("\n--- 2. IS-argmax 監査(IS<2022 robust/emp 較正で θ を選ぶと?) ---")
    print("   θ      IS_rob    IS_emp    OOS_rob   OOS_emp")
    print(f"  base  {base['is_rob_cagr']:+8.2%} {base['is_emp_cagr']:+8.2%} "
          f"{base['oos_rob_cagr']:+8.2%} {base['oos_emp_cagr']:+8.2%}")
    for th in THETAS:
        r = results[th]
        print(f"  {th:.3%} {r['is_rob_cagr']:+8.2%} {r['is_emp_cagr']:+8.2%} "
              f"{r['oos_rob_cagr']:+8.2%} {r['oos_emp_cagr']:+8.2%}")
    cands = ["base"] + list(THETAS)
    arg_rob = max(cands, key=lambda c: results[c]["is_rob_cagr"] if c != "base" else base["is_rob_cagr"])
    print(f"  IS-argmax(rob) = {arg_rob}")

    print("\n--- 3. 署名ブートシード監査(emp_k, seeds 0-2) ---")
    p95b = {sd: boot_dd(base["eq_of_k"](base["emp_k"]), n_boot=1500, seed=sd)["p95"]
            for sd in (0, 1, 2)}
    print("  base: " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95b.items()))
    for th in THETAS:
        r = results[th]
        p95c = {sd: boot_dd(r["eq_of_k"](r["emp_k"]), n_boot=1500, seed=sd)["p95"]
                for sd in (0, 1, 2)}
        n_sig = sum((r["emp_cagr"] > base["emp_cagr"]) and
                    (abs(p95c[sd]) > abs(p95b[sd]) + 0.005) for sd in p95c)
        r["gates"]["sig_seeds"] = n_sig
        print(f"  θ={th:.3%}: " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95c.items()) +
              f"  署名 {n_sig}/3")

    print("\n--- 4. M1 粒度監査(全θ, rob_m5 k) ---")
    grid_idx = pd.DatetimeIndex(load_m1("EURUSD").index.tz_localize(None))
    a52.MAX_POS = MAX_POS
    base_k5 = float(np.mean([base["rob"][sd]["k"] for sd in SEEDS]))
    aud_b = a52.m1_audit_one("base", pool, closes,
                             champion_sizing(pool, max_pos=MAX_POS), base_k5, grid_idx)
    print(f"  base: 谷比 {aud_b['ratio']:.3f} 掛け目 x{aud_b['haircut']:.3f} "
          f"実効 {aud_b['cagr_adj']:+.2%}")
    m1 = {"base": aud_b}
    for th in THETAS:
        r = results[th]
        aug = pd.concat([pool, addons[th]], ignore_index=True
                        ).sort_values("entry").reset_index(drop=True)
        k5 = float(np.mean([r["rob"][sd]["k"] for sd in SEEDS]))
        aud = a52.m1_audit_one(f"pyr{th:.3%}", aug, closes,
                               champion_sizing(aug, max_pos=MAX_POS), k5, grid_idx)
        m1[th] = aud
        print(f"  θ={th:.3%}: 谷比 {aud['ratio']:.3f} ({'PASS' if aud['ratio']<=1.15 else 'FAIL'}) "
              f"掛け目 x{aud['haircut']:.3f} 実効 {aud['cagr_adj']:+.2%} "
              f"(base比 {(aud['cagr_adj']-aud_b['cagr_adj'])*100:+.2f}pp)  [{time.time()-t0:.0f}s]")

    # --- 保存 ---
    rows = []
    for c in cands:
        r = results[c] if c != "base" else base
        row = {"cfg": str(c), "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_mean": r["rob_cagr_mean"],
               **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS},
               "is_rob": r["is_rob_cagr"], "oos_rob": r["oos_rob_cagr"],
               "is_emp": r["is_emp_cagr"], "oos_emp": r["oos_emp_cagr"],
               "worst_year": r["worst_year"],
               "m1_ratio": m1[c]["ratio"], "m1_haircut": m1[c]["haircut"],
               "eff_cagr": m1[c]["cagr_adj"]}
        if c != "base":
            row.update({f"gate_{k}": v for k, v in r["gates"].items() if k != "per_seed"})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp65_result.csv", index=False)
    payload = {"pool": prows,
               "is_argmax_rob": str(arg_rob),
               "results": {str(c): {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                                    for k, v in (results[c] if c != "base" else base).items()
                                    if k not in ("eq_of_k", "yr_emp", "yr_rob0")}
                           for c in cands},
               "m1": {str(c): {k: v for k, v in m1[c].items() if k != "episodes"} for c in m1}}
    (OUT_DIR / "exp65_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp65_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
