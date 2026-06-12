"""edge06: veto候補1「金曜遅バー(ラベル金曜20:00)エントリー禁止」の口座レベル用量検証。

候補ルール(edge05 が指名): pool.entry が金曜 20:00 ラベル(dayofweek==4 & hour==20)の
エントリーを見送る(スキップのみ。d>=2 繰延は禁止済み)。
機構: このバーだけ実約定(最終M1ティック)が金曜21:55頃 = UTC20-22 ロールオーバー窓の
ド真ん中+週末ギャップ直前という執行最悪コホート。
プール上 n=29 / sum -0.0154 / mean -5.3bps(×3再価格で -7.3bps)。
弱点: n=29<30、OOS 単独では正(+0.018)、2018 単年集中。

手順(検証規約準拠):
  0) ベース再現: mm_pool_v2d1_H4_19 + champion_sizing(mp8) + ペアシード較正(seeds 0-4)で
     rob 5シード平均 +18.63% / emp +27.50% を ±0.1pp で再現(失敗なら以降の比較は無効)。
  1) 変種 = 該当29件をプールから行ごと除外 → 同一シード集合で較正・再シミュレート
     (建玉枠の解放・較正 k の変化込み。PnL の単純引き算はしない)。
  2) rob 各シードΔ / 平均Δ / empΔ / p95 / 最悪年 / 年次差分。レバ偽装署名
     (emp CAGR↑ かつ p95 悪化 → reject。ブートシード 0-2 でも署名を測る)。
  3) G3: IS(<2022) 較正(emp / rob_s0)→ OOS 素成績。ベース vs 変種を同一手順で。
  4) 年次分割感度: 効果が僅差(<0.5pp)なら前半/後半・偶奇年で符号安定性
     (プール段=除外 ret の年次符号反転 / 口座段=emp較正・rob_s0較正の年次差分)。
  5) 判定: reports/22 判例(+0.34pp・全ゲート通過でも、年次分割感度で崩れ・効果量が
     ノイズ帯未満・複雑性に見合わない → 不採用)の採用バーを適用。
     ※「実弾の執行リスク回避」という機構的価値は判定対象外(数値のみで判定)。

実行: uv run python -m research.experiments.edge06_friday_latebar_account
出力: research/outputs/edge06_result.json / edge06_account.csv
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
from mm_production import build_pool_d1, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab import universe as uni  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)

SEEDS = (0, 1, 2, 3, 4)
MAX_POS = 8
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
BASE_N, BASE_SUM = 1207, 1.9622
REF = {"rob_mean": 0.1863, "emp_cagr": 0.2750, "rob_s0_k": 6.084, "rob_s0": 0.1824}
TOL = 0.001  # ±0.1pp
OUT_DIR = ROOT / "research" / "outputs"
OUT_JSON = OUT_DIR / "edge06_result.json"
OUT_CSV = OUT_DIR / "edge06_account.csv"


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


class Cfg:
    """k -> simulate 結果のキャッシュ付き設定(exp47/56 踏襲)。"""

    def __init__(self, label: str, pool: pd.DataFrame, closes: pd.DataFrame):
        self.label, self.pool, self.closes = label, pool, closes
        self.mk = champion_sizing(pool, max_pos=MAX_POS)
        self._c: dict[float, tuple] = {}

    def _sim(self, k: float):
        kk = round(float(k), 10)
        if kk not in self._c:
            self._c[kk] = mm.simulate(self.pool, self.closes, self.mk(kk), max_pos=MAX_POS)
        return self._c[kk]

    def eq_of_k(self, k):
        return self._sim(k)[0]

    def info_of_k(self, k):
        return self._sim(k)[2]


def full_eval(cfg: Cfg) -> dict:
    """empirical + robust(seeds 0-4) + 年次 + IS較正(emp/rob_s0)→OOS素検証 + skip率。"""
    r = protocol_eval(cfg.eq_of_k, label=cfg.label, seeds=SEEDS)
    eq_e = cfg.eq_of_k(r["emp_k"])
    yr_e = yearly_returns(eq_e)
    r["worst_year"] = float(yr_e.min())
    r["neg_years_emp"] = int((yr_e < 0).sum())
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    info_e = cfg.info_of_k(r["emp_k"])
    r["skip_emp"] = info_e["skipped"] / (info_e["skipped"] + info_e["n_taken"])
    k_r0 = r["rob"][SEEDS[0]]["k"]
    yr_r0 = yearly_returns(cfg.eq_of_k(k_r0))
    r["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    r["neg_years_rob0"] = int((yr_r0 < 0).sum())
    r["worst_year_rob0"] = float(yr_r0.min())
    info_r = cfg.info_of_k(k_r0)
    r["skip_rob0"] = info_r["skipped"] / (info_r["skipped"] + info_r["n_taken"])

    # IS(<2022) 較正 → OOS 素検証
    is_pool = cfg.pool[cfg.pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = cfg.pool[cfg.pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = cfg.closes[cfg.closes.index < OOS_START]
    oos_cl = cfg.closes[cfg.closes.index >= OOS_START]

    def eq_fn(pl, cl):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(pl, cl, cfg.mk(kk), max_pos=MAX_POS)[0]
            return c[kk]
        return f

    eq_is, eq_oos = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_ie = calibrate_empirical(eq_is, 0.20)
    r["k_is_emp"] = k_ie
    r["is_emp_cagr"] = cagr_of(eq_is(k_ie))
    r["oos_emp_cagr"] = cagr_of(eq_oos(k_ie))
    r["oos_emp_dd"] = max_dd(eq_oos(k_ie))
    k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
    r["k_is_rob"] = k_ir
    r["is_rob_cagr"] = cagr_of(eq_is(k_ir))
    r["oos_rob_cagr"] = cagr_of(eq_oos(k_ir))
    r["oos_rob_dd"] = max_dd(eq_oos(k_ir))
    print(f"      IS emp k={k_ie:5.2f} ISC={r['is_emp_cagr']:+7.2%} -> "
          f"OOS {r['oos_emp_cagr']:+7.2%} (DD {r['oos_emp_dd']:+5.1%}) | "
          f"IS rob k={k_ir:5.2f} ISC={r['is_rob_cagr']:+7.2%} -> "
          f"OOS {r['oos_rob_cagr']:+7.2%} (DD {r['oos_rob_dd']:+5.1%}) | "
          f"skip emp/rob0 {r['skip_emp']:.1%}/{r['skip_rob0']:.1%}")
    return r


def year_diff_audit(tag: str, yc: dict, yb: dict) -> dict:
    """年次差分(変種−ベース)の単年依存監査 + 分割感度(前半/後半・偶奇年)。"""
    d = (pd.Series(yc) - pd.Series(yb)).dropna()
    total = float(d.sum())
    best_y = int(d.idxmax())
    excl_best = float(d.drop(best_y).sum())
    yrs = sorted(d.index)
    front = [y for y in yrs[: (len(yrs) + 1) // 2]]
    back = [y for y in yrs[(len(yrs) + 1) // 2:]]
    return {
        "basis": tag, "total": total,
        "best_year": best_y, "best_year_diff": float(d[best_y]),
        "excl_best": excl_best,
        "keep_share_excl_best": excl_best / total if total != 0 else np.nan,
        "front_half": float(d.loc[front].sum()), "back_half": float(d.loc[back].sum()),
        "even_years": float(d.loc[[y for y in yrs if y % 2 == 0]].sum()),
        "odd_years": float(d.loc[[y for y in yrs if y % 2 == 1]].sum()),
        "n_pos_years": int((d > 0).sum()), "n_neg_years": int((d < 0).sum()),
        "yearly": {int(y): float(v) for y, v in d.items()},
    }


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1()  # cache: results/mm_pool_v2d1_H4_19.parquet
    closes = mm.load_closes()

    sec("0. プール検算 + 金曜遅バーコホートの確認")
    n, s = len(pool), float(pool["ret"].sum())
    print(f"pool n={n} sum(ret)={s:+.4f} (期待 {BASE_N}, {BASE_SUM:+.4f}) / grid {len(closes)}本")
    assert n == BASE_N and abs(s - BASE_SUM) < 1e-3, "ベースプール不一致 — 以降の比較は無効"

    fri = (pool["entry"].dt.dayofweek == 4) & (pool["entry"].dt.hour == 20)
    sub = pool[fri]
    print(f"金曜20:00ラベル: n={fri.sum()} sum={sub['ret'].sum():+.4f} "
          f"mean={sub['ret'].mean()*1e4:+.1f}bps (期待 29 / -0.0154 / -5.3bps)")
    assert int(fri.sum()) == 29, "コホート件数が edge05 指名(29件)と不一致"
    by_entry_y = sub.groupby(sub["entry"].dt.year)["ret"].agg(["count", "sum"])
    print("除外コホートの entry 年次:")
    print(by_entry_y.to_string())
    oos_m = sub["entry"] >= OOS_START
    print(f"IS sum={sub.loc[~oos_m,'ret'].sum():+.4f} / OOS sum={sub.loc[oos_m,'ret'].sum():+.4f}")

    # プール段差分(除外 = -ret を失う/得る)。年次は exp47 慣例の exit 年で。
    pool_yr_diff = (-sub["ret"]).groupby(sub["exit"].dt.year).sum()
    print("プール段 年次差分(exit年, 変種−ベース): " +
          "  ".join(f"{int(y)}:{v:+.4f}" for y, v in pool_yr_diff.items()))

    variant = pool[~fri].reset_index(drop=True)
    print(f"変種プール n={len(variant)} sum={variant['ret'].sum():+.4f}")

    sec(f"1. ベース再現 (seeds {SEEDS}, mp{MAX_POS}, champion_sizing)")
    cfg_b = Cfg("base_d1", pool, closes)
    base = full_eval(cfg_b)
    chk = {
        "rob_mean": (base["rob_cagr_mean"], REF["rob_mean"]),
        "emp_cagr": (base["emp_cagr"], REF["emp_cagr"]),
        "rob_s0": (base["rob"][0]["cagr"], REF["rob_s0"]),
    }
    ok_all = True
    for kk, (got, ref) in chk.items():
        ok = abs(got - ref) <= TOL
        ok_all &= ok
        print(f"  再現チェック {kk}: got {got:+.4%} vs ref {ref:+.4%} "
              f"(diff {(got-ref)*100:+.3f}pp) -> {'OK' if ok else 'FAIL'}")
    print(f"  rob_s0 k: got {base['rob'][0]['k']:.3f} vs ref {REF['rob_s0_k']:.3f}")
    if not ok_all:
        print("!! ベース再現失敗(±0.1pp 超)。手順を疑え。以降は参考値。")
    print(f"    [{time.time()-t0:.0f}s]")

    sec(f"2. 変種(金曜遅バー29件除外)の口座レベル (同一シード集合)")
    cfg_v = Cfg("no_fri_latebar", variant, closes)
    var = full_eval(cfg_v)
    print(f"    [{time.time()-t0:.0f}s]")

    sec("3. ペアシード比較 + レバ偽装署名")
    per_seed = {sd: var["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
    gain = var["rob_cagr_mean"] - base["rob_cagr_mean"]
    print("  seed |   base    |  variant   |   Δ")
    for sd in SEEDS:
        print(f"   s{sd}  | {base['rob'][sd]['cagr']:+.2%} (k{base['rob'][sd]['k']:5.2f}) "
              f"| {var['rob'][sd]['cagr']:+.2%} (k{var['rob'][sd]['k']:5.2f}) "
              f"| {per_seed[sd]*100:+.3f}pp")
    print(f"  mean | {base['rob_cagr_mean']:+.2%}   | {var['rob_cagr_mean']:+.2%}   "
          f"| {gain*100:+.3f}pp")
    emp_d = var["emp_cagr"] - base["emp_cagr"]
    print(f"  empirical: base {base['emp_cagr']:+.2%} (k={base['emp_k']:.3f}, "
          f"p95 {base['emp_p95']:+.1%}) -> variant {var['emp_cagr']:+.2%} "
          f"(k={var['emp_k']:.3f}, p95 {var['emp_p95']:+.1%})  Δ {emp_d*100:+.3f}pp")
    print(f"  最悪年(emp較正): base {base['worst_year']:+.2%} -> variant {var['worst_year']:+.2%}")
    print(f"  最悪年(rob_s0): base {base['worst_year_rob0']:+.2%} -> "
          f"variant {var['worst_year_rob0']:+.2%}")

    sig = (var["emp_cagr"] > base["emp_cagr"]) and \
          (abs(var["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
    print(f"  レバ偽装署名(seed0, 0.5pp閾): {'あり=reject' if sig else 'なし'}")
    # ブートシード 0-2 での署名監査(p95 はシード依存 ±0.4-0.8pp)
    sig_seeds = {}
    for sd in (0, 1, 2):
        pb = boot_dd(cfg_b.eq_of_k(base["emp_k"]), n_boot=1500, seed=sd)["p95"]
        pv = boot_dd(cfg_v.eq_of_k(var["emp_k"]), n_boot=1500, seed=sd)["p95"]
        sig_seeds[sd] = {"base_p95": float(pb), "var_p95": float(pv),
                         "sig": bool(emp_d > 0 and abs(pv) > abs(pb) + 0.005)}
        print(f"    boot s{sd}: base p95 {pb:+.2%} / variant p95 {pv:+.2%} "
              f"署名 {'X' if sig_seeds[sd]['sig'] else '-'}")
    n_sig = sum(v["sig"] for v in sig_seeds.values())

    sec("4. G3: IS(<2022)較正 -> OOS素成績(ベース vs 変種)")
    print(f"  emp較正: base IS {base['is_emp_cagr']:+.2%} -> OOS {base['oos_emp_cagr']:+.2%} "
          f"(DD {base['oos_emp_dd']:+.1%}) | variant IS {var['is_emp_cagr']:+.2%} -> "
          f"OOS {var['oos_emp_cagr']:+.2%} (DD {var['oos_emp_dd']:+.1%}) "
          f"Δ {(var['oos_emp_cagr']-base['oos_emp_cagr'])*100:+.3f}pp")
    print(f"  rob較正: base IS {base['is_rob_cagr']:+.2%} -> OOS {base['oos_rob_cagr']:+.2%} "
          f"(DD {base['oos_rob_dd']:+.1%}) | variant IS {var['is_rob_cagr']:+.2%} -> "
          f"OOS {var['oos_rob_cagr']:+.2%} (DD {var['oos_rob_dd']:+.1%}) "
          f"Δ {(var['oos_rob_cagr']-base['oos_rob_cagr'])*100:+.3f}pp")
    g3_rob = var["oos_rob_cagr"] > base["oos_rob_cagr"]
    g3_emp = var["oos_emp_cagr"] > base["oos_emp_cagr"]

    sec("5. 年次差分監査 + 分割感度(前半/後半・偶奇年)")
    a_emp = year_diff_audit("emp較正", var["yr_emp"], base["yr_emp"])
    a_rob = year_diff_audit("rob_s0較正", var["yr_rob0"], base["yr_rob0"])
    pool_front = float(pool_yr_diff[pool_yr_diff.index <= 2020].sum())
    pool_back = float(pool_yr_diff[pool_yr_diff.index >= 2021].sum())
    pool_even = float(pool_yr_diff[pool_yr_diff.index % 2 == 0].sum())
    pool_odd = float(pool_yr_diff[pool_yr_diff.index % 2 == 1].sum())
    for a in (a_emp, a_rob):
        print(f"\n  [{a['basis']}] 年次差分合計 {a['total']*100:+.3f}pp / "
              f"最良年 {a['best_year']}({a['best_year_diff']*100:+.3f}pp) "
              f"除外後 {a['excl_best']*100:+.3f}pp (残存率 {a['keep_share_excl_best']:.0%})")
        print(f"    前半 {a['front_half']*100:+.3f}pp / 後半 {a['back_half']*100:+.3f}pp | "
              f"偶数年 {a['even_years']*100:+.3f}pp / 奇数年 {a['odd_years']*100:+.3f}pp | "
              f"+年 {a['n_pos_years']} / -年 {a['n_neg_years']}")
        print("    年次: " + "  ".join(f"{y}:{v*100:+.2f}" for y, v in a["yearly"].items()))
    print(f"\n  [プール段(exit年)] 合計 {float(pool_yr_diff.sum()):+.4f} | "
          f"前半(≤2020) {pool_front:+.4f} / 後半(≥2021) {pool_back:+.4f} | "
          f"偶数年 {pool_even:+.4f} / 奇数年 {pool_odd:+.4f}")

    sec("6. ゲート判定 + 採用バー(reports/22 判例)")
    noise_band = 0.4  # pp(較正ノイズ下限。ペアシードで縮むが効果量の物差しは維持)
    gates = {
        "g0_baseline_repro": bool(ok_all),
        "g1_gain_pp": gain * 100,
        "g1_above_noise": bool(gain * 100 >= noise_band),
        "g1_all_seeds_pos": bool(all(v > 0 for v in per_seed.values())),
        "g2_no_signature_s0": bool(not sig),
        "g2_sig_seeds_n": int(n_sig),
        "g3_oos_rob": bool(g3_rob), "g3_oos_emp": bool(g3_emp),
        "g3_oos": bool(g3_rob and g3_emp),
        "g4_all_years_pos": bool(var["neg_years_emp"] == 0 and var["neg_years_rob0"] == 0),
        "g5_emp_keep": a_emp["keep_share_excl_best"],
        "g5_rob_keep": a_rob["keep_share_excl_best"],
        "g6_split_stable_emp": bool(a_emp["front_half"] * a_emp["back_half"] > 0
                                    and a_emp["even_years"] * a_emp["odd_years"] > 0),
        "g6_split_stable_rob": bool(a_rob["front_half"] * a_rob["back_half"] > 0
                                    and a_rob["even_years"] * a_rob["odd_years"] > 0),
        "g6_split_stable_pool": bool(pool_front * pool_back > 0 and pool_even * pool_odd > 0),
    }
    for kk, v in gates.items():
        print(f"  {kk}: {v if not isinstance(v, float) else f'{v:+.3f}'}")

    if gain * 100 >= noise_band and gates["g1_all_seeds_pos"] and not sig \
            and gates["g3_oos"] and gates["g6_split_stable_emp"] \
            and gates["g6_split_stable_rob"] and gates["g6_split_stable_pool"]:
        verdict = "adopt_candidate"
    elif abs(gain) * 100 < noise_band:
        verdict = "noise_band"
    else:
        verdict = "reject"
    print(f"\n  verdict: {verdict}  (rob mean Δ {gain*100:+.3f}pp, "
          f"emp Δ {emp_d*100:+.3f}pp, noise band ±{noise_band}pp)")

    # --- 保存 -------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for nm, r in [("base_d1", base), ("no_fri_latebar", var)]:
        rows.append({
            "cfg": nm, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
            "emp_dd": r["emp_dd"], "emp_p95": r["emp_p95"],
            **{f"rob_s{sd}_k": r["rob"][sd]["k"] for sd in SEEDS},
            **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS},
            "rob_mean": r["rob_cagr_mean"], "worst_year": r["worst_year"],
            "worst_year_rob0": r["worst_year_rob0"],
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            "skip_emp": r["skip_emp"], "skip_rob0": r["skip_rob0"],
            "k_is_emp": r["k_is_emp"], "is_emp_cagr": r["is_emp_cagr"],
            "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
            "k_is_rob": r["k_is_rob"], "is_rob_cagr": r["is_rob_cagr"],
            "oos_rob_cagr": r["oos_rob_cagr"], "oos_rob_dd": r["oos_rob_dd"],
        })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    payload = {
        "cohort": {"n": 29, "sum_ret": float(sub["ret"].sum()),
                   "mean_bps": float(sub["ret"].mean() * 1e4),
                   "by_entry_year": {int(y): {"count": int(c), "sum": float(v)}
                                     for y, (c, v) in by_entry_y.iterrows()},
                   "is_sum": float(sub.loc[~oos_m, "ret"].sum()),
                   "oos_sum": float(sub.loc[oos_m, "ret"].sum())},
        "baseline_check": {k: {"got": float(g), "ref": float(rf), "ok": bool(abs(g - rf) <= TOL)}
                           for k, (g, rf) in chk.items()},
        "base": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                 for k, v in base.items()},
        "variant": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                    for k, v in var.items()},
        "per_seed_delta_pp": {str(sd): float(v * 100) for sd, v in per_seed.items()},
        "gain_pp": float(gain * 100), "emp_delta_pp": float(emp_d * 100),
        "sig_seeds": {str(s): v for s, v in sig_seeds.items()},
        "year_audit": {"emp": a_emp, "rob0": a_rob},
        "pool_year_diff": {int(y): float(v) for y, v in pool_yr_diff.items()},
        "pool_splits": {"front": pool_front, "back": pool_back,
                        "even": pool_even, "odd": pool_odd},
        "gates": gates, "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n      -> {OUT_JSON}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
