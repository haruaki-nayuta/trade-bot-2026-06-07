"""edge07: veto候補2「末尾5分限界クロス拒否」の口座レベル検証(チャンピオン v2_d1)。

候補ルール(edge01 が指名した veto 候補 2):
  シグナルバー close を「バー終了5分前時点の M1 close」に置換して z(50) を再計算し、
  ±2.0 クロス不成立なら見送る(= edge01 の spike_made_l5 コホート n=98 を除外)。
  プール上 n=98 / mean +1.8bps / 除外PnL +0.0181(=総PnLは微減)。期待利得は
  per-trade 質(残存 mean 16.3→17.5bps)と DD 形状→較正 k のみ。
  OOS(2022-)では除外コホートが -0.0107 なので OOS 改善の可能性があるが、
  コホート年次(edge01)は 2022 単年 -0.0442 が支配的 = 単年依存の疑いを年次分割で監査。

手順(exp47 の同一テールプロトコルを踏襲・veto候補1=edge06 と同一):
  0) ベース再現(rob 5シード(0-4)平均 +18.63% / emp k≈8.895 +27.50% / rob_s0 k=6.084 +18.24%)
  1) 変種プール = spike_made_l5=True の 98 行をプールから落として**再シミュレート**
     (建玉枠の解放・較正 k の変化込み。PnL の単純引き算は禁止)
  2) ペアシード robust(p95=20%, seeds 0-4)+ empirical 20% + レバ偽装署名(+ブートシード監査)
  3) IS(<2022)較正→OOS素検証(emp/rob 両較正, G3)
  4) 年次差分監査(最良年除外・2022除外)+ 年次分割感度(偶奇年/前後半, reports/22 規約)
  5) 複雑性コスト対効果(本番 bot に M1 フィード+バー終了5分前スナップショットの z 再計算が必要)

実行: uv run python research/experiments/edge07_l5cross_account.py
出力: research/outputs/edge07_result.json / edge07_account.csv
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
    boot_dd,
    calibrate_empirical,
    calibrate_robust_seeded,
    cagr_of,
    max_dd,
    protocol_eval,
    yearly_returns,
)

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

POOL_PATH = ROOT / "results" / "mm_pool_v2d1_H4_19.parquet"
EDGE01_CSV = ROOT / "research" / "outputs" / "edge01_trades.csv"
EXPECT_N = 1207
EXPECT_SUM = 1.9622
MAX_POS = 8
SEEDS_FULL = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
OUT_JSON = OUT_DIR / "edge07_result.json"
OUT_ACC = OUT_DIR / "edge07_account.csv"

# 確定ベースライン(検証済み, exp47/52)
REF_ROB_MEAN = 0.1863
REF_EMP_CAGR = 0.2750
REF_ROB_S0_K = 6.084
REF_ROB_S0_CAGR = 0.1824


def sec(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --- 口座レベル評価(exp47 踏襲・k キャッシュ) ----------------------------
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
    """empirical + robust(seeds) + 年次 + IS較正(emp/rob)→OOS素検証 + skip率。"""
    res = protocol_eval(cfg.eq_of_k, label=cfg.label, seeds=seeds)
    eq_emp = cfg.eq_of_k(res["emp_k"])
    yr_emp = yearly_returns(eq_emp)
    res["worst_year"] = float(yr_emp.min())
    res["neg_years_emp"] = int((yr_emp < 0).sum())
    res["yr_emp"] = {int(y): float(v) for y, v in yr_emp.items()}
    info_e = cfg.info_of_k(res["emp_k"])
    res["skip_emp"] = info_e["skipped"] / (info_e["skipped"] + info_e["n_taken"])
    k_r0 = res["rob"][seeds[0]]["k"]
    yr_r0 = yearly_returns(cfg.eq_of_k(k_r0))
    res["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}
    res["neg_years_rob0"] = int((yr_r0 < 0).sum())
    res["worst_year_rob0"] = float(yr_r0.min())
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
    res["yr_oos_emp"] = {int(y): float(v) for y, v in yearly_returns(eqo).items()}

    k_is_rob = calibrate_robust_seeded(eq_is_of_k, 0.20, seed=0)
    res["k_is_rob"] = k_is_rob
    res["is_rob_cagr"] = cagr_of(eq_is_of_k(k_is_rob))
    eqor = eq_oos_of_k(k_is_rob)
    res["oos_rob_cagr"] = cagr_of(eqor)
    res["oos_rob_dd"] = max_dd(eqor)
    res["yr_oos_rob"] = {int(y): float(v) for y, v in yearly_returns(eqor).items()}
    print(f"      IS emp k={k_is_emp:5.2f} ISC={res['is_emp_cagr']:+7.2%} -> "
          f"OOS {res['oos_emp_cagr']:+7.2%} (DD {res['oos_emp_dd']:+5.1%}) | "
          f"IS rob k={k_is_rob:5.2f} ISC={res['is_rob_cagr']:+7.2%} -> "
          f"OOS {res['oos_rob_cagr']:+7.2%} (DD {res['oos_rob_dd']:+5.1%}) | "
          f"skip emp/rob0 {res['skip_emp']:.1%}/{res['skip_rob0']:.1%}")
    return res


def year_diff_audit(tag: str, yc: dict, yb: dict) -> dict:
    """年次差分(候補−ベース)の単年依存 + 分割感度(偶奇年/前後半)監査。"""
    d = (pd.Series(yc) - pd.Series(yb)).dropna()
    total = float(d.sum())
    best_y = int(d.idxmax())
    excl_best = float(d.drop(best_y).sum())
    excl_2022 = float(d.drop(2022).sum()) if 2022 in d.index else total
    keep_share = excl_best / total if total > 0 else np.nan
    years = d.index.to_numpy()
    even = float(d[years % 2 == 0].sum())
    odd = float(d[years % 2 == 1].sum())
    mid = int(np.median(years))
    first = float(d[years <= mid].sum())
    second = float(d[years > mid].sum())
    return {"basis": tag, "total": total, "best_year": best_y,
            "best_year_diff": float(d[best_y]), "excl_best": excl_best,
            "excl_2022": excl_2022, "keep_share_excl_best": keep_share,
            "even_sum": even, "odd_sum": odd, "even_odd_same_sign": even * odd > 0,
            "half_split_year": mid, "first_half": first, "second_half": second,
            "half_same_sign": first * second > 0,
            "yearly": {int(y): float(v) for y, v in d.items()}}


def main() -> int:
    t0 = time.time()

    sec("0. プール検算 + edge01 整列検証 + ベースライン再現")
    pool = pd.read_parquet(POOL_PATH).reset_index(drop=True)
    closes = mm.load_closes()
    ok_pool = len(pool) == EXPECT_N and abs(pool["ret"].sum() - EXPECT_SUM) < 1e-3
    print(f"pool {len(pool)} trades  sum(ret)={pool['ret'].sum():+.4f} "
          f"(基準 n={EXPECT_N}, {EXPECT_SUM:+.4f}): {ok_pool}  grid {len(closes)} bars")
    tr = pd.read_csv(EDGE01_CSV, parse_dates=["entry", "exit"])
    aligned = (
        len(tr) == len(pool)
        and (tr["instr"].to_numpy() == pool["instr"].to_numpy()).all()
        and (pd.DatetimeIndex(tr["entry"]) == pd.DatetimeIndex(pool["entry"])).all()
        and np.allclose(tr["ret"].to_numpy(), pool["ret"].to_numpy())
    )
    mask = tr["spike_made_l5"].to_numpy().astype(bool)
    print(f"edge01 1:1 整列(instr/entry/ret): {aligned} / spike_made_l5 n={mask.sum()}")
    if not (ok_pool and aligned and mask.sum() == 98):
        print("!! 前提検証失敗。以降の比較は無効。")
        return 1

    sec("1. プールレベル断面(98件除外の差分)")
    exc, keep = pool[mask], pool[~mask]
    diff_pool = float(-exc["ret"].sum())  # 除外の利得 = -除外PnL
    r0, r1 = pool["ret"], keep["ret"]
    pf0 = float(r0[r0 > 0].sum() / abs(r0[r0 < 0].sum()))
    pf1 = float(r1[r1 > 0].sum() / abs(r1[r1 < 0].sum()))
    yr_d = (-exc["ret"]).groupby(exc["exit"].dt.year).sum()
    is_m = (exc["entry"] < OOS_START).to_numpy()
    pool_cs = {
        "n_excluded": int(mask.sum()), "excluded_ret_sum": float(exc["ret"].sum()),
        "excluded_mean_bps": float(exc["ret"].mean() * 1e4),
        "n_keep": len(keep), "sum_keep": float(r1.sum()),
        "diff": diff_pool, "pf_base": pf0, "pf_variant": pf1,
        "mean_bps_base": float(r0.mean() * 1e4), "mean_bps_variant": float(r1.mean() * 1e4),
        "diff_is": float(-exc["ret"][is_m].sum()), "diff_oos": float(-exc["ret"][~is_m].sum()),
        "yearly_diff": {int(y): float(v) for y, v in yr_d.items()},
    }
    yd = pd.Series(pool_cs["yearly_diff"])
    best_y = int(yd.idxmax())
    pool_cs["best_year"] = best_y
    pool_cs["best_year_diff"] = float(yd[best_y])
    pool_cs["excl_best"] = float(yd.drop(best_y).sum())
    pool_cs["excl_2022"] = float(yd.drop(2022).sum()) if 2022 in yd.index else diff_pool
    ys = yd.index.to_numpy()
    pool_cs["even_sum"], pool_cs["odd_sum"] = float(yd[ys % 2 == 0].sum()), float(yd[ys % 2 == 1].sum())
    print(f"  除外 98件: PnL {pool_cs['excluded_ret_sum']:+.4f} (mean {pool_cs['excluded_mean_bps']:+.1f}bps)"
          f" -> 変種 n={pool_cs['n_keep']} sum={pool_cs['sum_keep']:+.4f}  diff={diff_pool:+.4f}")
    print(f"  PF {pf0:.3f} -> {pf1:.3f} / mean {pool_cs['mean_bps_base']:+.1f} -> "
          f"{pool_cs['mean_bps_variant']:+.1f} bps")
    print(f"  IS差 {pool_cs['diff_is']:+.4f} / OOS差 {pool_cs['diff_oos']:+.4f} | "
          f"最良年 {best_y}({pool_cs['best_year_diff']:+.4f}) 除外後 {pool_cs['excl_best']:+.4f} / "
          f"2022除外 {pool_cs['excl_2022']:+.4f} | 偶数年 {pool_cs['even_sum']:+.4f} / "
          f"奇数年 {pool_cs['odd_sum']:+.4f}")
    print("  年次差分: " + "  ".join(f"{int(y)}:{v:+.4f}" for y, v in yd.items()))

    variant_pool = keep.reset_index(drop=True)

    sec(f"2. 口座シミュ: ベース + 変種 (seeds {SEEDS_FULL}, mp{MAX_POS}, P=4.0)")
    cfg_b = CfgEval("base_v2d1", pool, closes)
    cfg_v = CfgEval("l5cross_veto", variant_pool, closes)
    base = evaluate(cfg_b, seeds=SEEDS_FULL)
    print(f"    [{time.time()-t0:.0f}s]")
    var = evaluate(cfg_v, seeds=SEEDS_FULL)
    print(f"    [{time.time()-t0:.0f}s]")

    rep_ok = (abs(base["rob_cagr_mean"] - REF_ROB_MEAN) < 0.005
              and abs(base["emp_cagr"] - REF_EMP_CAGR) < 0.005)
    print(f"\n  ベース再現: rob_mean {base['rob_cagr_mean']:+.2%} (基準 {REF_ROB_MEAN:+.2%}) / "
          f"emp {base['emp_cagr']:+.2%} k={base['emp_k']:.3f} (基準 {REF_EMP_CAGR:+.2%} k≈8.895) / "
          f"rob_s0 k={base['rob'][0]['k']:.3f} {base['rob'][0]['cagr']:+.2%} "
          f"(基準 k={REF_ROB_S0_K} {REF_ROB_S0_CAGR:+.2%}) -> {'OK' if rep_ok else 'NG'}")

    print("\n--- ペアシード robust(p95=20%) CAGR 比較(同一シード集合) ---")
    print("  seed |   base    |  variant   (diff)")
    for sd in SEEDS_FULL:
        b, v = base["rob"][sd]["cagr"], var["rob"][sd]["cagr"]
        print(f"   s{sd}  | {b:+.2%}  | {v:+.2%}  ({(v-b)*100:+.2f}pp)  "
              f"k {base['rob'][sd]['k']:.3f}->{var['rob'][sd]['k']:.3f}")
    rob_gain = var["rob_cagr_mean"] - base["rob_cagr_mean"]
    rob_gain_rel = rob_gain / base["rob_cagr_mean"]
    print(f"  mean | {base['rob_cagr_mean']:+.2%}  | {var['rob_cagr_mean']:+.2%}  "
          f"({rob_gain*100:+.2f}pp, 相対 {rob_gain_rel:+.1%})")
    print(f"  empirical 20%: base {base['emp_cagr']:+.2%} (k={base['emp_k']:.3f}, "
          f"p95 {base['emp_p95']:+.1%}) | variant {var['emp_cagr']:+.2%} "
          f"(k={var['emp_k']:.3f}, p95 {var['emp_p95']:+.1%})")

    sec("3. ゲート判定(G2署名 + ブートシード監査 / G3 OOS / G4全年+ / G5単年依存+分割感度)")
    sig = (var["emp_cagr"] > base["emp_cagr"]) and \
          (abs(var["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
    # 署名ブートシード監査(emp_k 固定, n_boot=1500, seeds 0-2)
    p95_b = {sd: boot_dd(cfg_b.eq_of_k(base["emp_k"]), n_boot=1500, seed=sd)["p95"]
             for sd in (0, 1, 2)}
    p95_v = {sd: boot_dd(cfg_v.eq_of_k(var["emp_k"]), n_boot=1500, seed=sd)["p95"]
             for sd in (0, 1, 2)}
    emp_up = var["emp_cagr"] > base["emp_cagr"]
    sigs = {sd: emp_up and (abs(p95_v[sd]) > abs(p95_b[sd]) + 0.005) for sd in p95_b}
    print("  p95(emp_k固定): base " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95_b.items()))
    print("                  var  " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95_v.items()) +
          f"   署名: {sum(sigs.values())}/3 シード")

    a_emp = year_diff_audit("emp", var["yr_emp"], base["yr_emp"])
    a_rob = year_diff_audit("rob0", var["yr_rob0"], base["yr_rob0"])
    for tag, a in [("emp較正", a_emp), ("rob_s0較正", a_rob)]:
        print(f"\n  {tag}: 年次差分合計 {a['total']:+.2%} / 最良年 {a['best_year']}"
              f"({a['best_year_diff']:+.2%}) 除外後 {a['excl_best']:+.2%} "
              f"(残存率 {a['keep_share_excl_best'] if np.isfinite(a['keep_share_excl_best']) else float('nan'):.0%}) "
              f"/ 2022除外 {a['excl_2022']:+.2%}")
        print(f"    分割感度: 偶数年 {a['even_sum']:+.2%} / 奇数年 {a['odd_sum']:+.2%} "
              f"(同符号 {a['even_odd_same_sign']}) | ~{a['half_split_year']} {a['first_half']:+.2%} / "
              f"以降 {a['second_half']:+.2%} (同符号 {a['half_same_sign']})")
        print("    年次: " + "  ".join(f"{y}:{v*100:+.2f}pp" for y, v in a["yearly"].items()))

    g5_pool = (diff_pool > 0 and pool_cs["excl_best"] >= 0.5 * diff_pool
               and pool_cs["excl_2022"] > 0)
    g = {
        "g0_base_repro": bool(rep_ok),
        "g1_rob_gain_pp": rob_gain * 100, "g1_rob_gain_rel": rob_gain_rel,
        "g1_above_noise": abs(rob_gain) >= 0.004,  # 較正ノイズ±0.4-0.8pp
        "g1_adopt_line": rob_gain_rel >= 0.10,
        "g2_no_signature": not sig, "g2_sig_seeds": int(sum(sigs.values())),
        "g3_oos_rob": var["oos_rob_cagr"] > base["oos_rob_cagr"],
        "g3_oos_emp": var["oos_emp_cagr"] > base["oos_emp_cagr"],
        "g4_all_years_pos": (var["neg_years_emp"] == 0) and (var["neg_years_rob0"] == 0),
        "g5_emp": (a_emp["total"] > 0 and a_emp["keep_share_excl_best"] >= 0.5
                   and a_emp["excl_2022"] > 0),
        "g5_rob0": (a_rob["total"] > 0 and a_rob["keep_share_excl_best"] >= 0.5
                    and a_rob["excl_2022"] > 0),
        "g5_pool": g5_pool,
        "g5_split_emp": bool(a_emp["even_odd_same_sign"] and a_emp["half_same_sign"]),
        "g5_split_rob0": bool(a_rob["even_odd_same_sign"] and a_rob["half_same_sign"]),
    }
    g["g3_oos"] = g["g3_oos_rob"] and g["g3_oos_emp"]
    g["g5_single_year"] = g["g5_emp"] and g["g5_rob0"] and g["g5_pool"]
    g["all_pass"] = (g["g1_adopt_line"] and g["g2_no_signature"] and g["g3_oos"]
                     and g["g4_all_years_pos"] and g["g5_single_year"]
                     and g["g5_split_emp"] and g["g5_split_rob0"])
    print(f"\n  G3 OOS: rob base {base['oos_rob_cagr']:+.2%} -> var {var['oos_rob_cagr']:+.2%} "
          f"({'pass' if g['g3_oos_rob'] else 'FAIL'}) | emp base {base['oos_emp_cagr']:+.2%} -> "
          f"var {var['oos_emp_cagr']:+.2%} ({'pass' if g['g3_oos_emp'] else 'FAIL'})")
    print(f"  G4 負け年: emp {var['neg_years_emp']} (worst {var['worst_year']:+.1%}) / "
          f"rob_s0 {var['neg_years_rob0']} (base: {base['neg_years_emp']}/{base['neg_years_rob0']}, "
          f"worst {base['worst_year']:+.1%})")
    print(f"  ゲート: G0再現={g['g0_base_repro']} G1noise超={g['g1_above_noise']} "
          f"G1adopt={g['g1_adopt_line']} G2署名なし={g['g2_no_signature']} "
          f"G3oos={g['g3_oos']} G4全年+={g['g4_all_years_pos']} "
          f"G5単年={g['g5_single_year']} G5分割(emp/rob0)={g['g5_split_emp']}/{g['g5_split_rob0']} "
          f"-> 総合 {'PASS' if g['all_pass'] else 'FAIL'}")

    sec("4. 複雑性コスト対効果(実装の正直な見積もり)")
    cap = 300_000  # ユーザーの実弾想定(¥30万)
    yen = rob_gain * cap
    print(f"  実装要件: 本番 bot に M1 フィード追加 + 各シグナルバー終了5分前に z(50) を"
          f"スナップショット再計算する状態管理。")
    print(f"  robust 利得 {rob_gain*100:+.2f}pp -> ¥{cap:,} 口座で年あたり ¥{yen:+,.0f}")
    print(f"  (reports/22 判例: 便益年¥1,200 に状態管理の複雑性が見合わない -> 不採用)")

    # --- 保存 -------------------------------------------------------------
    rows = []
    for lbl, r in [("base", base), ("l5cross_veto", var)]:
        rows.append({
            "label": lbl,
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"], "worst_year": r["worst_year"],
            "neg_years_emp": r["neg_years_emp"], "neg_years_rob0": r["neg_years_rob0"],
            **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS_FULL},
            **{f"rob_k{sd}": r["rob"][sd]["k"] for sd in SEEDS_FULL},
            "rob_mean": r["rob_cagr_mean"],
            "skip_emp": r["skip_emp"], "skip_rob0": r["skip_rob0"],
            "k_is_emp": r["k_is_emp"], "is_emp_cagr": r["is_emp_cagr"],
            "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
            "k_is_rob": r["k_is_rob"], "is_rob_cagr": r["is_rob_cagr"],
            "oos_rob_cagr": r["oos_rob_cagr"], "oos_rob_dd": r["oos_rob_dd"],
        })
    adf = pd.DataFrame(rows)
    adf.to_csv(OUT_ACC, index=False)
    payload = {
        "baseline": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                     for k, v in base.items()},
        "variant": {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                    for k, v in var.items()},
        "pool_cross_section": pool_cs,
        "year_audit": {"emp": a_emp, "rob0": a_rob},
        "sig_audit": {"p95_base": {str(s): v for s, v in p95_b.items()},
                      "p95_var": {str(s): v for s, v in p95_v.items()},
                      "n_sig": int(sum(sigs.values()))},
        "gates": g,
        "complexity": {"capital_jpy": cap, "rob_gain_pp": rob_gain * 100,
                       "yen_per_year": yen},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nsaved -> {OUT_ACC}\n      -> {OUT_JSON}")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print("\n=== 口座レベル最終表 ===")
        print(adf.to_string(index=False))
    print(f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
