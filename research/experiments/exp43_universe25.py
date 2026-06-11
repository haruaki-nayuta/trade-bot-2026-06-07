"""exp43: ユニバース25銘柄化 — 「その他6クロス」追加の同一テール判定(レバー検証3)。

背景(reports/15 gap_1 / anatomy_gap_1.py): 19銘柄ユニバースの除外9ペアのうち、
キャリー/リスクオンJPY3本(AUDJPY/NZDJPY/CADJPY)の除外は域外検証を通過した正当ルール。
残る6本(CHFJPY/NZDCHF/CADCHF/GBPCAD/GBPNZD/EURNZD)はプールレベルで合計+0.1403(+7.4%)の
取り逃しと実測済み。境界ルールは事前登録:「キャリー/リスクオンJPYクロスのみ除外、
他は全クロス採用」= in-sample 選別をしない。

検証内容:
  1) 25銘柄プール構築(専用キャッシュ) + 19銘柄部分が参照プールと一致する検算
  2) プールレベル: 新6銘柄の n/sum/PF/年次、6pips(2倍)ストレス生存
  3) 口座レベル: champion_sizing(P=4) × mp8/mp11、ペアシード(0-4) robust + empirical
  4) IS(<2022)較正→OOS素検証(empirical/robust双方)、IS-argmax 監査
  5) 2022除外チェック(改善幅の年次分解)・全年プラス維持(2018含む)
  6) レバ偽装署名(emp CAGR↑ + p95悪化)チェック、新6銘柄のmm層実寄与(除外シミュ差分)

判定: robust 5シード平均 +0.5pp 以上 + 全チェック通過で adopt。

実行: PYTHONPATH=. uv run python research/experiments/exp43_universe25.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    boot_dd,
    cagr_of,
    calibrate_empirical,
    calibrate_robust_seeded,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab import config, universe as uni  # noqa: E402

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 40)

TF = "H4"
SEEDS = (0, 1, 2, 3, 4)
CROSS_SPREAD = 3.0
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
POOL19_PATH = config.RESULTS_DIR / "mm_pool_v2_H4_19.parquet"
OUT_DIR = ROOT / "research" / "outputs"
POOL25_CACHE = OUT_DIR / "exp43_pool25.parquet"
OUT_JSON = OUT_DIR / "exp43_universe25.json"
OUT_CSV = OUT_DIR / "exp43_universe25.csv"

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

# 事前登録ルール: キャリー/リスクオンJPY(AUDJPY/NZDJPY/CADJPY)のみ除外。他は全クロス採用。
# 6新クロスは USD建てメジャーから合成(anatomy_gap_1 と同一規約・同一クォート方向)。
EXTRA6: dict[str, tuple[str, str, str]] = {
    "CHFJPY": ("USDJPY", "/", "USDCHF"),
    "NZDCHF": ("NZDUSD", "*", "USDCHF"),
    "CADCHF": ("USDCHF", "/", "USDCAD"),
    "GBPCAD": ("GBPUSD", "*", "USDCAD"),
    "GBPNZD": ("GBPUSD", "/", "NZDUSD"),
    "EURNZD": ("EURUSD", "/", "NZDUSD"),
}
NEW6 = list(EXTRA6)


# --- ユーティリティ -------------------------------------------------------
def pool_stats(pool: pd.DataFrame, label: str) -> dict:
    r = pool["ret"]
    gp, gl = r[r > 0].sum(), -r[r < 0].sum()
    y = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    return {
        "group": label, "n": len(r), "sum_ret": round(float(r.sum()), 4),
        "mean_bps": round(float(r.mean() * 1e4), 1) if len(r) else np.nan,
        "win_%": round(float((r > 0).mean() * 100), 1) if len(r) else np.nan,
        "PF": round(float(gp / gl), 3) if gl > 0 else np.inf,
        "pos_years": f"{int((y > 0).sum())}/{len(y)}",
    }


def load_closes_local(instruments: list[str]) -> pd.DataFrame:
    """mm_lab.load_closes と同一規約(results/ のキャッシュには触れない)。"""
    closes = pd.DataFrame({nm: uni.instrument_close(nm, TF) for nm in instruments})
    return closes.sort_index().ffill()


def eval_config(label: str, pool: pd.DataFrame, closes: pd.DataFrame, max_pos: int,
                seeds=SEEDS) -> dict:
    """ベースライン/候補を同一手順でフル評価(empirical + ペアシード robust + IS/OOS)。"""
    mk = champion_sizing(pool, max_pos=max_pos)
    cache: dict[float, pd.Series] = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, mk(kk), max_pos=max_pos)
            cache[kk] = eqm
        return cache[kk]

    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    eq_emp = eq_of_k(res["emp_k"])
    res["yearly_emp"] = yearly_returns(eq_emp)
    eq_rob = eq_of_k(res["rob_k_mean"])
    res["yearly_rob"] = yearly_returns(eq_rob)
    res["rob_dd"] = max_dd(eq_rob)
    res["worst_year_emp"] = float(res["yearly_emp"].min())
    res["neg_years_emp"] = int((res["yearly_emp"] < 0).sum())
    res["neg_years_rob"] = int((res["yearly_rob"] < 0).sum())

    # IS(<2022) 較正 → OOS 素検証(empirical / robust seed0 の両較正)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    is_cache: dict[float, pd.Series] = {}

    def eq_is(k):
        kk = round(float(k), 10)
        if kk not in is_cache:
            eqm, _, _ = mm.simulate(is_pool, is_cl, mk(kk), max_pos=max_pos)
            is_cache[kk] = eqm
        return is_cache[kk]

    k_is_emp = calibrate_empirical(eq_is, 0.20)
    k_is_rob = calibrate_robust_seeded(eq_is, 0.20, seed=0)
    res["k_is_emp"], res["k_is_rob"] = k_is_emp, k_is_rob
    res["is_emp_cagr"] = cagr_of(eq_is(k_is_emp))
    res["is_rob_cagr"] = cagr_of(eq_is(k_is_rob))
    for tag, k_is in [("emp", k_is_emp), ("rob", k_is_rob)]:
        eqo, _, _ = mm.simulate(oos_pool, oos_cl, mk(k_is), max_pos=max_pos)
        res[f"oos_{tag}_cagr"] = cagr_of(eqo)
        res[f"oos_{tag}_dd"] = max_dd(eqo)
    print(f"      IS emp k={k_is_emp:5.2f} (IS CAGR={res['is_emp_cagr']:+.2%}) -> "
          f"OOS CAGR={res['oos_emp_cagr']:+7.2%} DD={res['oos_emp_dd']:+6.1%}")
    print(f"      IS rob k={k_is_rob:5.2f} (IS CAGR={res['is_rob_cagr']:+.2%}) -> "
          f"OOS CAGR={res['oos_rob_cagr']:+7.2%} DD={res['oos_rob_dd']:+6.1%}")
    res["_mk"] = mk
    res["_eq_of_k"] = eq_of_k
    return res


def main() -> int:
    # --- セットアップ: CROSS_DEFS 実行時拡張(プロセス内のみ)+ 3pips 登録 ---
    uni.CROSS_DEFS.update(EXTRA6)
    uni.register_cross_spreads(CROSS_SPREAD)

    ref = pd.read_parquet(POOL19_PATH)
    instr19 = MAJORS + [c for c in uni.CROSS_DEFS if c != "AUDJPY" and c not in EXTRA6]
    instr25 = instr19 + NEW6
    assert sorted(set(ref["instr"])) == sorted(instr19), "参照プールの銘柄集合が想定と不一致"
    print(f"instr19={len(instr19)} instr25={len(instr25)} 新規={NEW6}")

    # ========== 1) 25銘柄プール構築 + 19銘柄部分の検算 ==========
    print("\n" + "=" * 90)
    print("[1] 25銘柄プール構築(専用キャッシュ) + 19銘柄部分 vs 参照プール検算")
    print("=" * 90)
    if POOL25_CACHE.exists():
        pool25 = pd.read_parquet(POOL25_CACHE)
        print(f"キャッシュ読込: {POOL25_CACHE.name}")
    else:
        pool25 = mm.build_pool(tf=TF, instruments=instr25, cross_spread=CROSS_SPREAD, cache=False)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pool25.to_parquet(POOL25_CACHE)
    pool19 = ref  # ベースラインは参照プールそのもの
    sub19 = pool25[~pool25["instr"].isin(NEW6)]
    m = ref.merge(sub19, on=["instr", "entry"], suffixes=("_ref", "_new"), how="outer",
                  indicator=True)
    n_unmatched = int((m["_merge"] != "both").sum())
    max_dret = float((m.loc[m._merge == "both", "ret_ref"]
                      - m.loc[m._merge == "both", "ret_new"]).abs().max())
    ok = (len(ref) == len(sub19)) and n_unmatched == 0 and max_dret < 1e-9
    print(f"参照 n={len(ref)} sum={ref['ret'].sum():+.4f} / 再生19部分 n={len(sub19)} "
          f"sum={sub19['ret'].sum():+.4f} / 不一致行={n_unmatched} / 最大|Δret|={max_dret:.2e}")
    print(f"検算: {'PASS(完全一致)' if ok else 'FAIL'}")
    verify_pool = ok

    # ========== 2) プールレベル: 新6銘柄の成績 + 6pips ストレス ==========
    print("\n" + "=" * 90)
    print("[2] プールレベル: 新6銘柄 個別成績(3pips) / 年次 / 6pips(2倍)ストレス")
    print("=" * 90)
    pool_new6 = pool25[pool25["instr"].isin(NEW6)].reset_index(drop=True)
    rows = [pool_stats(pool_new6[pool_new6["instr"] == nm], nm) for nm in NEW6]
    rows.append(pool_stats(pool_new6, "新6合計"))
    rows.append(pool_stats(pool19, "既存19合計"))
    rows.append(pool_stats(pool25, "25合計"))
    tbl_new6 = pd.DataFrame(rows)
    print(tbl_new6.to_string(index=False))

    ymat = pool_new6.assign(year=pd.to_datetime(pool_new6["exit"]).dt.year) \
        .pivot_table(index="year", columns="instr", values="ret", aggfunc="sum").round(4)
    print("\n新6 × 決済年 sum(ret):")
    print(ymat.reindex(columns=NEW6).to_string())
    y25 = pool25.groupby(pd.to_datetime(pool25["exit"]).dt.year)["ret"].sum()
    print("\n25銘柄プール年次 sum(ret):", {int(k): round(float(v), 4) for k, v in y25.items()})
    print(f"25銘柄プール: 全暦年プラス = {bool((y25 > 0).all())}")

    # 6pips ストレス(新6のみ再生成)
    pool_new6_6p = mm.build_pool(tf=TF, instruments=NEW6, cross_spread=6.0, cache=False)
    uni.register_cross_spreads(CROSS_SPREAD)  # 3pips へ戻す
    rows6 = [pool_stats(pool_new6_6p[pool_new6_6p["instr"] == nm], nm) for nm in NEW6]
    rows6.append(pool_stats(pool_new6_6p, "新6合計@6pips"))
    tbl_6p = pd.DataFrame(rows6)
    print("\n6pips(2倍コスト)ストレス:")
    print(tbl_6p.to_string(index=False))
    stress_survive = float(pool_new6_6p["ret"].sum()) > 0

    # ========== 3) 口座レベル: ペアシード protocol_eval(mp8 / mp11) ==========
    print("\n" + "=" * 90)
    print(f"[3] 口座レベル: champion_sizing(P=4) ペアシード seeds={SEEDS}")
    print("=" * 90)
    closes19 = load_closes_local(instr19)
    closes25 = load_closes_local(instr25)
    same_grid = closes19.index.equals(closes25.index)
    print(f"グリッド: 19={len(closes19)}本 / 25={len(closes25)}本 / 同一index={same_grid}")

    results: dict[str, dict] = {}
    for mp in (8, 11):
        for tag, pool, closes in [(f"base19_mp{mp}", pool19, closes19),
                                  (f"univ25_mp{mp}", pool25, closes25)]:
            print(f"\n--- {tag} ---")
            results[tag] = eval_config(tag, pool, closes, max_pos=mp)

    # ペアシード比較表
    print("\n" + "=" * 90)
    print("[3b] ペアシード比較(robust p95=20% CAGR, per-seed)")
    print("=" * 90)
    cmp_rows = []
    for mp in (8, 11):
        b, c = results[f"base19_mp{mp}"], results[f"univ25_mp{mp}"]
        row = {"mp": mp}
        for sd in SEEDS:
            row[f"s{sd}_base"] = b["rob"][sd]["cagr"]
            row[f"s{sd}_univ25"] = c["rob"][sd]["cagr"]
            row[f"s{sd}_diff"] = c["rob"][sd]["cagr"] - b["rob"][sd]["cagr"]
        row["rob_mean_base"] = b["rob_cagr_mean"]
        row["rob_mean_univ25"] = c["rob_cagr_mean"]
        row["rob_mean_diff"] = c["rob_cagr_mean"] - b["rob_cagr_mean"]
        row["emp_base"] = b["emp_cagr"]
        row["emp_univ25"] = c["emp_cagr"]
        row["emp_diff"] = c["emp_cagr"] - b["emp_cagr"]
        row["p95_base"] = b["emp_p95"]
        row["p95_univ25"] = c["emp_p95"]
        cmp_rows.append(row)
        print(f"mp{mp}: robust mean {b['rob_cagr_mean']:+.2%} -> {c['rob_cagr_mean']:+.2%} "
              f"(diff {row['rob_mean_diff']:+.2%}) | per-seed diff "
              + " ".join(f"s{sd}:{row[f's{sd}_diff']:+.2%}" for sd in SEEDS))
        print(f"      empirical {b['emp_cagr']:+.2%} -> {c['emp_cagr']:+.2%} "
              f"(diff {row['emp_diff']:+.2%}) | emp_p95 {b['emp_p95']:+.1%} -> {c['emp_p95']:+.1%}")
    cmp_df = pd.DataFrame(cmp_rows)

    # ========== 4) IS-argmax 監査 / 2022除外 / 全年プラス ==========
    print("\n" + "=" * 90)
    print("[4] IS-argmax 監査(IS<2022 のみで 19 vs 25 を選ぶと?) + 年次分解")
    print("=" * 90)
    b8, c8 = results["base19_mp8"], results["univ25_mp8"]
    is_pick = "univ25" if c8["is_rob_cagr"] > b8["is_rob_cagr"] else "base19"
    print(f"IS robust CAGR: base19={b8['is_rob_cagr']:+.2%} univ25={c8['is_rob_cagr']:+.2%} "
          f"-> IS選択 = {is_pick}")
    print(f"IS emp CAGR  : base19={b8['is_emp_cagr']:+.2%} univ25={c8['is_emp_cagr']:+.2%}")
    print(f"OOS(rob較正k): base19={b8['oos_rob_cagr']:+.2%} univ25={c8['oos_rob_cagr']:+.2%} "
          f"(diff {c8['oos_rob_cagr']-b8['oos_rob_cagr']:+.2%})")
    print(f"OOS(emp較正k): base19={b8['oos_emp_cagr']:+.2%} univ25={c8['oos_emp_cagr']:+.2%} "
          f"(diff {c8['oos_emp_cagr']-b8['oos_emp_cagr']:+.2%})")

    # 年次分解(empirical較正 と robust mean-k 較正の両方)
    print("\n年次リターン(mp8):")
    ytab = pd.DataFrame({
        "base19_emp": b8["yearly_emp"], "univ25_emp": c8["yearly_emp"],
        "diff_emp": c8["yearly_emp"] - b8["yearly_emp"],
        "base19_rob": b8["yearly_rob"], "univ25_rob": c8["yearly_rob"],
        "diff_rob": c8["yearly_rob"] - b8["yearly_rob"],
    })
    print((ytab * 100).round(2).to_string())
    for col in ("diff_emp", "diff_rob"):
        d = ytab[col].dropna()
        best_y = int(d.idxmax())
        excl_best = d.drop(best_y)
        excl_2022 = d.drop(2022) if 2022 in d.index else d
        print(f"{col}: 合計={d.sum()*100:+.2f}pp / 最良年={best_y}({d.max()*100:+.2f}pp) / "
              f"最良年除外合計={excl_best.sum()*100:+.2f}pp / 2022除外合計={excl_2022.sum()*100:+.2f}pp")
    excl_ok = bool((ytab["diff_rob"].dropna().drop(int(ytab["diff_rob"].dropna().idxmax()))
                    .sum()) > 0)

    print(f"\n全年プラス維持(mp8): base19 emp負け年={b8['neg_years_emp']} rob負け年={b8['neg_years_rob']} / "
          f"univ25 emp負け年={c8['neg_years_emp']} rob負け年={c8['neg_years_rob']}")
    y2018_emp = float(c8["yearly_emp"].get(2018, np.nan))
    y2018_rob = float(c8["yearly_rob"].get(2018, np.nan))
    print(f"univ25 mp8 2018年(プール最弱年): emp={y2018_emp:+.2%} rob={y2018_rob:+.2%}")

    # ========== 5) レバ偽装署名 + 新6のmm層実寄与 ==========
    print("\n" + "=" * 90)
    print("[5] レバ偽装署名チェック + 新6銘柄のmm層実寄与(除外シミュ差分)")
    print("=" * 90)
    tail_flags = {}
    for mp in (8, 11):
        b, c = results[f"base19_mp{mp}"], results[f"univ25_mp{mp}"]
        emp_up = c["emp_cagr"] > b["emp_cagr"]
        p95_worse = c["emp_p95"] < b["emp_p95"] - 0.005  # 0.5pp超の悪化を「悪化」とみなす
        flag = emp_up and p95_worse
        tail_flags[mp] = flag
        print(f"mp{mp}: emp CAGR {'↑' if emp_up else '↓'} ({b['emp_cagr']:+.2%}->{c['emp_cagr']:+.2%}), "
              f"p95 {b['emp_p95']:+.1%}->{c['emp_p95']:+.1%} "
              f"=> レバ偽装署名 {'検出!' if flag else 'なし'}")
    tail_ok = not any(tail_flags.values())

    # 新6の実寄与: univ25 mp8 の emp_k を固定し、新6トレードを除いた同一シミュとの差分
    mk25 = c8["_mk"]
    k_fix = c8["emp_k"]
    eq_full, _, info_full = mm.simulate(pool25, closes25, mk25(k_fix), max_pos=8)
    pool25_no6 = pool25[~pool25["instr"].isin(NEW6)].reset_index(drop=True)
    eq_no6, _, info_no6 = mm.simulate(pool25_no6, closes25, mk25(k_fix), max_pos=8)
    contrib = cagr_of(eq_full) - cagr_of(eq_no6)
    print(f"\n新6実寄与(mp8, k={k_fix:.2f}固定): full CAGR={cagr_of(eq_full):+.2%} "
          f"(DD={max_dd(eq_full):+.1%}, taken={info_full['n_taken']}, skip={info_full['skipped']}) "
          f"vs 新6除外 CAGR={cagr_of(eq_no6):+.2%} (DD={max_dd(eq_no6):+.1%}, "
          f"taken={info_no6['n_taken']}, skip={info_no6['skipped']}) => 寄与 {contrib:+.2%}")

    # ========== 6) 判定 ==========
    print("\n" + "=" * 90)
    print("[6] 判定サマリ")
    print("=" * 90)
    rob_diff_mp8 = c8["rob_cagr_mean"] - b8["rob_cagr_mean"]
    checks = {
        "pool19検算PASS": verify_pool,
        "robust平均diff>=+0.5pp(mp8)": rob_diff_mp8 >= 0.005,
        "per-seed全シードでdiff>0(mp8)": all(
            c8["rob"][sd]["cagr"] > b8["rob"][sd]["cagr"] for sd in SEEDS),
        "レバ偽装署名なし": tail_ok,
        "IS選択がOOSで持続": (is_pick == "univ25"
                              and c8["oos_rob_cagr"] > b8["oos_rob_cagr"]),
        "最良年除外でも改善符号維持(rob)": excl_ok,
        "負け年が増えない(emp/rob)": (c8["neg_years_emp"] <= b8["neg_years_emp"]
                                      and c8["neg_years_rob"] <= b8["neg_years_rob"]),
        "新6 6pipsストレス生存": stress_survive,
    }
    for k_, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k_}")
    verdict = "adopt" if all(checks.values()) else "reject/inconclusive"
    print(f"\n=> 判定: {verdict} (robust mean diff mp8 = {rob_diff_mp8:+.2%})")

    # ========== 保存 ==========
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cmp_df.to_csv(OUT_CSV, index=False)
    dump = {}
    for lab, r in results.items():
        dump[lab] = {k_: v for k_, v in r.items()
                     if k_ not in ("rob", "yearly_emp", "yearly_rob", "_mk", "_eq_of_k")}
        dump[lab]["rob"] = {str(s): w for s, w in r["rob"].items()}
        dump[lab]["yearly_emp"] = {int(y): float(v) for y, v in r["yearly_emp"].items()}
        dump[lab]["yearly_rob"] = {int(y): float(v) for y, v in r["yearly_rob"].items()}
    dump["checks"] = {k_: bool(v) for k_, v in checks.items()}
    dump["verdict"] = verdict
    dump["new6_pool"] = tbl_new6.to_dict("records")
    dump["new6_pool_6pips"] = tbl_6p.to_dict("records")
    dump["new6_mm_contrib_mp8"] = float(contrib)
    OUT_JSON.write_text(json.dumps(dump, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n      -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
