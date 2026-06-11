"""exp37: シグナル品質グレード・サイジング — ER(40) 連続グレーディング × z-power。

仮説: エントリーフィルタ ER(40)≤0.55 は二値であり、ER が低い(=レンジ性が強い)ほど
平均回帰の質が高いなら、f(z)×g(ER) の連続グレーディングで同テールのまま利益密度が上がる。

事前診断(本ファイルが最初に出力): ER 五分位ごとのプール平均 ret/PF。
  → 実測は仮説と**逆勾配**(最高 ER 帯が最良 PF、中位が最悪の U 字)だったため、
    指示の低ER増し構成に加えて逆方向(高ER増し)も参考として計測する。
    ※逆方向は診断を見てから足した構成なので選択リークの疑いを併記する。

乗数候補(全てエントリー時 ER のみ参照=因果。clip[0.3,3.0]→プール平均で正規化):
  線形   g = 1 + a*(0.55-ER)/0.55, a∈{0.5,1.0,1.5}   (低ERを増す=仮説方向)
  凸     g = ((0.6-ER)/0.6)^q,     q∈{0.5,1.0}
  二値逆 g = 0.5 if ER>0.4 else 1.0
  逆線形 g = 1 + a*ER/0.55,        a∈{0.5,1.0}        (高ERを増す=診断方向, 参考)
  ついで(z形状の高原確認のみ): CLIP_HI=4.0 / P=2.5

評価: mp11 固定。tail_protocol 2段階(全構成 empirical+robust seed0 → 上位のみ seeds1,2)
+ IS(<2022)較正→OOS素検証。per-trade 乗数は exp30 と同じ (instr, round(ret,12), bars_held)
キーで ctx に紐付け(衝突数を報告)。

実行: PYTHONPATH=. uv run python research/experiments/exp37_quality_sizing.py
      PYTHONPATH=. uv run python research/experiments/exp37_quality_sizing.py --zshape
        (ついで枠の深掘り: z-power P∈[2,5]+step極限+z0近傍の高原マップ → exp37_zshape_plateau.csv)

結論(2026-06-11 実行):
  ER グレーディング = reject。勾配は弱い正(仮説と逆)で IS/OOS 不安定(U字が反転)。
  低ER増し全構成は emp CAGR↑ + p95 悪化(-30.1〜-33.3% vs -29.4%)=レバ偽装署名、
  robust はベースライン以下。高ER増しは単純に CAGR 減。
  ついで枠の z-power P が本物の改善方向: P=4.0 で robust mean3 +17.21%(基準 +14.98%)、
  p95 フラット(-29.2%)、OOS +38.1%(基準 +32.5%)、3シード全て>+16.8%、z0 近傍も高原。
  ※ P>4.5 は全期間ではまだ伸びるが OOS は P3.5-4.0 がピーク、step 極限は最悪年がマイナス化
    (-5.4%)するため P=4.0 を高原中央とみなす。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import _fz, champion_sizing  # noqa: E402
from tail_protocol import (  # noqa: E402
    calibrate_empirical,
    cagr_of,
    max_dd,
    protocol_eval,
    yearly_returns,
)
from fxlab import universe as uni  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAX_POS = 11
STAGE2_TH = 0.156  # robust seed0 がこれ未満なら seeds1,2 は省略
OUT_JSON = ROOT / "research" / "outputs" / "exp37_quality_sizing.json"
OUT_CSV = ROOT / "research" / "outputs" / "exp37_quality_sizing.csv"
OUT_DIAG = ROOT / "research" / "outputs" / "exp37_er_quintiles.csv"

Z0, P_BASE, CLIP_LO, CLIP_HI = 2.2, 2.0, 0.3, 3.0


# --- ER(40) at entry ------------------------------------------------------
def entry_er(pool: pd.DataFrame, win=40) -> np.ndarray:
    """各トレードのエントリー時 Kaufman ER(40)。終値のみ・asof(ffill)=因果。"""
    uni.register_cross_spreads(3.0)
    out = np.full(len(pool), np.nan)
    for nm, idx in pool.groupby("instr").groups.items():
        c = uni.instrument_data(nm, "H4")["close"]
        er = (c - c.shift(win)).abs() / c.diff().abs().rolling(win).sum()
        ii = np.asarray(idx)
        out[ii] = er.reindex(pd.to_datetime(pool["entry"].iloc[ii]), method="ffill").to_numpy()
    return out


def quintile_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    def pf(x):
        g = x[x > 0].sum()
        l = -x[x <= 0].sum()
        return g / l if l > 0 else np.inf
    d = df.copy()
    d["q"] = pd.qcut(d["er"], 5, labels=False)
    tab = d.groupby("q").agg(n=("ret", "size"), er_lo=("er", "min"), er_hi=("er", "max"),
                             mean_ret=("ret", "mean"), med_ret=("ret", "median"),
                             win=("ret", lambda x: (x > 0).mean()), PF=("ret", pf),
                             mean_z=("z_entry", "mean"))
    tab.insert(0, "sample", label)
    return tab


# --- 乗数 → サイジング(per-trade キーハック) ---------------------------
def graded_sizing_factory(pool, mult: np.ndarray, max_pos=MAX_POS, fz=_fz):
    """champion z-power × per-trade 乗数 mult(正規化済)。fz 差替で z 形状変種も作る。"""
    fbar = float(np.mean([fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0
    key = {}
    instr = pool["instr"].to_numpy()
    ret = pool["ret"].to_numpy()
    bh = pool["bars_held"].to_numpy()
    for i in range(len(pool)):
        key[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = float(mult[i])

    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            gm = key.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), 1.0)
            return ctx["equity_real"] * base * gm * (fz(ctx["z"]) / fbar)
        return sizing
    return make_sizing


def normalize_mult(g_raw: np.ndarray) -> np.ndarray:
    """nan→1 → clip[0.3,3.0] → プール平均=1 に正規化(k 較正と直交)。"""
    g = np.where(np.isfinite(g_raw), g_raw, 1.0)
    g = np.clip(g, CLIP_LO, CLIP_HI)
    return g / g.mean()


def fz_variant(z0=Z0, p=P_BASE, lo=CLIP_LO, hi=CLIP_HI):
    def f(z):
        return float(np.clip((z / z0) ** p, lo, hi)) if np.isfinite(z) else 1.0
    return f


# --- 評価 -----------------------------------------------------------------
def eval_config(label, pool, closes, make_sizing, seeds=(0,)) -> dict:
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            eqm, _, _ = mm.simulate(pool, closes, make_sizing(kk), max_pos=MAX_POS)
            cache[kk] = eqm
        return cache[kk]

    res = protocol_eval(eq_of_k, label=label, seeds=seeds)
    eq_emp = eq_of_k(res["emp_k"])
    res["worst_year"] = float(yearly_returns(eq_emp).min())

    # IS 較正 → OOS 素検証(empirical)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]

    def eq_is(k):
        eqm, _, _ = mm.simulate(is_pool, is_cl, make_sizing(k), max_pos=MAX_POS)
        return eqm

    k_is = calibrate_empirical(eq_is, 0.20)
    eqo, _, _ = mm.simulate(oos_pool, oos_cl, make_sizing(k_is), max_pos=MAX_POS)
    res["k_is"] = k_is
    res["oos_emp_cagr"] = cagr_of(eqo)
    res["oos_emp_dd"] = max_dd(eqo)
    print(f"      IS k={k_is:5.2f} -> OOS CAGR={res['oos_emp_cagr']:+7.2%} DD={res['oos_emp_dd']:+6.1%}")
    return res


def zshape_main() -> int:
    """ついで枠の深掘り: z-power の P 方向 + step 極限 + z0 近傍の高原マップ(mean3)。"""
    pool = mm.build_pool()
    closes = mm.load_closes()
    ones = np.ones(len(pool))

    def fz_step(z):
        return 3.0 if (np.isfinite(z) and z >= 2.2) else 0.3

    rows = []
    cfgs = [(f"z_p{p}", fz_variant(p=p)) for p in [2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0]]
    cfgs += [("z_step_lim", fz_step), ("z0_2.0_p4", fz_variant(z0=2.0, p=4.0)),
             ("z0_2.4_p4", fz_variant(z0=2.4, p=4.0))]
    for tag, fz in cfgs:
        mk = graded_sizing_factory(pool, ones, fz=fz)
        r = eval_config(tag, pool, closes, mk, seeds=(0, 1, 2))
        rows.append({"cfg": tag, "emp_cagr": r["emp_cagr"], "emp_p95": r["emp_p95"],
                     "rob_mean3": r["rob_cagr_mean"], "worst_year": r["worst_year"],
                     "oos_emp_cagr": r["oos_emp_cagr"]})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "research" / "outputs" / "exp37_zshape_plateau.csv", index=False)
    print(df.round(4).to_string(index=False))
    return 0


def main() -> int:
    if "--zshape" in sys.argv:
        return zshape_main()
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {len(pool)} trades / grid {len(closes)} bars / max_pos={MAX_POS}")

    # per-trade キー衝突チェック
    keys = list(zip(pool["instr"], pool["ret"].round(12), pool["bars_held"]))
    n_dup = len(keys) - len(set(keys))
    print(f"per-trade キー衝突: {n_dup} 件")

    # --- 診断: ER 五分位のエッジ勾配 ------------------------------------
    er = entry_er(pool)
    df = pool.copy()
    df["er"] = er
    tabs = [quintile_table(df, "full"),
            quintile_table(df[df["entry"] < OOS_START], "IS"),
            quintile_table(df[df["entry"] >= OOS_START], "OOS")]
    diag = pd.concat(tabs)
    print("\n=== 診断: ER(40) 五分位 × プール成績(勾配が仮説方向か?) ===")
    print(diag.round(4).to_string())
    sp = df["er"].corr(df["ret"], method="spearman")
    print(f"spearman(ER, ret) = {sp:+.4f}  (仮説は負を予測、正なら逆勾配)")
    OUT_DIAG.parent.mkdir(parents=True, exist_ok=True)
    diag.to_csv(OUT_DIAG)

    # --- 構成定義 ---------------------------------------------------------
    configs: list[tuple[str, object, str]] = []  # (label, make_sizing, notes)
    configs.append(("baseline_mp11", champion_sizing(pool, max_pos=MAX_POS), "基準"))
    for a in [0.5, 1.0, 1.5]:
        g = normalize_mult(1 + a * (0.55 - er) / 0.55)
        configs.append((f"lin_a{a}", graded_sizing_factory(pool, g), "線形・低ER増し(仮説方向)"))
    for q in [0.5, 1.0]:
        g = normalize_mult(np.power(np.clip((0.6 - er) / 0.6, 0.0, None), q))
        configs.append((f"cvx_q{q}", graded_sizing_factory(pool, g), "凸・低ER増し"))
    g = normalize_mult(np.where(er > 0.4, 0.5, 1.0))
    configs.append(("bin_er0.4_half", graded_sizing_factory(pool, g), "二値逆(ER>0.4半減)"))
    for a in [0.5, 1.0]:
        g = normalize_mult(1 + a * er / 0.55)
        configs.append((f"rev_a{a}", graded_sizing_factory(pool, g),
                        "逆線形・高ER増し(診断方向=選択リーク疑い併記)"))
    ones = np.ones(len(pool))
    configs.append(("z_clip4.0", graded_sizing_factory(pool, ones, fz=fz_variant(hi=4.0)),
                    "z形状高原確認(CLIP_HI=4)"))
    configs.append(("z_p2.5", graded_sizing_factory(pool, ones, fz=fz_variant(p=2.5)),
                    "z形状高原確認(P=2.5)"))

    # --- ステージ1: 全構成 empirical + robust seed0 ------------------------
    print("\n=== ステージ1: empirical較正 + robust seed0 + IS/OOS ===")
    results = {}
    for label, mk, notes in configs:
        res = eval_config(label, pool, closes, mk, seeds=(0,))
        res["notes"] = notes
        results[label] = res
        OUT_JSON.write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk != "rob"} |
                {"rob": {str(s): w for s, w in v["rob"].items()}}
             for k, v in results.items()}, indent=2, default=float))

    # --- ステージ2: robust seed0 上位 ≤3 に seeds 1,2 ----------------------
    cand = [(lab, r) for lab, r in results.items()
            if lab != "baseline_mp11" and r["rob"][0]["cagr"] >= STAGE2_TH]
    cand.sort(key=lambda x: -x[1]["rob"][0]["cagr"])
    cand = cand[:3]
    print(f"\n=== ステージ2: seeds(1,2) 追加対象 = {[c[0] for c in cand] or 'なし(早期終了)'} ===")
    mk_of = {label: mk for label, mk, _ in configs}
    for label, _ in cand:
        res = eval_config(label + "_mean3", pool, closes, mk_of[label], seeds=(0, 1, 2))
        results[label]["rob"] = res["rob"]
        results[label]["rob_cagr_mean"] = res["rob_cagr_mean"]
        results[label]["mean3"] = True

    # ベースラインも(候補があれば)同条件 mean3 を出してペアシード比較
    if cand:
        res = eval_config("baseline_mp11_mean3", pool, closes, mk_of["baseline_mp11"],
                          seeds=(0, 1, 2))
        results["baseline_mp11"]["rob"] = res["rob"]
        results["baseline_mp11"]["rob_cagr_mean"] = res["rob_cagr_mean"]
        results["baseline_mp11"]["mean3"] = True

    # --- 保存 ---------------------------------------------------------------
    rows = []
    for lab, r in results.items():
        rows.append({
            "label": lab, "notes": r.get("notes", ""),
            "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"], "emp_dd": r["emp_dd"],
            "emp_p95": r["emp_p95"], "worst_year": r["worst_year"],
            "rob_s0": r["rob"][0]["cagr"],
            "rob_s1": r["rob"].get(1, {}).get("cagr"),
            "rob_s2": r["rob"].get(2, {}).get("cagr"),
            "rob_mean": r.get("rob_cagr_mean") if r.get("mean3") else None,
            "k_is": r["k_is"], "oos_emp_cagr": r["oos_emp_cagr"], "oos_emp_dd": r["oos_emp_dd"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items() if kk != "rob"} |
            {"rob": {str(s): w for s, w in v["rob"].items()}}
         for k, v in results.items()}, indent=2, default=float))
    print(f"\nsaved -> {OUT_CSV}\n        -> {OUT_JSON}")
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
