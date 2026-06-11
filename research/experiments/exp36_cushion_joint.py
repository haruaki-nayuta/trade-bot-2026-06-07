"""exp36: DD連動トレンド・クッション × max_pos の連立最適化 + USDゲート複合。

背景(reports/10 §5, reports/11 exp30):
  ・ADXトレンド(short 30/100)を「チャンピオンが dd_mtm < -gate のDD中だけ」発火させる
    DD連動オーバーレイは CAGR中立のまま p95 を約5pp削減した。ただし当時は mp 固定でしか測っていない。
  ・仮説: テールクッションで高 mp(13-15)+高 k 領域が robust 較正で解禁され、
    連立(overlay × mp × k)に非自明な前進があるかもしれない。
  ・さらに exp30 の USDファクター・ゲート(usd-only)を champ エントリーにだけ重ねた複合も実測。

判定 = 同一テール判定プロトコル(reports/11):
  stage1: 全構成 empirical較正 + そのkでの boot p95(n_boot=600, seed0) + robust(p95=20%) seed0
  stage2: robust seed0 ≥ +15.6% の上位≤3のみ seeds 1,2 を追加 → mean3
  合格: robust mean3 ≥ +16.6%(ベースライン mp12 robust 平均 +15.04% の +10%相対)+ OOS維持
  レバ偽装署名: empirical CAGR↑ かつ empirical較正kでの p95 が -29.4% より有意悪化 → 前進と認めない

実行: PYTHONPATH=. uv run python research/experiments/exp36_cushion_joint.py
出力: research/outputs/exp36_cushion_joint.csv / .json
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
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd, yearly_returns,
)
from mm_production import champion_sizing, _fz  # noqa: E402
from fxlab.data import load  # noqa: E402

pd.set_option("display.width", 260)

OUT_CSV = ROOT / "research" / "outputs" / "exp36_cushion_joint.csv"
OUT_JSON = ROOT / "research" / "outputs" / "exp36_cushion_joint.json"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]

OVL_PARAMS = {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}

PASS_STAGE2 = 0.156   # robust seed0 がこれ以上なら seeds1,2 を追加
PASS_FULL = 0.166     # robust mean3 合格ライン
BASE_P95 = -0.294     # mp11 empirical較正kでのブートp95(レバ偽装チェックの物差し)


# --- exp30 の USD ファクター(因果・終値のみ) ---------------------------------
def usd_factor_er(win=40) -> pd.Series:
    legs = []
    for p in MAJORS:
        c = np.log(load(p, "H4")["close"])
        legs.append(c if p.startswith("USD") else -c)
    F = pd.concat(legs, axis=1).ffill().dropna().mean(axis=1)
    direction = (F - F.shift(win)).abs()
    volatility = F.diff().abs().rolling(win).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


# --- exp21d の champ+overlay 統合プール --------------------------------------
def build_both(pool_c: pd.DataFrame, overlay_pool: pd.DataFrame):
    """champ/ovl を結合し、ctx 照合キー(instr, round(ret,12), bars_held)→src の辞書を作る。"""
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)
    fbar = float(np.mean([_fz(z) for z in pool_c["z_entry"].to_numpy()])) or 1.0
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    src = both["src"].to_numpy()
    keysrc = {}
    for i in range(len(both)):
        keysrc[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = src[i]
    return both, fbar, keysrc


def usd_gate_mult(pool_c: pd.DataFrame, er_f: pd.Series, th: float, g: float) -> dict:
    """champ プール各トレードの USDゲート乗数(usd-only)。エントリー時点の確定 ER_F のみ参照(因果)。"""
    er_at = er_f.reindex(pd.to_datetime(pool_c["entry"]), method="ffill").to_numpy()
    is_usd = pool_c["instr"].astype(str).str.contains("USD").to_numpy()
    mult = np.where(np.isfinite(er_at) & (er_at > th) & is_usd, g, 1.0)
    instr = pool_c["instr"].to_numpy(); ret = pool_c["ret"].to_numpy(); bh = pool_c["bars_held"].to_numpy()
    out = {}
    for i in range(len(pool_c)):
        out[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = float(mult[i])
    return out


def make_sizing_factory(fbar, keysrc, w, gate, max_pos, gatemult=None):
    """champ=z-power(×USDゲート乗数)、overlay=dd_mtm<-gate のとき weight w で建玉。"""
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
            s = keysrc.get(key, "champ")
            if s == "champ":
                gm = gatemult.get(key, 1.0) if gatemult is not None else 1.0
                if gm == 0.0:
                    return 0.0
                return ctx["equity_real"] * base * gm * (_fz(ctx["z"]) / fbar)
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * w
            return 0.0
        return sizing
    return make_sizing


def champ_gated_sizing(pool_c, fbar, max_pos, gatemult):
    """チャンピオン単独 × USDゲート(combined プール無し版)。"""
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
            gm = gatemult.get(key, 1.0)
            if gm == 0.0:
                return 0.0
            return ctx["equity_real"] * base * gm * (_fz(ctx["z"]) / fbar)
        return sizing
    return make_sizing


# --- 2段階プロトコル ----------------------------------------------------------
def stage1(label: str, eq_of_k, meta: dict) -> dict:
    t0 = time.time()
    k_emp = calibrate_empirical(eq_of_k, target=0.20, hi=24.0)
    eq_e = eq_of_k(k_emp)
    bs = boot_dd(eq_e, n_boot=600, seed=0)
    yr = yearly_returns(eq_e)
    k_r0 = calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, seed=0)
    eq_r0 = eq_of_k(k_r0)
    row = {"label": label, **meta,
           "emp_k": k_emp, "emp_cagr": cagr_of(eq_e), "emp_dd": max_dd(eq_e),
           "emp_p95": bs["p95"], "worst_year_emp": float(yr.min()),
           "rob_k0": k_r0, "rob_cagr0": cagr_of(eq_r0)}
    print(f"  {label:36s} emp k={k_emp:5.2f} CAGR={row['emp_cagr']:+7.2%} "
          f"p95={bs['p95']:+6.1%} wy={row['worst_year_emp']:+6.1%} | "
          f"rob s0 k={k_r0:4.2f} CAGR={row['rob_cagr0']:+7.2%}  ({time.time()-t0:.0f}s)",
          flush=True)
    return row


def stage2(row: dict, eq_of_k) -> dict:
    for sd in (1, 2):
        k_r = calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, seed=sd)
        row[f"rob_k{sd}"] = k_r
        row[f"rob_cagr{sd}"] = cagr_of(eq_of_k(k_r))
    row["rob_mean3"] = float(np.mean([row["rob_cagr0"], row["rob_cagr1"], row["rob_cagr2"]]))
    print(f"  {row['label']:36s} rob s1={row['rob_cagr1']:+.2%} s2={row['rob_cagr2']:+.2%} "
          f"mean3={row['rob_mean3']:+.2%}", flush=True)
    return row


def is_oos(pool_any, closes, make_sizing, mp, keysrc=None) -> dict:
    """IS(<2022)empirical較正 → OOS素検証。keysrc 指定時は overlay トレードの OOS PnL を分離集計。"""
    is_pool = pool_any[pool_any["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool_any[pool_any["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is = calibrate_empirical(
        lambda k: mm.simulate(is_pool, is_cl, make_sizing(k), max_pos=mp)[0], target=0.20, hi=24.0)
    taken = []
    sz = make_sizing(k_is)

    def logsz(ctx):
        a = sz(ctx)
        if a > 0:
            taken.append((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]), a))
        return a

    eqo, _, _ = mm.simulate(oos_pool, oos_cl, logsz, max_pos=mp)
    out = {"k_is": k_is, "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo)}
    if keysrc is not None:
        ovl = [(i, r, b, a) for (i, r, b, a) in taken if keysrc.get((i, r, b), "champ") == "ovl"]
        out["ovl_oos_n"] = len(ovl)
        out["ovl_oos_pnl_frac"] = float(sum(a * r for (_, r, _, a) in ovl) / eqo.iloc[0])
    return out


def main() -> int:
    t_all = time.time()
    pool = mm.build_pool()
    closes = mm.load_closes()
    import strategies.adx_trend as adx
    ovl = mm.build_pool_for(adx, OVL_PARAMS, tf="H4", side="short",
                            tag="adx_trend_30_100_14_20_short")
    er_f = usd_factor_er(40)
    both, fbar, keysrc = build_both(pool, ovl)
    print(f"champ pool {len(pool)} / overlay pool {len(ovl)} / grid {len(closes)}本", flush=True)

    rows = []
    eqfn = {}   # label -> eq_of_k(stage2/OOS 再利用)
    ctxmap = {}  # label -> (pool_any, make_sizing, mp, keysrc or None)

    # --- 1) チャンピオン単独 mp11(ペアシード健全性)/ mp13 / mp15 ----------
    print("\n=== チャンピオン単独(mp スイープ) ===", flush=True)
    for mp in (11, 13, 15):
        mk = champion_sizing(pool, max_pos=mp)
        fn = (lambda mk=mk, mp=mp: (lambda k: mm.simulate(pool, closes, mk(k), max_pos=mp)[0]))()
        lab = f"champ mp{mp}"
        rows.append(stage1(lab, fn, {"mp": mp, "gate": None, "w": None, "th": None, "g": None}))
        eqfn[lab] = fn
        ctxmap[lab] = (pool, mk, mp, None)

    # --- 2) 連立スイープ: mp × gate × w --------------------------------------
    print("\n=== DD連動クッション連立スイープ(adx_trend 30/100 short) ===", flush=True)
    for mp in (11, 13, 15):
        for gate in (0.05, 0.08):
            for w in (0.5, 1.0):
                mk = make_sizing_factory(fbar, keysrc, w, gate, mp)
                fn = (lambda mk=mk, mp=mp: (lambda k: mm.simulate(both, closes, mk(k), max_pos=mp)[0]))()
                lab = f"cushion mp{mp} g{gate} w{w}"
                rows.append(stage1(lab, fn, {"mp": mp, "gate": gate, "w": w, "th": None, "g": None}))
                eqfn[lab] = fn
                ctxmap[lab] = (both, mk, mp, keysrc)

    # --- 3) USDゲート複合(上位2構成の champ エントリーにのみ重ねる) --------
    df1 = pd.DataFrame(rows)
    top_for_usd = df1.sort_values("rob_cagr0", ascending=False).head(2)
    print("\n=== USDゲート複合(usd-only, champ側のみ)対象: "
          + ", ".join(top_for_usd["label"]) + " ===", flush=True)
    for _, r in top_for_usd.iterrows():
        base_lab = r["label"]
        mp = int(r["mp"])
        for th in (0.30, 0.35):
            for g in (0.0, 0.5):
                gm = usd_gate_mult(pool, er_f, th, g)
                if base_lab.startswith("cushion"):
                    mk = make_sizing_factory(fbar, keysrc, float(r["w"]), float(r["gate"]), mp,
                                             gatemult=gm)
                    pool_any, ks = both, keysrc
                else:
                    mk = champ_gated_sizing(pool, fbar, mp, gm)
                    pool_any, ks = pool, None
                fn = (lambda mk=mk, mp=mp, pa=pool_any: (
                    lambda k: mm.simulate(pa, closes, mk(k), max_pos=mp)[0]))()
                lab = f"{base_lab} +usd th{th} g{g}"
                rows.append(stage1(lab, fn, {"mp": mp, "gate": r["gate"], "w": r["w"],
                                             "th": th, "g": g}))
                eqfn[lab] = fn
                ctxmap[lab] = (pool_any, mk, mp, ks)

    # --- 4) stage2: robust seed0 ≥ +15.6% の上位≤3 に seeds 1,2 --------------
    df = pd.DataFrame(rows)
    cand = df[df["rob_cagr0"] >= PASS_STAGE2].sort_values("rob_cagr0", ascending=False).head(3)
    print(f"\n=== stage2 対象({len(cand)}構成, robust s0 ≥ {PASS_STAGE2:+.1%}) ===", flush=True)
    if cand.empty:
        print("  該当なし → 早期終了(reject)", flush=True)
    for idx in cand.index:
        rows[idx] = stage2(rows[idx], eqfn[rows[idx]["label"]])

    # --- 5) IS/OOS: ベースライン参照(champ mp11)+ 上位構成 ------------------
    print("\n=== IS較正→OOS素検証 ===", flush=True)
    df = pd.DataFrame(rows)
    oos_targets = ["champ mp11"]
    best_order = df.sort_values(
        df.get("rob_mean3", df["rob_cagr0"]).name if "rob_mean3" in df else "rob_cagr0",
        ascending=False)
    # rob_mean3 があればそれ優先、無ければ rob_cagr0 で上位2構成
    if "rob_mean3" in df.columns and df["rob_mean3"].notna().any():
        best_order = df.sort_values("rob_mean3", ascending=False, na_position="last")
    oos_targets += [l for l in best_order["label"].head(3) if l not in oos_targets][:2]
    for lab in oos_targets:
        pa, mk, mp, ks = ctxmap[lab]
        o = is_oos(pa, closes, mk, mp, keysrc=ks)
        i = df.index[df["label"] == lab][0]
        rows[i].update({f"oos_{k2}" if not k2.startswith(("oos", "ovl", "k_is")) else k2: v
                        for k2, v in o.items()})
        ext = (f" ovlN={o['ovl_oos_n']} ovlPnL={o['ovl_oos_pnl_frac']:+.2%}"
               if "ovl_oos_n" in o else "")
        print(f"  {lab:36s} k_is={o['k_is']:5.2f} OOS CAGR={o['oos_cagr']:+7.2%} "
              f"DD={o['oos_dd']:+6.1%}{ext}", flush=True)

    # --- 保存 -----------------------------------------------------------------
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(rows, indent=2, default=lambda x: None if pd.isna(x) else x))
    print(f"\nsaved -> {OUT_CSV}\n total {time.time()-t_all:.0f}s", flush=True)

    # サマリ
    print("\n=== サマリ(robust 降順) ===")
    cols = [c for c in ["label", "emp_k", "emp_cagr", "emp_p95", "rob_cagr0", "rob_mean3",
                        "worst_year_emp", "oos_cagr"] if c in out.columns]
    print(out.sort_values("rob_cagr0", ascending=False)[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
