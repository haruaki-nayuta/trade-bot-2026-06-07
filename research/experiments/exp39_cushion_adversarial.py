"""exp39: 発見B(DDクッション×mp15)の敵対検証 — overlay脚の選択バイアスと2022一発依存の監査。

スタンス: 懐疑者。exp36 の「cushion mp15 g0.05 w1.0 = robust +17%級」を反証するつもりで監査する。

検証項目:
  1. overlay脚の近傍頑健性: adx_trend {fast,slow}∈{(20,50),(30,100),(40,150)} × adx_th∈{15,20,25}
     (side=short固定)の9近傍で cushion mp15 g0.05 w1.0 robust seed0。1点だけ良ければ選択バイアス。
  2. 脚ファミリー交換: ma_cross(30/100, short) / donchian_breakout(20/10, short)。機構が横断成立するか。
  3. 2022一発依存: (a) overlay発火トレードの年次PnL分解 (b) 2022除外プール+グリッドで
     cushion vs champ mp11/mp15 の robust seed0 差分が残るか。
  4. gate帯域: gate∈{0.045,0.05,0.055,0.06,0.065} × w1.0 × mp15 robust seed0。
  5. 追加シード: 最良構成(mp15 g0.05 w1.0) seeds{0..4}。全シード ≥ +16.6% か。
  6. mp外挿: cushion mp17/mp19 robust seed0(飽和位置の確認)。

実装は exp36(exp21d系 make_sizing_factory)のコピー。champ=z-power(P=2.0, mm_production._fz)、
overlay=dd_mtm<-gate のとき equity*base*w で建玉。数値は全て出し直す。

実行: PYTHONPATH=. uv run python research/experiments/exp39_cushion_adversarial.py
出力: research/outputs/exp39_cushion_adversarial.csv / .json
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
from mm_production import _fz  # noqa: E402

pd.set_option("display.width", 260)

OUT_CSV = ROOT / "research" / "outputs" / "exp39_cushion_adversarial.csv"
OUT_JSON = ROOT / "research" / "outputs" / "exp39_cushion_adversarial.json"
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")

MP, GATE, W = 15, 0.05, 1.0      # 検証対象の最良構成
PASS = 0.166                      # 合格ライン(robust ≥ +16.6%)
BASE_P95 = -0.294                 # mp11 empirical較正kのブートp95(レバ偽装の物差し)


# --- exp36 からのコピー(コアは変更しない方針のため) -------------------------
def build_both(pool_c: pd.DataFrame, overlay_pool: pd.DataFrame):
    """champ/ovl を結合し、ctx 照合キー(instr, round(ret,12), bars_held)→src の辞書を作る。"""
    pc = pool_c.copy(); pc["src"] = "champ"
    po = overlay_pool.copy(); po["src"] = "ovl"
    both = pd.concat([pc, po], ignore_index=True).sort_values("entry").reset_index(drop=True)
    fbar = float(np.mean([_fz(z) for z in pool_c["z_entry"].to_numpy()])) or 1.0
    instr = both["instr"].to_numpy(); ret = both["ret"].to_numpy(); bh = both["bars_held"].to_numpy()
    src = both["src"].to_numpy(); ent = both["entry"].to_numpy()
    keysrc, keyentry = {}, {}
    for i in range(len(both)):
        key = (instr[i], round(float(ret[i]), 12), int(bh[i]))
        keysrc[key] = src[i]
        keyentry[key] = pd.Timestamp(ent[i])
    return both, fbar, keysrc, keyentry


def make_sizing_factory(fbar, keysrc, w, gate, max_pos):
    """champ=z-power、overlay=dd_mtm<-gate のとき weight w で建玉(exp36 と同一)。"""
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
            if keysrc.get(key, "champ") == "champ":
                return ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
            if ctx["dd_mtm"] < -gate:
                return ctx["equity_real"] * base * w
            return 0.0
        return sizing
    return make_sizing


def champ_only_sizing(pool, max_pos):
    fbar = float(np.mean([_fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0
    def make_sizing(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
    return make_sizing


# --- 評価ヘルパ ---------------------------------------------------------------
def eval_cfg(label, eq_of_k, meta=None, seeds=(0,), emp=True):
    t0 = time.time()
    row = {"label": label, **(meta or {})}
    if emp:
        k_e = calibrate_empirical(eq_of_k, target=0.20, hi=24.0)
        eq_e = eq_of_k(k_e)
        bs = boot_dd(eq_e, n_boot=600, seed=0)
        row.update({"emp_k": k_e, "emp_cagr": cagr_of(eq_e), "emp_p95": bs["p95"],
                    "worst_year_emp": float(yearly_returns(eq_e).min())})
    for sd in seeds:
        k_r = calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, seed=sd)
        row[f"rob_k{sd}"] = k_r
        row[f"rob_cagr{sd}"] = cagr_of(eq_of_k(k_r))
    rc = " ".join(f"s{sd}:{row[f'rob_cagr{sd}']:+.2%}" for sd in seeds)
    emp_s = (f"emp k={row['emp_k']:5.2f} CAGR={row['emp_cagr']:+7.2%} "
             f"p95={row['emp_p95']:+6.1%} wy={row['worst_year_emp']:+6.1%} | " if emp else "")
    print(f"  {label:44s} {emp_s}rob {rc}  ({time.time()-t0:.0f}s)", flush=True)
    return row


def cushion_fn(pool, ovl, mp, gate, w, closes):
    both, fbar, keysrc, keyentry = build_both(pool, ovl)
    mk = make_sizing_factory(fbar, keysrc, w, gate, mp)
    fn = lambda k: mm.simulate(both, closes, mk(k), max_pos=mp)[0]  # noqa: E731
    return fn, both, mk, keysrc, keyentry


def drop_year(pool, closes, year=2022, strict=False):
    """entry(strict なら entry/exit いずれか)が year のトレードと、グリッドの year を除外。"""
    ey = pd.to_datetime(pool["entry"]).dt.year
    keep = ey != year
    if strict:
        xy = pd.to_datetime(pool["exit"]).dt.year
        keep &= xy != year
    return pool[keep].reset_index(drop=True), closes[closes.index.year != year]


def robust_is_to_oos(pool_any, closes, mk, mp, seed=0):
    """robust(IS)較正 → OOS素検証(exp36c 流のペア検証)。"""
    is_pool = pool_any[pool_any["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool_any[pool_any["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is = calibrate_robust_seeded(
        lambda k: mm.simulate(is_pool, is_cl, mk(k), max_pos=mp)[0],
        target=0.20, n_boot=600, seed=seed)
    eqo, _, _ = mm.simulate(oos_pool, oos_cl, mk(k_is), max_pos=mp)
    return {"k_is_rob": k_is, "oos_cagr": cagr_of(eqo), "oos_dd": max_dd(eqo)}


def main() -> int:
    t_all = time.time()
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"champ pool {len(pool)} / grid {len(closes)}本", flush=True)

    import strategies.adx_trend as adx
    import strategies.ma_cross as mac
    import strategies.donchian_breakout as don

    # --- overlay プール群(近傍 + ファミリー) -------------------------------
    print("\n=== overlay プール構築 ===", flush=True)
    ovl_pools = {}
    for (f, s) in ((20, 50), (30, 100), (40, 150)):
        for th in (15, 20, 25):
            tag = f"adx_trend_{f}_{s}_14_{th}"
            t0 = time.time()
            ovl_pools[("adx", f, s, th)] = mm.build_pool_for(
                adx, {"fast": f, "slow": s, "adx_period": 14, "adx_th": th},
                tf="H4", side="short", tag=tag + "_short")
            print(f"  {tag:28s} {len(ovl_pools[('adx', f, s, th)]):4d} trades "
                  f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    ovl_pools[("ma", 30, 100, None)] = mm.build_pool_for(
        mac, {"fast": 30, "slow": 100}, tf="H4", side="short", tag="ma_cross_30_100_short")
    print(f"  {'ma_cross_30_100':28s} {len(ovl_pools[('ma', 30, 100, None)]):4d} trades "
          f"({time.time()-t0:.0f}s)", flush=True)
    t0 = time.time()
    ovl_pools[("don", 20, 10, None)] = mm.build_pool_for(
        don, {"entry": 20, "exit": 10}, tf="H4", side="short", tag="donchian_20_10_short")
    print(f"  {'donchian_20_10':28s} {len(ovl_pools[('don', 20, 10, None)]):4d} trades "
          f"({time.time()-t0:.0f}s)", flush=True)

    canon = ovl_pools[("adx", 30, 100, 20)]
    sections = {}

    # --- 0. 参照: champ単独 mp11 / mp15(robust s0 再現) ---------------------
    print("\n=== 0. 参照 champ単独(robust s0) ===", flush=True)
    refs = []
    for mp in (11, 15):
        mk = champ_only_sizing(pool, mp)
        fn = (lambda mk=mk, mp=mp: (lambda k: mm.simulate(pool, closes, mk(k), max_pos=mp)[0]))()
        refs.append(eval_cfg(f"champ mp{mp}", fn, {"sec": "ref", "mp": mp}))
    sections["ref"] = refs

    # --- 1+2. overlay脚の近傍 + ファミリー交換(cushion mp15 g0.05 w1.0) -----
    print("\n=== 1. adx 近傍9点 + 2. ファミリー交換(cushion mp15 g0.05 w1.0, rob s0) ===",
          flush=True)
    legs = []
    for key, ovl in ovl_pools.items():
        fam, f, s, th = key
        lab = (f"cushion adx({f}/{s},th{th})" if fam == "adx"
               else f"cushion {'ma_cross' if fam == 'ma' else 'donchian'}({f}/{s})")
        fn, *_ = cushion_fn(pool, ovl, MP, GATE, W, closes)
        legs.append(eval_cfg(lab, fn, {"sec": "legs", "family": fam, "fast": f, "slow": s,
                                       "adx_th": th, "n_ovl": len(ovl)}))
    sections["legs"] = legs

    # --- 5. 最良構成の5シード + robust-IS→OOS ペア ---------------------------
    print("\n=== 5. canonical(adx 30/100 th20) seeds 0..4 + robust-IS→OOSペア ===", flush=True)
    fn_c, both_c, mk_c, keysrc_c, keyentry_c = cushion_fn(pool, canon, MP, GATE, W, closes)
    canon_row = eval_cfg(f"cushion mp{MP} g{GATE} w{W} (canonical)", fn_c,
                         {"sec": "canonical"}, seeds=(0, 1, 2, 3, 4))
    canon_row["rob_mean5"] = float(np.mean([canon_row[f"rob_cagr{s}"] for s in range(5)]))
    oo = robust_is_to_oos(both_c, closes, mk_c, MP)
    canon_row.update(oo)
    # ベースライン mp11 のペア(同条件参照)
    mk_b = champ_only_sizing(pool, 11)
    oo_b = robust_is_to_oos(pool, closes, mk_b, 11)
    print(f"  canonical rob mean5={canon_row['rob_mean5']:+.2%} | "
          f"robust-IS k={oo['k_is_rob']:.2f} OOS={oo['oos_cagr']:+.2%} DD={oo['oos_dd']:+.1%} | "
          f"baseline mp11 robust-IS k={oo_b['k_is_rob']:.2f} OOS={oo_b['oos_cagr']:+.2%} "
          f"DD={oo_b['oos_dd']:+.1%}", flush=True)
    sections["canonical"] = [canon_row,
                             {"label": "champ mp11 (paired oos)", "sec": "canonical", **oo_b}]

    # --- 3a. overlay発火トレードの年次PnL分解(robust s0 の k で実走) --------
    print("\n=== 3a. overlay発火の年次PnL分解(canonical, k=rob_k0) ===", flush=True)
    k0 = canon_row["rob_k0"]
    fired = []
    sz = mk_c(k0)

    def logsz(ctx):
        a = sz(ctx)
        key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
        if a > 0 and keysrc_c.get(key, "champ") == "ovl":
            fired.append({"year": int(keyentry_c[key].year), "instr": ctx["instr"],
                          "ret": float(ctx["ret"]), "alloc": float(a),
                          "pnl": float(a * ctx["ret"]),
                          "pnl_frac": float(a * ctx["ret"] / ctx["equity_mtm"])})
        return a

    eq_full, _, _ = mm.simulate(both_c, closes, logsz, max_pos=MP)
    fdf = pd.DataFrame(fired)
    if fdf.empty:
        print("  (overlay発火ゼロ)", flush=True)
        ydec = pd.DataFrame()
    else:
        ydec = fdf.groupby("year").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                       pnl_frac=("pnl_frac", "sum"),
                                       win_rate=("pnl", lambda x: float((x > 0).mean())))
        print(ydec.to_string(), flush=True)
        is_dec = fdf[fdf["year"] <= 2021]
        print(f"  IS期(<=2021): n={len(is_dec)} pnl_frac合計={is_dec['pnl_frac'].sum():+.2%} "
              f"| 2022: n={len(fdf[fdf['year'] == 2022])} "
              f"pnl_frac={fdf[fdf['year'] == 2022]['pnl_frac'].sum():+.2%} "
              f"| 2023以降: pnl_frac={fdf[fdf['year'] >= 2023]['pnl_frac'].sum():+.2%}", flush=True)
    sections["yearly_decomp"] = ydec.reset_index().to_dict("records") if not ydec.empty else []

    # --- 3b. 2022除外(プール+グリッド) -------------------------------------
    print("\n=== 3b. 2022除外: cushion vs champ(robust s0, 同一no-2022条件) ===", flush=True)
    no22 = []
    for strict in (False, True):
        tag = "no22strict" if strict else "no22"
        p22, c22 = drop_year(pool, closes, 2022, strict=strict)
        o22, _ = drop_year(canon, closes, 2022, strict=strict)
        for mp, label, use_cushion in ((11, f"champ mp11 [{tag}]", False),
                                       (15, f"champ mp15 [{tag}]", False),
                                       (MP, f"cushion mp15 g0.05 w1.0 [{tag}]", True)):
            if use_cushion:
                fn, *_ = cushion_fn(p22, o22, mp, GATE, W, c22)
            else:
                mk = champ_only_sizing(p22, mp)
                fn = (lambda mk=mk, mp=mp, p=p22, c=c22:
                      (lambda k: mm.simulate(p, c, mk(k), max_pos=mp)[0]))()
            no22.append(eval_cfg(label, fn, {"sec": tag, "mp": mp,
                                             "n_pool": len(p22), "n_ovl": len(o22)}))
        sections[tag] = [r for r in no22 if r["sec"] == tag]

    # --- 4. gate帯域(canonical脚, w1.0, mp15, rob s0) ------------------------
    print("\n=== 4. gate帯域 {0.045..0.065} (rob s0) ===", flush=True)
    gates = []
    for g in (0.045, 0.05, 0.055, 0.06, 0.065):
        fn, *_ = cushion_fn(pool, canon, MP, g, W, closes)
        gates.append(eval_cfg(f"cushion mp15 g{g} w1.0", fn, {"sec": "gate", "gate": g}))
    sections["gate"] = gates

    # --- 6. mp外挿(canonical脚, g0.05 w1.0) ----------------------------------
    print("\n=== 6. mp外挿 mp17/mp19 (rob s0) ===", flush=True)
    mps = []
    for mp in (17, 19):
        fn, *_ = cushion_fn(pool, canon, mp, GATE, W, closes)
        mps.append(eval_cfg(f"cushion mp{mp} g0.05 w1.0", fn, {"sec": "mp_ext", "mp": mp}))
    sections["mp_ext"] = mps

    # --- 保存 -----------------------------------------------------------------
    flat = []
    for sec, rows_ in sections.items():
        if sec == "yearly_decomp":
            continue
        flat.extend(rows_)
    out = pd.DataFrame(flat)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    OUT_JSON.write_text(json.dumps(
        {k: v for k, v in sections.items()}, indent=2,
        default=lambda x: None if (isinstance(x, float) and pd.isna(x)) else
        (x.item() if hasattr(x, "item") else str(x))))
    print(f"\nsaved -> {OUT_CSV}\n total {time.time()-t_all:.0f}s", flush=True)

    cols = [c for c in ["label", "emp_k", "emp_cagr", "emp_p95", "worst_year_emp",
                        "rob_cagr0", "rob_mean5", "oos_cagr"] if c in out.columns]
    print("\n=== サマリ ===")
    print(out[cols].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
