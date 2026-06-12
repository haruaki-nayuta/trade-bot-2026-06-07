"""exp56: 第3ラウンド候補のフルプロトコル(seeds 0-4 ペア較正 + 6ゲート + IS-argmax)。

exp54/55 スカウトの生き残り+用量再審候補:
  - h20decon : d1+h20除染(exp47 §7 で gate 通過済みの任意装備。+0.4pp 級)
  - P4.5     : z-power 指数 4.0→4.5(ban は P>4.5。採用には IS-argmax が 4.5 以上を
               選ぶこと+全ゲートを要求。d0 時代の IS-argmax は 3.5 だった)
  - mp9/mp10 : max_pos 用量再審。mp11 は d0 で M1 粒度ゲート(谷比1.161>1.15)死。
               d1 は M1 谷比 1.05 と大幅改善しており中間用量の再審に正当性がある。
               (採用には別途 M1 粒度監査が必須=本実験は H4 段の選別のみ)
  - tau1.25  : スカウト+0.59pp・署名なしだがプール段 2021 集中(65%)=多シードで判定
  - スタック  : クリーンな構成要素の組合せ(相互作用込みで実測)
参考(棄却確認用): tau1.5(署名+2022集中), dz0.75(署名+1点スパイク)は多シードに載せない。
  用量曲線の端の形だけ tau1.75/2.0 のプール段断面で記録する。

実行: PYTHONPATH=. uv run python research/experiments/exp56_round3_protocol.py
出力: research/outputs/exp56_result.csv / exp56_result.json
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
from tail_protocol import (  # noqa: E402
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, max_dd,
    protocol_eval, yearly_returns,
)
from exp47_entry_delay import reconstruct, delayed_pool  # noqa: E402
from exp55_d1_refinements import zsigned_at, pool_audit  # noqa: E402
from fxlab import universe as uni  # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
OUT_DIR = ROOT / "research" / "outputs"
Z0, CLIP_LO, CLIP_HI = 2.2, 0.3, 3.0


def make_sizing(pool, *, p=4.0, max_pos=8):
    def fz(z):
        return float(np.clip((z / Z0) ** p, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0
    fbar = float(np.mean([fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0

    def make(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (fz(ctx["z"]) / fbar)
    return make


class Cfg:
    def __init__(self, label, pool, closes, *, p=4.0, max_pos=8):
        self.label, self.pool, self.closes = label, pool, closes
        self.p, self.max_pos = p, max_pos
        self.mk = make_sizing(pool, p=p, max_pos=max_pos)
        self._c = {}

    def eq_of_k(self, k):
        kk = round(float(k), 10)
        if kk not in self._c:
            self._c[kk] = mm.simulate(self.pool, self.closes, self.mk(kk),
                                      max_pos=self.max_pos)[0]
        return self._c[kk]


def full_eval(cfg: Cfg, base=None) -> dict:
    r = protocol_eval(cfg.eq_of_k, label=cfg.label, seeds=SEEDS)
    eq_e = cfg.eq_of_k(r["emp_k"])
    yr_e = yearly_returns(eq_e)
    r["worst_year"] = float(yr_e.min())
    r["neg_years_emp"] = int((yr_e < 0).sum())
    yr_r0 = yearly_returns(cfg.eq_of_k(r["rob"][0]["k"]))
    r["neg_years_rob0"] = int((yr_r0 < 0).sum())
    r["yr_emp"] = {int(y): float(v) for y, v in yr_e.items()}
    r["yr_rob0"] = {int(y): float(v) for y, v in yr_r0.items()}

    # IS較正→OOS素検証
    is_pool = cfg.pool[cfg.pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = cfg.pool[cfg.pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = cfg.closes[cfg.closes.index < OOS_START]
    oos_cl = cfg.closes[cfg.closes.index >= OOS_START]

    def eq_fn(pl, cl):
        c = {}

        def f(k):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(pl, cl, cfg.mk(kk), max_pos=cfg.max_pos)[0]
            return c[kk]
        return f
    eq_is, eq_oos = eq_fn(is_pool, is_cl), eq_fn(oos_pool, oos_cl)
    k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
    r["is_rob_cagr"] = cagr_of(eq_is(k_ir))
    r["oos_rob_cagr"] = cagr_of(eq_oos(k_ir))
    r["oos_rob_dd"] = max_dd(eq_oos(k_ir))
    k_ie = calibrate_empirical(eq_is, 0.20)
    r["is_emp_cagr"] = cagr_of(eq_is(k_ie))
    r["oos_emp_cagr"] = cagr_of(eq_oos(k_ie))
    r["oos_emp_dd"] = max_dd(eq_oos(k_ie))

    if base is not None:
        gain = r["rob_cagr_mean"] - base["rob_cagr_mean"]
        per_seed = {sd: r["rob"][sd]["cagr"] - base["rob"][sd]["cagr"] for sd in SEEDS}
        sig = (r["emp_cagr"] > base["emp_cagr"]) and \
              (abs(r["emp_p95"]) > abs(base["emp_p95"]) + 0.005)
        r["gain_pp"] = gain * 100
        r["gain_per_seed_pp"] = {sd: v * 100 for sd, v in per_seed.items()}
        r["all_seeds_pos"] = all(v > 0 for v in per_seed.values())
        r["signature"] = bool(sig)
        r["g3_oos"] = (r["oos_rob_cagr"] > base["oos_rob_cagr"]) and \
                      (r["oos_emp_cagr"] > base["oos_emp_cagr"])
        r["g4_all_years_pos"] = (r["neg_years_emp"] == 0) and (r["neg_years_rob0"] == 0)
        print(f"      gain {gain*100:+.2f}pp (seeds: " +
              " ".join(f"s{sd}:{v*100:+.2f}" for sd, v in per_seed.items()) +
              f") 署名={'あり' if sig else 'なし'} OOS={'+' if r['g3_oos'] else 'x'} "
              f"全年+={'+' if r['g4_all_years_pos'] else 'x'} "
              f"IS rob {r['is_rob_cagr']:+.1%}->OOS {r['oos_rob_cagr']:+.1%}")
    return r


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool0 = mm.build_pool()
    closes = mm.load_closes()
    rc = reconstruct(pool0)
    ret0 = pool0["ret"].to_numpy()
    dirs = pool0["dir"].to_numpy().astype(float)

    # 共通プール: d1 / d1+h20decon
    mod1, kept1, ret_new1, ex1 = delayed_pool(pool0, rc, 1)
    mod_h, kept_h, ret_h, ex_h = delayed_pool(pool0, rc, 1, skip_h20=True)
    z_exec = zsigned_at(pd.Series(ex1["dts"]), pool0["instr"].to_numpy())

    print(f"=== exp56: 第3ラウンド フルプロトコル (seeds {SEEDS}) ===")

    # --- 0. τ 用量曲線の端(プール段のみ・記録用) ---------------------------
    print("\n--- τ 用量曲線の端(プール段断面のみ) ---")
    for tau in (1.5, 1.75, 2.0):
        keep = kept1 & (dirs * z_exec <= -tau)
        row, _ = pool_audit(f"tau{tau}", pool0, keep, ret_new1, ret0)
        print(f"  tau{tau}: n={row['n']} diff_vs_d0={row['diff_vs_d0']:+.4f} "
              f"best_year={row['best_year']}({row['best_year_diff']:+.4f}) "
              f"excl_best={row['excl_best']:+.4f} excl_2022={row['excl_2022']:+.4f}")

    # --- 1. ベースライン --------------------------------------------------------
    print("\n--- 1. ベースライン d1 (mp8, P4.0) ---")
    cfg_base = Cfg("base_d1", mod1, closes)
    base = full_eval(cfg_base)
    print(f"    [{time.time()-t0:.0f}s]")

    results = {"base_d1": base}

    # --- 2. 単独レバー -----------------------------------------------------------
    print("\n--- 2. 単独レバー ---")
    singles = {
        "h20decon": Cfg("h20decon", mod_h, closes),
        "P4.5": Cfg("P4.5", mod1, closes, p=4.5),
        "mp9": Cfg("mp9", mod1, closes, max_pos=9),
        "mp10": Cfg("mp10", mod1, closes, max_pos=10),
    }
    # tau1.25(プール変更)
    keep125 = kept1 & (dirs * z_exec <= -1.25)
    mod125 = pool0.copy()
    mod125["entry"] = ex1["dts"]
    mod125["entry_price"] = ex1["dclose"] * rc["slip"]
    mod125["ret"] = ret_new1
    mod125["bars_held"] = np.maximum(pool0["bars_held"].to_numpy() - 1, 1)
    mod125 = mod125[keep125].sort_values("entry").reset_index(drop=True)
    singles["tau1.25"] = Cfg("tau1.25", mod125, closes)

    for name, cfg in singles.items():
        results[name] = full_eval(cfg, base)
        print(f"    [{time.time()-t0:.0f}s]")

    # --- 3. P の IS-argmax 監査(d1 プール, IS<2022, robust seed0 較正) -----------
    print("\n--- 3. P IS-argmax 監査 (d1 プール) ---")
    is_pool = mod1[mod1["entry"] < OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_pool = mod1[mod1["entry"] >= OOS_START].reset_index(drop=True)
    oos_cl = closes[closes.index >= OOS_START]
    p_is = {}
    for p in (3.0, 3.5, 4.0, 4.5):
        mk = make_sizing(mod1, p=p, max_pos=8)
        c = {}

        def eq_is(k, mk=mk, c=c):
            kk = round(float(k), 10)
            if kk not in c:
                c[kk] = mm.simulate(is_pool, is_cl, mk(kk), max_pos=8)[0]
            return c[kk]
        k_ir = calibrate_robust_seeded(eq_is, 0.20, seed=0)
        cag_is = cagr_of(eq_is(k_ir))
        co = {}

        def eq_oos(k, mk=mk, co=co):
            kk = round(float(k), 10)
            if kk not in co:
                co[kk] = mm.simulate(oos_pool, oos_cl, mk(kk), max_pos=8)[0]
            return co[kk]
        cag_oos = cagr_of(eq_oos(k_ir))
        p_is[p] = {"is_rob": cag_is, "oos_rob": cag_oos, "k_is": k_ir}
        print(f"  P={p}: IS rob {cag_is:+.2%} -> OOS {cag_oos:+.2%} (k_is={k_ir:.2f})")
    arg = max(p_is, key=lambda p: p_is[p]["is_rob"])
    print(f"  IS-argmax: P={arg}  (d0 時代は 3.5。4.5 採用には argmax>=4.5 を要求)")

    # --- 4. スタック(クリーン要素のみ動的に構成) --------------------------------
    print("\n--- 4. スタック ---")
    def clean(name):
        r = results[name]
        return (r["gain_pp"] > 0 and r["all_seeds_pos"] and not r["signature"]
                and r["g3_oos"] and r["g4_all_years_pos"])
    stacks = {}
    # h20 + P4.5(P監査が通る場合のみ意味を持つが相互作用の記録として実測)
    stacks["h20+P4.5"] = Cfg("h20+P4.5", mod_h, closes, p=4.5)
    stacks["h20+mp9"] = Cfg("h20+mp9", mod_h, closes, max_pos=9)
    stacks["h20+mp10"] = Cfg("h20+mp10", mod_h, closes, max_pos=10)
    stacks["h20+P4.5+mp9"] = Cfg("h20+P4.5+mp9", mod_h, closes, p=4.5, max_pos=9)
    stacks["h20+P4.5+mp10"] = Cfg("h20+P4.5+mp10", mod_h, closes, p=4.5, max_pos=10)
    for name, cfg in stacks.items():
        results[name] = full_eval(cfg, base)
        print(f"    [{time.time()-t0:.0f}s]")

    # --- 保存 -------------------------------------------------------------------
    rows = []
    for name, r in results.items():
        rows.append({
            "cfg": name, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
            "emp_p95": r["emp_p95"], "rob_mean": r["rob_cagr_mean"],
            **{f"rob_s{sd}": r["rob"][sd]["cagr"] for sd in SEEDS},
            "gain_pp": r.get("gain_pp", 0.0),
            "all_seeds_pos": r.get("all_seeds_pos"), "signature": r.get("signature"),
            "g3_oos": r.get("g3_oos"), "g4_all_years_pos": r.get("g4_all_years_pos"),
            "worst_year": r["worst_year"], "neg_emp": r["neg_years_emp"],
            "neg_rob0": r["neg_years_rob0"],
            "is_rob_cagr": r["is_rob_cagr"], "oos_rob_cagr": r["oos_rob_cagr"],
            "oos_rob_dd": r["oos_rob_dd"], "oos_emp_cagr": r["oos_emp_cagr"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "exp56_result.csv", index=False)
    payload = {"p_is_argmax": {str(p): v for p, v in p_is.items()}, "argmax": arg,
               "results": {n: {k: ({str(s): vv for s, vv in v.items()} if k == "rob" else v)
                               for k, v in r.items() if k not in ("yr_emp", "yr_rob0")}
                           for n, r in results.items()}}
    (OUT_DIR / "exp56_result.json").write_text(json.dumps(payload, indent=2, default=float))
    print("\n=== 最終表 ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT_DIR / 'exp56_result.csv'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
