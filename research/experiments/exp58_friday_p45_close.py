"""exp58: 第3ラウンドの残課題2件 — 週末エントリー衛生のスカウト + P4.5 の完全文書化。

A) 週末エントリー衛生: d1 執行バーが「週の最終バー」(次グリッドバーまで >8h = close後に
   週末ギャップ)なら、もう1本送って週明け最初のバー close で建玉する(h20 と同型の
   執行衛生規則)。機構仮説: 金曜深夜の新規建玉は週末ギャップのテールを建玉直後に浴びる。
   M1 監査の深掘り局面が月曜早朝に集中している(2022-03-07 01:51 等)ことと整合するか。
   判定: プール段 G5(単年依存) → 口座 seed0 → 生き残れば多シード。

B) P4.5 の完全文書化(採否は別途判断。現時点の方針は棄却=既知勾配の再訴訟・IS差+0.05pp
   はノイズ・raw G3-emp fail):
   1. 署名ブートシード監査(seeds 0-2)
   2. 口座段 年次差分 G5(emp較正/rob_s0較正, 対ベース)
   3. M1 粒度監査(rob_m5 k)→ 掛け目・実効CAGR

実行: PYTHONPATH=. uv run python research/experiments/exp58_friday_p45_close.py
出力: research/outputs/exp58_result.json
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
    boot_dd, cagr_of, calibrate_empirical, calibrate_robust_seeded, protocol_eval,
    yearly_returns,
)
from exp47_entry_delay import reconstruct, delayed_pool, year_diff_audit  # noqa: E402
from exp55_d1_refinements import pool_audit  # noqa: E402
from exp56_round3_protocol import make_sizing  # noqa: E402
import exp52_d1_m1audit as a52  # noqa: E402
from fxlab.data import load_m1  # noqa: E402
from fxlab import universe as uni  # noqa: E402

OUT_DIR = ROOT / "research" / "outputs"
E56 = json.loads((OUT_DIR / "exp56_result.json").read_text())["results"]


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def eq_fn(pool, closes, mk, max_pos=8):
    c = {}

    def f(k):
        kk = round(float(k), 10)
        if kk not in c:
            c[kk] = mm.simulate(pool, closes, mk(kk), max_pos=max_pos)[0]
        return c[kk]
    return f


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool0 = mm.build_pool()
    closes = mm.load_closes()
    rc = reconstruct(pool0)
    ret0 = pool0["ret"].to_numpy()
    dirs = pool0["dir"].to_numpy().astype(float)
    mod1, kept1, ret_new1, ex1 = delayed_pool(pool0, rc, 1)
    out = {}

    # ---------------- A) 週末エントリー衛生 ----------------
    sec("A. 週末エントリー衛生(d1 執行バーが週最終バーなら週明けへ送る)")
    n = len(pool0)
    dclose = np.full(n, np.nan)
    dts = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    ie_w = np.full(n, -1)
    n_pushed = 0
    for instr, g in pool0.groupby("instr"):
        rows = g.index.to_numpy()
        s = rc["closes_by"][instr]
        sidx = s.index.values
        # 各バーから次バーへのギャップ(時間)。最終バーは0埋め(押し出さない)
        gap_h = np.zeros(len(s))
        gap_h[:-1] = (sidx[1:] - sidx[:-1]).astype("timedelta64[m]").astype(float) / 60.0
        ie_d = rc["idx_e"][rows] + 1                       # d1 執行バー
        ie_cl = np.minimum(ie_d, len(s) - 1)
        is_last_of_week = gap_h[ie_cl] > 8.0               # 次バーまで >8h = 週末/休場ギャップ
        n_pushed += int(is_last_of_week.sum())
        ie_d = ie_d + is_last_of_week.astype(int)          # もう1本送る(週明け初バー)
        ie_cl = np.minimum(ie_d, len(s) - 1)
        dclose[rows] = s.to_numpy()[ie_cl]
        dts[rows] = sidx[ie_cl]
        ie_w[rows] = ie_d
    kept_w = ie_w < rc["idx_x"]
    ret_w = dirs * (rc["exit_close"] / dclose - 1.0) - rc["cost"]
    print(f"週末押し出し対象: {n_pushed}件 / 消滅 {int((~kept_w).sum())}件 (d1 は {int((~kept1).sum())}件)")
    row, yr = pool_audit("friday_push", pool0, kept_w, ret_w, ret0)
    d1row, _ = pool_audit("d1_base", pool0, kept1, ret_new1, ret0)
    print(f"  プール: n={row['n']} diff_vs_d0={row['diff_vs_d0']:+.4f} (d1 {d1row['diff_vs_d0']:+.4f})")
    # 増分(対 d1)の年次
    diff_inc = np.where(kept_w & kept1, ret_w - ret_new1,
                        np.where(kept1 & ~kept_w, -ret_new1,
                                 np.where(kept_w & ~kept1, ret_w, 0.0)))
    yri = pd.Series(diff_inc).groupby(pool0["exit"].dt.year).sum()
    tot = float(diff_inc.sum())
    if abs(tot) > 1e-9:
        by = int(yri.idxmax())
        ks = float(yri.drop(by).sum() / tot) if tot > 0 else np.nan
        print(f"  増分(対d1) {tot:+.4f}  年次: " +
              "  ".join(f"{int(y)}:{v:+.4f}" for y, v in yri.items() if abs(v) > 1e-4))
        print(f"  最良年 {by}({float(yri[by]):+.4f}) 残存率 {ks:.0%} -> "
              f"{'pass' if (tot > 0 and ks >= 0.5) else 'FAIL'}")
    out["friday_pool"] = {"n_pushed": n_pushed, "diff_vs_d0": row["diff_vs_d0"],
                          "inc_vs_d1": tot,
                          "inc_yearly": {int(y): float(v) for y, v in yri.items()}}
    # 口座 seed0
    mod_w = pool0.copy()
    mod_w["entry"] = pd.DatetimeIndex(dts).tz_localize("UTC")
    mod_w["entry_price"] = dclose * rc["slip"]
    mod_w["ret"] = ret_w
    mod_w["bars_held"] = np.maximum(pool0["bars_held"].to_numpy() - 1, 1)
    mod_w = mod_w[kept_w].sort_values("entry").reset_index(drop=True)
    mk_w = make_sizing(mod_w, p=4.0, max_pos=8)
    r_w = protocol_eval(eq_fn(mod_w, closes, mk_w), label="friday_push", seeds=(0,))
    base_r0 = E56["base_d1"]["rob"]["0"]["cagr"]
    print(f"  口座 seed0: rob {r_w['rob'][0]['cagr']:+.2%} (base {base_r0:+.2%}, "
          f"{(r_w['rob'][0]['cagr']-base_r0)*100:+.2f}pp)")
    out["friday_account_s0"] = {"rob_s0": r_w["rob"][0]["cagr"], "emp": r_w["emp_cagr"],
                                "emp_p95": r_w["emp_p95"]}

    # ---------------- B) P4.5 完全文書化 ----------------
    sec("B. P4.5 文書化(署名シード/年次G5/M1監査)")
    mk40 = make_sizing(mod1, p=4.0, max_pos=8)
    mk45 = make_sizing(mod1, p=4.5, max_pos=8)
    f40 = eq_fn(mod1, closes, mk40)
    f45 = eq_fn(mod1, closes, mk45)
    k40e = calibrate_empirical(f40, 0.20)
    k45e = calibrate_empirical(f45, 0.20)
    # 1) 署名シード
    p95b = {sd: boot_dd(f40(k40e), n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)}
    p95c = {sd: boot_dd(f45(k45e), n_boot=1500, seed=sd)["p95"] for sd in (0, 1, 2)}
    emp_up = cagr_of(f45(k45e)) > cagr_of(f40(k40e))
    sigs = {sd: emp_up and (abs(p95c[sd]) > abs(p95b[sd]) + 0.005) for sd in p95c}
    print("  p95 base: " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95b.items()))
    print("  p95 P4.5: " + " / ".join(f"s{sd}:{v:+.2%}" for sd, v in p95c.items()) +
          f"   署名 {sum(sigs.values())}/3")
    out["p45_signature"] = {"n_sig": sum(sigs.values()),
                            "p95_base": {str(s): float(v) for s, v in p95b.items()},
                            "p95_p45": {str(s): float(v) for s, v in p95c.items()}}
    # 2) 年次 G5(emp / rob_s0)
    k40r = calibrate_robust_seeded(f40, 0.20, seed=0)
    k45r = calibrate_robust_seeded(f45, 0.20, seed=0)
    for tag, fb, fc, kb, kc in [("emp", f40, f45, k40e, k45e), ("rob0", f40, f45, k40r, k45r)]:
        yb = {int(y): float(v) for y, v in yearly_returns(fb(kb)).items()}
        yc = {int(y): float(v) for y, v in yearly_returns(fc(kc)).items()}
        a = year_diff_audit(tag, yc, yb)
        out[f"p45_g5_{tag}"] = a
        print(f"  G5({tag}): 合計 {a['total']:+.2%} 最良年 {a['best_year']}"
              f"({a['best_year_diff']:+.2%}) 除外後 {a['excl_best']:+.2%} "
              f"(残存率 {a['keep_share_excl_best']:.0%}) 2022除外 {a['excl_2022']:+.2%}")
    # 3) M1 監査(rob_m5 k)
    k_m5_45 = float(np.mean([E56["P4.5"]["rob"][str(s)]["k"] for s in range(5)]))
    grid_idx = pd.DatetimeIndex(load_m1("EURUSD").index.tz_localize(None))
    a52.MAX_POS = 8
    aud = a52.m1_audit_one("d1mp8_P4.5", mod1, closes, mk45, k_m5_45, grid_idx)
    print(f"  M1: 谷比 {aud['ratio']:.3f}  掛け目 x{aud['haircut']:.3f}  "
          f"実効CAGR {aud['cagr_adj']:+.2%}  p95_M1近似 {aud['p95_m1_approx']:+.1%}")
    out["p45_m1"] = {k: v for k, v in aud.items() if k != "episodes"}

    (OUT_DIR / "exp58_result.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {OUT_DIR / 'exp58_result.json'}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
