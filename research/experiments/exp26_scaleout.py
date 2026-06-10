"""exp26: 分割決済(scale-out)— 鉄の三角形の未踏領域②。

チャンピオンv2 の出口は「短期Zが −exit_z(=0.5) を終値で回帰したら全量決済」。
本実験は1トレードを2レグに分割し、早いしきい値(exit_z=a)で w を先に決済、
残り (1−w) を遅いしきい値(exit_z=b)で決済する。

効果の仮説: 早期レグが塩漬けの保有期間/含み損エクスポージャを削る → MtM DD 低下 →
同じ DD=20% でより高いレバ k → CAGR 増。対価は早期決済分の取り損ない。
プールPF単体ではなく **統合口座の CAGR@DD20%** で判定する(これが正しい物差し)。

レグのプールは「entry_z 等のエントリー条件を完全固定し exit_z だけ変えた」2つの
バックテストから (instr, entry) で内部結合して作る(再エントリー差分のトレードは落とす=保守的)。

実行: PYTHONPATH=. uv run python research/experiments/exp26_scaleout.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

import mm_lab as mm  # noqa: E402
import strategies.confluence_meanrev_v2 as v2  # noqa: E402
from mm_production import champion_sizing, _fz  # noqa: E402

pd.set_option("display.width", 220)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def build_leg_pool(exit_z: float) -> pd.DataFrame:
    p = dict(v2.PARAMS)
    p["exit_z"] = exit_z
    return mm.build_pool_for(v2, p, tf="H4", tag=f"v2_xz{int(round(exit_z*100)):03d}")


def merge_legs(pa: pd.DataFrame, pb: pd.DataFrame) -> pd.DataFrame:
    """(instr, entry) で内部結合。レグA=早い出口, レグB=遅い出口。"""
    a = pa.set_index(["instr", "entry"])
    b = pb.set_index(["instr", "entry"])
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    j = j.reset_index()
    # 共有エントリー情報は A 側を使用(entry_price/z_entry は同一のはず)
    out = pd.DataFrame({
        "instr": j["instr"], "entry": j["entry"],
        "entry_price": j["entry_price_a"],
        "dir": j["dir_a"],
        "z_entry": j["z_entry_a"],
        "exit_a": j["exit_a"], "ret_a": j["ret_a"], "bars_a": j["bars_held_a"],
        "exit_b": j["exit_b"], "ret_b": j["ret_b"], "bars_b": j["bars_held_b"],
    }).sort_values("entry").reset_index(drop=True)
    return out


def simulate_scaleout(pool: pd.DataFrame, closes: pd.DataFrame, k: float, w_early: float,
                      *, fbar: float, init=10_000.0, max_pos=8):
    """2レグ建玉の口座シミュ(mm_lab.simulate 互換ロジック+レグ分割)。"""
    grid = closes.index
    col_of = {c: i for i, c in enumerate(closes.columns)}
    carr = closes.to_numpy()
    n = len(grid)
    gi = grid.to_numpy()
    e_pos = np.clip(np.searchsorted(gi, pool["entry"].to_numpy(), side="left"), 0, n - 1)
    xa_pos = np.clip(np.searchsorted(gi, pool["exit_a"].to_numpy(), side="left"), 0, n - 1)
    xb_pos = np.clip(np.searchsorted(gi, pool["exit_b"].to_numpy(), side="left"), 0, n - 1)

    by_entry = {}
    for ti in range(len(pool)):
        by_entry.setdefault(int(e_pos[ti]), []).append(ti)

    instr_arr = pool["instr"].to_numpy()
    dir_arr = pool["dir"].to_numpy().astype(float)
    ep_arr = pool["entry_price"].to_numpy()
    ra = pool["ret_a"].to_numpy()
    rb = pool["ret_b"].to_numpy()
    z_arr = pool["z_entry"].to_numpy()

    base = k / max_pos
    equity = init
    open_pos = []   # dict(col,dir,eprice,legs=[(exit_pos, alloc, ret), ...])
    eq_mtm = np.empty(n)
    eq_real = np.empty(n)
    skipped = 0
    conc = []

    for b in range(n):
        if open_pos:
            still = []
            for p in open_pos:
                legs_left = []
                for (xp, alloc, ret) in p["legs"]:
                    if xp <= b:
                        equity += alloc * ret
                    else:
                        legs_left.append((xp, alloc, ret))
                p["legs"] = legs_left
                if legs_left:
                    still.append(p)
            open_pos = still
        unreal = 0.0
        for p in open_pos:
            px = carr[b, p["col"]]
            rr = p["dir"] * (px / p["eprice"] - 1.0)
            unreal += sum(alloc for (_, alloc, _) in p["legs"]) * rr
        mtm = equity + unreal
        eq_mtm[b] = mtm
        eq_real[b] = equity

        if b in by_entry:
            for ti in by_entry[b]:
                if len(open_pos) >= max_pos:
                    skipped += 1
                    continue
                alloc = equity * base * (_fz(float(z_arr[ti])) / fbar)
                if alloc <= 0:
                    skipped += 1
                    continue
                legs = []
                if w_early > 0:
                    legs.append((int(xa_pos[ti]), alloc * w_early, float(ra[ti])))
                if w_early < 1:
                    legs.append((int(xb_pos[ti]), alloc * (1 - w_early), float(rb[ti])))
                open_pos.append({"col": col_of[instr_arr[ti]], "dir": dir_arr[ti],
                                 "eprice": ep_arr[ti], "legs": legs})
                conc.append(len(open_pos))

    info = {"final": equity, "skipped": skipped, "n_taken": len(conc),
            "max_conc": max(conc) if conc else 0,
            "avg_conc": float(np.mean(conc)) if conc else 0.0}
    return pd.Series(eq_mtm, index=grid), pd.Series(eq_real, index=grid), info


def calibrate_scaleout(pool, closes, w_early, fbar, target_dd=0.20, max_pos=8,
                       lo=0.02, hi=14.0, iters=22):
    def dd_of(k):
        eqm, _, _ = simulate_scaleout(pool, closes, k, w_early, fbar=fbar, max_pos=max_pos)
        return abs(float((eqm / eqm.cummax() - 1.0).min()))
    if dd_of(hi) <= target_dd:
        eqm, eqr, info = simulate_scaleout(pool, closes, hi, w_early, fbar=fbar, max_pos=max_pos)
        return hi, eqm, eqr, info
    for _ in range(iters):
        mid = (lo + hi) / 2
        if dd_of(mid) > target_dd:
            hi = mid
        else:
            lo = mid
    eqm, eqr, info = simulate_scaleout(pool, closes, lo, w_early, fbar=fbar, max_pos=max_pos)
    return lo, eqm, eqr, info


def eval_combo(pool, closes, w_early, fbar, label, max_pos=8):
    k, eqm, eqr, info = calibrate_scaleout(pool, closes, w_early, fbar, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=800)
    # IS較正→OOS素検証
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]
    oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = calibrate_scaleout(is_pool, is_cl, w_early, fbar, max_pos=max_pos)
    eqo, ero, io = simulate_scaleout(oos_pool, oos_cl, k_is, w_early, fbar=fbar, max_pos=max_pos)
    so = mm.stats(eqo, ero, io)
    print(f"  {label:28s} k={k:5.2f} CAGR={s['cagr']:+7.2%} DD={s['maxdd_mtm']:+6.1%} "
          f"Sharpe={s['sharpe']:4.2f} boot95={bs['p95']:+6.1%} +年={s['pos_year_rate']:3.0%} "
          f"最悪年={s['worst_year']:+5.1%} | OOS CAGR={so['cagr']:+7.2%} DD={so['maxdd_mtm']:+6.1%}")
    return {"label": label, "k": k, "cagr": s["cagr"], "dd": s["maxdd_mtm"],
            "boot95": bs["p95"], "oos_cagr": so["cagr"], "oos_dd": so["maxdd_mtm"]}


def main() -> int:
    closes = mm.load_closes()
    pool_std = mm.build_pool()
    fbar = float(np.mean([_fz(z) for z in pool_std["z_entry"].to_numpy()])) or 1.0

    # === 基準: 標準(exit_z=0.5 全量)を同じシミュレータで(プロトコル整合確認) ===
    print("=== 基準(標準出口 exit_z=0.5, w_early=0 で B レグのみ) ===")
    leg_b_std = build_leg_pool(0.5)
    merged_std = merge_legs(leg_b_std, leg_b_std)
    base = eval_combo(merged_std, closes, 0.0, fbar, "baseline xz=0.5")

    # === レグ組合せ ===
    combos = [
        (1.0, 0.5), (1.25, 0.5), (1.5, 0.5),   # 早出し + 標準
        (1.0, 0.25), (0.75, 0.25),             # 早出し + 深め
        (0.5, 0.0),                            # 標準 + フル平均回帰
    ]
    weights = [0.25, 0.5, 0.75]
    for (a, b) in combos:
        pa = build_leg_pool(a)
        pb = build_leg_pool(b)
        merged = merge_legs(pa, pb)
        n_match = len(merged)
        print(f"\n=== legs exit_z=({a}, {b})  共有トレード {n_match} ===")
        for w in weights:
            eval_combo(merged, closes, w, fbar, f"a={a} b={b} w_early={w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
