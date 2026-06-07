"""資金管理 最終統合・決着スクリプト — チャンピオンv2(confluence_meanrev_v2)。

Workflow(13エージェント)の結論を、リーク無し&厳密な理論DD較正で**自前再検証**し確定する。

主要論点:
  1. per-instrument ケリーの重みは全期間で推定=先読み(リーク)。→ ウォークフォワード(因果)で再評価し
     「本当に運用可能な honest CAGR」を出す。リークの寄与を切り分ける。
  2. 「理論上の最大DD≤20%」の2解釈で全手法を較正・比較:
       (A) 経験的(単一バックテストパス)MtM 最大DD = 20%
       (B) 堅牢: ブロックブートストラップ理論DD(p95, 20回に1回級) = 20%
  3. リーク無しで頑健な勝者(乖離連動z / max_pos / WFケリー)を確定し、本番サイジングを提示。

実行: uv run python mm_synthesis.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 50)

BASELINE_CAGR = 0.1593


# ============ レバー(全てmake_sizing(k)=総建玉をkに線形スケール) ============
def make_flat(max_pos=6):
    """固定比率(ベースライン)。"""
    def make_sizing(k):
        w = k / max_pos
        return lambda ctx: ctx["equity_real"] * w
    return make_sizing


def _fz(z, z0=2.2, p=2.0, lo=0.3, hi=3.0):
    return float(np.clip((z / z0) ** p, lo, hi)) if np.isfinite(z) else 1.0


def make_zsize(pool, max_pos=6, z0=2.2, p=2.0, lo=0.3, hi=3.0):
    """乖離連動サイズ(因果=エントリー|Z|のみ。リーク無し)。"""
    zvals = pool["z_entry"].to_numpy()
    fbar = float(np.mean([_fz(z, z0, p, lo, hi) for z in zvals])) or 1.0

    def make_sizing(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (_fz(ctx["z"], z0, p, lo, hi) / fbar)
    return make_sizing


# --- per-instrument ケリー(2版: フル期間=リーク / ウォークフォワード=因果) ---
def _kelly_logopt(r, fmax=60.0, n=2000):
    if len(r) < 5:
        return 0.0
    best_f, best_g = 0.0, -1e18
    for f in np.linspace(0.0, fmax, n):
        x = 1.0 + f * r
        if (x <= 0).any():
            break
        g = float(np.mean(np.log(x)))
        if g > best_g:
            best_g, best_f = g, f
    return best_f


def _instr_weights_full(pool, min_trades=15, shrink=0.5):
    """全期間データで銘柄別ケリー重み(=先読み/リーク版)。"""
    glob = _kelly_logopt(pool["ret"].to_numpy())
    w = {}
    for nm, g in pool.groupby("instr"):
        r = g["ret"].to_numpy()
        fi = glob if len(r) < min_trades else shrink * _kelly_logopt(r) + (1 - shrink) * glob
        w[nm] = fi
    arr = np.array(list(w.values()))
    mean_w = arr[arr > 0].mean() if (arr > 0).any() else 1.0
    return {k: (v / mean_w if mean_w > 0 else 1.0) for k, v in w.items()}


def _per_trade_weight_wf(pool, min_trades=15, shrink=0.5, refit_every=20):
    """ウォークフォワード(因果)銘柄別ケリー重み: 各トレードの重みは、そのトレードの
    entry 時点までに **exit 済み** の同銘柄トレードのみで推定(拡張窓)。先読み無し。

    返り値: pool 行順に整列した重み配列(平均≈1へ正規化、ただし因果なので厳密平均1ではない)。
    実装メモ: 較正の単調性のため、全体スケールは別途 k に吸収。ここでは相対形のみ与える。
    """
    n = len(pool)
    entry = pool["entry"].to_numpy()
    exitt = pool["exit"].to_numpy()
    instr = pool["instr"].to_numpy()
    ret = pool["ret"].to_numpy()
    order = np.argsort(entry, kind="stable")

    # 各銘柄について、exit 順に ret を蓄積。あるトレードの entry 時点で exit<=entry の ret を使う。
    # 効率のため: 銘柄別に (exit, ret) を保持し、entry 時に二分探索で「exit<=entry」を集める。
    by_instr_exits = {}
    by_instr_rets = {}
    for nm in np.unique(instr):
        m = instr == nm
        ex = exitt[m]
        rr = ret[m]
        srt = np.argsort(ex, kind="stable")
        by_instr_exits[nm] = ex[srt]
        by_instr_rets[nm] = rr[srt]

    glob_rets_sorted_ex = np.sort(exitt)
    glob_rets_by_ex = ret[np.argsort(exitt, kind="stable")]

    w = np.ones(n)
    for ti in order:
        e = entry[ti]
        nm = instr[ti]
        # グローバル(全銘柄)の既知 ret(exit<=e)で glob ケリー
        gpos = np.searchsorted(glob_rets_sorted_ex, e, side="right")
        gr = glob_rets_by_ex[:gpos]
        glob = _kelly_logopt(gr) if len(gr) >= min_trades else 0.0
        # 銘柄別
        ex_i = by_instr_exits[nm]
        ri = by_instr_rets[nm]
        ipos = np.searchsorted(ex_i, e, side="right")
        r_known = ri[:ipos]
        if len(r_known) < min_trades:
            fi = glob if glob > 0 else 1.0
        else:
            fi_raw = _kelly_logopt(r_known)
            fi = shrink * fi_raw + (1 - shrink) * (glob if glob > 0 else fi_raw)
        w[ti] = fi if fi > 0 else 1.0
    # 平均1へ正規化(相対形)
    pos = w[w > 0]
    w = w / (pos.mean() if len(pos) else 1.0)
    return w


def _weight_lookup(pool, w_arr):
    """(instr, ret丸め, bars_held) → 重み の辞書(ctx から引くため。1214件で一意)。"""
    d = {}
    instr = pool["instr"].to_numpy(); ret = pool["ret"].to_numpy(); bh = pool["bars_held"].to_numpy()
    for i in range(len(pool)):
        d[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = float(w_arr[i])
    return d


def make_kelly(pool, max_pos=6, mode="full", shrink=0.5, min_trades=15, clip=4.0):
    """per-instrument ケリー。mode='full'(リーク) / 'wf'(ウォークフォワード=因果)。"""
    if mode == "full":
        weights = _instr_weights_full(pool, min_trades=min_trades, shrink=shrink)

        def make_sizing(k):
            base = k / max_pos
            return lambda ctx: ctx["equity_real"] * base * min(weights.get(ctx["instr"], 1.0), clip)
        return make_sizing
    # walk-forward: per-trade 重み → ctx 一意キーで引く
    w_arr = _per_trade_weight_wf(pool, min_trades=min_trades, shrink=shrink)
    lut = _weight_lookup(pool, w_arr)

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            key = (ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"]))
            wi = min(lut.get(key, 1.0), clip)
            return ctx["equity_real"] * base * wi
        return sizing
    return make_sizing


def make_combo(pool, max_pos=6, kelly_mode="wf", shrink=0.5, min_trades=15, clip=4.0,
               z0=2.2, p=2.0, lo=0.3, hi=3.0):
    """ケリー(wf/full) × 乖離連動z の乗算合成。"""
    zvals = pool["z_entry"].to_numpy()
    fbar = float(np.mean([_fz(z, z0, p, lo, hi) for z in zvals])) or 1.0
    if kelly_mode == "full":
        weights = _instr_weights_full(pool, min_trades=min_trades, shrink=shrink)
        wfun = lambda ctx: weights.get(ctx["instr"], 1.0)
    else:
        w_arr = _per_trade_weight_wf(pool, min_trades=min_trades, shrink=shrink)
        lut = _weight_lookup(pool, w_arr)
        wfun = lambda ctx: lut.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), 1.0)

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            wi = min(wfun(ctx), clip)
            return ctx["equity_real"] * base * wi * (_fz(ctx["z"], z0, p, lo, hi) / fbar)
        return sizing
    return make_sizing


# ============ 評価 ============
def emp_row(name, pool, closes, mk, max_pos=6):
    r = mm.evaluate_method(name, pool, closes, mk, target_dd=0.20, max_pos=max_pos, n_boot=1500)
    return r


def robust_k(pool, closes, mk, max_pos=6):
    k, eqm, eqr, info, p95 = mm.calibrate_robust(pool, closes, mk, target_dd=0.20,
                                                 max_pos=max_pos, n_boot=600)
    s = mm.stats(eqm, eqr, info)
    return {"k": k, "cagr": s["cagr"], "maxdd_mtm": s["maxdd_mtm"], "p95": -p95,
            "sharpe": s["sharpe"], "pos_year_rate": s["pos_year_rate"]}


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"プール {len(pool)} トレード / グリッド {len(closes)} 本\n")

    methods = [
        ("固定比率 mp6 (baseline)",        make_flat(6), 6),
        ("乖離連動z mp6",                   make_zsize(pool, 6), 6),
        ("乖離連動z mp8",                   make_zsize(pool, 8), 8),
        ("max_pos=8 (z無)",                make_flat(8), 8),
        ("Kelly[full=リーク] mp6",          make_kelly(pool, 6, "full"), 6),
        ("Kelly[WF=因果] mp6",             make_kelly(pool, 6, "wf"), 6),
        ("Kelly[WF]×z mp6",                make_combo(pool, 6, "wf"), 6),
        ("Kelly[WF]×z mp8",                make_combo(pool, 8, "wf"), 8),
        ("Kelly[full=リーク]×z mp8",        make_combo(pool, 8, "full"), 8),
    ]

    print("=" * 120)
    print("【基準A: 経験的 MtM最大DD = 20% に較正 → CAGR最大化】")
    print("=" * 120)
    print(f"{'手法':<26}{'k':>7}{'CAGR':>9}{'MtM_DD':>8}{'実現DD':>8}{'Sharpe':>7}{'+年':>5}"
          f"{'p95':>8}{'p99':>8}{'OOS_CAGR':>9}{'OOS_DD':>8}{'OOS+年':>7}")
    emp = {}
    for name, mk, mp in methods:
        r = emp_row(name, pool, closes, mk, mp)
        emp[name] = r
        beat = "✓" if r["cagr"] > BASELINE_CAGR + 0.001 else ("=" if abs(r["cagr"]-BASELINE_CAGR) <= 0.001 else "✗")
        print(f"{name:<26}{r['k']:>7.2f}{r['cagr']:>+9.2%}{r['maxdd_mtm']:>+8.1%}{r['maxdd_real']:>+8.1%}"
              f"{r['sharpe']:>7.2f}{r['pos_year_rate']:>5.0%}{r['boot_p95']:>+8.1%}{r['boot_p99']:>+8.1%}"
              f"{r.get('oos_cagr',float('nan')):>+9.2%}{r.get('oos_maxdd_mtm',float('nan')):>+8.1%}"
              f"{r.get('oos_pos_year',float('nan')):>7.0%} {beat}")

    print("\n" + "=" * 120)
    print("【基準B: 堅牢 — ブートストラップ理論DD(p95, 20回に1回級)= 20% に較正】(真の'理論上の最大DD'解釈)")
    print("=" * 120)
    print(f"{'手法':<26}{'k':>7}{'CAGR':>9}{'経験DD':>8}{'p95':>8}{'Sharpe':>7}{'+年':>5}")
    for name, mk, mp in [methods[0], methods[1], methods[2], methods[7]]:
        rb = robust_k(pool, closes, mk, mp)
        print(f"{name:<26}{rb['k']:>7.2f}{rb['cagr']:>+9.2%}{rb['maxdd_mtm']:>+8.1%}{rb['p95']:>+8.1%}"
              f"{rb['sharpe']:>7.2f}{rb['pos_year_rate']:>5.0%}")

    print("\n" + "=" * 120)
    print("【ケリーのリーク切り分け(基準A・mp6)】")
    print("=" * 120)
    f_full = emp["Kelly[full=リーク] mp6"]
    f_wf = emp["Kelly[WF=因果] mp6"]
    print(f"  Kelly full(リーク): CAGR={f_full['cagr']:+.2%}  OOS={f_full.get('oos_cagr',float('nan')):+.2%}")
    print(f"  Kelly WF(因果)    : CAGR={f_wf['cagr']:+.2%}  OOS={f_wf.get('oos_cagr',float('nan')):+.2%}")
    print(f"  → リークの寄与 ≈ {(f_full['cagr']-f_wf['cagr'])*100:+.1f}pp(これがリークで水増しされていた分)")


if __name__ == "__main__":
    main()
