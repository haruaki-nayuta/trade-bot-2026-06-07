"""資金管理: 分数ケリー(Kelly criterion)サイジング — チャンピオンv2。

目的: トレードプールの ret 群から成長最適レバレッジ(ケリー)を推定し、
**分数ケリー**を make_sizing(k) に対応させて 20%DD較正でCAGRを最大化できるか実測する。

ケリーの3つの顔(すべて実測):
  1) 連続版 f* = mean(ret)/var(ret)   … 2次近似(テール無視で過大)
  2) 厳密 単一ベット f* = argmax E[log(1+f*ret)] … 真の log成長最適レバ(テール込み)
  3) 勝敗版 f* = W - (1-W)/R           … 二値ベット近似

サイジングの shape(配分の"形"):
  - "flat"   : 全トレード等ケリー重み = 各トレードに base_frac の配分。
               → 定数レバ ⇒ 数学的に固定比率(baseline)と一致するはず(検証)。
  - "perinst": 銘柄別ケリー。各銘柄の自前 ret 群から log最適 f_i を推定し、
               配分を f_i に比例(全体平均で正規化)。"形"を変える唯一の自由度。
  - "zkelly" : z(乖離の深さ)で配分を傾ける(深い乖離ほどケリー比例で厚く)。実験的。

make_sizing(k) は必ず総建玉を k に線形スケール(較正の単調性を保つ)。
recent_vol は使わない(ケリーは vol ターゲットではなく "edge/odds" ベース)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mm_lab import (build_pool, load_closes, default_instruments,
                    evaluate_method, fixed_fractional, simulate, stats,
                    bootstrap_maxdd, calibrate)

MAX_POS = 6


# --- ケリー推定子 -------------------------------------------------------
def kelly_continuous(r: np.ndarray) -> float:
    """連続版 f* = mean/var(2次近似・テール無視で過大)。"""
    v = r.var()
    return float(r.mean() / v) if v > 0 else 0.0


def kelly_logopt(r: np.ndarray, fmax=60.0, n=4000) -> float:
    """厳密な単一ベット log成長最適 f* = argmax E[log(1+f*ret)]。テール込み。"""
    if len(r) < 5:
        return 0.0
    fs = np.linspace(0.0, fmax, n)
    # E[log(1+f*r)]; 1+f*r<=0 は -inf 扱い
    best_f, best_g = 0.0, -1e18
    for f in fs:
        x = 1.0 + f * r
        if (x <= 0).any():
            break  # これ以上 f を上げると破産項が出る(単調)
        g = np.mean(np.log(x))
        if g > best_g:
            best_g, best_f = g, f
    return float(best_f)


def kelly_winloss(r: np.ndarray) -> float:
    """勝敗版 f* = W - (1-W)/R, R=avg_win/avg_loss。"""
    W = float((r > 0).mean())
    aw = r[r > 0].mean() if (r > 0).any() else 0.0
    al = -r[r < 0].mean() if (r < 0).any() else 1e-9
    R = aw / al if al > 0 else 0.0
    return float(W - (1 - W) / R) if R > 0 else 0.0


# --- per-instrument ケリー重み(銘柄別の自前 edge) ---------------------
def per_instrument_weights(pool: pd.DataFrame, min_trades=15, shrink=0.5) -> dict:
    """各銘柄の自前 ret 群から log最適 f_i を推定 → 平均1.0 へ正規化した相対重み。

    取引数が少ない銘柄は shrink で全体推定へ縮小(過剰最適化防止)。
    """
    glob = kelly_logopt(pool["ret"].to_numpy())
    w = {}
    for nm, g in pool.groupby("instr"):
        r = g["ret"].to_numpy()
        if len(r) < min_trades:
            fi = glob
        else:
            fi = kelly_logopt(r)
            fi = shrink * fi + (1 - shrink) * glob  # 縮小推定
        w[nm] = fi
    arr = np.array(list(w.values()))
    mean_w = arr[arr > 0].mean() if (arr > 0).any() else 1.0
    return {k: (v / mean_w if mean_w > 0 else 1.0) for k, v in w.items()}


# --- make_sizing 群 -----------------------------------------------------
def make_flat(max_pos=MAX_POS):
    """定数ケリー(全トレード等重み)。alloc = equity * k / max_pos。
    → 固定比率と数学的に同一(ケリーは定数レバ)。baseline 一致の検証用。"""
    def make_sizing(k):
        w = k / max_pos
        def sizing(ctx):
            return ctx["equity_real"] * w
        return sizing
    return make_sizing


def make_perinst(pool: pd.DataFrame, max_pos=MAX_POS, clip=4.0, **kw):
    """銘柄別ケリー。alloc = equity * (k/max_pos) * w_instr。w は平均1へ正規化済。"""
    weights = per_instrument_weights(pool, **kw)
    # 平均1なので総建玉は k に線形(銘柄ミックスが一定なら厳密、概ね線形)
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            wi = weights.get(ctx["instr"], 1.0)
            wi = min(wi, clip)
            return ctx["equity_real"] * base * wi
        return sizing
    return make_sizing


def make_zkelly(pool: pd.DataFrame, max_pos=MAX_POS, beta=0.3, clip=4.0):
    """z(乖離の深さ)連動。w(z)=1+beta*(z/zbar -1)。深い乖離ほど厚く。"""
    zbar = float(pool["z_entry"].mean())
    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            z = ctx.get("z", zbar)
            if not np.isfinite(z):
                z = zbar
            w = 1.0 + beta * (z / zbar - 1.0)
            w = float(np.clip(w, 0.2, clip))
            return ctx["equity_real"] * base * w
        return sizing
    return make_sizing


def _row(name, res):
    return (f"{name:<26} k={res['k']:>6.2f}  CAGR={res['cagr']:>+7.2%}  "
            f"DD={res['maxdd_mtm']:>+6.1%}  real={res['maxdd_real']:>+6.1%}  "
            f"Sh={res['sharpe']:>4.2f}  +yr={res['pos_year_rate']:>4.0%}  "
            f"p95={res['boot_p95']:>+6.1%} p99={res['boot_p99']:>+6.1%}  "
            f"OOS:CAGR={res.get('oos_cagr',float('nan')):>+7.2%} DD={res.get('oos_maxdd_mtm',float('nan')):>+6.1%} +yr={res.get('oos_pos_year',float('nan')):>4.0%}")


def main():
    pool = build_pool()
    closes = load_closes()
    r = pool["ret"].to_numpy()

    print("=== ケリー推定(全プール) ===")
    print(f"  n={len(r)}  win_rate={(r>0).mean():.3f}  "
          f"avg_win={r[r>0].mean():.4f}  avg_loss={-r[r<0].mean():.4f}  "
          f"R={r[r>0].mean()/-r[r<0].mean():.3f}")
    f_cont = kelly_continuous(r)
    f_log = kelly_logopt(r)
    f_wl = kelly_winloss(r)
    print(f"  連続版 f*(mean/var)       = {f_cont:6.2f}  (2次近似・テール無視で過大)")
    print(f"  厳密 単一ベット f*(logopt) = {f_log:6.2f}  (真の log成長最適・テール込み)")
    print(f"  勝敗版 f*(W-(1-W)/R)       = {f_wl:6.3f}  (二値ベット bankroll比率)")
    print(f"  → フルケリー(logopt) の worst-trade equity 変化 = {f_log*r.min():+.1%}(破産級)")
    print()

    print("=== per-instrument ケリー(銘柄別 logopt f_i, 平均1へ正規化) ===")
    w = per_instrument_weights(pool)
    for nm in sorted(w, key=lambda x: -w[x]):
        nt = int((pool["instr"] == nm).sum())
        print(f"  {nm:<8} w={w[nm]:>5.2f}  (n={nt})")
    print()

    print("=== 20%DD較正 → CAGR比較(スイープ n_boot=400) ===")
    results = {}

    res = evaluate_method("Kelly-flat(=固定比率)", pool, closes, make_flat(), n_boot=400)
    results["flat"] = res
    print(_row("Kelly-flat (定数レバ)", res))

    res = evaluate_method("Kelly-perinst", pool, closes, make_perinst(pool), n_boot=400)
    results["perinst"] = res
    print(_row("Kelly-perinst (銘柄別)", res))

    # perinst の縮小度/クリップ近傍(高原性チェック)
    for shrink in [0.3, 0.7]:
        res = evaluate_method(f"perinst shrink={shrink}", pool, closes,
                              make_perinst(pool, shrink=shrink), n_boot=400)
        print(_row(f"  perinst shrink={shrink}", res))
    for clip in [3.0, 6.0]:
        res = evaluate_method(f"perinst clip={clip}", pool, closes,
                              make_perinst(pool, clip=clip), n_boot=400)
        print(_row(f"  perinst clip={clip}", res))

    for beta in [0.2, 0.4]:
        res = evaluate_method(f"zkelly beta={beta}", pool, closes,
                              make_zkelly(pool, beta=beta), n_boot=400)
        print(_row(f"  zkelly beta={beta}", res))

    return results


if __name__ == "__main__":
    main()
