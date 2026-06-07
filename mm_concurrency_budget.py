"""資金管理手法: 同時建玉数に応じたリスクバジェット配分(凹関数で多玉時に絞る)。

狙い: 多数の逆張り玉が同時に積み上がる局面(2022トレンド期=DDの主因)で総エクスポージャの
膨張を抑え、テールDDを削る。固定比率(alloc=equity*k/max_pos)は満玉でも各玉一定だが、
本手法は「積み上がる過程」の途中リスクを n_open に応じて動的に絞る。

サイジング関数は全て総建玉が k に対して線形(較正の単調性を保証)。
make_sizing(k, **shape) を evaluate_method に渡す。shape は budget 形状と m0/alpha/gamma。

budget 形状(いずれも alloc = equity_real * k * w(n_open)、w は n_open に対して非増加):
  - "harmonic"  : w = 1 / max(n_open+1, m0)   満玉時に総枠が k へ漸近(等リスク総枠の発想)
  - "lindecay"  : w = (1/max_pos) * (1 - alpha * n_open / max_pos)  k/max_pos を線形減衰
  - "power"     : w = (1/max_pos) * (1 - n_open/max_pos)**gamma     凹/凸を gamma で調整
  - "fixed"     : w = 1/max_pos                                    ベースライン(対照)

立ち上がり期(recent_vol nan)は budget だけで建玉(vol非依存なので問題なし)。
clip上限は alloc <= equity_real * cap で頭打ち回避のため十分広く設定。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm


def make_sizing(k, shape="harmonic", m0=1.0, alpha=0.5, gamma=1.0, max_pos=6, cap=3.0):
    """総建玉を k 倍に線形スケールする「同時建玉数連動バジェット」サイジングを返す。

    w(n_open) は n_open に対して非増加(凹的)。alloc = equity_real * k * w(n_open)。
    全 shape で alloc は k に線形 → calibrate の単調性が成立。
    """
    inv_mp = 1.0 / max_pos

    def _w(n_open):
        if shape == "harmonic":
            return 1.0 / max(n_open + 1, m0)
        if shape == "lindecay":
            return inv_mp * max(0.0, 1.0 - alpha * n_open / max_pos)
        if shape == "power":
            frac = max(0.0, 1.0 - n_open / max_pos)
            return inv_mp * (frac ** gamma)
        if shape == "fixed":
            return inv_mp
        raise ValueError(shape)

    def _f(ctx):
        n = ctx["n_open"]
        w = _w(n)
        alloc = ctx["equity_real"] * k * w
        # clip 上限(較正が頭打ちにならぬよう十分広く)
        cap_dollars = ctx["equity_real"] * cap
        return min(alloc, cap_dollars)

    return _f


def sweep():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"pool {pool.shape} closes {closes.shape}\n")

    configs = []
    # harmonic: m0 で「漸近する満玉総枠」を制御。m0小=より凹(序盤に厚く)
    for m0 in [1.0, 2.0, 3.0, 4.0]:
        configs.append(("harmonic", dict(shape="harmonic", m0=m0), f"harmonic m0={m0}"))
    # linear decay: alpha で減衰の急さ。alpha=0 は fixed と同じ
    for alpha in [0.3, 0.5, 0.7, 0.9]:
        configs.append(("lindecay", dict(shape="lindecay", alpha=alpha), f"lindecay alpha={alpha}"))
    # power: gamma>1 で多玉時を強く絞る(凸減衰), gamma<1 で緩やか
    for gamma in [0.5, 1.0, 1.5, 2.0]:
        configs.append(("power", dict(shape="power", gamma=gamma), f"power gamma={gamma}"))
    # 対照: fixed (ベースライン)
    configs.append(("fixed", dict(shape="fixed"), "fixed (baseline)"))

    print(f"{'config':<22} {'k':>6} {'CAGR':>8} {'DD_mtm':>8} {'DD_real':>8} {'Sharpe':>7} "
          f"{'posYr':>6} {'p95':>7} {'p99':>7} {'oCAGR':>7} {'oDD':>7} {'maxC':>5}")
    rows = []
    for tag, shape, label in configs:
        ms = lambda k, shape=shape: make_sizing(k, **shape)
        r = mm.evaluate_method(label, pool, closes, ms, n_boot=400)
        rows.append(r)
        print(f"{label:<22} {r['k']:>6.2f} {r['cagr']:>+8.2%} {r['maxdd_mtm']:>+8.2%} "
              f"{r['maxdd_real']:>+8.2%} {r['sharpe']:>7.2f} {r['pos_year_rate']:>6.0%} "
              f"{r['boot_p95']:>+7.1%} {r['boot_p99']:>+7.1%} "
              f"{r.get('oos_cagr', float('nan')):>+7.1%} {r.get('oos_maxdd_mtm', float('nan')):>+7.1%} "
              f"{r['max_conc']:>5}")
    return rows


def make_sizing_tilt(k, beta=0.0, max_pos=6, cap=3.0):
    """w(n) = (1/max_pos) * (1 + beta*(1 - 2*n/(max_pos-1))) を正規化なしで使う一般化チルト。
    beta>0 = 少玉時に厚く・多玉時に薄く(凹的減衰), beta<0 = 逆(凸的)。beta=0 で fixed。
    n は 0..max_pos-1 を想定。clamp で非負。線形 in k は保たれる。
    """
    inv_mp = 1.0 / max_pos
    denom = max(max_pos - 1, 1)

    def _f(ctx):
        n = ctx["n_open"]
        tilt = 1.0 + beta * (1.0 - 2.0 * n / denom)  # n=0 -> 1+beta, n=max-1 -> 1-beta
        w = inv_mp * max(0.0, tilt)
        alloc = ctx["equity_real"] * k * w
        return min(alloc, ctx["equity_real"] * cap)

    return _f


def sweep_fine():
    """fixed 近傍を細かく: 軽いチルト beta と軽い lindecay alpha。"""
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"=== fine sweep around fixed === pool {pool.shape}\n")
    print(f"{'config':<22} {'k':>6} {'CAGR':>8} {'DD_mtm':>8} {'Sharpe':>7} "
          f"{'posYr':>6} {'p95':>7} {'p99':>7} {'oCAGR':>7} {'oDD':>7}")
    rows = []
    cfgs = []
    for beta in [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]:
        cfgs.append((lambda k, b=beta: make_sizing_tilt(k, beta=b), f"tilt beta={beta}"))
    for alpha in [0.1, 0.2]:
        cfgs.append((lambda k, a=alpha: make_sizing(k, shape="lindecay", alpha=a),
                     f"lindecay alpha={alpha}"))
    for r_ms, label in cfgs:
        r = mm.evaluate_method(label, pool, closes, r_ms, n_boot=400)
        rows.append((label, r))
        print(f"{label:<22} {r['k']:>6.2f} {r['cagr']:>+8.2%} {r['maxdd_mtm']:>+8.2%} "
              f"{r['sharpe']:>7.2f} {r['pos_year_rate']:>6.0%} "
              f"{r['boot_p95']:>+7.1%} {r['boot_p99']:>+7.1%} "
              f"{r.get('oos_cagr', float('nan')):>+7.1%} {r.get('oos_maxdd_mtm', float('nan')):>+7.1%}")
    return rows


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fine":
        sweep_fine()
    else:
        sweep()
