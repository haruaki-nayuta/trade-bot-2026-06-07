"""乖離連動サイズ(未踏フロンティア#1)— |エントリーZ| に建玉サイズを比例させる資金管理。

狙い: 平均回帰は乖離(|Z|)が深いほど期待反転が大きい(プールでも z>2.5 バケットの平均 ret は
z 2.0-2.5 の約2倍)。深い乖離の勝ちトレードに厚く張れば、同じ MtM最大DD=20% 較正下で CAGR を
押し上げられるか? を実測する。

  単一エントリー・分割しない(ナンピン非該当)。総量は k で制御。配分の"形" f(z) を固定し、
  各トレード alloc = equity_real * (k/max_pos) * f(z) / fbar。
  fbar は f(z) の経験平均で正規化 → f の絶対水準が k 較正に吸収されず、"形だけ"を変える。

リスク: 深い乖離は「一直線トレンドへの逆張り=塩漬け」でもありうる。深 z に厚く張ると含み損が
膨らみ MtM DD を悪化させ、較正 k が下がって CAGR が伸びない可能性。データで決着をつける。

実行: uv run python mm_zsize.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm

# --- f(z) 形状ファミリ -------------------------------------------------------
# z0: 基準乖離。z<z0 で f<1(薄め), z>z0 で f>1(厚め)。lo/hi で clip。

def _make_fz(kind, z0=2.2, p=1.0, lo=0.3, hi=3.0,
             s1=2.5, s2=3.0, m1=1.2, m2=1.5):
    """f(z) を返す。kind: 'flat'(=z無依存,sanity), 'linear', 'power', 'step'。"""
    if kind == "flat":
        return lambda z: 1.0
    if kind == "linear":
        return lambda z: float(np.clip(z / z0, lo, hi))
    if kind == "power":
        return lambda z: float(np.clip((z / z0) ** p, lo, hi))
    if kind == "step":
        def _f(z):
            if z >= s2:
                return m2
            if z >= s1:
                return m1
            return 1.0
        return _f
    raise ValueError(kind)


def make_sizing_factory(pool, kind, **kw):
    """evaluate_method に渡す make_sizing(k) を返すファクトリ。

    fbar = f(z) のプール平均(取引が来る順の重み付けは等加重で近似)で正規化するので、
    総建玉の平均水準は kind に依らず k に比例 → calibrate の単調性を保つ。
    """
    fz = _make_fz(kind, **kw)
    zvals = pool["z_entry"].to_numpy()
    fbar = float(np.mean([fz(z) if np.isfinite(z) else 1.0 for z in zvals]))
    if fbar <= 0:
        fbar = 1.0

    def make_sizing(k):
        base = k / 6.0  # max_pos=6 を想定(evaluate_method の既定)

        def sizing(ctx):
            z = ctx["z"]
            f = fz(z) if np.isfinite(z) else 1.0
            return ctx["equity_real"] * base * (f / fbar)

        return sizing

    return make_sizing


# --- スイープ -----------------------------------------------------------------
def sweep():
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"プール {len(pool)} トレード / グリッド {len(closes)} 本\n")

    configs = [
        ("flat (sanity=baseline形)", dict(kind="flat")),
        ("linear z0=2.2 lo.3 hi3", dict(kind="linear", z0=2.2, lo=0.3, hi=3.0)),
        ("linear z0=2.2 lo.5 hi2", dict(kind="linear", z0=2.2, lo=0.5, hi=2.0)),
        ("linear z0=2.0 lo.3 hi3", dict(kind="linear", z0=2.0, lo=0.3, hi=3.0)),
        ("power p0.5 z0=2.2", dict(kind="power", p=0.5, z0=2.2, lo=0.3, hi=3.0)),
        ("power p2 z0=2.2", dict(kind="power", p=2.0, z0=2.2, lo=0.3, hi=3.0)),
        ("power p3 z0=2.2", dict(kind="power", p=3.0, z0=2.2, lo=0.2, hi=4.0)),
        ("step 2.5->1.2 3.0->1.5", dict(kind="step", s1=2.5, s2=3.0, m1=1.2, m2=1.5)),
        ("step 2.5->1.5 3.0->2.0", dict(kind="step", s1=2.5, s2=3.0, m1=1.5, m2=2.0)),
        # 逆張り(浅 z に厚く)= 仮説の反証用
        ("INV power p-1 z0=2.2", dict(kind="power", p=-1.0, z0=2.2, lo=0.3, hi=3.0)),
    ]

    rows = []
    for name, kw in configs:
        mk = make_sizing_factory(pool, **kw)
        r = mm.evaluate_method(name, pool, closes, mk, n_boot=400)
        rows.append(r)
        print(f"{name:28s} k={r['k']:5.2f} CAGR={r['cagr']:+.2%} "
              f"DD={r['maxdd_mtm']:+.1%} Sh={r['sharpe']:.2f} "
              f"posY={r['pos_year_rate']:.0%} p95={r['boot_p95']:+.1%} "
              f"p99={r['boot_p99']:+.1%} | OOS CAGR={r.get('oos_cagr',float('nan')):+.2%} "
              f"OOSdd={r.get('oos_maxdd_mtm',float('nan')):+.1%} OOSposY={r.get('oos_pos_year',float('nan')):.0%}")
    return pool, closes, rows


if __name__ == "__main__":
    sweep()
