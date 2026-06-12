"""exp54: サイジング層スカウト(d1プール固定・seed0)— CAGR20%越え第3ラウンドの候補出し。

背景: d1 採用(reports/18)でシグナルバー(t)と執行バー(t+1)が分離したが、サイジングの
z 入力は t 時点の |z| のまま(検証済み構成の踏襲)。執行バー t+1 の z は執行時点で確定済み
(=因果)であり、逆行第1波を経た「より新しい乖離の深さ」を持つ。reports/14 の
「深い乖離ほど期待リターン勾配が大きい」が正しければ、入力の鮮度更新はタダで取れる利得の候補。
併せて、d0 時代に較正された配分形状(P/z0/clip)が d1 プールでもプラトー中央のままかを OAT で点検する。

変更は全て「配分の形」のみ(プール=トレード集合は cached d1 1207件で固定、ret 不変)。
fbar 正規化は各変種の入力で再計算(形だけ比較、総量は k 較正が吸収)。

判定(スカウト段): empirical較正 + robust seed0。生き残りは exp56 でフルプロトコル。

実行: PYTHONPATH=. uv run python research/experiments/exp54_sizing_scout.py
出力: research/outputs/exp54_scout.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
from mm_production import build_pool_d1  # noqa: E402
from tail_protocol import protocol_eval  # noqa: E402
from fxlab import universe as uni  # noqa: E402

MAX_POS = 8
OUT = ROOT / "research" / "outputs" / "exp54_scout.csv"


def z_exec_of(pool: pd.DataFrame, win: int = 50) -> np.ndarray:
    """執行バー(pool.entry=遅延先バー)時点の |z|(シフトなし=そのバーの確定 close まで)。"""
    out = np.full(len(pool), np.nan)
    for instr, g in pool.groupby("instr"):
        s = uni.instrument_close(instr, "H4")
        z = (s - s.rolling(win).mean()) / s.rolling(win).std()
        out[g.index.to_numpy()] = np.abs(z.reindex(g["entry"]).to_numpy())
    return out


def make_sizing_with(zvals: np.ndarray, pool: pd.DataFrame, *, p=4.0, z0=2.2,
                     lo=0.3, hi=3.0, max_pos=MAX_POS):
    """champion_sizing と同形だが z 入力ベクトル・P/z0/clip を差し替え可能にした版。

    simulate() の ctx["z"] はプール列 z_entry 由来なので、トレード順の zvals を
    エントリーのタイムスタンプ経由でなく「プールをコピーして z_entry を差し替える」
    方式で渡す(呼び出し側で pool2 を使う)。本関数は f(z) と fbar のみ定義。
    """
    def fz(z):
        return float(np.clip((z / z0) ** p, lo, hi)) if np.isfinite(z) else 1.0
    fbar = float(np.mean([fz(z) for z in zvals])) or 1.0

    def make(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (fz(ctx["z"]) / fbar)
    return make


def eval_cfg(label, pool, closes, mk, seeds=(0,)):
    cache = {}

    def eq_of_k(k):
        kk = round(float(k), 10)
        if kk not in cache:
            cache[kk] = mm.simulate(pool, closes, mk(kk), max_pos=MAX_POS)[0]
        return cache[kk]
    return protocol_eval(eq_of_k, label=label, seeds=seeds)


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1()
    closes = mm.load_closes()
    print(f"=== exp54: サイジング層スカウト (d1 pool n={len(pool)}, mp{MAX_POS}, seed0) ===")

    z_sig = pool["z_entry"].to_numpy()
    z_ex = z_exec_of(pool)
    n_nan = int((~np.isfinite(z_ex)).sum())
    print(f"z_exec: nan {n_nan}件 / corr(z_sig, z_exec) = "
          f"{np.corrcoef(z_sig[np.isfinite(z_ex)], z_ex[np.isfinite(z_ex)])[0,1]:.3f} / "
          f"mean z_sig {np.nanmean(z_sig):.3f} -> z_exec {np.nanmean(z_ex):.3f} / "
          f"z_exec>z_sig 割合 {np.nanmean((z_ex > z_sig).astype(float)):.1%}")

    # z 入力変種(z_entry 列を差し替えたプールで simulate に流す)
    z_max = np.fmax(z_sig, z_ex)
    z_mean = np.where(np.isfinite(z_ex), 0.5 * (z_sig + z_ex), z_sig)

    cfgs = []
    # 1) z 入力の鮮度
    for tag, zv in [("base(z_sig)", z_sig), ("z_exec", np.where(np.isfinite(z_ex), z_ex, z_sig)),
                    ("z_max", z_max), ("z_mean", z_mean)]:
        cfgs.append((tag, zv, dict()))
    # 2) 配分形状 OAT(z 入力は base のまま)
    for tag, kw in [("P=3.5", dict(p=3.5)), ("P=4.5", dict(p=4.5)),
                    ("z0=2.0", dict(z0=2.0)), ("z0=2.4", dict(z0=2.4)),
                    ("clip_hi=2.5", dict(hi=2.5)), ("clip_hi=4.0", dict(hi=4.0)),
                    ("clip_hi=5.0", dict(hi=5.0)),
                    ("clip_lo=0.2", dict(lo=0.2)), ("clip_lo=0.5", dict(lo=0.5))]:
        cfgs.append((tag, z_sig, kw))

    rows = []
    base_rob = None
    for tag, zv, kw in cfgs:
        pool2 = pool.copy()
        pool2["z_entry"] = zv
        mk = make_sizing_with(zv, pool2, **kw)
        r = eval_cfg(tag, pool2, closes, mk, seeds=(0,))
        row = {"cfg": tag, "emp_k": r["emp_k"], "emp_cagr": r["emp_cagr"],
               "emp_p95": r["emp_p95"], "rob_s0": r["rob"][0]["cagr"],
               "rob_k0": r["rob"][0]["k"]}
        if tag.startswith("base"):
            base_rob = row["rob_s0"]
        row["d_rob_s0_pp"] = (row["rob_s0"] - base_rob) * 100 if base_rob is not None else np.nan
        rows.append(row)
        print(f"    [{time.time()-t0:.0f}s]")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\n=== スカウト結果(seed0, base 比 pp) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4f}"):
        print(df.to_string(index=False))
    print(f"\nsaved -> {OUT}\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
