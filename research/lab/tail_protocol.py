"""tail_protocol — 同一テール判定プロトコルの共有実装(reports/11 で確立)。

判定の物差し:
  1) empirical 較正: 単一パス MtM 最大DD = 20% に k を二分探索 → CAGR
  2) robust 較正  : ブロックブートストラップ p95 DD = 20% に k を二分探索 → CAGR
     ※ p95 推定はシード依存(±0.4〜0.8pp の較正ノイズ)。**必ず複数シードで、
       ベースラインと候補を同一シード集合で較正して比較する(ペアシード)**。
  3) レバ偽装署名: 「empirical CAGR↑ かつ p95 悪化」は前進と認めない([[same-tail-protocol]])。

使い方(候補ごとに eq_of_k: k -> MtM equity Series を渡すだけ):
    from tail_protocol import protocol_eval, paired_table
    res = protocol_eval(lambda k: my_simulate(k), label="carry w=0.5", seeds=(0,1,2))
短保有ストリーム(保有 ≲ 1 H4バー)を含む場合は M1 粒度 DD 監査が別途必要(exp24/31)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_dd(eq: pd.Series) -> float:
    return float((eq / eq.cummax() - 1.0).min())


def cagr_of(eq: pd.Series) -> float:
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    final = eq.iloc[-1] / eq.iloc[0]
    return final ** (1 / years) - 1 if final > 0 else -1.0


def boot_dd(eq: pd.Series, n_boot=600, block=63, seed=0) -> dict:
    """mm_lab.bootstrap_maxdd と同一ロジック(シード明示)。p95 = 5%分位の最大DD。"""
    r = eq.pct_change().replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    n = len(r)
    if n < block * 2:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts_all = rng.integers(0, n - block, size=(n_boot, n_blocks))
    dds = np.empty(n_boot)
    for i in range(n_boot):
        idx = (starts_all[i][:, None] + np.arange(block)).ravel()[:n]
        path = np.cumprod(1.0 + r[idx])
        peak = np.maximum.accumulate(path)
        dds[i] = (path / peak - 1.0).min()
    return {"p50": float(np.percentile(dds, 50)), "p95": float(np.percentile(dds, 5)),
            "p99": float(np.percentile(dds, 1)), "worst": float(dds.min())}


def calibrate_empirical(eq_of_k, target=0.20, lo=0.02, hi=16.0, iters=22) -> float:
    """単一パス MtM 最大DD == target になる最大の k。"""
    if abs(max_dd(eq_of_k(hi))) <= target:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if abs(max_dd(eq_of_k(mid))) > target:
            hi = mid
        else:
            lo = mid
    return lo


def calibrate_robust_seeded(eq_of_k, target=0.20, n_boot=600, block=63, seed=0,
                            lo=0.02, hi=16.0, iters=16) -> float:
    """ブート p95 DD == target になる k(シード明示)。"""
    def p95(k):
        return abs(boot_dd(eq_of_k(k), n_boot=n_boot, block=block, seed=seed)["p95"])
    if p95(hi) <= target:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if p95(mid) > target:
            hi = mid
        else:
            lo = mid
    return lo


def protocol_eval(eq_of_k, label="", target=0.20, seeds=(0, 1, 2), n_boot_cal=600,
                  n_boot_verify=1500, verbose=True) -> dict:
    """1候補のフル評価: empirical較正 + 複数シードrobust較正。

    返り値 dict:
      emp_k, emp_cagr, emp_dd, emp_p95(=1500本,seed0で測った理論テール)
      rob: {seed: {k, cagr}}, rob_cagr_mean
    """
    k_emp = calibrate_empirical(eq_of_k, target)
    eq_e = eq_of_k(k_emp)
    bs = boot_dd(eq_e, n_boot=n_boot_verify, seed=0)
    out = {"label": label, "emp_k": k_emp, "emp_cagr": cagr_of(eq_e),
           "emp_dd": max_dd(eq_e), "emp_p95": bs["p95"], "emp_p99": bs["p99"]}
    rob = {}
    for sd in seeds:
        k_r = calibrate_robust_seeded(eq_of_k, target, n_boot=n_boot_cal, seed=sd)
        eq_r = eq_of_k(k_r)
        rob[sd] = {"k": k_r, "cagr": cagr_of(eq_r)}
    out["rob"] = rob
    out["rob_cagr_mean"] = float(np.mean([v["cagr"] for v in rob.values()]))
    out["rob_k_mean"] = float(np.mean([v["k"] for v in rob.values()]))
    if verbose:
        rc = " ".join(f"s{sd}:{v['cagr']:+.2%}" for sd, v in rob.items())
        print(f"  {label:34s} emp k={k_emp:5.2f} CAGR={out['emp_cagr']:+7.2%} "
              f"p95={bs['p95']:+6.1%} | robust mean={out['rob_cagr_mean']:+7.2%} ({rc})")
    return out


def yearly_returns(eq: pd.Series) -> pd.Series:
    y = eq.groupby(eq.index.year).last()
    r = y.pct_change()
    r.iloc[0] = y.iloc[0] / eq.iloc[0] - 1
    return r
