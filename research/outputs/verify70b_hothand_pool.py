"""verify70b: exp70 の未検定セル「直前5(exit確定済)5/5勝ち→次勝率78%」の検定。

exp70 は last5_wins_to_next の表を出力したが検定していない。素朴な2標本比率
検定では p=0.007 に見えるが、連続トレードは条件集合(直前5)を4/5共有するため
独立性が破れており素朴 p は過大有意。ここでは:
  (a) 銘柄内置換(exp70 と同じ帰無系) — ただし条件状態 wins5 をシャッフル後の
      ret から毎回再計算(状態が ret の関数である点を正しく扱う)
  (b) 銘柄×年内置換 — 年次レジーム(2025-26 の高勝率等)を保存した上で
      ストリーク固有の情報が残るか
統計量: Δ勝率(5/5 vs それ以外)と、wins5 と次結果の点双列相関(全セル使用)。

実行: PYTHONPATH=. uv run python research/outputs/verify70b_hothand_pool.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from mm_production import build_pool_d1  # noqa: E402
from fxlab import universe as uni  # noqa: E402


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    psort = pool.sort_values("entry").reset_index(drop=True)
    ret_arr = psort["ret"].to_numpy()
    instr_arr = psort["instr"].to_numpy()
    ent_ns = psort["entry"].astype("int64").to_numpy()
    ext_ns = psort["exit"].astype("int64").to_numpy()
    years = psort["entry"].dt.year.to_numpy()

    # 直前5(exit < entry)の条件集合インデックスを事前計算(時刻のみ依存=固定)
    order = np.argsort(ext_ns, kind="stable")
    ext_sorted = ext_ns[order]
    cond_sets = []
    for i in range(len(psort)):
        kk = int(np.searchsorted(ext_sorted, ent_ns[i], side="left"))
        cond_sets.append(order[kk - 5:kk] if kk >= 5 else None)

    def stats_for(r: np.ndarray):
        w5 = np.full(len(r), -1)
        for i, cs in enumerate(cond_sets):
            if cs is not None:
                w5[i] = int((r[cs] > 0).sum())
        valid = w5 >= 0
        m5 = w5 == 5
        d_wr = float((r[m5] > 0).mean() - (r[valid & ~m5] > 0).mean())
        # 点双列相関(wins5 と次の勝敗)
        x = w5[valid].astype(float)
        y = (r[valid] > 0).astype(float)
        corr = float(np.corrcoef(x, y)[0, 1])
        d_mr = float(r[m5].mean() - r[valid & ~m5].mean()) * 1e4
        return d_wr, corr, d_mr

    d_obs, c_obs, dm_obs = stats_for(ret_arr)
    print(f"観測: Δ勝率(5/5 vs 他)={d_obs:+.3f}  corr(wins5,次勝敗)={c_obs:+.4f}  "
          f"Δ平均ret={dm_obs:+.1f}bps")

    rng = np.random.default_rng(7)
    n_boot = 2000
    res = {"obs": {"delta_wr": d_obs, "corr": c_obs, "delta_mr_bps": dm_obs}}
    for label, strat in (("instr_shuffle", instr_arr),
                         ("instr_year_shuffle",
                          np.char.add(instr_arr.astype(str), years.astype(str)))):
        slots = [np.where(strat == s)[0] for s in np.unique(strat)]
        d_null = np.empty(n_boot)
        c_null = np.empty(n_boot)
        dm_null = np.empty(n_boot)
        for b in range(n_boot):
            shuf = ret_arr.copy()
            for s in slots:
                shuf[s] = rng.permutation(shuf[s])
            d_null[b], c_null[b], dm_null[b] = stats_for(shuf)
        p_d = float((np.abs(d_null) >= abs(d_obs)).mean())
        p_c = float((np.abs(c_null) >= abs(c_obs)).mean())
        p_dm = float((np.abs(dm_null) >= abs(dm_obs)).mean())
        # 片側(hot-hand 方向)も併記
        p_d1 = float((d_null >= d_obs).mean())
        res[label] = {"p_delta_wr_2s": p_d, "p_delta_wr_1s": p_d1,
                      "p_corr_2s": p_c, "p_delta_mr_2s": p_dm}
        print(f"{label:20s}: Δ勝率 p(両側)={p_d:.3f} p(片側)={p_d1:.3f} | "
              f"corr p={p_c:.3f} | Δret p={p_dm:.3f}")

    out = ROOT / "research" / "outputs" / "verify70b_result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"saved -> {out}  経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
