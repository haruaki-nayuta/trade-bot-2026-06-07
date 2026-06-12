"""verify70c: 「直前5全勝→次勝率78%」セルの事後選択補正と頑健性監査。

verify70b で 5/5 セル単体は置換帰無 p≈0.012 を生き残った。しかしこれは
6 セル中の極値を事後に選んだ統計量。ここで決着をつける:
  (1) max-cell 置換検定 — 帰無下でも「どこかのセル」が大きく振れる確率
      (n>=30 のセルのみ対象、統計量 = max_w |wr_w - wr_rest|)
  (2) 期間半割(2016-2020 / 2021-2026)での 5/5 効果の再現性
  (3) ストリーク長感度 — 直前 k 全勝(k=3,4,5,6,7)の Δ勝率と素朴 z。
      k=5 だけ立つなら事後選択の証拠
実行: PYTHONPATH=. uv run python research/outputs/verify70c_streak_robust.py
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


def cond_sets_for(ent_ns, ext_ns, k):
    order = np.argsort(ext_ns, kind="stable")
    ext_sorted = ext_ns[order]
    cs = []
    for i in range(len(ent_ns)):
        kk = int(np.searchsorted(ext_sorted, ent_ns[i], side="left"))
        cs.append(order[kk - k:kk] if kk >= k else None)
    return cs


def wins_state(cs, r, k):
    w = np.full(len(r), -1)
    for i, c in enumerate(cs):
        if c is not None:
            w[i] = int((r[c] > 0).sum())
    return w


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    psort = pool.sort_values("entry").reset_index(drop=True)
    r = psort["ret"].to_numpy()
    instr_arr = psort["instr"].to_numpy()
    ent_ns = psort["entry"].astype("int64").to_numpy()
    ext_ns = psort["exit"].astype("int64").to_numpy()
    years = psort["entry"].dt.year.to_numpy()
    res = {}

    # ---- (1) max-cell 置換検定 -------------------------------------------------
    cs5 = cond_sets_for(ent_ns, ext_ns, 5)

    def maxcell_stat(rr):
        w5 = wins_state(cs5, rr, 5)
        valid = w5 >= 0
        best = 0.0
        for w in range(6):
            m = w5 == w
            if m.sum() < 30:
                continue
            d = abs((rr[m] > 0).mean() - (rr[valid & ~m] > 0).mean())
            best = max(best, float(d))
        return best

    obs = maxcell_stat(r)
    rng = np.random.default_rng(11)
    slots = [np.where(instr_arr == s)[0] for s in np.unique(instr_arr)]
    n_boot = 2000
    null = np.empty(n_boot)
    for b in range(n_boot):
        shuf = r.copy()
        for s in slots:
            shuf[s] = rng.permutation(shuf[s])
        null[b] = maxcell_stat(shuf)
    p_max = float((null >= obs).mean())
    res["maxcell"] = {"obs": obs, "p": p_max}
    print(f"(1) max-cell 統計量: 観測 {obs:.3f} → 置換 p={p_max:.3f}")

    # ---- (2) 期間半割 -----------------------------------------------------------
    w5 = wins_state(cs5, r, 5)
    valid = w5 >= 0
    halves = {}
    for label, m_per in (("2016-2020", years <= 2020), ("2021-2026", years >= 2021)):
        m5 = (w5 == 5) & m_per
        mr = valid & ~(w5 == 5) & m_per
        halves[label] = {
            "n_5of5": int(m5.sum()),
            "wr_5of5": float((r[m5] > 0).mean()),
            "wr_rest": float((r[mr] > 0).mean()),
            "delta": float((r[m5] > 0).mean() - (r[mr] > 0).mean()),
            "mr_5of5_bps": float(r[m5].mean() * 1e4),
            "mr_rest_bps": float(r[mr].mean() * 1e4),
        }
        h = halves[label]
        print(f"(2) {label}: 5/5 n={h['n_5of5']} 勝率 {h['wr_5of5']:.1%} vs 他 "
              f"{h['wr_rest']:.1%} (Δ={h['delta']:+.1%}) | 平均 {h['mr_5of5_bps']:+.1f} "
              f"vs {h['mr_rest_bps']:+.1f}bps")
    res["halves"] = halves

    # ---- (3) ストリーク長感度 ----------------------------------------------------
    sens = {}
    for k in (3, 4, 5, 6, 7):
        csk = cond_sets_for(ent_ns, ext_ns, k)
        wk = wins_state(csk, r, k)
        vk = wk >= 0
        mk = wk == k
        nk = int(mk.sum())
        wr_s = float((r[mk] > 0).mean())
        wr_o = float((r[vk & ~mk] > 0).mean())
        # 素朴2標本比率 z(参考値)
        pbar = float((r[vk] > 0).mean())
        se = np.sqrt(pbar * (1 - pbar) * (1 / nk + 1 / int((vk & ~mk).sum())))
        sens[f"k{k}"] = {"n_streak": nk, "wr_streak": wr_s, "wr_rest": wr_o,
                         "delta": wr_s - wr_o, "naive_z": float((wr_s - wr_o) / se)}
        s = sens[f"k{k}"]
        print(f"(3) 直前{k}全勝: n={nk:>4} 勝率 {wr_s:.1%} vs 他 {wr_o:.1%} "
              f"(Δ={s['delta']:+.1%}, 素朴z={s['naive_z']:+.2f})")
    res["streak_len_sensitivity"] = sens

    out = ROOT / "research" / "outputs" / "verify70c_result.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"saved -> {out}  経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
