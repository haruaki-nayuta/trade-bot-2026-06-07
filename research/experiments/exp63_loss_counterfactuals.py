"""exp63: 損失テールへの対症療法の反実仮想(プール段) — exp62 の補完。

exp62 の発見: 損失は (a) 入口特徴量では勝者と判別不能 (b) 事後のボラ膨張・週末跨ぎ数と
単調に悪化(ただし両方とも「収束しなかった」ことの事後署名) (c) 98%は一度も+0.5%に
届かない緩慢ブリード(単発ギャップ寄与は13-17%)。

自然な対症療法2つを**反実仮想で実測**する(出口層は閉鎖済みだが、現行 d1 プールで
具体的な数字を出して「なぜ閉鎖が正しいか」をエッジケース込みで確認する):
  A) ボラ膨張カット: 保有中に vol20 > θ×エントリー時vol となった最初のバー close で決済
     (θ ∈ {2.0, 2.5, 3.0})
  B) 時間ストップ: N 本で強制決済(N ∈ {30, 40, 60})
評価: プール段の総リターン差・ワースト10%への影響・救済/誤殺の分解。
(プール段で純減なら口座段は測るまでもなく悪化=較正kも下がる)

実行: PYTHONPATH=. uv run python research/experiments/exp63_loss_counterfactuals.py
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
sys.path.insert(0, str(ROOT / "research" / "experiments"))

from mm_production import build_pool_d1  # noqa: E402
from fxlab import universe as uni  # noqa: E402


def main() -> int:
    t0 = time.time()
    uni.register_cross_spreads(3.0)
    pool = build_pool_d1().copy()
    n = len(pool)
    print(f"=== exp63: 損失対症療法の反実仮想 (d1 pool n={n}) ===")

    # パスとボラ系列を前計算
    paths, vols = {}, {}
    for instr, g in pool.groupby("instr"):
        d = uni.instrument_data(instr, "H4")
        close = d["close"]
        vol20 = close.pct_change().rolling(20).std()
        idx = close.index
        pos_of = pd.Series(np.arange(len(idx)), index=idx)
        e_pos = pos_of.reindex(g["entry"]).to_numpy()
        x_pos = pos_of.reindex(g["exit"]).to_numpy()
        carr = close.to_numpy()
        varr = vol20.to_numpy()
        for ti, e, x in zip(g.index.to_numpy(), e_pos, x_pos):
            e, x = int(e), int(x)
            dirv = float(pool.at[ti, "dir"])
            seg = carr[e:x + 1]
            paths[ti] = dirv * (seg / seg[0] - 1.0)
            vols[ti] = varr[e:x + 1]

    ret0 = pool["ret"].to_numpy()
    gross0 = np.array([paths[i][-1] for i in range(n)])
    cost = gross0 - ret0  # 往復コスト(再構成と同じ恒等式)
    worst_idx = set(pool["ret"].nsmallest(120).index)

    def evaluate(tag, cut_bar):
        """cut_bar[i] = 強制決済バー(パス内インデックス)。None=変更なし。"""
        ret_new = ret0.copy()
        n_cut = 0
        for i in range(n):
            cb = cut_bar(i)
            if cb is not None and cb < len(paths[i]) - 1:
                ret_new[i] = paths[i][cb] - cost[i]
                n_cut += 1
        diff = ret_new - ret0
        saved = diff[diff > 0].sum()
        killed = diff[diff < 0].sum()
        w_mask = np.array([i in worst_idx for i in range(n)])
        print(f"  [{tag}] 介入 {n_cut}件  プール差 {diff.sum():+.4f} "
              f"(救済 +{saved:.4f} / 誤殺 {killed:+.4f})")
        print(f"      ワースト10%への効果 {diff[w_mask].sum():+.4f} / "
              f"それ以外 {diff[~w_mask].sum():+.4f}  新プール合計 {ret_new.sum():+.4f} "
              f"(元 {ret0.sum():+.4f})")
        return diff.sum()

    print("\n--- A) ボラ膨張カット(vol20 > θ×入口vol で決済) ---")
    for th in (2.0, 2.5, 3.0):
        def cut(i, th=th):
            v = vols[i]
            v0 = v[0]
            if not np.isfinite(v0) or v0 <= 0:
                return None
            hit = np.where(v > th * v0)[0]
            return int(hit[0]) if len(hit) else None
        evaluate(f"vol_cut x{th}", cut)

    print("\n--- B) 時間ストップ(N 本で強制決済) ---")
    for N in (30, 40, 60):
        def cut(i, N=N):
            return N if len(paths[i]) - 1 > N else None
        evaluate(f"time_stop {N}本", cut)

    print(f"\n総経過 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
