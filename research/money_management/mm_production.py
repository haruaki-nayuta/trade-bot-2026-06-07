"""本番・資金管理(マネーマネジメント)— チャンピオンv2 の DD≤20% 利益最大化サイジング。

Workflow(13エージェント・10手法)+ リーク無し自前再検証(mm_synthesis.py)の確定結論を実装する
**正式な資金管理ツール**。account_sim.py の後継(資金配分の"形"を最適化した版)。

────────────────────────────────────────────────────────────────────────
確定した最適資金管理 = 乖離連動サイズ(z-power)× 同時建玉上限 max_pos
  alloc = equity * (k / max_pos) * f(z)/f̄,   f(z) = clip((|Z_entry| / z0)^p, lo, hi)
  ・|Z_entry| = エントリー時の短期Zスコアの絶対値(=乖離の深さ)。深い乖離=期待反転大に厚く張る。
  ・単一エントリー・分割しない(ナンピン非該当)。f̄ で正規化し総量は k のみで制御。
  ・**完全に因果**(エントリー時点の値のみ。先読み無し)。max_pos で分散を増やし1玉を縮小。

なぜこれか(棄却した手法との対比):
  ・ボラターゲット/エントリーボラ逆比例 = ベースライン未満(戻り局面で減らす/高エッジ局面を削る)。
  ・CPPI・DD制御スロットル = 経験CAGRは上がるが理論テール(p95)悪化=過去波形フィット。
  ・per-instrument ケリー = 見かけ +9pp だが**ほぼ全部が重みの先読み(リーク)**。因果(WF)版は ≈ベースライン。
  ・乖離連動z だけが「リーク無し・テール中立・高原頑健」で両DD解釈下でベースラインを +36% 相対で上回る。

DD≤20% の2解釈(--dd-mode で選択):
  empirical : 経験的(単一バックテストパス)MtM最大DD = 20%。最も攻めた利益。k≈8(mp8)→ CAGR≈+21.6%。
  robust    : ブートストラップ理論DD(p95, 20回に1回級)= 20%。テール自体を縛る保守解。k≈5.4(mp8)→ CAGR≈+14.3%。
  ※ empirical で20%に張ると理論テール(p95)は約-28%。「経験20%=理論テール28%」を運用前提とせよ。

実行:
  uv run python mm_production.py                          # 推奨: mp8, empirical 20%
  uv run python mm_production.py --dd-mode robust         # テール自体を20%に縛る保守運用
  uv run python mm_production.py --max-pos 10 --dd-target 0.20   # より攻める(テール悪化と引換)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import mm_lab as mm

pd.set_option("display.width", 200)

# 確定パラメータ(mm_synthesis.py の高原中央)
Z0 = 2.2
P = 2.0
CLIP_LO, CLIP_HI = 0.3, 3.0


def _fz(z):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def champion_sizing(pool, max_pos=8):
    """確定サイジング make_sizing(k): 乖離連動z(因果)。総建玉は k に線形。"""
    fbar = float(np.mean([_fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0

    def make_sizing(k):
        base = k / max_pos
        return lambda ctx: ctx["equity_real"] * base * (_fz(ctx["z"]) / fbar)
    return make_sizing


def report(eqm, eqr, info, init, p95, p99, k, dd_mode, dd_target):
    s = mm.stats(eqm, eqr, info)
    print(f"\n{'='*64}")
    print(f"  資金管理: 乖離連動z(z0={Z0},p={P}) × max_pos / 較正={dd_mode} {dd_target:.0%}")
    print(f"{'='*64}")
    print(f"  較正レバレッジ k : {k:.2f}x(満玉時の総建玉=資産×{k:.2f})")
    print(f"  通算リターン     : {s['total_return']:+.1%}   最終資産 {eqm.iloc[-1]:,.0f}(初期{init:,.0f})")
    print(f"  CAGR             : {s['cagr']:+.2%}")
    print(f"  最大DD(MtM含み損): {s['maxdd_mtm']:+.1%}  ← 口座が経験する真のDD")
    print(f"  最大DD(実現のみ) : {s['maxdd_real']:+.1%}")
    print(f"  理論DD(ブート)   : p95 {p95:+.1%} / p99 {p99:+.1%}  ← 20回/100回に1回級の上振れ")
    print(f"  Sharpe / Sortino : {s['sharpe']:.2f} / {s['sortino']:.2f}")
    print(f"  プラス年率       : {s['pos_year_rate']:.0%}  ({int(round(s['pos_year_rate']*s['n_years']))}/{s['n_years']}年)  最悪年 {s['worst_year']:+.1%}")
    print(f"  同時建玉 最大/見送り: {s['max_conc']} / {s['skipped']}件")
    print("\n  年次リターン:")
    print((s["yr_ret"] * 100).round(1).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description="チャンピオンv2 本番資金管理(DD≤20%利益最大化)")
    ap.add_argument("--max-pos", type=int, default=8, help="同時建玉上限(推奨8。10で攻め/6で保守)")
    ap.add_argument("--dd-target", type=float, default=0.20, help="DD上限(既定0.20)")
    ap.add_argument("--dd-mode", choices=["empirical", "robust"], default="empirical",
                    help="empirical=経験的MtM最大DD / robust=ブート理論DD(p95)")
    ap.add_argument("--init", type=float, default=10_000.0)
    args = ap.parse_args()

    print(f"=== チャンピオンv2 本番資金管理 (max_pos={args.max_pos}) ===")
    pool = mm.build_pool()
    closes = mm.load_closes()
    print(f"トレード {len(pool)}(年{len(pool)/11:.0f}) / グリッド {len(closes)}本")

    mk = champion_sizing(pool, max_pos=args.max_pos)
    if args.dd_mode == "robust":
        k, eqm, eqr, info, p95 = mm.calibrate_robust(pool, closes, mk, target_dd=args.dd_target,
                                                     max_pos=args.max_pos, n_boot=1000)
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500)
        p95, p99 = bs["p95"], bs["p99"]
    else:
        k, eqm, eqr, info = mm.calibrate(pool, closes, mk, target_dd=args.dd_target,
                                         max_pos=args.max_pos)
        bs = mm.bootstrap_maxdd(eqm, n_boot=1500)
        p95, p99 = bs["p95"], bs["p99"]

    # init でスケール(複利なので相対は不変、最終資産表示用)
    eqm = eqm / eqm.iloc[0] * args.init
    eqr = eqr / eqr.iloc[0] * args.init
    report(eqm, eqr, info, args.init, p95, p99, k, args.dd_mode, args.dd_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
