"""本番・資金管理(マネーマネジメント)— チャンピオンv2 の DD≤20% 利益最大化サイジング。

Workflow(13エージェント・10手法)+ リーク無し自前再検証(mm_synthesis.py)の確定結論を実装する
**正式な資金管理ツール**。account_sim.py の後継(資金配分の"形"を最適化した版)。

────────────────────────────────────────────────────────────────────────
確定した最適資金管理 = 乖離連動サイズ(z-power)× 同時建玉上限 max_pos
  alloc = equity * (k / max_pos) * f(z)/f̄,   f(z) = clip((|Z_entry| / z0)^p, lo, hi)
  ・|Z_entry| = エントリー時の短期Zスコアの絶対値(=乖離の深さ)。深い乖離=期待反転大に厚く張る。
  ・単一エントリー・分割しない(ナンピン非該当)。f̄ で正規化し総量は k のみで制御。
  ・**完全に因果**(エントリー時点の値のみ。先読み無し)。max_pos で分散を増やし1玉を縮小。

エントリー1バー遅延 d1 の採用(2026-06-12, exp47/51/52/53, reports/18): プールの既定を
  **confluence_meanrev_v2_d1**(シグナルの次バー close で建玉、遅延先で z が exit 域なら見送り)に更新。
  逆行第1波を建玉前に外すことで robust(p95=20%較正) mp8 **+16.41%→+18.63%** / empirical
  **+24.64%→+27.50%**、p95 は -27.8→-27.3% と改善(=レバ偽装でない)。敵対検証3本
  (独立再実装一致 / M1粒度谷比1.05でゲート通過 / 近傍OAT 10/10正+固定kでもDD・p95とも浅い)全通過。
  ※ **d≥2 は禁止**(d2=2020単年依存88%+レバ偽装署名、d3=最良年除外で符号反転。IS-argmax は d3 を
    選ぶがゲートで死ぬ=P>4.5禁止と同型の線引き)。旧 d0 プールは --legacy-d0 で再現可。
  ※ M1粒度監査の掛け目は d1 構成で **k×0.955**(exp52。d0 の×0.965 より約1pp 深い)。

指数 P の再較正(2026-06-11, exp37/exp38, reports/14): P=2.0 → **4.0** に更新。
  深い乖離ほど期待リターンが大きい勾配は P=2 では刈り取り切れておらず、P を上げると robust
  (ブートp95=20%較正)CAGR が単調改善する(mp11 5シード平均 +15.0%→+17.4%、empirical 20% でも
  +23.8%→+27.6%、p95 はフラット=レバ偽装でない)。敵対検証7項目(独立再実装/IS単独argmax=3.5が
  OOSで+2.7pp持続/5シード全合格/配分集中=1玉レバ1.85x<P2の1.92x/ブロック長/mp非依存/年次8/11年
  改善)を全通過。高原は P∈[3.0,5.0]・z0∈[2.0,2.4] で滑らか。
  ※ **P>4.5 への引き上げ禁止**(IS単独証拠は3.5でピーク、二値step極限は最悪年マイナス化で崩壊。
    「もっと上げる」はフル期間後知恵=逐次探索バイアスの典型経路)。
  ※ 本番 k は robust 較正を複数シードで取り平均(較正ノイズ±0.3)。M1粒度監査(exp24)の掛け目
    k×0.95 も併用推奨。

なぜこれか(棄却した手法との対比):
  ・ボラターゲット/エントリーボラ逆比例 = ベースライン未満(戻り局面で減らす/高エッジ局面を削る)。
  ・CPPI・DD制御スロットル = 経験CAGRは上がるが理論テール(p95)悪化=過去波形フィット。
  ・per-instrument ケリー = 見かけ +9pp だが**ほぼ全部が重みの先読み(リーク)**。因果(WF)版は ≈ベースライン。
  ・乖離連動z だけが「リーク無し・テール中立・高原頑健」で両DD解釈下でベースラインを +36% 相対で上回る。

DD≤20% の2解釈(--dd-mode で選択)。数値は d1+P=4.0 更新後(括弧内は d0 時代):
  empirical : 経験的(単一バックテストパス)MtM最大DD = 20%。最も攻めた利益。
              mp8 k≈8.9 → CAGR≈+27.5%(d0 +24.6%)。
  robust    : ブートストラップ理論DD(p95, 20回に1回級)= 20%。テール自体を縛る保守解。
              mp8 → CAGR≈+18.6%(d0 +16.4%、5シード平均)。
  ※ empirical で20%に張ると理論テール(p95)は約-27〜-28%。「経験20%=理論テール28%」を運用前提とせよ。
  ※ mp11 は reject 確定(reports/16 exp44: M1粒度で実効CAGR逆転)。mp8 を維持する。

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

# 確定パラメータ(z0/clip は mm_synthesis.py の高原中央、P は exp37/38 の再較正値=reports/14)
Z0 = 2.2
P = 4.0
CLIP_LO, CLIP_HI = 0.3, 3.0


def _fz(z):
    return float(np.clip((z / Z0) ** P, CLIP_LO, CLIP_HI)) if np.isfinite(z) else 1.0


def build_pool_d1(tf="H4", instruments=None, cross_spread=3.0, cache=True):
    """d1(エントリー1バー遅延)戦略のトレードプール(mm_lab.build_pool と同形式)。

    z_entry は **シグナルバー(=エントリーバーの1本前)時点の |z|**。検証済み構成
    (exp47/51)と同じ規約: サイジングはシグナル時点の乖離の深さで決める(因果)。
    """
    from fxlab import config, universe as uni
    from fxlab.backtest import run
    from fxlab.trades import trade_table
    from strategies.confluence_meanrev_v2_d1 import PARAMS, generate_signals

    uni.register_cross_spreads(cross_spread)
    instruments = instruments or mm.default_instruments()
    cache_path = config.RESULTS_DIR / f"mm_pool_v2d1_{tf}_{len(instruments)}.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    win = PARAMS["window"]
    frames = []
    for nm in instruments:
        data = uni.instrument_data(nm, tf)
        pf = run(nm, tf, generate_signals, dict(PARAMS), data=data, size_mode="value")
        tt = trade_table(pf, data)
        if tt.empty:
            continue
        close = data["close"]
        z_sig = ((close - close.rolling(win).mean()) / close.rolling(win).std()).shift(1)
        vol_sig = close.pct_change().rolling(20).std().shift(1)
        frames.append(pd.DataFrame({
            "instr": nm,
            "entry": tt["entry"].to_numpy(),
            "exit": tt["exit"].to_numpy(),
            "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
            "entry_price": tt["entry_price"].to_numpy(),
            "ret": tt["return_pct"].to_numpy() / 100.0,
            "bars_held": tt["bars_held"].to_numpy(),
            "z_entry": np.abs(z_sig.reindex(tt["entry"]).to_numpy()),
            "vol_entry": vol_sig.reindex(tt["entry"]).to_numpy(),
        }))
    pool = pd.concat(frames, ignore_index=True).sort_values("entry").reset_index(drop=True)
    if cache:
        pool.to_parquet(cache_path)
    return pool


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
    ap.add_argument("--legacy-d0", action="store_true",
                    help="旧: 遅延なし(v2 素)のプールで評価(reports/18 以前の挙動)")
    args = ap.parse_args()

    tag = "v2(d0)" if args.legacy_d0 else "v2_d1(エントリー1バー遅延)"
    print(f"=== チャンピオン{tag} 本番資金管理 (max_pos={args.max_pos}) ===")
    pool = mm.build_pool() if args.legacy_d0 else build_pool_d1()
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
