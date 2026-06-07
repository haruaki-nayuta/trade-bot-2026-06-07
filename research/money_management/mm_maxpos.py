"""固定比率サイジング × 同時建玉上限(max_pos)の最適化 — "真のベースライン"確定。

手法: mm_lab.fixed_fractional(満玉時に deploy 比率, alloc=equity*deploy/max_pos)を基準に、
max_pos を {4,5,6,8,10,12} で変えて各々 MtM最大DD=20% に較正→CAGR を比較する。

問い: 建玉上限を上げる(=分散を増やし1玉を小さく)と、同じ20%DDでCAGRが上がるか?
      それとも見送り(skipped)が減るだけで頭打ちか?

make_sizing(k) は「総建玉を k 倍に線形スケール」する契約。fixed_fractional では
deploy=k がそのまま総エクスポージャ倍率なので、make_sizing(k)=fixed_fractional(deploy=k, max_pos)
とすれば総建玉は k に線形(alloc = equity * k / max_pos、満玉時の総建玉 = equity*k)。
これは max_pos に依らず「総建玉 = equity*k(満玉時)」で正規化されるため、max_pos 間で公平に較正・比較できる。

実行: uv run python mm_maxpos.py
"""

from __future__ import annotations

import numpy as np

import mm_lab as mm


def make_sizing_factory(max_pos):
    """指定 max_pos に対する make_sizing(k) を返す。

    make_sizing(k) は固定比率サイジング関数を返す: alloc = equity_real * k / max_pos。
    満玉(n_open=max_pos)時の総建玉 = equity_real * k なので総エクスポージャは k に線形。
    """
    def make_sizing(k):
        w = k / max_pos
        def _sizing(ctx):
            return ctx["equity_real"] * w
        return _sizing
    return make_sizing


def main():
    instruments = mm.default_instruments()
    print(f"=== mm_maxpos: 固定比率 × max_pos 最適化 (対象{len(instruments)}) ===")
    pool = mm.build_pool(instruments=instruments)
    closes = mm.load_closes(instruments=instruments)
    print(f"トレード総数 {len(pool)} / グリッド {len(closes)}本\n")

    MAXPOS_SWEEP = [4, 5, 6, 8, 10, 12]

    print("=== スイープ: 各 max_pos を MtM最大DD=20% に較正 (n_boot=400) ===")
    header = ("max_pos    k    CAGR    DD(MtM)  DD(real)  Sharpe  プラス年  "
              "boot_p95  boot_p99  n_taken  skipped  max_conc  "
              "OOS_CAGR  OOS_DD  OOS_pos%")
    print(header)
    results = {}
    for mp in MAXPOS_SWEEP:
        mk = make_sizing_factory(mp)
        r = mm.evaluate_method(f"fixed_mp{mp}", pool, closes, mk,
                               target_dd=0.20, max_pos=mp, n_boot=400)
        results[mp] = r
        oc = r.get("oos_cagr", float("nan"))
        od = r.get("oos_maxdd_mtm", float("nan"))
        op = r.get("oos_pos_year", float("nan"))
        print(f"  {mp:>4d}  {r['k']:>5.2f}  {r['cagr']:>+6.1%}  {r['maxdd_mtm']:>7.1%}  "
              f"{r['maxdd_real']:>7.1%}  {r['sharpe']:>5.2f}  {r['pos_year_rate']:>6.0%}  "
              f"{r['boot_p95']:>7.1%}  {r['boot_p99']:>7.1%}  {r['n_taken']:>6d}  "
              f"{r['skipped']:>6d}  {r['max_conc']:>6d}  "
              f"{oc:>+7.1%}  {od:>6.1%}  {op:>6.0%}")

    # 最良 max_pos = CAGR 最大(20%DD較正は全構成で揃っている前提)
    best_mp = max(results, key=lambda m: results[m]["cagr"])
    print(f"\n最良 max_pos = {best_mp} (CAGR={results[best_mp]['cagr']:+.2%})")

    # 最終: 最良構成を n_boot=1500 で確定評価
    print(f"\n=== 確定評価: max_pos={best_mp}, n_boot=1500 ===")
    mk = make_sizing_factory(best_mp)
    final = mm.evaluate_method(f"fixed_mp{best_mp}", pool, closes, mk,
                              target_dd=0.20, max_pos=best_mp, n_boot=1500)
    for key in ["method", "k", "cagr", "maxdd_mtm", "maxdd_real", "sharpe", "sortino",
                "pos_year_rate", "worst_year", "boot_p95", "boot_p99", "boot_worst",
                "max_conc", "n_taken", "skipped",
                "k_is", "oos_cagr", "oos_maxdd_mtm", "oos_pos_year", "oos_sharpe"]:
        if key in final:
            v = final[key]
            if isinstance(v, float):
                print(f"  {key:>16s}: {v:+.4f}")
            else:
                print(f"  {key:>16s}: {v}")

    return results, final


if __name__ == "__main__":
    main()
