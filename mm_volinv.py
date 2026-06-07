"""資金管理: エントリー時ボラ逆比例サイズ(リスク均等化)。

チャンピオンv2(無ストップ平均回帰・H4・19対象)の口座サイジングを、各トレードの
エントリー時ボラ vol_entry に逆比例させて「1トレードあたりリスクを均等化」する案を実測する。

  alloc = equity_real * (k/max_pos) * (vol_ref / vol_entry)^p   (clip 後)

狙い: 急変時の深い逆張り(=塩漬けの含み損 / テールDD の温床)は vol_entry が大きいので alloc を絞り、
平穏なエントリーを厚くする。1トレードあたりのMtM変動(=DD寄与)を均し、テールを削れれば、
較正 k を上げられて CAGR が伸びる…という仮説を検証する。

別案:
  - cut_decile: vol_entry 上位 decile のトレードを alloc=0 で見送り(サイジングでの選別)。
  - power p∈{0.5,1.0}: 逆比例の強さ。

== vol_entry を sizing から引く方法 ==
simulate が渡す ctx には vol_entry が無い(z/ret/bars_held/instr のみ)。そこで pool から
キー (instr, ret, bars_held, z) → vol_entry の辞書を作りクロージャに取り込む。このキーは
全1214トレードで一意(検証済み)なので衝突しない。ctx の (instr, ret, bars_held, z) で引く。

== 較正の単調性(契約)==
make_sizing(k) は総建玉を k に線形スケールする。shape 部 (vol_ref/vol_entry)^p は k に依存しない
純粋な「配分の形」なので、k に対する総エクスポージャは厳密に線形 → calibrate の二分探索が成立する。
vol_ref は pool の vol_entry 中央値(定数)。clip は較正が頭打ちにならないよう十分広く取る。

実行: uv run python mm_volinv.py            # shape スイープ + ベスト構成のフル評価
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import mm_lab as mm


def _vol_lookup(pool: pd.DataFrame) -> tuple[dict, float]:
    """(instr, ret_rounded, bars_held, z_rounded) -> vol_entry の辞書と vol_ref(中央値)。"""
    lut = {}
    for instr, ret, bars, z, vol in zip(
        pool["instr"].to_numpy(),
        pool["ret"].to_numpy(),
        pool["bars_held"].to_numpy(),
        pool["z_entry"].to_numpy(),
        pool["vol_entry"].to_numpy(),
    ):
        lut[(instr, round(float(ret), 10), int(bars), round(float(z), 8))] = float(vol)
    vol_ref = float(np.nanmedian(pool["vol_entry"].to_numpy()))
    return lut, vol_ref


def _ctx_vol(ctx: dict, lut: dict) -> float:
    key = (ctx["instr"], round(float(ctx["ret"]), 10), int(ctx["bars_held"]),
           round(float(ctx["z"]), 8))
    return lut.get(key, float("nan"))


def make_volinv(pool: pd.DataFrame, *, p=1.0, mult_lo=0.25, mult_hi=4.0,
                cut_top_decile=False, max_pos=6):
    """ボラ逆比例サイズの make_sizing(k) を返す。

    p          : 逆比例の強さ。alloc ∝ (vol_ref/vol_entry)^p。
    mult_lo/hi : shape 乗数のクリップ(較正が頭打ちにならないよう十分広く)。
    cut_top_decile : True なら vol_entry 上位 decile を見送り(alloc=0)。
    """
    lut, vol_ref = _vol_lookup(pool)
    cut_thr = float(np.nanpercentile(pool["vol_entry"].to_numpy(), 90)) if cut_top_decile else None

    def make_sizing(k):
        base = k / max_pos

        def sizing(ctx):
            vol = _ctx_vol(ctx, lut)
            if not np.isfinite(vol) or vol <= 0:
                # フォールバック: 固定 weight(立上り/欠損時)
                return ctx["equity_real"] * base
            if cut_thr is not None and vol > cut_thr:
                return 0.0
            shape = (vol_ref / vol) ** p
            shape = min(max(shape, mult_lo), mult_hi)
            return ctx["equity_real"] * base * shape

        return sizing

    return make_sizing


def _row(res):
    return (f"{res['method']:28s} k={res['k']:5.2f}  CAGR={res['cagr']:+6.2%}  "
            f"DD_mtm={res['maxdd_mtm']:+6.2%}  DD_real={res['maxdd_real']:+6.2%}  "
            f"Sh={res['sharpe']:.2f}  pos_yr={res['pos_year_rate']:.0%}  "
            f"p95={res['boot_p95']:+6.2%}  p99={res['boot_p99']:+6.2%}  "
            f"OOS_CAGR={res.get('oos_cagr', float('nan')):+6.2%}  "
            f"OOS_DD={res.get('oos_maxdd_mtm', float('nan')):+6.2%}  "
            f"OOS_posyr={res.get('oos_pos_year', float('nan')):.0%}")


def main():
    pool = mm.build_pool()
    closes = mm.load_closes()

    # baseline
    def make_fixed(max_pos=6):
        def mk(k):
            w = k / max_pos
            return lambda ctx: ctx["equity_real"] * w
        return mk

    print("=== ボラ逆比例サイズ(リスク均等化)スイープ — 20%DD較正 ===\n")
    base = mm.evaluate_method("BASELINE_fixed", pool, closes, make_fixed(), n_boot=400)
    print(_row(base), "\n")

    configs = [
        dict(p=0.5, mult_lo=0.3, mult_hi=3.0, cut_top_decile=False),
        dict(p=1.0, mult_lo=0.25, mult_hi=4.0, cut_top_decile=False),
        dict(p=1.0, mult_lo=0.2, mult_hi=6.0, cut_top_decile=False),
        dict(p=1.5, mult_lo=0.15, mult_hi=8.0, cut_top_decile=False),
        dict(p=0.0, mult_lo=0.0, mult_hi=1.0, cut_top_decile=True),   # cut only
        dict(p=1.0, mult_lo=0.25, mult_hi=4.0, cut_top_decile=True),  # volinv + cut
        dict(p=0.5, mult_lo=0.3, mult_hi=3.0, cut_top_decile=True),
    ]
    results = []
    for cfg in configs:
        name = f"volinv_p{cfg['p']}_lo{cfg['mult_lo']}_hi{cfg['mult_hi']}_cut{int(cfg['cut_top_decile'])}"
        mk = make_volinv(pool, max_pos=6, **cfg)
        res = mm.evaluate_method(name, pool, closes, mk, n_boot=400)
        results.append((cfg, res))
        print(_row(res))

    best_cfg, best = max(results, key=lambda cr: cr[1]["cagr"])
    print(f"\n=== ベスト構成(CAGR最大): {best_cfg} → 最終フル評価(n_boot=1500)===")
    mk = make_volinv(pool, max_pos=6, **best_cfg)
    final = mm.evaluate_method("volinv_BEST", pool, closes, mk, n_boot=1500)
    print(_row(final))
    return best_cfg, final


if __name__ == "__main__":
    main()
