"""exp30: USD ファクター・トレンドゲート — テール共通モードの根本(ドル主導トレンド)を狙い撃つ。

発見(本セッション): scale-out/TFアンサンブル/scale-in は全て「同一テール基準で純増ゼロ」。
理由はテール episodes(2022型 USD ラリー等)が族内の全ストリームに共通だから。
既存フィルタは全て銘柄単位(ER40・ボラ・slow_z)。**ポートフォリオの共通モード=USD ファクター
自身のトレンド強度**でエントリーを絞るゲートは未検証だった(見落とし候補)。

設計(全て因果・終値のみ):
  USD 因子 F_t = mean_i ±log(close_i)  (USDが分子のペアは+、分母のペアは−)→ H4 系列
  ゲート指標 = Kaufman ER(F, 40)(因子が一直線に動いている度合い)
  新規エントリー時に ER_F > th なら建玉サイズを g 倍(g=0 でスキップ)。クロスにも適用
  (クロスはUSD両建てだがリスクオン相関で同時失血するため全銘柄一律も試す/USD脚のみも試す)。

判定: empirical 20% と robust(p95=20%) の両較正で champion mp11 と比較。
IS較正→OOS素検証つき。ゲートのしきい値は高原を見る(0.25/0.30/0.35/0.40)。

実行: PYTHONPATH=. uv run python research/experiments/exp30_usd_factor_gate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))
sys.path.insert(0, str(ROOT / "research" / "lab"))

import mm_lab as mm  # noqa: E402
import ens_lab as ens  # noqa: E402
from fxlab.data import load  # noqa: E402

pd.set_option("display.width", 240)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")
MAJORS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDJPY", "USDCHF", "USDCAD"]


def usd_factor_er(win=40) -> pd.Series:
    """USD 因子(等加重 log)の Kaufman ER。H4・因果。"""
    legs = []
    for p in MAJORS:
        c = np.log(load(p, "H4")["close"])
        legs.append(c if p.startswith("USD") else -c)
    F = pd.concat(legs, axis=1).ffill().dropna().mean(axis=1)
    direction = (F - F.shift(win)).abs()
    volatility = F.diff().abs().rolling(win).sum()
    return (direction / volatility).replace([np.inf, -np.inf], np.nan)


def gated_sizing_factory(pool, er_f: pd.Series, th: float, g: float, max_pos: int,
                         usd_only: bool):
    """champion z-power サイジング × USDファクターゲート。エントリー時点の ER_F のみ参照(因果)。"""
    from mm_production import _fz
    fbar = float(np.mean([_fz(z) for z in pool["z_entry"].to_numpy()])) or 1.0
    # エントリー時刻 → 直近の確定 ER_F(asof)
    er_at = er_f.reindex(pd.to_datetime(pool["entry"]), method="ffill").to_numpy()
    is_usd = pool["instr"].astype(str).str.contains("USD").to_numpy()
    gate_mult = np.where(np.isfinite(er_at) & (er_at > th),
                         np.where(is_usd | (not usd_only), g, 1.0), 1.0)
    # ctx には行番号がないため (instr, ret, bars_held) で照合(exp21d と同じ手法)
    key = {}
    instr = pool["instr"].to_numpy(); ret = pool["ret"].to_numpy(); bh = pool["bars_held"].to_numpy()
    for i in range(len(pool)):
        key[(instr[i], round(float(ret[i]), 12), int(bh[i]))] = gate_mult[i]

    def make_sizing(k):
        base = k / max_pos
        def sizing(ctx):
            gm = key.get((ctx["instr"], round(float(ctx["ret"]), 12), int(ctx["bars_held"])), 1.0)
            return ctx["equity_real"] * base * gm * (_fz(ctx["z"]) / fbar)
        return sizing
    return make_sizing


def eval_gate(pool, closes, make_sizing, label, max_pos=11, n_boot=800):
    # empirical 20%
    k, eqm, eqr, info = mm.calibrate(pool, closes, make_sizing, target_dd=0.20, max_pos=max_pos)
    s = mm.stats(eqm, eqr, info)
    bs = mm.bootstrap_maxdd(eqm, n_boot=n_boot)
    # robust p95=20%
    kr, eqr2, eqrr, infor, p95 = mm.calibrate_robust(pool, closes, make_sizing, target_dd=0.20,
                                                     max_pos=max_pos, n_boot=600)
    sr = mm.stats(eqr2, eqrr, infor)
    # IS較正→OOS(empirical)
    is_pool = pool[pool["entry"] < OOS_START].reset_index(drop=True)
    oos_pool = pool[pool["entry"] >= OOS_START].reset_index(drop=True)
    is_cl = closes[closes.index < OOS_START]; oos_cl = closes[closes.index >= OOS_START]
    k_is, *_ = mm.calibrate(is_pool, is_cl, make_sizing, target_dd=0.20, max_pos=max_pos)
    eqo, ero, io = mm.simulate(oos_pool, oos_cl, make_sizing(k_is), max_pos=max_pos)
    so = mm.stats(eqo, ero, io)
    print(f"  {label:30s} emp: CAGR={s['cagr']:+7.2%} p95={bs['p95']:+6.1%} worst={s['worst_year']:+5.1%}"
          f" | robust: CAGR={sr['cagr']:+7.2%} | OOS emp: CAGR={so['cagr']:+7.2%} DD={so['maxdd_mtm']:+6.1%}")
    return {"label": label, "emp": s["cagr"], "p95": bs["p95"], "rob": sr["cagr"],
            "oos": so["cagr"]}


def main() -> int:
    pool = mm.build_pool()
    closes = mm.load_closes()
    er_f = usd_factor_er(40)
    print(f"ER_F 分布: p50={er_f.median():.3f} p75={er_f.quantile(.75):.3f} p90={er_f.quantile(.90):.3f}")

    from mm_production import champion_sizing
    print("=== 基準: champion mp11 ===")
    eval_gate(pool, closes, champion_sizing(pool, max_pos=11), "baseline mp11")

    print("\n=== USDファクター・ゲート(全銘柄一律) ===")
    for th in [0.25, 0.30, 0.35, 0.40]:
        for g in [0.0, 0.5]:
            mk = gated_sizing_factory(pool, er_f, th, g, 11, usd_only=False)
            eval_gate(pool, closes, mk, f"all th={th} g={g}")

    print("\n=== USD脚ペアのみゲート ===")
    for th in [0.30, 0.35]:
        for g in [0.0, 0.5]:
            mk = gated_sizing_factory(pool, er_f, th, g, 11, usd_only=True)
            eval_gate(pool, closes, mk, f"usd-only th={th} g={g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
