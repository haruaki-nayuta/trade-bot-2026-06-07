"""exp25: H1 スケールのチャンピオン構造(confluence_meanrev_v2)は単体で成立するか。

仮説 H3(TFアンサンブル)の前段。同一のエッジ構造(短期Z×RSI×平穏×長期Z×ER)を
H1 足・同一バー数パラメータで回す=「より速い時間スケールの平均回帰ストリーム」。
H4 ストリームとは時間スケールが違うため相関が下がり、統合すれば同一DD予算でCAGRを
積み増せる可能性がある。ただし H1 は1トレードの値幅が小さくコスト比が重い。

判定基準(単体): プールPF / 年次プラス率 / 年間取引数 / IS・OOS 分割 / H4ストリームとの月次相関。

実行: PYTHONPATH=.:research/money_management uv run python research/experiments/exp25_h1_stream.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research" / "money_management"))

import mm_lab as mm  # noqa: E402
import strategies.confluence_meanrev_v2 as v2  # noqa: E402

pd.set_option("display.width", 220)

OOS_START = pd.Timestamp("2022-01-01", tz="UTC")


def pool_report(pool: pd.DataFrame, label: str, bars_per_day: float):
    if pool.empty:
        print(f"[{label}] トレード0件")
        return
    r = pool["ret"]
    gp = r[r > 0].sum()
    gl = -r[r < 0].sum()
    years = (pool["exit"].max() - pool["entry"].min()).days / 365.25
    exit_year = pd.to_datetime(pool["exit"]).dt.year
    yearly = pool.groupby(exit_year)["ret"].sum()
    is_r = pool.loc[pd.to_datetime(pool["entry"]) < OOS_START, "ret"]
    oos_r = pool.loc[pd.to_datetime(pool["entry"]) >= OOS_START, "ret"]

    def pf(x):
        g = x[x > 0].sum(); l = -x[x < 0].sum()
        return g / l if l > 0 else float("inf")

    print(f"\n[{label}] trades={len(pool)} ({len(pool)/years:.0f}/年)  ΣR={r.sum():+.3f}  "
          f"プールPF={pf(r):.3f}  勝率={(r>0).mean():.1%}  平均={r.mean()*1e4:+.1f}bps  "
          f"平均保有={pool['bars_held'].mean():.1f}bars({pool['bars_held'].mean()/bars_per_day:.1f}日)")
    print(f"  IS  PF={pf(is_r):.3f} ΣR={is_r.sum():+.3f} (n={len(is_r)})")
    print(f"  OOS PF={pf(oos_r):.3f} ΣR={oos_r.sum():+.3f} (n={len(oos_r)})")
    pos = (yearly > 0)
    print(f"  年次プラス: {pos.sum()}/{len(yearly)}  最悪年 {yearly.min():+.3f} ({yearly.idxmin()})")
    print("  年次ΣR:", {int(y): round(float(v), 3) for y, v in yearly.items()})


def monthly_pnl(pool: pd.DataFrame) -> pd.Series:
    m = pd.to_datetime(pool["exit"]).dt.to_period("M")
    return pool.groupby(m)["ret"].sum()


def main() -> int:
    params = dict(v2.PARAMS)

    print("=== H4 ストリーム(基準) ===")
    pool_h4 = mm.build_pool()
    pool_report(pool_h4, "H4 v2", 6)

    print("\n=== H1 ストリーム(同一バー数パラメータ = H1スケール) ===")
    pool_h1 = mm.build_pool_for(v2, params, tf="H1", tag="v2_h1scale")
    pool_report(pool_h1, "H1 v2(同一バー数)", 24)

    if not pool_h1.empty:
        a = monthly_pnl(pool_h4)
        b = monthly_pnl(pool_h1)
        idx = a.index.union(b.index)
        corr = a.reindex(idx).fillna(0).corr(b.reindex(idx).fillna(0))
        print(f"\n  月次PnL相関 (H4 vs H1): {corr:.3f}")

    print("\n=== H1 ストリーム(entry_z を上げた絞り込み版: 2.25 / 2.5) ===")
    for ez in [2.25, 2.5]:
        p = dict(params); p["entry_z"] = ez
        pl = mm.build_pool_for(v2, p, tf="H1", tag=f"v2_h1z{int(ez*100)}")
        pool_report(pl, f"H1 v2 entry_z={ez}", 24)
        if not pl.empty:
            a = monthly_pnl(pool_h4); b = monthly_pnl(pl)
            idx = a.index.union(b.index)
            print(f"  月次PnL相関 vs H4: {a.reindex(idx).fillna(0).corr(b.reindex(idx).fillna(0)):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
