"""exp32: XAUUSD(金)ストリーム — 「別資産」によるテール分散の最初の実証。

reports/10/11 の結論: FX 内では何を足してもテール episodes が共通で同一テール基準の前進はない。
残るフロンティア = 別資産。金は (a) Dukascopy で同品質の M1 が取れる (b) リスクオフで USD と
逆相関になることが多い=テールの起源が違う (c) スプレッドが価格比で FX メジャー並みに薄い、
の3点で最初の候補。

検証:
  1. チャンピオン構造(confluence_meanrev_v2, パラメータ完全固定=再最適化なし)を XAUUSD H4 に適用
     → 単体プール品質(PF/IS/OOS/年次)+ チャンピオンポートとの月次相関
  2. スプレッド感応度($0.40/$0.60/$0.80 フル)
  3. 成立すれば ens_lab でストリーム統合(budget 1-2, w 掃引)→ empirical/robust 両較正で
     champion mp11 と比較(同一テール判定プロトコル)

コスト実装: config.SPREADS_PIPS["XAUUSD"] に「pip=0.0001 換算で絶対額になる」値を登録
(例 $0.40 → 4000pips)。エンジンは spread_pips×pip_size の絶対価格幅しか使わないため正確。

実行: PYTHONPATH=. uv run python research/experiments/exp32_gold_stream.py
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
import strategies.confluence_meanrev_v2 as v2  # noqa: E402
from fxlab import config  # noqa: E402
from fxlab.backtest import run  # noqa: E402
from fxlab.data import load  # noqa: E402
from fxlab.trades import trade_table  # noqa: E402

pd.set_option("display.width", 240)

OOS = pd.Timestamp("2022-01-01", tz="UTC")
COLS = ["instr", "entry", "exit", "dir", "entry_price", "ret", "z_entry", "stream", "w"]


def _zscore(s, w):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()


def build_gold_pool(spread_usd: float) -> pd.DataFrame:
    config.SPREADS_PIPS["XAUUSD"] = spread_usd / 0.0001  # 絶対額→擬似pips
    data = load("XAUUSD", "H4")
    pf = run("XAUUSD", "H4", v2.generate_signals, dict(v2.PARAMS), data=data, size_mode="value")
    tt = trade_table(pf, data)
    if tt.empty:
        return pd.DataFrame(columns=COLS)
    close = data["close"]
    z = _zscore(close, v2.PARAMS.get("window", 50))
    g = pd.DataFrame({
        "instr": "XAUUSD",
        "entry": tt["entry"].to_numpy(),
        "exit": tt["exit"].to_numpy(),
        "dir": np.where(tt["dir"].to_numpy() == "Long", 1, -1),
        "entry_price": tt["entry_price"].to_numpy(),
        "ret": tt["return_pct"].to_numpy() / 100.0,
        "z_entry": np.abs(z.reindex(tt["entry"]).to_numpy()),
    })
    g["stream"] = "gold"
    g["w"] = 1.0
    return g.sort_values("entry").reset_index(drop=True)


def pool_quick(pool, label):
    if pool.empty:
        print(f"  [{label}] 0 trades")
        return
    r = pool["ret"]
    def pf(x):
        g = x[x > 0].sum(); l = -x[x < 0].sum()
        return g / l if l > 0 else float("inf")
    is_r = pool.loc[pool["entry"] < OOS, "ret"]
    oos_r = pool.loc[pool["entry"] >= OOS, "ret"]
    yearly = pool.groupby(pd.to_datetime(pool["exit"]).dt.year)["ret"].sum()
    years = max((pool["exit"].max() - pool["entry"].min()).days / 365.25, 1e-9)
    print(f"  [{label}] n={len(pool)} ({len(pool)/years:.0f}/年) ΣR={r.sum():+.3f} PF={pf(r):.3f} "
          f"勝率={(r>0).mean():.0%} 平均={r.mean()*1e4:+.1f}bps")
    print(f"    IS PF={pf(is_r):.3f}(n={len(is_r)}) OOS PF={pf(oos_r):.3f}(n={len(oos_r)}) "
          f"年次プラス {int((yearly>0).sum())}/{len(yearly)} 最悪 {yearly.min():+.3f}({yearly.idxmin()})")
    print(f"    年次: {dict((int(y), round(float(v),3)) for y,v in yearly.items())}")
    print(f"    Long {(pool['dir']>0).sum()} / Short {(pool['dir']<0).sum()}  "
          f"ΣR long={pool.loc[pool['dir']>0,'ret'].sum():+.3f} short={pool.loc[pool['dir']<0,'ret'].sum():+.3f}")


def main() -> int:
    print("=== 1. 単体品質(スプレッド感応度) ===")
    pools = {}
    for sp in [0.40, 0.60, 0.80]:
        pools[sp] = build_gold_pool(sp)
        pool_quick(pools[sp], f"XAUUSD v2 spread=${sp:.2f}")

    gold = pools[0.40]
    if gold.empty or gold["ret"].sum() <= 0:
        print("\n→ 単体で不成立。統合はスキップ。")
        return 0

    champ = mm.build_pool().copy()
    champ["stream"] = "champ"; champ["w"] = 1.0
    a = champ.groupby(pd.to_datetime(champ["exit"]).dt.tz_localize(None).dt.to_period("M"))["ret"].sum()
    b = gold.groupby(pd.to_datetime(gold["exit"]).dt.tz_localize(None).dt.to_period("M"))["ret"].sum()
    idx = a.index.union(b.index)
    print(f"\n月次PnL相関 vs チャンピオン: {a.reindex(idx).fillna(0).corr(b.reindex(idx).fillna(0)):.3f}")

    print("\n=== 2. 統合(empirical / robust 両較正) ===")
    closes = mm.load_closes().copy()
    config.SPREADS_PIPS["XAUUSD"] = 0.40 / 0.0001
    gc = load("XAUUSD", "H4")["close"]
    closes["XAUUSD"] = gc.reindex(closes.index, method="ffill")

    def robust_streams(pool, budgets, target=0.20, n_boot=1000, lo=0.02, hi=20.0, iters=15):
        fbars = ens.stream_fbars(pool)
        def p95_of(k):
            eqm, _, _ = ens.simulate_streams(pool, closes, k, budgets, fbars=fbars)
            return abs(mm.bootstrap_maxdd(eqm, n_boot=n_boot)["p95"])
        for _ in range(iters):
            mid = (lo + hi) / 2
            if p95_of(mid) > target: hi = mid
            else: lo = mid
        eqm, eqr, info = ens.simulate_streams(pool, closes, lo, budgets, fbars=fbars)
        return lo, mm.stats(eqm, eqr, info)

    def emp_streams(pool, budgets):
        fbars = ens.stream_fbars(pool)
        k, eqm, eqr, info = ens.calibrate_streams(pool, closes, budgets, fbars=fbars, target_dd=0.20)
        return k, mm.stats(eqm, eqr, info), mm.bootstrap_maxdd(eqm, n_boot=1000)

    k, s, bs = emp_streams(champ[COLS], {"champ": 11})
    print(f"  基準 mp11 emp: CAGR={s['cagr']:+.2%} p95={bs['p95']:+.1%}")
    k, s = robust_streams(champ[COLS], {"champ": 11})
    print(f"  基準 mp11 rob: CAGR={s['cagr']:+.2%}")

    for w in [0.5, 1.0, 1.5]:
        g = gold.copy(); g["w"] = w
        pool = pd.concat([champ[COLS], g[COLS]], ignore_index=True).sort_values("entry").reset_index(drop=True)
        for bg in [1, 2]:
            k, s, bs = emp_streams(pool, {"champ": 11, "gold": bg})
            print(f"  champ+gold w={w} b={bg} emp: CAGR={s['cagr']:+.2%} p95={bs['p95']:+.1%} "
                  f"worst={s['worst_year']:+.1%} skip={s['skipped']}")
            k, s = robust_streams(pool, {"champ": 11, "gold": bg})
            print(f"  champ+gold w={w} b={bg} rob: CAGR={s['cagr']:+.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
