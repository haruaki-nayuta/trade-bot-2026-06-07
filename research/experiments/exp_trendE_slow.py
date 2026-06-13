"""再構築E: 低回転・長期TF トレンドフォロー。

核心: W1(週足)/ D1超低回転(200日チャネル, 12ヶ月momの月次rebalance)で
回転を1/5以下に落としたとき、コスト最小化が net を breakeven へ近づけるか。
GROSS が負のままなら「コストではなくエッジ不在」が確定する。

測定:
  - 7メジャー10年・size=value 両建て
  - GROSS(コスト0) と NET を別プロセス(subprocess)で測り比較
  - 7ペア平均 sharpe / total_return
  - IS(前半)/ OOS(後半) を NET で
  - turnover(num_trades)をベースライン tsmom D1 lb100 と比較
  - plateau(パラメータ近傍で符号維持か)

実行:
  uv run python research/experiments/exp_trendE_slow.py            # 本体(GROSS+NET, IS/OOS, turnover)
  GROSS=1 uv run python research/experiments/exp_trendE_slow.py    # GROSSのみ(内部用 subprocess)
"""

from __future__ import annotations

import os
import sys
import json
import subprocess

import numpy as np
import pandas as pd

# リポジトリ root を import path に追加(subprocess 直叩きでも fxlab を解決)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fxlab import load, run, metrics
import fxlab.config as C

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]


# ----------------------------------------------------------------------
# 戦略定義(全て低回転)
# ----------------------------------------------------------------------
def sig_tsmom(data, lookback=100, band=0.0):
    """タイムシリーズ・モメンタム(ドテン)。"""
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0
    long_state = mom > band
    short_state = mom < -band
    le = long_state & ~long_state.shift(fill_value=False)
    se = short_state & ~short_state.shift(fill_value=False)
    return le, se, se, le  # le, lx(=se), se, sx(=le)


def sig_donchian(data, entry=52, exit=26):
    """ドンチャン・ブレイク(順張りトレーリング)。"""
    high, low, close = data["high"], data["low"], data["close"]
    upper = high.rolling(entry).max().shift()
    lower = low.rolling(entry).min().shift()
    exit_lower = low.rolling(exit).min().shift()
    exit_upper = high.rolling(exit).max().shift()
    le = close > upper
    se = close < lower
    lx = close < exit_lower
    sx = close > exit_upper
    return le, lx, se, sx


def sig_channel200(data, entry=200, exit=100):
    """D1 超低回転: 200日チャネルブレイク + 100日トレーリング。"""
    return sig_donchian(data, entry=entry, exit=exit)


def sig_mom12m_monthly(data, lookback=252, band=0.0):
    """12ヶ月モメンタムを月次rebalance(ドテン状態を月初だけ更新)。

    毎営業日ではなく月の最初のバーでのみ状態(ロング/ショート)を確定し、
    その状態を月内維持する。回転を最小化する古典的TSMOM運用。
    先読みなし: 過去 lookback 本の確定リターンで判断。
    """
    close = data["close"]
    mom = close / close.shift(lookback) - 1.0
    raw_state = np.where(mom > band, 1, np.where(mom < -band, -1, 0))
    raw_state = pd.Series(raw_state, index=close.index)

    # 月初(その月の最初のバー)だけ状態更新、他は前方埋め
    month_id = close.index.to_period("M")
    is_month_start = pd.Series(month_id, index=close.index).ne(
        pd.Series(month_id, index=close.index).shift()
    )
    state = raw_state.where(is_month_start).ffill().fillna(0)

    long_state = state > 0
    short_state = state < 0
    le = long_state & ~long_state.shift(fill_value=False)
    se = short_state & ~short_state.shift(fill_value=False)
    return le, se, se, le


STRATS = {
    # name: (sig_func, params, pair_tf_map_resolved_below)
    "tsmom_W1_lb52": (sig_tsmom, {"lookback": 52, "band": 0.0}, "W1"),
    "tsmom_W1_lb26": (sig_tsmom, {"lookback": 26, "band": 0.0}, "W1"),
    "donchian_W1_52_26": (sig_donchian, {"entry": 52, "exit": 26}, "W1"),
    "channel200_D1": (sig_channel200, {"entry": 200, "exit": 100}, "D1"),
    "mom12m_monthly_D1": (sig_mom12m_monthly, {"lookback": 252, "band": 0.0}, "D1"),
}

# plateau 用の近傍パラメータ
PLATEAU = {
    "tsmom_W1_lb52": [{"lookback": 39}, {"lookback": 52}, {"lookback": 65}],
    "donchian_W1_52_26": [
        {"entry": 40, "exit": 20},
        {"entry": 52, "exit": 26},
        {"entry": 60, "exit": 30},
    ],
    "channel200_D1": [
        {"entry": 150, "exit": 75},
        {"entry": 200, "exit": 100},
        {"entry": 250, "exit": 125},
    ],
    "mom12m_monthly_D1": [
        {"lookback": 189}, {"lookback": 252}, {"lookback": 315},
    ],
}


def zero_costs():
    C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
    C.COMMISSION_FRACTION = 0.0


def eval_one(pair, tf, sig, params, data=None, period=None):
    if data is None:
        data = load(pair, tf)
    d = data if period is None else data.loc[period[0]:period[1]]
    pf = run(pair, tf, sig, params, data=d, size_mode="value", side="both")
    m = metrics(pf)
    row = m.iloc[0] if isinstance(m, pd.DataFrame) else m
    return {
        "total_return": float(row["total_return"]),
        "sharpe": float(row["sharpe"]),
        "num_trades": int(row["num_trades"]),
        "max_drawdown": float(row["max_drawdown"]),
        "profit_factor": float(row["profit_factor"]) if np.isfinite(row["profit_factor"]) else float("nan"),
    }


def measure_all():
    """全戦略 x 7ペア。full期間 + IS/OOS。current cost setting で測る。"""
    out = {}
    for name, (sig, params, tf) in STRATS.items():
        rows_full, rows_is, rows_oos = [], [], []
        # IS/OOS split: 10年のうち前半/後半。日付で分ける。
        for pair in PAIRS:
            data = load(pair, tf)
            idx = data.index
            mid = idx[len(idx) // 2]
            rows_full.append(eval_one(pair, tf, sig, params, data=data))
            rows_is.append(eval_one(pair, tf, sig, params, data=data,
                                    period=(idx[0], mid)))
            rows_oos.append(eval_one(pair, tf, sig, params, data=data,
                                     period=(mid, idx[-1])))
        out[name] = {
            "full": rows_full, "is": rows_is, "oos": rows_oos, "tf": tf,
        }
    return out


def avg(rows, key):
    vals = [r[key] for r in rows if np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def measure_plateau():
    """NET で plateau: 近傍パラメータの7ペア平均total_return符号維持か。"""
    res = {}
    for name, grid in PLATEAU.items():
        sig, _, tf = STRATS[name]
        per_param = []
        for params in grid:
            rows = [eval_one(p, tf, sig, params) for p in PAIRS]
            per_param.append({
                "params": params,
                "avg_total_return": avg(rows, "total_return"),
                "avg_sharpe": avg(rows, "sharpe"),
            })
        signs = [1 if pp["avg_total_return"] > 0 else -1 for pp in per_param]
        res[name] = {
            "per_param": per_param,
            "sign_stable": len(set(signs)) == 1,
        }
    return res


def main():
    if os.environ.get("GROSS") == "1":
        zero_costs()
        data = measure_all()
        print("__JSON__" + json.dumps(data))
        return

    # NET 本測定
    net = measure_all()
    plateau = measure_plateau()

    # GROSS は別プロセス
    env = dict(os.environ, GROSS="1")
    proc = subprocess.run(
        [sys.executable, __file__],
        capture_output=True, text=True, env=env,
    )
    gross = None
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            gross = json.loads(line[len("__JSON__"):])
    if gross is None:
        print("GROSS subprocess failed:\n", proc.stdout[-2000:], proc.stderr[-2000:])
        return

    # baseline turnover: tsmom D1 lb100 (NET) — full期間の7ペア合計取引数
    base_rows = [eval_one(p, "D1", sig_tsmom, {"lookback": 100, "band": 0.0}) for p in PAIRS]
    base_trades = sum(r["num_trades"] for r in base_rows)

    print("=" * 78)
    print("再構築E: 低回転・長期TF トレンドフォロー (7メジャー10年, size=value, both)")
    print("=" * 78)
    print(f"\nベースライン turnover: tsmom D1 lb100 = {base_trades} trades (7ペア合計, 10年)")
    print(f"  (≒ {base_trades/7:.0f} trades/ペア/10年)\n")

    summary = {}
    for name in STRATS:
        tf = STRATS[name][2]
        g, n = gross[name], net[name]
        g_tr = avg(g["full"], "total_return"); n_tr = avg(n["full"], "total_return")
        g_sh = avg(g["full"], "sharpe"); n_sh = avg(n["full"], "sharpe")
        n_is = avg(n["is"], "total_return"); n_oos = avg(n["oos"], "total_return")
        tot_trades = sum(r["num_trades"] for r in n["full"])
        cost_drag = g_tr - n_tr
        pos_pairs = sum(1 for r in n["full"] if r["total_return"] > 0)
        turnover_ratio = tot_trades / base_trades if base_trades else float("nan")

        summary[name] = {
            "tf": tf,
            "gross_total_return_avg": g_tr,
            "net_total_return_avg": n_tr,
            "gross_sharpe_avg": g_sh,
            "net_sharpe_avg": n_sh,
            "net_is": n_is,
            "net_oos": n_oos,
            "total_trades_7pairs": tot_trades,
            "turnover_vs_baseline": turnover_ratio,
            "cost_drag": cost_drag,
            "net_pos_pairs": pos_pairs,
            "plateau_sign_stable": plateau.get(name, {}).get("sign_stable"),
        }

        print(f"--- {name} [{tf}] ---")
        print(f"  trades(7p,10y)={tot_trades:4d}  turnover vs base={turnover_ratio:.2f}x")
        print(f"  GROSS: sharpe={g_sh:+.3f}  total_return={g_tr:+.3f}")
        print(f"  NET  : sharpe={n_sh:+.3f}  total_return={n_tr:+.3f}  pos_pairs={pos_pairs}/7")
        print(f"  cost_drag(g-n total_return)= {cost_drag:+.4f}")
        print(f"  NET IS={n_is:+.3f}  OOS={n_oos:+.3f}")
        print(f"  plateau sign_stable={plateau.get(name, {}).get('sign_stable')}")
        print()

    print("=" * 78)
    print("plateau 詳細 (NET, 近傍パラメータの7ペア平均total_return):")
    for name, pr in plateau.items():
        print(f"  {name}: stable={pr['sign_stable']}")
        for pp in pr["per_param"]:
            print(f"    {pp['params']} -> tr={pp['avg_total_return']:+.3f} sh={pp['avg_sharpe']:+.3f}")

    print("\n__SUMMARY__" + json.dumps(summary))


if __name__ == "__main__":
    main()
