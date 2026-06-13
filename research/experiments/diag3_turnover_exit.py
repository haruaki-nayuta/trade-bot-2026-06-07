"""診断3: 回転とエグジット品質。

tsmom(lb100) と breakout_trend を 7メジャー D1/H4 (+W1で回転落とし) で回し:
  - num_trades, 平均保有バー, payoff比(平均勝ち額/平均負け額), コストドラッグ(gross-net total_return)
を測る。GROSS(コスト0) と NET(通常コスト) を別プロセスで両方測るため、
本スクリプトは環境変数 DIAG3_MODE=gross/net で切り替え、結果を JSON で stdout に吐く。
ランナー(main)が両モードを subprocess で起動して集計する。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")

PAIRS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]

# (strategy module, params)
STRATS = {
    "tsmom_lb100": ("strategies.tsmom", {"lookback": 100, "band": 0.0}),
    "breakout_trend": ("strategies.breakout_trend", {"entry": 40, "exit": 20, "trend": 200}),
}

# 足ごとの「1年あたりおよそのバー数」≒ 営業日換算(回転倍率の補助)
BARS_PER_YEAR = {"H4": 252 * 6, "D1": 252, "W1": 52}


def _measure(mode: str) -> dict:
    """1プロセス内で gross または net を測る。"""
    import importlib

    import pandas as pd

    import fxlab.config as C

    if mode == "gross":
        C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
        C.COMMISSION_FRACTION = 0.0

    # config をいじった後に backtest を import（_slippage_series がモジュール属性を読む）
    from fxlab import load, run

    out = {}
    for sname, (mod_path, params) in STRATS.items():
        mod = importlib.import_module(mod_path)
        gen = mod.generate_signals
        for tf in ["H4", "D1", "W1"]:
            try:
                data = load("EURUSD", tf)  # noqa: F841  (cache warm; per-pair below)
            except Exception:
                pass
            for pair in PAIRS:
                key = f"{sname}|{tf}|{pair}"
                try:
                    data = load(pair, tf)
                    pf = run(pair, tf, gen, params, data=data,
                             size_mode="value", side="both")
                    total_return = float(pf.total_return())
                    tr = pf.trades.records_readable
                    n = int(len(tr))
                    if n == 0:
                        out[key] = {
                            "total_return": total_return, "num_trades": 0,
                            "avg_hold_bars": None, "payoff": None,
                            "win_rate": None, "avg_win": None, "avg_loss": None,
                        }
                        continue
                    pnl = tr["PnL"].astype(float)
                    wins = pnl[pnl > 0]
                    losses = pnl[pnl < 0]
                    avg_win = float(wins.mean()) if len(wins) else 0.0
                    avg_loss = float(losses.mean()) if len(losses) else 0.0  # 負値
                    payoff = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
                    win_rate = float(len(wins) / n)
                    # 保有バー数: (Exit - Entry) / バー間隔。バー数で測るためインデックス位置差を使う
                    idx = data.index
                    pos = pd.Series(range(len(idx)), index=idx)
                    ent = tr["Entry Timestamp"].map(pos)
                    ext = tr["Exit Timestamp"].map(pos)
                    hold = (ext - ent).dropna()
                    avg_hold = float(hold.mean()) if len(hold) else None
                    out[key] = {
                        "total_return": total_return, "num_trades": n,
                        "avg_hold_bars": avg_hold, "payoff": payoff,
                        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
                    }
                except Exception as e:  # noqa: BLE001
                    out[key] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    mode = os.environ.get("DIAG3_MODE")
    if mode in ("gross", "net"):
        res = _measure(mode)
        print("###JSON###")
        print(json.dumps(res))
        return

    # ランナー: 両モードを subprocess で起動
    here = os.path.abspath(__file__)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(here), "..", ".."))
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = repo_root + os.pathsep + base_env.get("PYTHONPATH", "")
    env_g = dict(base_env, DIAG3_MODE="gross")
    env_n = dict(base_env, DIAG3_MODE="net")

    def _runp(env):
        p = subprocess.run([sys.executable, here], env=env, cwd=repo_root,
                           capture_output=True, text=True)
        if p.returncode != 0:
            sys.stderr.write(p.stderr)
            raise SystemExit(f"subprocess failed rc={p.returncode}")
        marker = "###JSON###"
        txt = p.stdout
        return json.loads(txt[txt.index(marker) + len(marker):].strip())

    gross = _runp(env_g)
    net = _runp(env_n)

    # ---- 集計 ----
    import statistics as st

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    report = {"per_pair": {}, "summary": {}}
    for sname in STRATS:
        for tf in ["H4", "D1", "W1"]:
            drags, nets, grosses, ntr, holds, payoffs_n, payoffs_g, wr = [], [], [], [], [], [], [], []
            for pair in PAIRS:
                key = f"{sname}|{tf}|{pair}"
                g = gross.get(key, {})
                nn = net.get(key, {})
                if "error" in g or "error" in nn:
                    report["per_pair"][key] = {"gross": g, "net": nn}
                    continue
                drag = g["total_return"] - nn["total_return"]
                drags.append(drag)
                nets.append(nn["total_return"])
                grosses.append(g["total_return"])
                if nn["num_trades"]:
                    ntr.append(nn["num_trades"])
                holds.append(nn["avg_hold_bars"])
                payoffs_n.append(nn["payoff"])
                payoffs_g.append(g["payoff"])
                wr.append(nn["win_rate"])
                report["per_pair"][key] = {
                    "num_trades": nn["num_trades"],
                    "avg_hold_bars": nn["avg_hold_bars"],
                    "payoff_net": nn["payoff"],
                    "payoff_gross": g["payoff"],
                    "win_rate": nn["win_rate"],
                    "gross_total_return": round(g["total_return"], 4),
                    "net_total_return": round(nn["total_return"], 4),
                    "cost_drag": round(drag, 4),
                }
            skey = f"{sname}|{tf}"
            report["summary"][skey] = {
                "avg_cost_drag": avg(drags),
                "avg_net_return": avg(nets),
                "avg_gross_return": avg(grosses),
                "avg_num_trades": avg(ntr),
                "avg_hold_bars": avg(holds),
                "avg_payoff_net": avg(payoffs_n),
                "avg_payoff_gross": avg(payoffs_g),
                "avg_win_rate": avg(wr),
                "trades_per_year": (avg(ntr) / (10) if avg(ntr) else None),  # データ約10年
            }

    print("###REPORT###")
    print(json.dumps(report, indent=2))

    # ---- 人間可読サマリ(stderr へ) ----
    sys.stderr.write("\n==== SUMMARY (per strategy|tf, 7-pair avg) ====\n")
    for skey, s in report["summary"].items():
        sys.stderr.write(
            f"{skey:28s} n~{_f(s['avg_num_trades'],0):>6} "
            f"hold~{_f(s['avg_hold_bars'],1):>6}bars "
            f"payoff net {_f(s['avg_payoff_net'],2)}/gross {_f(s['avg_payoff_gross'],2)} "
            f"WR {_f(s['avg_win_rate'],3)}  "
            f"ret g {_f(s['avg_gross_return'],3)} / n {_f(s['avg_net_return'],3)}  "
            f"drag {_f(s['avg_cost_drag'],3)}\n"
        )

    # ---- D1 vs H4 倍率と回転落とし改善 ----
    sys.stderr.write("\n==== cost-drag H4 vs D1 vs W1 (ratio) ====\n")
    for sname in STRATS:
        d = {tf: report["summary"][f"{sname}|{tf}"]["avg_cost_drag"] for tf in ["H4", "D1", "W1"]}
        nr = {tf: report["summary"][f"{sname}|{tf}"]["avg_net_return"] for tf in ["H4", "D1", "W1"]}
        ratio_h4_d1 = (d["H4"] / d["D1"]) if d["D1"] not in (None, 0) else None
        sys.stderr.write(
            f"{sname}: drag H4 {_f(d['H4'],3)} D1 {_f(d['D1'],3)} W1 {_f(d['W1'],3)} | "
            f"H4/D1 drag x{_f(ratio_h4_d1,2)} | "
            f"net H4 {_f(nr['H4'],3)} -> D1 {_f(nr['D1'],3)} -> W1 {_f(nr['W1'],3)}\n"
        )


def _f(x, nd):
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


if __name__ == "__main__":
    main()
