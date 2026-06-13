"""検証3: USDJPY H1 lb24 tsmom の執行現実性。

(a) スプレッド感度: spread in {0.1,0.2,0.5,0.7(retail),1.4} pip で NET Sharpe / total_return / 損益分岐
(b) 1バー遅延執行: entries/exits を shift(1) で NET 維持されるか
(c) 取引頻度: 年間取引数
(d) 往復コスト: 1取引あたり必要 pip と平均利益 pip に対する比率

実行: uv run python -m research.experiments.exp_tsmom_execution
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import fxlab.config as C
from fxlab import run, metrics
from fxlab import universe as uni
from strategies.tsmom import generate_signals

PAIR = "USDJPY"
TF = "H1"
LB = 24
BAND = 0.0
PARAMS = {"lookback": LB, "band": BAND}

data = uni.instrument_data(PAIR, TF)
close = data["close"]
print(f"data: {PAIR} {TF}  rows={len(data)}  {data.index[0]} -> {data.index[-1]}")
years = (data.index[-1] - data.index[0]).days / 365.25
pip = C.pip_size(PAIR)
print(f"span_years={years:.2f}  pip_size={pip}")


def f(x):
    return float(x.iloc[0]) if isinstance(x, pd.Series) else float(x)


def signals_delayed(d, lookback=LB, band=BAND):
    le, lx, se, sx = generate_signals(d, lookback=lookback, band=band)
    # 1バー遅延執行: シグナル発生の次バーで約定
    return (le.shift(1, fill_value=False), lx.shift(1, fill_value=False),
            se.shift(1, fill_value=False), sx.shift(1, fill_value=False))


def eval_at_spread(spread_pips, gen, side="both"):
    orig = dict(C.SPREADS_PIPS)
    C.SPREADS_PIPS[PAIR] = spread_pips
    try:
        pf = run(PAIR, TF, gen, PARAMS, data=data, size_mode="value", side=side)
        m = metrics(pf)
        out = {
            "spread": spread_pips,
            "sharpe": f(m["sharpe"]),
            "total_return": f(m["total_return"]),
            "num_trades": int(f(m["num_trades"])),
            "win_rate": f(m["win_rate"]),
            "profit_factor": f(m["profit_factor"]),
            "expectancy": f(m["expectancy"]),
            "max_dd": f(m["max_drawdown"]),
        }
    finally:
        C.SPREADS_PIPS.clear()
        C.SPREADS_PIPS.update(orig)
    return out


print("\n===== (a) スプレッド感度 (immediate exec, value sizing, both sides) =====")
spreads = [0.0, 0.1, 0.2, 0.5, 0.7, 1.4]
rows = []
for sp in spreads:
    r = eval_at_spread(sp, generate_signals)
    rows.append(r)
    print(f"  spread={sp:>4} pip | Sharpe={r['sharpe']:+.3f} | ret={r['total_return']*100:+7.2f}% "
          f"| trades={r['num_trades']:4d} | PF={r['profit_factor']:.3f} | WR={r['win_rate']*100:4.1f}% "
          f"| exp={r['expectancy']:+.4f} | DD={r['max_dd']*100:.1f}%")

# 損益分岐スプレッド (total_return が 0 を跨ぐ点を線形補間)
def breakeven_spread(rows, key="total_return"):
    xs = [r["spread"] for r in rows]
    ys = [r[key] for r in rows]
    for i in range(len(xs) - 1):
        if ys[i] > 0 >= ys[i + 1] or (ys[i] >= 0 > ys[i + 1]):
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            return x0 + (x1 - x0) * (y0 - 0) / (y0 - y1)
    if all(y > 0 for y in ys):
        return float("inf")
    if all(y <= 0 for y in ys):
        return 0.0
    return None

be_ret = breakeven_spread(rows, "total_return")
be_shp = breakeven_spread(rows, "sharpe")
print(f"\n  損益分岐スプレッド (total_return=0): {be_ret if be_ret==float('inf') else f'{be_ret:.3f}'} pip")
print(f"  Sharpe=0 となるスプレッド        : {be_shp if be_shp==float('inf') else f'{be_shp:.3f}'} pip")


print("\n===== (b) 1バー遅延執行 (shift(1)) スプレッド感度 =====")
rows_d = []
for sp in spreads:
    r = eval_at_spread(sp, signals_delayed)
    rows_d.append(r)
    print(f"  spread={sp:>4} pip | Sharpe={r['sharpe']:+.3f} | ret={r['total_return']*100:+7.2f}% "
          f"| trades={r['num_trades']:4d} | PF={r['profit_factor']:.3f}")
be_ret_d = breakeven_spread(rows_d, "total_return")
print(f"  遅延後 損益分岐スプレッド: {be_ret_d if be_ret_d==float('inf') else (f'{be_ret_d:.3f}' if be_ret_d is not None else 'N/A')} pip")

# retail(0.7)での即時 vs 遅延 比較
imm07 = next(r for r in rows if r["spread"] == 0.7)
del07 = next(r for r in rows_d if r["spread"] == 0.7)
print(f"\n  @retail 0.7pip  即時: Sharpe={imm07['sharpe']:+.3f} ret={imm07['total_return']*100:+.2f}%")
print(f"  @retail 0.7pip  遅延: Sharpe={del07['sharpe']:+.3f} ret={del07['total_return']*100:+.2f}%")


print("\n===== (c) 取引頻度 =====")
n_tr = next(r for r in rows if r["spread"] == 0.7)["num_trades"]
print(f"  総取引数(両建て,10年): {n_tr}")
print(f"  年間取引数            : {n_tr / years:.1f} trades/yr")
print(f"  月間取引数            : {n_tr / years / 12:.1f} trades/mo")
print(f"  平均保有バー(H1)       : 約 {len(data) / max(n_tr,1):.1f} bars (~{len(data)/max(n_tr,1):.1f} 時間)")


print("\n===== (d) 往復コスト pip と平均利益 pip =====")
# GROSS(spread=0)のトレードから平均利益pip(絶対値)を測る
orig = dict(C.SPREADS_PIPS)
C.SPREADS_PIPS[PAIR] = 0.0
try:
    pf0 = run(PAIR, TF, generate_signals, PARAMS, data=data, size_mode="value", side="both")
    tr = pf0.trades.records_readable
finally:
    C.SPREADS_PIPS.clear(); C.SPREADS_PIPS.update(orig)

# pnl in price terms per trade -> approximate pip move = |entry-exit price| / pip
entry_px = tr["Avg Entry Price"].values
exit_px = tr["Avg Exit Price"].values
abs_pip_move = np.abs(exit_px - entry_px) / pip
pnl = tr["PnL"].values
gross_win_pip = abs_pip_move[pnl > 0].mean() if (pnl > 0).any() else 0.0
gross_loss_pip = abs_pip_move[pnl <= 0].mean() if (pnl <= 0).any() else 0.0
mean_abs_pip = abs_pip_move.mean()
# 平均利益pip(符号付き, 価格動だけ。GROSSなのでネット期待値の代理)
signed_pip = (exit_px - entry_px) / pip
# ロングは+方向,ショートは-方向が利益なので side で符号反転が必要 -> PnL符号で代用
# 期待損益pip = 勝ち平均pip*WR - 負け平均pip*(1-WR) のグロス
wr = (pnl > 0).mean()
exp_gross_pip = gross_win_pip * wr - gross_loss_pip * (1 - wr)

print(f"  GROSS トレード数: {len(tr)}")
print(f"  平均 |値幅| / トレード: {mean_abs_pip:.2f} pip")
print(f"  勝ちトレード平均値幅: {gross_win_pip:.2f} pip  (WR={wr*100:.1f}%)")
print(f"  負けトレード平均値幅: {gross_loss_pip:.2f} pip")
print(f"  GROSS 1取引あたり期待値幅: {exp_gross_pip:+.3f} pip")
print()
for sp in [0.1, 0.2, 0.5, 0.7, 1.4]:
    # 往復コスト = 1スプレッド分 (entry half + exit half)
    rt_cost = sp
    pct_of_exp = (rt_cost / exp_gross_pip * 100) if exp_gross_pip > 0 else float("inf")
    pct_of_win = rt_cost / gross_win_pip * 100
    net_exp = exp_gross_pip - rt_cost
    print(f"  spread={sp:>4}pip | 往復コスト={rt_cost:.2f}pip | GROSS期待値幅の {pct_of_exp:5.1f}% "
          f"| 勝ち値幅の {pct_of_win:4.1f}% | NET期待値幅={net_exp:+.3f}pip")

print("\n===== (補) long/short 分離 @ retail 0.7 と ECN 0.2 =====")
for sp in [0.2, 0.7]:
    rl = eval_at_spread(sp, generate_signals, side="long")
    rs = eval_at_spread(sp, generate_signals, side="short")
    print(f"  spread={sp}: long Sharpe={rl['sharpe']:+.3f} ret={rl['total_return']*100:+.2f}% "
          f"| short Sharpe={rs['sharpe']:+.3f} ret={rs['total_return']*100:+.2f}%")
