"""イテレーション20: 「資本効率は悪いが着実に利益が増えている」か? を実測検証。

ユーザー仮説: 資本効率↓ だが その分 着実(steady)に利益が積み上がる、のでは?
→ 「資本効率」と「着実さ(リターンの時間的安定)」は独立概念。データで切り分ける。

比較対象:
  champion = confluence_meanrev_v2 (現行チャンピオン, ERフィルタ)
  xsec     = クロスセクション平均回帰 (lb=9, hold=24, 片側4脚)  ※同尺 $10k/脚

着実さの指標(複数レンズ):
  (1) 年次   : プラス年率 / 年次PnLの変動係数CV
  (2) 月次   : プラス月率 / 最大DD / 最長連敗(月) / 月次Sharpe
  (3) ローリング: 12ヶ月移動PnLの安定度(CV, 最小値)
  (4) 微視(トレード単位): 単発最大損失 / 上位10%トレードの利益占有率 / 単発のテール
        =「多数の小さな貢献でコツコツ」型か「一部の大勝/大負け」型か
  (5) 資本効率: 平均投下資本 と 投下資本あたりリターン
実行: uv run python exp20.py
"""
from __future__ import annotations
import numpy as np, pandas as pd
from fxlab import universe as uni
from fxlab.backtest import run
from fxlab.trades import trade_table
from exp19c import xs_meanrev_trades, monthly, yearly_pnl


def champion_trades(tf, instruments, params):
    """v2(ERフィルタ)のトレード表 (exit, pnl) を全銘柄合算で返す。"""
    from strategies.confluence_meanrev_v2 import generate_signals
    frames = []
    for name in instruments:
        data = uni.instrument_data(name, tf)
        pf = run(name, tf, generate_signals, params, data=data, size_mode="value")
        tt = trade_table(pf, data)
        if not tt.empty:
            frames.append(tt[["exit", "pnl"]])
    return pd.concat(frames, ignore_index=True)

pd.set_option("display.width", 220)


def max_losing_streak(series: pd.Series) -> int:
    streak = best = 0
    for v in series:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return best


def avg_concurrent_capital(trades_with_entry: pd.DataFrame, bars_index, hold_bars=None):
    """近似: 各トレードが建玉している間 $10k を占有。建玉bar数で平均同時建玉を出す。"""
    pass  # champion は entry/exit を別途取得して算出(下で実装)


def analyze(name, tr):
    """tr: columns=[exit, pnl] のトレード表。着実さ指標一式を返す。"""
    y = yearly_pnl(tr)
    m = monthly(tr)
    pnl = tr["pnl"].values
    gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
    # 利益集中度: トレードを利益降順、上位10%が総"純"利益に占める割合
    sorted_pnl = np.sort(pnl)[::-1]
    top10 = sorted_pnl[:max(1, len(sorted_pnl)//10)].sum()
    net = pnl.sum()
    res = {
        "total_pnl": round(net, 0),
        "n_trades": len(pnl),
        # 年次の着実さ
        "pos_year_rate": round((y > 0).mean(), 2),
        "year_pnl_CV": round(y.std()/y.mean(), 2) if y.mean() != 0 else np.nan,
        "worst_year": round(y.min(), 0),
        # 月次の着実さ
        "pos_month_rate": round((m > 0).mean(), 2),
        "month_sharpe": round(m.mean()/m.std()*np.sqrt(12), 2),
        "max_dd": round((m.cumsum() - m.cumsum().cummax()).min(), 0),
        "longest_losing_streak_months": max_losing_streak(m),
        # 微視(トレード単位)
        "max_single_loss": round(pnl.min(), 0),
        "max_single_win": round(pnl.max(), 0),
        "top10pct_share_of_net": round(top10/net, 2) if net > 0 else np.nan,
        "avg_pnl_per_trade": round(pnl.mean(), 1),
    }
    return res, y, m


def rolling12(m):
    r = m.sort_index()
    # PeriodIndex 月次 → 12ヶ月移動合計
    s = r.rolling(12).sum().dropna()
    return s


def main():
    uni.register_cross_spreads(3.0)
    instruments = [x for x in uni.universe(crosses=True) if x != "AUDJPY"]
    close = pd.DataFrame({n: uni.instrument_close(n, "H4") for n in instruments}).dropna()
    params = dict(window=50, entry_z=2.0, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65,
                  vol_win=100, vol_pct=0.70, slow_win=250, slow_z=1.75,
                  er_win=40, er_max=0.55, adx_p=14, adx_max=None, slow_z_cap=None,
                  atr_p=14, atr_cap=None, slope_win=10, slope_max=None)

    print(">> トレード生成中(champion v2 + xsec)...")
    xs = xs_meanrev_trades(close, 9, 24, 0.0)               # 片側4脚
    ch = champion_trades("H4", instruments, params)         # v2

    rx, yx, mx = analyze("xsec", xs)
    rc, yc, mc = analyze("champion_v2", ch)
    comp = pd.DataFrame({"champion_v2": rc, "xsec_MR": rx})
    print("\n=== 着実さ指標の対比 ===")
    print(comp.to_string())

    print("\n=== 年次PnL ===")
    print(pd.DataFrame({"champion_v2": yc, "xsec_MR": yx}).round(0).to_string())

    print("\n=== ローリング12ヶ月PnL(着実さ=これが安定して正か)===")
    r12c, r12x = rolling12(mc), rolling12(mx)
    print(f"  champion_v2 : min={r12c.min():.0f}  最低が正?={'YES' if r12c.min()>0 else 'NO'}  "
          f"正の割合={ (r12c>0).mean():.0%}  CV={r12c.std()/r12c.mean():.2f}")
    print(f"  xsec_MR     : min={r12x.min():.0f}  最低が正?={'YES' if r12x.min()>0 else 'NO'}  "
          f"正の割合={ (r12x>0).mean():.0%}  CV={r12x.std()/r12x.mean():.2f}")
    print("  ※ 12ヶ月のどの窓を切ってもプラス=本当の意味で着実。NOなら『着実』は誤り。")


if __name__ == "__main__":
    main()
