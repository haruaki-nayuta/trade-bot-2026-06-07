"""診断2: レジーム依存(順張りは高ER年で勝ち低ER年で負けるか)。

tsmom(lb100) と breakout_trend を 7メジャー D1 で回し、各ペアの年次NET P&L を出す。
並行して各ペアの年次トレンド性 = 効率比 ER(w=40) の年平均を計算。
問い: 順張りエッジはレジーム(ER)にゲートされているか、年によらず一様に負けるか。
年次P&Lとその年ERの符号一致率/相関を定量化。
XAUUSD(金)で同じtsmomを回し、金のER水準とP&Lが7メジャーとどう違うか比較診断。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import fxlab.config as C
from fxlab import load, run, metrics
from strategies.tsmom import generate_signals as tsmom_sig
from strategies.breakout_trend import generate_signals as breakout_sig

MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
TF = "D1"
ER_W = 40

TSMOM_PARAMS = {"lookback": 100, "band": 0.0}
BREAKOUT_PARAMS = {"entry": 40, "exit": 20, "trend": 200}


def efficiency_ratio(close: pd.Series, w: int = ER_W) -> pd.Series:
    """Kaufman 効率比: |close - close.shift(w)| / Σ|close.diff()| over w bars."""
    direction = (close - close.shift(w)).abs()
    volatility = close.diff().abs().rolling(w).sum()
    er = direction / volatility
    return er


def annual_pnl_from_pf(pf) -> pd.Series:
    """Portfolio のトレード記録から、エグジット年ごとの net P&L 合計を返す。"""
    tr = pf.trades.records_readable
    if len(tr) == 0:
        return pd.Series(dtype=float)
    # 列名を確認して exit timestamp と pnl を取る
    exit_col = [c for c in tr.columns if "Exit" in c and ("Timestamp" in c or "Index" in c)]
    pnl_col = [c for c in tr.columns if c == "PnL" or "PnL" in c]
    ec = exit_col[0]
    pc = pnl_col[0]
    exit_ts = pd.to_datetime(tr[ec])
    pnl = tr[pc].astype(float)
    g = pd.DataFrame({"year": exit_ts.dt.year, "pnl": pnl})
    return g.groupby("year")["pnl"].sum()


def annual_er(close: pd.Series, w: int = ER_W) -> pd.Series:
    er = efficiency_ratio(close, w)
    return er.groupby(er.index.year).mean()


def run_strategy_net(pair: str, sig, params: dict):
    """NET(通常コスト)で run。"""
    data = load(pair, TF)
    pf = run(pair, TF, sig, params, data=data, size_mode="value", side="both")
    m = metrics(pf)
    overall_sharpe = float(m["sharpe"].iloc[0])
    apnl = annual_pnl_from_pf(pf)
    return overall_sharpe, apnl, data


def run_strategy_gross(pair: str, sig, params: dict):
    """GROSS(コスト0)で run。呼び出し前にコストを0化しておくこと。"""
    data = load(pair, TF)
    pf = run(pair, TF, sig, params, data=data, size_mode="value", side="both")
    m = metrics(pf)
    return float(m["sharpe"].iloc[0])


def analyze(strat_name, sig, params, pairs):
    """各ペアの年次P&L と 年次ER を並べ、符号一致率と相関を集計。"""
    rows = []
    per_pair_detail = {}
    net_sharpes = []
    for pair in pairs:
        sharpe, apnl, data = run_strategy_net(pair, sig, params)
        net_sharpes.append(sharpe)
        aer = annual_er(data["close"])
        # 共通年で揃える
        common = sorted(set(apnl.index) & set(aer.index))
        sub = pd.DataFrame(
            {"year": common,
             "pnl": [apnl.get(y, np.nan) for y in common],
             "er": [aer.get(y, np.nan) for y in common]}
        ).dropna()
        per_pair_detail[pair] = sub
        if len(sub) >= 3:
            # 符号一致: pnl>0 と er が「その戦略の中央値ER」より高いか
            er_med = sub["er"].median()
            high_er = sub["er"] > er_med
            win = sub["pnl"] > 0
            agree = (high_er == win).mean()  # 高ER年=勝ち / 低ER年=負け の一致率
            corr = sub["pnl"].corr(sub["er"]) if sub["pnl"].std() > 0 else np.nan
            # 高ER年 vs 低ER年 の平均P&L
            high_pnl = sub.loc[high_er, "pnl"].mean()
            low_pnl = sub.loc[~high_er, "pnl"].mean()
        else:
            agree = corr = high_pnl = low_pnl = np.nan
        rows.append({
            "pair": pair, "net_sharpe": round(sharpe, 3),
            "n_years": len(sub),
            "mean_ER": round(sub["er"].mean(), 3) if len(sub) else np.nan,
            "agree_rate": round(agree, 3) if agree == agree else np.nan,
            "pnl_er_corr": round(corr, 3) if corr == corr else np.nan,
            "highER_mean_pnl": round(high_pnl, 1) if high_pnl == high_pnl else np.nan,
            "lowER_mean_pnl": round(low_pnl, 1) if low_pnl == low_pnl else np.nan,
        })
    summary = pd.DataFrame(rows)
    return summary, per_pair_detail, net_sharpes


def main():
    print("=" * 70)
    print("診断2: レジーム依存 (順張りは高ER年で勝ち低ER年で負けるか)")
    print(f"TF={TF}  ER window={ER_W}")
    print("=" * 70)

    results = {}

    # --- NET 解析(通常コスト) ---
    for strat_name, sig, params in [
        ("tsmom_lb100", tsmom_sig, TSMOM_PARAMS),
        ("breakout_trend", breakout_sig, BREAKOUT_PARAMS),
    ]:
        print(f"\n### {strat_name} (NET, 通常コスト) params={params}")
        summ, detail, net_sharpes = analyze(strat_name, sig, params, MAJORS)
        print(summ.to_string(index=False))
        avg_agree = summ["agree_rate"].mean()
        avg_corr = summ["pnl_er_corr"].mean()
        net_sh_avg = float(np.mean(net_sharpes))
        print(f"--- 7ペア平均: agree_rate={avg_agree:.3f}  pnl_ER_corr={avg_corr:.3f}  net_sharpe_avg={net_sh_avg:.3f}")
        # 全ペア・全年プールでの高ER vs 低ER 平均P&L
        all_sub = pd.concat(detail.values(), ignore_index=True)
        med = all_sub["er"].median()
        pooled_high = all_sub.loc[all_sub["er"] > med, "pnl"]
        pooled_low = all_sub.loc[all_sub["er"] <= med, "pnl"]
        print(f"--- プール: 高ER年(>{med:.3f}) mean_pnl={pooled_high.mean():.1f} (n={len(pooled_high)}, win%={(pooled_high>0).mean()*100:.0f})"
              f"  低ER年 mean_pnl={pooled_low.mean():.1f} (n={len(pooled_low)}, win%={(pooled_low>0).mean()*100:.0f})")
        results[strat_name] = {
            "summary": summ, "net_sharpe_avg": net_sh_avg,
            "avg_agree": avg_agree, "avg_corr": avg_corr,
            "pooled_high_mean": pooled_high.mean(), "pooled_low_mean": pooled_low.mean(),
            "pooled_high_win": (pooled_high > 0).mean(), "pooled_low_win": (pooled_low > 0).mean(),
        }

    # --- GROSS 解析(コスト0) ---
    print("\n" + "=" * 70)
    print("GROSS (コスト0) 7ペア平均 Sharpe")
    print("=" * 70)
    orig_spreads = dict(C.SPREADS_PIPS)
    orig_comm = C.COMMISSION_FRACTION
    C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
    C.SPREADS_PIPS["XAUUSD"] = 0.0
    C.COMMISSION_FRACTION = 0.0
    gross_results = {}
    for strat_name, sig, params in [
        ("tsmom_lb100", tsmom_sig, TSMOM_PARAMS),
        ("breakout_trend", breakout_sig, BREAKOUT_PARAMS),
    ]:
        gross_sharpes = []
        for pair in MAJORS:
            gs = run_strategy_gross(pair, sig, params)
            gross_sharpes.append(gs)
        gross_avg = float(np.mean(gross_sharpes))
        gross_results[strat_name] = {"gross_sharpe_avg": gross_avg, "per_pair": dict(zip(MAJORS, [round(x, 3) for x in gross_sharpes]))}
        print(f"{strat_name}: GROSS 7ペア平均 Sharpe = {gross_avg:.3f}  per-pair={gross_results[strat_name]['per_pair']}")
    # restore
    C.SPREADS_PIPS = orig_spreads
    C.COMMISSION_FRACTION = orig_comm

    # --- XAUUSD 比較診断(同じ tsmom) ---
    print("\n" + "=" * 70)
    print("XAUUSD(金) 比較診断: 同じ tsmom lb100")
    print("=" * 70)
    gold_close = load("XAUUSD", TF)["close"]
    gold_aer = annual_er(gold_close)
    print(f"金 ER 年平均: 全期間mean={gold_aer.mean():.3f}")
    print("金 年次ER:")
    print(gold_aer.round(3).to_string())

    # 金: NET と GROSS の両方
    gold_net_sharpe, gold_apnl, _ = run_strategy_net("XAUUSD", tsmom_sig, TSMOM_PARAMS)
    print(f"\n金 tsmom NET sharpe(default cost, spread=1.0pip近似)= {gold_net_sharpe:.3f}")
    gold_detail = pd.DataFrame({
        "year": sorted(set(gold_apnl.index) & set(gold_aer.index)),
    })
    gold_detail["pnl"] = gold_detail["year"].map(gold_apnl)
    gold_detail["er"] = gold_detail["year"].map(gold_aer)
    gold_detail = gold_detail.dropna()
    print("金 年次 P&L vs ER:")
    print(gold_detail.round(2).to_string(index=False))
    if len(gold_detail) >= 3:
        gmed = gold_detail["er"].median()
        ghigh = gold_detail["er"] > gmed
        gwin = gold_detail["pnl"] > 0
        gagree = (ghigh == gwin).mean()
        gcorr = gold_detail["pnl"].corr(gold_detail["er"])
        print(f"金: agree_rate={gagree:.3f}  pnl_ER_corr={gcorr:.3f}"
              f"  高ER年mean_pnl={gold_detail.loc[ghigh,'pnl'].mean():.1f}"
              f"  低ER年mean_pnl={gold_detail.loc[~ghigh,'pnl'].mean():.1f}")

    # 金 GROSS
    C.SPREADS_PIPS = {k: 0.0 for k in C.SPREADS_PIPS}
    C.SPREADS_PIPS["XAUUSD"] = 0.0
    C.COMMISSION_FRACTION = 0.0
    gold_gross = run_strategy_gross("XAUUSD", tsmom_sig, TSMOM_PARAMS)
    C.SPREADS_PIPS = orig_spreads
    C.COMMISSION_FRACTION = orig_comm
    print(f"金 tsmom GROSS sharpe = {gold_gross:.3f}")

    # 7メジャー平均ER との比較
    majors_mean_er = np.mean([annual_er(load(p, TF)["close"]).mean() for p in MAJORS])
    print(f"\n7メジャー平均ER = {majors_mean_er:.3f}  vs  金平均ER = {gold_aer.mean():.3f}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (machine-readable)")
    print("=" * 70)
    for sn in ["tsmom_lb100", "breakout_trend"]:
        r = results[sn]
        g = gross_results[sn]
        print(f"{sn}: GROSS_7pair_avg_sharpe={g['gross_sharpe_avg']:.3f}  NET_7pair_avg_sharpe={r['net_sharpe_avg']:.3f}"
              f"  avg_agree={r['avg_agree']:.3f}  avg_corr={r['avg_corr']:.3f}"
              f"  pooled_highER_mean={r['pooled_high_mean']:.1f}(win{r['pooled_high_win']*100:.0f}%)"
              f"  pooled_lowER_mean={r['pooled_low_mean']:.1f}(win{r['pooled_low_win']*100:.0f}%)")
    print(f"GOLD tsmom: GROSS_sharpe={gold_gross:.3f} NET_sharpe={gold_net_sharpe:.3f} mean_ER={gold_aer.mean():.3f}")
    print(f"MAJORS_mean_ER={majors_mean_er:.3f}")


if __name__ == "__main__":
    main()
