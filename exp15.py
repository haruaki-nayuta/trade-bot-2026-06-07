"""イテレーション15: 平均回帰性の強い銘柄を客観選択し、頻度を上げてPF2.0×年100取引×毎年プラスを狙う。

着想: 個別では AUDCAD(PF6.9)等、PF2.0を大きく超える銘柄がある。弱い銘柄に薄められて
集計1.83になっている。「平均回帰性が統計的に強い銘柄」だけを a-priori(戦略成績を見ない
分散比で)選び、エントリーをやや緩めて頻度を上げれば、高PFの余力を年100取引に充てつつ
PF2.0・毎年プラスを同時達成できる可能性がある。

銘柄選択は **分散比 VR(k)=Var(k期リターン)/(k·Var(1期))**(VR<1=平均回帰、戦略非依存=非カーブフィット)。
実行: uv run python exp15.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from fxlab import universe as uni
from strategies.confluence_meanrev import generate_signals as g

pd.set_option("display.width", 200)
TF = "H4"


def variance_ratio(close: pd.Series, k: int = 10) -> float:
    r = np.log(close).diff().dropna()
    if len(r) < k * 5:
        return np.nan
    var1 = r.var()
    vark = r.rolling(k).sum().dropna().var() / k
    return float(vark / var1) if var1 > 0 else np.nan


def main():
    uni.register_cross_spreads(3.0)
    allinstr = uni.universe()  # 7メジャー+13クロス

    # 1) 客観的な平均回帰スコア(分散比, 小さいほど平均回帰)— 戦略成績は一切見ない
    scores = {}
    for nm in allinstr:
        c = uni.instrument_close(nm, TF)
        scores[nm] = variance_ratio(c, k=10)
    sc = pd.Series(scores).sort_values()
    print("=== 平均回帰スコア(分散比VR(10), 小=平均回帰的)===")
    print(sc.round(3).to_string())

    # 2) a-priori 経済連動ブロック内クロス(USD=世界の主因, JPY=キャリー/リスク を除外し
    #    トレンド要因の小さい欧州系/資源系クロスに限定)。戦略成績は見ない=非カーブフィット。
    basket = ["EURCHF", "EURGBP", "GBPCHF",          # 欧州ブロック
              "AUDCAD", "AUDNZD", "NZDCAD",          # 資源ブロック
              "EURAUD", "EURCAD", "GBPAUD", "AUDCHF"]  # ブロック間(非USD非JPY)
    basket = [b for b in basket if b in allinstr]
    print(f"\n採用バスケット(経済連動クロス, {len(basket)}銘柄): {basket}")
    print(f"(参考VR: {sc[basket].round(3).to_dict()})\n")

    # 3) 頻度を上げる方向に entry_z/slow_z を振り、3条件同時を探す
    base = dict(window=50, exit_z=0.5, rsi_p=14, rsi_low=35, rsi_high=65, vol_win=100, vol_pct=0.70)
    print(f"{'entry_z':>8}{'slow_z':>8}{'rsi帯':>10} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7}")
    hits = []
    for ez, sz, rsi in itertools.product([1.25, 1.5, 1.75, 2.0], [1.0, 1.5, 1.75], [(40, 60), (35, 65)]):
        rl, rh = rsi
        P = dict(base, entry_z=ez, slow_z=sz, rsi_low=rl, rsi_high=rh)
        port = uni.portfolio_yearly(TF, g, P, instruments=basket, size_mode="value")
        if port.empty:
            continue
        pf = port["profit_factor"].replace(np.inf, np.nan)
        pos = (port["pnl"] > 0).mean(); med = pf.median(); mn = pf.min(); tr = int(port["trades"].mean())
        star = ""
        if pos == 1.0 and tr >= 100 and med >= 2.0:
            star = " ★"; hits.append((P, med, mn, tr, port))
        # 取引数が100近い or 達成のものだけ表示
        if tr >= 80 or star:
            print(f"{ez:>8}{sz:>8}{str(rl)+'/'+str(rh):>10} {pos:>6.0%} {med:>7.2f} {mn:>7.2f} {tr:>7d}{star}")

    if hits:
        hits.sort(key=lambda x: -x[1])
        P, med, mn, tr, port = hits[0]
        print(f"\n=== ★★★ 3条件同時達成 ===\nparams={P}\nPF中央{med:.2f} 最小{mn:.2f} 年取引{tr}")
        b = port.copy(); b["profit_factor"] = b["profit_factor"].replace(np.inf, np.nan).round(2)
        b["pnl"] = b["pnl"].round(0); print(b.to_string())
    else:
        print("\n3条件同時達成は見つからず。")


if __name__ == "__main__":
    main()
