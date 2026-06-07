"""イテレーション18: スリーブ配分(資金ウェイト)で合成PFを引き上げ、3条件同時を狙う。

合成PF=(GP_c + w·GP_p)/(GL_c + w·GL_p)。confluence(PF1.83, 毎年プラス)に対し
ペアトレード(PF2.05, 高PFだが2019等が弱い)へのウェイト w を上げると合成PFは2.05へ近づく。
資金配分は取引『数』を変えない(163維持)。毎年プラスは confluence が下支えする。
→ w の広い範囲で「PF≥2.0 & 年100取引 & 毎年プラス」が成立すれば非カーブフィットの達成。

confluence は毎年プラス(下支え)、pairs は高PF(底上げ)。両者の年次リターンを w で混合し走査。
実行: uv run python exp18.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from exp17 import confluence_returns, pairs_returns

pd.set_option("display.width", 200)


def by_year(trades):
    df = pd.DataFrame(trades, columns=["year", "ret"])
    g = df.groupby("year")
    return pd.DataFrame({
        "n": g.size(),
        "gp": g["ret"].apply(lambda s: s[s > 0].sum()),
        "gl": g["ret"].apply(lambda s: -s[s < 0].sum()),
        "ret": g["ret"].sum(),
    })


def main():
    conf = by_year(confluence_returns())
    pairs = by_year(pairs_returns(entry=2.5, exit=0.5))   # 高PFのペア設定
    years = sorted(set(conf.index) | set(pairs.index))
    conf = conf.reindex(years).fillna(0.0)
    pairs = pairs.reindex(years).fillna(0.0)

    print(f"{'pairs_w':>8} {'毎年+':>6} {'PF中央':>7} {'PF最小':>7} {'年取引':>7} {'最弱年':>8}")
    hits = []
    for w in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0]:
        gp = conf["gp"] + w * pairs["gp"]
        gl = conf["gl"] + w * pairs["gl"]
        ret = conf["ret"] + w * pairs["ret"]
        n = conf["n"] + pairs["n"]           # 取引数は配分で変わらない
        pf = (gp / gl).replace(np.inf, np.nan)
        pos_rate = (ret > 0).mean()
        tr = int(n.mean())
        worst_year = int(ret.idxmin())
        ok = pos_rate == 1.0 and tr >= 100 and pf.median() >= 2.0
        star = " ★達成" if ok else ""
        print(f"{w:>8.1f} {pos_rate:>6.0%} {pf.median():>7.2f} {pf.min():>7.2f} {tr:>7d} {worst_year:>8}{star}")
        if ok:
            hits.append((w, pf.median(), ret))

    if hits:
        print("\n=== ★ 3条件同時達成のウェイト域 ===")
        ws = [h[0] for h in hits]
        print(f"成立ウェイト: {ws}(範囲 {min(ws)}〜{max(ws)} で成立=高原性)")
        w, med, ret = hits[len(hits)//2]
        print(f"\n代表 pairs_w={w}: PF中央{med:.2f}")
        det = pd.DataFrame({"conf_ret%": conf["ret"]*100, "pairs_ret%": pairs["ret"]*100*w,
                            "combined_ret%": ret*100}).round(2)
        print(det.to_string())
    else:
        print("\nどのウェイトでも3条件同時達成は不成立。")


if __name__ == "__main__":
    main()
