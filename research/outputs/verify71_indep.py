"""verify71: exp71(因果指値出口=純劣化)の敵対的独立再計算。

exp71 のコードは import しない。プール構築(build_pool_d1)と fxlab のみ利用可。
H4 系列は fxlab.data.resample ではなく groupby(floor("4h")) で自前構築(独立経路)。

監査項目:
  1. 指値水準の因果性: L_b = mean(close[b-50..b-1])(=b-1 close 確定時点の最新)か。
     stale 変種(L=mv[b-2])との差を実測し、古い情報で不利にしていないか確認。
  2. fill 判定の保守性: BID 系列で hi > L(long)/ lo < L(short)。スプレッド二重課金の有無を
     「指値に半スプレッドをクレジットした最甘ケース」で上限評価。
  3. Δret 式: exit_eff = entry_price*(1+dir*ret) を独立再構成し Cx − dir*sp/2 と突合。
     分母 Ce(生close) vs entry_eff の差も定量化。
  4. 決済バー内 fill の機構: メジャー3銘柄×直近100決済バーで M1 パスから
     「mean 到達時刻 → close がさらに突き抜けるか」を独立サンプル確認。
  5. 早期 fill の希少性: 毎バー更新実装(本スクリプト)と「初回バー固定 L」変種の fill 集合を比較。
  6. メジャー7銘柄の純効果(bps/トレード)を自前計算し exp71 の −0.89bps と突合。

追加(逃げ場探索): L を mean ± 0.25σ(因果, b-1)にずらした変種の純効果も実測。

実行: uv run python research/outputs/verify71_indep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "money_management"))

from mm_production import build_pool_d1  # noqa: E402
from fxlab import config, data  # noqa: E402
from fxlab import universe as uni  # noqa: E402

WIN = 50
MAJORS = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCHF", "USDCAD", "NZDUSD"]
SAMPLE_MAJORS = ["EURUSD", "USDJPY", "GBPUSD"]  # 点4 の M1 パス検査対象

pd.set_option("display.width", 220)


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    uni.register_cross_spreads(3.0)

    sec("0. プール検算")
    pool = build_pool_d1().reset_index(drop=True)
    n_all, sret = len(pool), float(pool["ret"].sum())
    print(f"pool n={n_all} sum(ret)={sret:+.4f}  (期待 1207 / +1.9622)")
    assert n_all == 1207 and abs(sret - 1.9622) < 1e-3

    mp = pool[pool["instr"].isin(MAJORS)].reset_index(drop=True)
    print(f"majors n={len(mp)}")

    # exp71 のトレード別出力(コードは import せず、出力のみ突合に使用)
    e71 = pd.read_csv(ROOT / "research" / "outputs" / "exp71_trades.csv",
                      parse_dates=["entry", "exit"])
    e71m = e71[e71["instr"].isin(MAJORS)].reset_index(drop=True)

    rows = []          # 自前 base 計算のトレード別結果
    sample4 = []       # 点4: 決済バー M1 パスのサンプル
    audit3_max_err = 0.0

    for nm in MAJORS:
        m1 = data.load_m1(nm)
        key = m1.index.floor("4h")
        # 独立経路: groupby で H4 構築(fxlab.resample 不使用)
        h4c = m1["close"].groupby(key).last()
        bhi = m1["high"].groupby(key).max()
        blo = m1["low"].groupby(key).min()
        idx = h4c.index
        cv = h4c.to_numpy()
        hv = bhi.to_numpy()
        lv = blo.to_numpy()
        mv = h4c.rolling(WIN).mean().to_numpy()
        sdv = h4c.rolling(WIN).std().to_numpy()
        sp = config.spread_pips(nm) * config.pip_size(nm)

        g = mp[mp["instr"] == nm]
        ie = idx.get_indexer(pd.DatetimeIndex(g["entry"]))
        ix = idx.get_indexer(pd.DatetimeIndex(g["exit"]))
        assert (ie >= 0).all() and (ix >= 0).all(), f"{nm}: timestamp miss"

        # ---- 点3: 価格規約の独立再構成 ------------------------------------
        d_arr = g["dir"].to_numpy().astype(float)
        Ce = cv[ie]
        Cx = cv[ix]
        entry_eff_re = Ce + d_arr * sp / 2.0
        err_e = np.abs(entry_eff_re - g["entry_price"].to_numpy()) / Ce
        exit_eff = g["entry_price"].to_numpy() * (1.0 + d_arr * g["ret"].to_numpy())
        exit_eff_re = Cx - d_arr * sp / 2.0
        err_x = np.abs(exit_eff - exit_eff_re) / Cx
        audit3_max_err = max(audit3_max_err, float(err_e.max()), float(err_x.max()))

        # ---- 自前スキャン: base / stale / fixed-L / deeper / shallower ----
        n_sampled = 0
        for k, (e, x, d, ret) in enumerate(zip(ie, ix, d_arr, g["ret"].to_numpy())):
            res = {}
            for tag, lag, q, fixed in [("base", 1, 0.0, False),
                                       ("stale", 2, 0.0, False),
                                       ("fixedL", 1, 0.0, True),
                                       ("deep", 1, +0.25, False),
                                       ("shallow", 1, -0.25, False)]:
                filled, fpos, Lf = False, -1, np.nan
                L_first = np.nan
                for b in range(e + 1, x + 1):
                    j = b - lag
                    if j < 0:
                        continue
                    L = mv[j] + d * q * sdv[j]
                    if fixed:
                        if not np.isfinite(L_first):
                            L_first = L
                        L = L_first
                    if not np.isfinite(L):
                        continue
                    hit = (hv[b] > L) if d > 0 else (lv[b] < L)
                    if hit:
                        filled, fpos, Lf = True, b, L
                        break
                res[tag] = (filled, fpos, Lf)

            fb, pb, Lb = res["base"]
            delta = d * (Lb - Cx[k]) / Ce[k] if fb else 0.0
            row = {"instr": nm, "entry": g["entry"].iloc[k], "exit": g["exit"].iloc[k],
                   "dir": d, "ret": ret, "Ce": Ce[k], "Cx": Cx[k], "sp": sp,
                   "e": e, "x": x,
                   "filled": fb, "fill_pos": pb, "L": Lb, "delta": delta}
            for tag in ("stale", "fixedL", "deep", "shallow"):
                ft, pt, Lt = res[tag]
                row[f"{tag}_filled"] = ft
                row[f"{tag}_delta"] = d * (Lt - Cx[k]) / Ce[k] if ft else 0.0
                row[f"{tag}_pos"] = pt
            rows.append(row)

            # ---- 点4: 決済バー内 fill の M1 パス検査(3銘柄×直近100) ----
            if nm in SAMPLE_MAJORS and fb and pb == x and n_sampled < 100:
                lab = idx[x]
                seg = m1.loc[lab: lab + pd.Timedelta(hours=4) - pd.Timedelta(minutes=1)]
                if d > 0:
                    crossed = seg["high"].to_numpy() > Lb
                else:
                    crossed = seg["low"].to_numpy() < Lb
                if crossed.any():
                    first_min = (seg.index[np.argmax(crossed)] - lab).total_seconds() / 60
                    bar_len = (seg.index[-1] - lab).total_seconds() / 60 + 1
                    close_beyond = d * (Cx[k] - Lb) > 0  # close が指値のさらに先で確定?
                    sample4.append({"instr": nm, "exit_bar": lab,
                                    "first_cross_min": first_min, "bar_minutes": bar_len,
                                    "delta_bps": delta * 1e4,
                                    "close_beyond_L": bool(close_beyond)})
                    n_sampled += 1

        del m1
        data.clear_cache()
        print(f"  {nm}: done (trades {len(g)})")

    tv = pd.DataFrame(rows)

    sec("1+5. 因果性とfill実装の確認(コード読解+変種実測)")
    print("L_b = rolling50.mean を b-1 で評価(=発注時点の最新確定値)。毎バー更新。")
    for tag, lab in [("base", "base(L=mv[b-1], 毎バー更新)"),
                     ("stale", "stale(L=mv[b-2], 1本古い)"),
                     ("fixedL", "fixed-L(エントリー時の L 固定・無更新)")]:
        f = tv[f"{tag}_filled"] if tag != "base" else tv["filled"]
        dd = tv[f"{tag}_delta"] if tag != "base" else tv["delta"]
        print(f"  {lab:42s}: fill率 {f.mean():.1%}  純 {dd.mean()*1e4:+.3f}bps")
    same_fill = (tv["filled"] == tv["fixedL_filled"]).all()
    print(f"  base と fixed-L の fill 集合は同一か: {same_fill}"
          f"(False=毎バー更新は fill 集合を実際に変える=更新は実装されている)")

    sec("3. 価格規約の独立再構成(点3)")
    print(f"  entry_eff=Ce+d·sp/2 / exit_eff=entry_price·(1+d·ret) vs Cx−d·sp/2 の"
          f"最大相対誤差 = {audit3_max_err:.2e}")
    print("  → 現行出口は BID close − d·sp/2。指値 fill も同じ BID 系列+同一半スプレッド規約なら")
    print("    Δret = d·(L−Cx)/分母 でスプレッド項は厳密に相殺(二重課金なし)。")

    sec("6. メジャー7銘柄の独立再計算 vs exp71")
    net_bps = tv["delta"].mean() * 1e4
    print(f"  自前 base: fill率 {tv['filled'].mean():.1%}  純効果 {net_bps:+.4f}bps/トレード"
          f"  ΣΔret {tv['delta'].sum():+.5f}")
    print(f"  exp71 majors: 純効果 {e71m['delta_bps'].mean():+.4f}bps"
          f"  fill率 {e71m['filled'].mean():.1%}")
    mg = tv.merge(e71m[["instr", "entry", "exit", "filled", "delta_ret", "fill_pos"]],
                  on=["instr", "entry", "exit"], suffixes=("", "_71"))
    print(f"  突合件数 {len(mg)}/{len(tv)}  filled 一致 {(mg['filled']==mg['filled_71']).mean():.1%}"
          f"  fill_pos 一致(filled時) "
          f"{(mg.loc[mg['filled'],'fill_pos']==mg.loc[mg['filled'],'fill_pos_71']).mean():.1%}")
    dmax = float(np.abs(mg["delta"] - mg["delta_ret"]).max())
    print(f"  Δret 最大差 {dmax:.2e}")

    # 分母 Ce vs entry_eff
    ee = tv["Ce"] + tv["dir"] * tv["sp"] / 2.0
    alt = (tv["dir"] * (tv["L"] - tv["Cx"]) / ee).where(tv["filled"], 0.0)
    print(f"  分母を entry_eff にした場合: {alt.mean()*1e4:+.4f}bps(差は無視可能か確認)")

    sec("2. スプレッド・クレジット上限(最甘ケース)— 偽陰性の上限監査")
    # 「指値はスリッページを払わない」と解釈し、fill トレードに半スプレッドをクレジット
    credit = tv["delta"] + np.where(tv["filled"], (tv["sp"] / 2.0) / tv["Ce"], 0.0)
    print(f"  majors base {net_bps:+.4f}bps → 半スプレッドクレジット後 {credit.mean()*1e4:+.4f}bps")
    print("  (ロングの真の機構では市場売り=BID・指値売り=BID で同一=クレジット根拠なし。")
    print("   ショートはむしろ ASK 到達が必要で base が甘い側。これは『あり得る最甘上限』)")

    sec("4. 決済バー内 fill の M1 パス独立サンプル(3銘柄×直近100)")
    s4 = pd.DataFrame(sample4)
    if len(s4):
        for nm, gg in s4.groupby("instr"):
            print(f"  {nm}: n={len(gg)}  mean到達分(バー開始から) 中央値 {gg['first_cross_min'].median():.0f}分"
                  f"  Δ平均 {gg['delta_bps'].mean():+.2f}bps"
                  f"  close が指値のさらに先で確定した率 {gg['close_beyond_L'].mean():.0%}")
        print(f"  全体: n={len(s4)}  Δ平均 {s4['delta_bps'].mean():+.2f}bps"
              f"  close_beyond_L 率 {s4['close_beyond_L'].mean():.0%}")
    # 自前計算での決済バー内 fill 全体(majors)
    inx = tv["filled"] & (tv["fill_pos"] == tv["x"])
    eaf = tv["filled"] & (tv["fill_pos"] < tv["x"])
    print(f"  majors 決済バー内 fill: n={int(inx.sum())} 平均 {tv.loc[inx,'delta'].mean()*1e4:+.2f}bps"
          f" / 早期 fill: n={int(eaf.sum())} 平均 "
          f"{tv.loc[eaf,'delta'].mean()*1e4 if eaf.any() else float('nan'):+.2f}bps")

    sec("追加: 指値水準をずらした変種(逃げ場探索, majors)")
    for tag, lab in [("deep", "deep(L=mean+0.25σ 利食い深め)"),
                     ("shallow", "shallow(L=mean−0.25σ 利食い浅め)")]:
        f = tv[f"{tag}_filled"]
        dd = tv[f"{tag}_delta"]
        print(f"  {lab:34s}: fill率 {f.mean():.1%}  純 {dd.mean()*1e4:+.3f}bps")

    sec("判定材料まとめ")
    print(f"  exp71 majors −0.89bps vs 自前 {net_bps:+.2f}bps / fill集合一致 / 規約誤差 {audit3_max_err:.0e}")
    out = ROOT / "research" / "outputs" / "verify71_trades_majors.csv"
    tv.to_csv(out, index=False)
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
