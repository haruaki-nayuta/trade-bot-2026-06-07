"""nb_session_time — 次足エッジ探索: 時間帯・セッション構造ファミリー (EURUSD M5/M1)

実行: uv run python -m research.experiments.nb_session_time

調査項目:
 1. UTC hour 別の次足ドリフト(train→test 持続性)
 2. hour×minute 季節性を特徴量化(train バケット平均→eval_signal)。q=0.02/0.05/0.10
 3. minute-of-hour 構造: 「毎時末 5-10 分の下落 → 毎時 10-15 分の上昇」
 4. セッション境界 (6,7,11,12,13 時) の V 字: 境界直前下落→直後上昇
 5. ロールオーバー 20:50-22:00 のマイクロ構造(BID スプレッド拡大アーティファクト疑い)
 6. 週末ギャップの埋め(イベントレベル・非重複)
 7. z20 平均回帰 × hour の分解(ベンチマークのエッジがどこに集中しているか)
 8. M1 での再検証(minute 季節性の train サwtooth アーティファクト確認含む)

死んだもの(このスクリプトでは再掲のみ・詳細探索済み):
 - 東京仲値 0:30-1:10 / WMR フィックス(ロンドン現地 16:00, DST 補正済) /
   NY オプションカット(NY 現地 10:00, DST 補正済): train→test で符号が反転、持続なし
 - 曜日効果(月〜金): 全て |t|<2。日曜オープンのみ +0.13p/bar (t=3.9) だが 21-23 時帯の
   スプレッド拡大ウィンドウと同一
 - 月末/月初(暦日 ±3 日): 全て持続なし
 - ロンドンオープンからの累積リターン(7-10時): IC_te -0.007, 死亡
 - NY オープンからの累積リターン(13:30-16時): IC_te -0.007, lo +0.45p t=1.5 のみ、有意水準未満
 - アジアレンジ(0-7時)ブレイク(7-10時評価): IC_te -0.006, 下抜けの h10 継続 -1.2p も t 弱
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.lab.nextbar_common import (
    SPLIT,
    eval_horizons,
    eval_signal,
    fmt_row,
    hour_mask,
    load_xy,
)


def bucket_table(keys: pd.Series, tgt: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"tgt": tgt, "k": keys}).dropna()
    tr = g[g.index < SPLIT].groupby("k")["tgt"].agg(["mean", "std", "count"])
    te = g[g.index >= SPLIT].groupby("k")["tgt"].agg(["mean", "std", "count"])
    return pd.DataFrame(
        {
            "tr_mean": tr["mean"],
            "tr_t": tr["mean"] / (tr["std"] / np.sqrt(tr["count"])),
            "te_mean": te["mean"],
            "te_t": te["mean"] / (te["std"] / np.sqrt(te["count"])),
            "te_n": te["count"],
        }
    )


def cond_mean(y: pd.Series) -> str:
    y = y.dropna()
    if len(y) < 3:
        return "n<3"
    t = y.mean() / (y.std() / np.sqrt(len(y)))
    return f"{y.mean():+.3f}p (t={t:+.1f}, n={len(y)})"


def main() -> None:
    df, tgt, pip = load_xy("EURUSD", "M5")
    idx = df.index
    c = df["close"]
    hr, mi = idx.hour, idx.minute
    tr_m = idx < SPLIT

    print("=" * 100)
    print("[1] UTC hour 別 次足ドリフト (M5) — 持続するのは 20/21/22 時のみ(ロールオーバー帯)")
    print(bucket_table(pd.Series(hr, index=idx), tgt).round(3).to_string())

    print("\n" + "=" * 100)
    print("[2] hour×minute 季節性特徴量 (train バケット平均→マップ)")
    key_hm = pd.Series(hr * 100 + mi, index=idx)
    mu = tgt[tr_m].groupby(key_hm[tr_m]).mean()
    seas = key_hm.map(mu)
    for q in (0.02, 0.05, 0.10):
        r = eval_signal(seas, tgt, q=q, name=f"seas_hm q={q}")
        print(fmt_row(r), f"| sig/day lo {r['lo_sig_per_day']:.2f} hi {r['hi_sig_per_day']:.2f}")
    h = eval_horizons(seas, df, "EURUSD", q=0.02)
    print("horizons(q=.02):", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in h.items()})
    for nm, hrs in [("active7-16", list(range(7, 17))), ("asia0-6", list(range(0, 7)))]:
        m = hour_mask(idx, hrs)
        r = eval_signal(seas[m], tgt[m], q=0.02, name=f"seas_hm@{nm}")
        print(fmt_row(r))
    # ロールオーバー帯を学習からも評価からも外した残差季節性
    m_day = pd.Series((hr >= 0) & (hr < 20), index=idx)
    mu2 = tgt[tr_m & m_day].groupby(key_hm[tr_m & m_day]).mean()
    seas_day = key_hm.map(mu2).where(m_day)
    r = eval_signal(seas_day, tgt, q=0.02, name="seas_hm_ex20-23")
    print(fmt_row(r), f"| sig/day lo {r['lo_sig_per_day']:.2f} hi {r['hi_sig_per_day']:.2f}")
    h = eval_horizons(seas_day, df, "EURUSD", q=0.02)
    print("horizons:", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in h.items()})

    print("\n" + "=" * 100)
    print("[3] minute-of-hour 構造 (日中 1-19 時): 毎時末ディップ → 毎時 10-15 分上昇")
    m_day19 = (hr >= 1) & (hr < 20)
    print(bucket_table(pd.Series(mi[m_day19], index=idx[m_day19]), tgt[m_day19]).round(3).to_string())

    print("\n" + "=" * 100)
    print("[4] セッション境界 V 字 (hours {6,7,11,12,13}): mi=50 bar の tgt = 毎時 55→00 の動き")
    for m_ in (45, 50, 55, 0):
        sel = pd.Series(np.isin(hr, [6, 7, 11, 12, 13]) & (mi == m_), index=idx)
        print(
            f"  boundary mi={m_:02d}: train {cond_mean(tgt[sel & tr_m])} | test {cond_mean(tgt[sel & ~tr_m])}"
        )
    # 境界直後ポップ(mi=55 bar)の horizon 減衰
    hts = {h_: c.diff(h_).shift(-h_) / pip for h_ in (1, 3, 5, 10, 20)}
    sel = pd.Series(np.isin(hr, [6, 7, 11, 12, 13]) & (mi == 55), index=idx) & ~tr_m
    print("  boundary_pop(mi=55) test horizons:",
          {f"h{h_}": cond_mean(ht[sel]) for h_, ht in hts.items()})

    print("\n" + "=" * 100)
    print("[5] ロールオーバー 20:30-22:30 マイクロ構造 (BIDアーティファクト疑い・取引不能枠)")
    hm = pd.Series(hr * 100 + mi, index=idx)
    sel = (hm >= 2030) & (hm <= 2230)
    tab = bucket_table(hm[sel], tgt[sel]).round(3)
    print(tab[(tab["tr_t"].abs() > 2.5) | (tab["te_t"].abs() > 2.5)].to_string())
    ret3 = (c - c.shift(3)) / pip
    at21 = pd.Series((hr == 21) & (mi <= 5), index=idx)
    r = eval_signal(ret3.where(at21), tgt, q=0.10, name="drop15m@21:00-05 q=.1")
    print(fmt_row(r), "← 直前15分の下落幅にバウンスが比例(BID回復)")

    print("\n" + "=" * 100)
    print("[6] 週末ギャップ埋め(イベントレベル・非重複)")
    ts = pd.Series(idx, index=idx)
    gap_bar = (ts - ts.shift(1)) > pd.Timedelta(hours=12)
    gap_pips = ((df["open"] - c.shift(1)) / pip).where(gap_bar)
    ev = gap_pips.dropna()
    thr = ev[ev.index < SPLIT].quantile([0.1, 0.25, 0.75, 0.9])
    print(f"  events: total {len(ev)}, train {(ev.index < SPLIT).sum()}, test {(ev.index >= SPLIT).sum()}")
    print(f"  train gap quantiles(pips): {thr.round(2).to_dict()}")
    hts = {h_: c.diff(h_).shift(-h_) / pip for h_ in (1, 3, 5, 10, 20, 50)}
    for tag, sel_ev in [
        ("gap_dn(<=q25)", ev <= thr[0.25]),
        ("gap_dn(<=q10)", ev <= thr[0.10]),
        ("gap_up(>=q75)", ev >= thr[0.75]),
        ("gap_up(>=q90)", ev >= thr[0.90]),
    ]:
        eidx = ev[sel_ev].index
        for st, ei in [("train", eidx[eidx < SPLIT]), ("test", eidx[eidx >= SPLIT])]:
            line = f"  {tag:<14} {st:<5} n={len(ei):3d} | "
            for h_, ht in hts.items():
                line += f"h{h_} {cond_mean(ht.reindex(ei))} | "
            print(line)

    print("\n" + "=" * 100)
    print("[7] z20 平均回帰 × hour 分解 (ベンチマークのエッジの所在)")
    z20 = (c - c.rolling(20).mean()) / c.rolling(20).std()
    for nm, hrs in [
        ("z20_all", list(range(24))),
        ("z20_no20-22", [h_ for h_ in range(24) if h_ not in (20, 21, 22)]),
        ("z20_only20-22", [20, 21, 22]),
        ("z20_active7-16", list(range(7, 17))),
        ("z20_asia0-6", list(range(0, 7))),
    ]:
        m = hour_mask(idx, hrs)
        r = eval_signal(z20[m], tgt[m], q=0.02, name=nm)
        print(fmt_row(r), f"| sig/day lo {r['lo_sig_per_day']:.2f} hi {r['hi_sig_per_day']:.2f}")
    rows = []
    for h_ in range(24):
        m = pd.Series(hr == h_, index=idx)
        r = eval_signal(z20[m], tgt[m], q=0.05, name=f"z20@h{h_:02d}")
        rows.append((h_, r["ic_test"], r["lo_test_mean_pips"], r["lo_test_t"], r["hi_test_mean_pips"], r["hi_test_t"]))
    print("hour | IC_te  | lo_te(t)       | hi_te(t)   (q=0.05/時間)")
    for h_, ic, lm, lt, hm_, ht_ in rows:
        print(f" {h_:3d} | {ic:+.3f} | {lm:+.3f} ({lt:+.1f}) | {hm_:+.3f} ({ht_:+.1f})")

    print("\n" + "=" * 100)
    print("[8] M1 再検証: minute-of-hour (日中 1-19 時)。train の交互サwtoothは test で消滅=アーティファクト")
    df1, tgt1, _ = load_xy("EURUSD", "M1")
    i1 = df1.index
    md = (i1.hour >= 1) & (i1.hour < 20)
    tab = bucket_table(pd.Series(i1.minute[md], index=i1[md]), tgt1[md]).round(4)
    print(tab[(tab["tr_t"].abs() > 3) | (tab["te_t"].abs() > 3)].to_string())
    # 毎時末ディップの M1 集約確認: 55-59 分の合計移動(test)
    print("  test 55-59min の合計(=M5 mi=50 tgt と一致するはず):",
          round(tab.loc[[54, 55, 56, 57, 58], "te_mean"].sum(), 3), "pips")
    # 週末ギャップを M1 でも(アンカー精度向上)
    ts1 = pd.Series(i1, index=i1)
    gb1 = (ts1 - ts1.shift(1)) > pd.Timedelta(hours=12)
    gp1 = ((df1["open"] - df1["close"].shift(1)) / pip).where(gb1).dropna()
    thr1 = gp1[gp1.index < SPLIT].quantile(0.25)
    hts1 = {h_: df1["close"].diff(h_).shift(-h_) / pip for h_ in (5, 15, 30, 60, 120)}
    ei = gp1[(gp1 <= thr1) & (gp1.index >= SPLIT)].index
    print(f"  M1 gap_dn(<=train q25={thr1:.1f}p) test n={len(ei)}:",
          {f"h{h_}m": cond_mean(ht.reindex(ei)) for h_, ht in hts1.items()})


if __name__ == "__main__":
    main()
