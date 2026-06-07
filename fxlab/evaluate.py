"""10年スパンの総合評価バッテリー + 改善提案の自動生成。

「雑にアイデアを放り込む → 10年検証結果 + 改善提案」を実現する中核。
1 つの戦略に対して以下を自動実行し、データ駆動で改善案を出す:

  A. 主ペア×主足のパラメータ探索(過剰最適化チェック用にグリッド全体も保持)
  B. イン/アウトサンプル(IS/OOS)検証 — 期間前半で最適化→後半で素の成績を確認
  C. 7ペア横断 — 通貨依存(まぐれ)でないか
  D. マルチ時間足 — どの足が効くか
  E. ロング/ショート分離 — 片側だけ効いていないか
  F. 損切り/利確/トレーリングの自動付与テスト — 改善余地を“実測”

そのうえで diagnose() がヒューリスティクスで具体的な改善提案を生成する。
"""

from __future__ import annotations

import pandas as pd

from . import config
from .backtest import metrics, run, sweep
from .data import available_pairs, load


# --- ユーティリティ ------------------------------------------------------
def _best_params(res: pd.DataFrame) -> dict:
    """sweep 結果(objective 降順済み)の先頭からパラメータ dict を復元。"""
    idx = res.index
    top = idx[0]
    names = list(idx.names)
    if not isinstance(top, tuple):
        top = (top,)
    return {n: (int(v) if float(v).is_integer() else float(v)) for n, v in zip(names, top)}


def _time_split(df: pd.DataFrame, frac: float = 0.7):
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]


def _m(pf) -> pd.Series:
    return metrics(pf).iloc[0]


# --- 総合評価 ------------------------------------------------------------
def evaluate(
    strategy_name: str,
    module,
    *,
    primary_pair: str = "EURUSD",
    primary_tf: str = "H1",
    timeframes: list[str] | None = None,
    objective: str = "sharpe",
    size_mode: str = "full",
    size_value: float | None = None,
) -> dict:
    gen = module.generate_signals
    grid = getattr(module, "PARAM_GRID", None)
    default_params = getattr(module, "PARAMS", {})
    timeframes = timeframes or ["M15", "H1", "H4", "D1"]
    szkw = {"size_mode": size_mode, "size_value": size_value}  # サイジングを全検証へ伝播

    full = load(primary_pair, primary_tf)
    period = (full.index.min(), full.index.max())
    years = round((period[1] - period[0]).days / 365.25, 1)

    out: dict = {
        "strategy": strategy_name,
        "primary_pair": primary_pair,
        "primary_tf": primary_tf,
        "objective": objective,
        "period": period,
        "years": years,
        "bars": len(full),
        "size_mode": size_mode,
        "size_value": size_value,
    }

    # A. 主ペア×主足の探索(グリッドが無ければデフォルト1点)
    if grid:
        res = sweep(primary_pair, primary_tf, gen, grid, objective=objective, **szkw)
        best = _best_params(res)
        out["sweep"] = res
        out["grid"] = grid
    else:
        res = None
        best = dict(default_params)
        out["sweep"] = None
        out["grid"] = None
    out["best_params"] = best
    out["best_metrics"] = _m(run(primary_pair, primary_tf, gen, best, **szkw))

    # B. IS / OOS
    is_df, oos_df = _time_split(full, 0.7)
    if grid:
        is_res = sweep(primary_pair, primary_tf, gen, grid, data=is_df, objective=objective, **szkw)
        is_best = _best_params(is_res)
    else:
        is_best = dict(default_params)
    is_m = _m(run(primary_pair, primary_tf, gen, is_best, data=is_df, **szkw))
    oos_m = _m(run(primary_pair, primary_tf, gen, is_best, data=oos_df, **szkw))
    out["is_oos"] = {
        "is_params": is_best,
        "is": is_m,
        "oos": oos_m,
        "is_period": (is_df.index.min(), is_df.index.max()),
        "oos_period": (oos_df.index.min(), oos_df.index.max()),
    }

    # C. 7ペア横断(best_params 固定)
    rows = {}
    for pair in available_pairs():
        try:
            rows[pair] = _m(run(pair, primary_tf, gen, best, **szkw))
        except Exception:  # noqa: BLE001
            pass
    out["all_pairs"] = pd.DataFrame(rows).T if rows else pd.DataFrame()

    # D. マルチ時間足(best_params 固定)
    rows = {}
    for tf in timeframes:
        try:
            rows[tf] = _m(run(primary_pair, tf, gen, best, **szkw))
        except Exception:  # noqa: BLE001
            pass
    out["timeframes"] = pd.DataFrame(rows).T if rows else pd.DataFrame()

    # E. ロング / ショート分離
    rows = {}
    for side in ("both", "long", "short"):
        rows[side] = _m(run(primary_pair, primary_tf, gen, best, side=side, **szkw))
    out["sides"] = pd.DataFrame(rows).T

    # F. 損切り/利確/トレーリング 自動テスト(改善余地の実測)
    variants = [
        ("baseline", {}),
        ("SL1%", {"sl_stop": 0.01}),
        ("SL2%", {"sl_stop": 0.02}),
        ("SL3%", {"sl_stop": 0.03}),
        ("SL1%+TP2%", {"sl_stop": 0.01, "tp_stop": 0.02}),
        ("SL2%+TP4%", {"sl_stop": 0.02, "tp_stop": 0.04}),
        ("TSL2%", {"tsl_stop": 0.02}),
    ]
    rows = {}
    for name, kw in variants:
        rows[name] = _m(run(primary_pair, primary_tf, gen, best, **kw, **szkw))
    stops = pd.DataFrame(rows).T
    out["stops"] = stops

    out["diagnostics"] = diagnose(out)
    return out


# --- ウォークフォワード最適化(ローリング) ----------------------------
def walk_forward(
    strategy_name: str,
    module,
    *,
    pair: str = "EURUSD",
    tf: str = "H1",
    n_folds: int = 5,
    objective: str = "sharpe",
) -> dict:
    """アンカード・ウォークフォワード。各 fold で「過去全部で最適化→次の窓で素検証」。

    単一 IS/OOS より厳しく、毎回その時点までの情報だけで選んだパラメータの
    アウトオブサンプル成績を積み上げる。再最適化を繰り返しても通用するかを見る。

    返り値: {folds: DataFrame, summary: Series, consistency: float}
    """
    gen = module.generate_signals
    grid = getattr(module, "PARAM_GRID", None)
    full = load(pair, tf)
    n = len(full)
    seg = n // (n_folds + 1)

    rows = []
    for i in range(n_folds):
        train = full.iloc[: (i + 1) * seg]
        test = full.iloc[(i + 1) * seg : (i + 2) * seg]
        if len(test) < 30:
            continue
        if grid:
            res = sweep(pair, tf, gen, grid, data=train, objective=objective)
            bp = _best_params(res)
        else:
            bp = dict(getattr(module, "PARAMS", {}))
        m = _m(run(pair, tf, gen, bp, data=test))
        rows.append({
            "fold": i + 1,
            "test_start": test.index.min(),
            "test_end": test.index.max(),
            "params": bp,
            "total_return": m["total_return"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["max_drawdown"],
            "num_trades": m["num_trades"],
        })

    folds = pd.DataFrame(rows)
    if folds.empty:
        return {"folds": folds, "summary": pd.Series(dtype=float), "consistency": float("nan")}
    summary = folds[["total_return", "sharpe", "max_drawdown", "num_trades"]].mean()
    consistency = float((folds["sharpe"] > 0).mean())  # OOSでプラスだったfoldの割合
    return {"folds": folds, "summary": summary, "consistency": consistency,
            "strategy": strategy_name, "pair": pair, "tf": tf}


# --- 改善提案(データ駆動ヒューリスティクス) --------------------------
def diagnose(r: dict, objective: str = "sharpe") -> list[dict]:
    """評価結果から具体的な改善提案を生成。{issue, evidence, suggestion} のリスト。"""
    props: list[dict] = []
    best = r["best_metrics"]

    # 1. 過剰最適化(IS→OOS の劣化)
    io = r["is_oos"]
    iss, oos = io["is"]["sharpe"], io["oos"]["sharpe"]
    if pd.notna(iss) and pd.notna(oos):
        if oos <= 0 < iss:
            props.append({
                "issue": "過剰最適化の疑い(強)",
                "evidence": f"IS Sharpe {iss:+.2f} → OOS {oos:+.2f}(アウトサンプルで崩壊)",
                "suggestion": "パラメータ数を減らす/探索範囲を狭める。固定値化や指標の単純化を検討。"
                              "OOS でプラスを維持できる頑健な領域だけを採用する。",
            })
        elif iss > 0 and oos < 0.5 * iss:
            props.append({
                "issue": "過剰最適化の疑い",
                "evidence": f"IS Sharpe {iss:+.2f} → OOS {oos:+.2f}(約 {oos/iss*100:.0f}% に劣化)",
                "suggestion": "パラメータ近傍が滑らかに良い領域を選ぶ。グリッドを粗くして頑健性を優先。",
            })

    # 2. パラメータがグリッド端 → 範囲拡張
    grid = r.get("grid")
    if grid:
        edges = []
        for k, v in r["best_params"].items():
            if k in grid and len(grid[k]) > 1:
                if v == min(grid[k]):
                    edges.append(f"{k}={v}(下限)")
                elif v == max(grid[k]):
                    edges.append(f"{k}={v}(上限)")
        if edges:
            props.append({
                "issue": "最適パラメータがグリッド端",
                "evidence": "、".join(edges),
                "suggestion": "PARAM_GRID の探索範囲をその方向へ広げて、真の最適が外側にないか確認。",
            })

    # 3. 高原 vs 突出(まぐれ最適化)
    res = r.get("sweep")
    if res is not None and "sharpe" in res:
        sh = res["sharpe"].dropna()
        if len(sh) >= 4:
            bs = sh.max()
            good = (sh > max(0, bs * 0.6)).sum()
            if bs > 0 and good <= max(1, int(0.15 * len(sh))):
                props.append({
                    "issue": "最適点が孤立(高原でない)",
                    "evidence": f"良好な組合せは {good}/{len(sh)} 件のみ。1点突出はまぐれの可能性。",
                    "suggestion": "近傍パラメータでも成績が保たれる領域を採用。突出値は実運用で再現しにくい。",
                })

    # 4. 取引数が少ない
    nt = best.get("num_trades")
    if pd.notna(nt) and nt < 30:
        props.append({
            "issue": "取引数が少なく統計的に不安定",
            "evidence": f"10年で {int(nt)} 取引",
            "suggestion": "時間足を下げる/エントリー条件を緩める。または複数ペアに分散して試行回数を稼ぐ。",
        })

    # 5. ドローダウン過大 + 損切りテストの実測改善
    dd = best.get("max_drawdown")
    stops = r.get("stops")
    if stops is not None and objective in stops:
        base = stops.loc["baseline", objective]
        bestvar = stops[objective].idxmax()
        bestval = stops[objective].max()
        if bestvar != "baseline" and pd.notna(base) and bestval > base * 1.1 + 1e-9:
            props.append({
                "issue": "損切り/利確で改善余地あり(実測済み)",
                "evidence": f"{bestvar} で {objective} {base:+.2f} → {bestval:+.2f} に改善",
                "suggestion": f"{bestvar} 相当の損切り/利確/トレーリングを戦略に組み込む。"
                              "ATR 連動にするとより頑健。",
            })
    if pd.notna(dd) and dd < -0.25:
        props.append({
            "issue": "最大ドローダウン過大",
            "evidence": f"max_drawdown {dd:.1%}",
            "suggestion": "損切り導入・ポジションサイズ縮小・トレンド/ボラフィルタで深いDDを抑制。",
        })

    # 6. 通貨依存(横断のばらつき)
    ap = r.get("all_pairs")
    if ap is not None and not ap.empty and "sharpe" in ap:
        sh = ap["sharpe"].dropna()
        win = sh[sh > 0].index.tolist()
        lose = sh[sh <= 0].index.tolist()
        if win and lose:
            props.append({
                "issue": "通貨ペア依存",
                "evidence": f"プラス: {', '.join(win)} / マイナス: {', '.join(lose)}",
                "suggestion": "効くペアに限定するか、レジーム/ボラフィルタで効かない局面を除外。"
                              "全ペアで負けなら手法自体を見直す。",
            })

    # 7. 最適な時間足の提示
    tfs = r.get("timeframes")
    if tfs is not None and not tfs.empty and "sharpe" in tfs:
        sh = tfs["sharpe"].dropna()
        if len(sh):
            best_tf = sh.idxmax()
            if best_tf != r["primary_tf"] and sh.max() > sh.get(r["primary_tf"], -99):
                props.append({
                    "issue": "より相性の良い時間足あり",
                    "evidence": f"{best_tf} の Sharpe {sh.max():+.2f}(検証足 {r['primary_tf']} より良い)",
                    "suggestion": f"主戦場を {best_tf} に変更して再検証する。",
                })

    # 8. ロング/ショート非対称
    sides = r.get("sides")
    if sides is not None and "sharpe" in sides:
        ls, ss = sides["sharpe"].get("long"), sides["sharpe"].get("short")
        if pd.notna(ls) and pd.notna(ss):
            if ls > 0.3 and ss < 0:
                props.append({
                    "issue": "ショートが足を引っ張る",
                    "evidence": f"long Sharpe {ls:+.2f} / short {ss:+.2f}",
                    "suggestion": "ロング専用にする、またはショート側に別ロジック/フィルタを用意。",
                })
            elif ss > 0.3 and ls < 0:
                props.append({
                    "issue": "ロングが足を引っ張る",
                    "evidence": f"long Sharpe {ls:+.2f} / short {ss:+.2f}",
                    "suggestion": "ショート専用にする、またはロング側に別ロジック/フィルタを用意。",
                })

    # 9. コスト負け気味
    pf_ = best.get("profit_factor")
    if pd.notna(pf_) and 1.0 < pf_ < 1.2:
        props.append({
            "issue": "取引コスト負け気味",
            "evidence": f"profit_factor {pf_:.2f}(コスト計上後、薄利)",
            "suggestion": "より大きい時間足で取引頻度を下げる、エントリーを厳選してコスト負けを回避。",
        })

    # 総括
    if not props:
        props.append({
            "issue": "大きな弱点は検出されず",
            "evidence": f"OOS Sharpe {oos:+.2f}、DD {best.get('max_drawdown', float('nan')):.1%}" if pd.notna(oos) else "—",
            "suggestion": "ロット管理・複数ペア分散・別レジームでの追加検証で実運用へ。",
        })
    return props
