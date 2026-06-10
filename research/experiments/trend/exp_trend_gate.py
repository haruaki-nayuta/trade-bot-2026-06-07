"""exp_trend_gate — トレンド強度ゲート族(ADX / Kaufman効率比ER)の固定グリッド検証。

族: トレンド強度ゲート型。
(a) 既存 strategies/adx_trend.py を再利用(MAクロス方向 + ADX>th ゲート)。
    注意: ADX は high/low を使うため、クロス(OHLC=close複製)では TR/DM が
    close 差分のみから計算される退化形になる。
(b) 新規 ER トレンド(close のみ=クロスにも正しく適用):
    ER(er_win) = |close - close[-er_win]| / Σ|Δ1| を .shift(1)(確定バーのみ)。
    ER > er_min かつ close>SMA(slow) でロング / close<SMA(slow) でショート。
    手仕舞い: ER < 0.25 または SMA 側の反転。

固定グリッド(追い込み禁止・全構成報告):
  ADX H4: (fast,slow,adx_th) = (20,50,20),(30,100,20),(30,100,25)
  ADX D1: (10,50,20),(20,100,20)
  ER  H4: (er_win,er_min,slow) = (40,0.45,100),(40,0.55,100),(20,0.45,50)
  ER  D1: (40,0.45,50),(20,0.45,50)

実行: リポジトリ直下で PYTHONPATH=. uv run python research/experiments/trend/exp_trend_gate.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")
import trend_lab as tl  # noqa: E402

from strategies.adx_trend import generate_signals as adx_gen  # noqa: E402


def er_trend_signals(data, er_win: int = 40, er_min: float = 0.45,
                     slow: int = 100, er_exit: float = 0.25):
    """Kaufman ER ゲート付き SMA トレンドフォロー(close のみ使用)。

    先読み防止: ER は .shift(1) で確定バーのみ。SMA は現在バーの確定 close まで。
    """
    close = data["close"]
    direction = (close - close.shift(er_win)).abs()
    volatility = close.diff().abs().rolling(er_win).sum()
    er = (direction / volatility).shift(1)  # 確定バーの ER のみ使用

    sma = close.rolling(slow).mean()
    up = close > sma
    down = close < sma

    long_state = (er > er_min) & up
    short_state = (er > er_min) & down
    long_entries = long_state & ~long_state.shift(fill_value=False)
    short_entries = short_state & ~short_state.shift(fill_value=False)
    long_exits = (er < er_exit) | down   # ER 失速 or SMA 側の反転
    short_exits = (er < er_exit) | up
    return (long_entries.fillna(False), long_exits.fillna(False),
            short_entries.fillna(False), short_exits.fillna(False))


CONFIGS = [
    # (a) ADX ゲート MA トレンド — 既存 strategies/adx_trend.py
    ("adx_f20s50t20_H4_both", adx_gen, {"fast": 20, "slow": 50, "adx_period": 14, "adx_th": 20}, "H4"),
    ("adx_f30s100t20_H4_both", adx_gen, {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 20}, "H4"),
    ("adx_f30s100t25_H4_both", adx_gen, {"fast": 30, "slow": 100, "adx_period": 14, "adx_th": 25}, "H4"),
    ("adx_f10s50t20_D1_both", adx_gen, {"fast": 10, "slow": 50, "adx_period": 14, "adx_th": 20}, "D1"),
    ("adx_f20s100t20_D1_both", adx_gen, {"fast": 20, "slow": 100, "adx_period": 14, "adx_th": 20}, "D1"),
    # (b) ER ゲート SMA トレンド — close のみ
    ("er_w40m045s100_H4_both", er_trend_signals, {"er_win": 40, "er_min": 0.45, "slow": 100}, "H4"),
    ("er_w40m055s100_H4_both", er_trend_signals, {"er_win": 40, "er_min": 0.55, "slow": 100}, "H4"),
    ("er_w20m045s50_H4_both", er_trend_signals, {"er_win": 20, "er_min": 0.45, "slow": 50}, "H4"),
    ("er_w40m045s50_D1_both", er_trend_signals, {"er_win": 40, "er_min": 0.45, "slow": 50}, "D1"),
    ("er_w20m045s50_D1_both", er_trend_signals, {"er_win": 20, "er_min": 0.45, "slow": 50}, "D1"),
]


def main() -> None:
    results = []
    for label, gen, params, tf in CONFIGS:
        pool = tl.build_pool(gen, params, tf=tf, side="both")
        st = tl.pool_stats(pool)
        st["label"] = label
        st["tf"] = tf
        st["params"] = json.dumps(params)
        results.append(st)
        print(json.dumps(st, ensure_ascii=False), flush=True)
    print("\n=== summary ===")
    for st in results:
        print(f"{st['label']:28s} n={st.get('n', 0):5d} pf={st.get('pool_pf')} "
              f"is={st.get('is_pf')} oos={st.get('oos_pf')} mean_bps={st.get('mean_bps')} "
              f"yearly={st.get('yearly_pos')} worst={st.get('worst_year')}")


if __name__ == "__main__":
    main()
