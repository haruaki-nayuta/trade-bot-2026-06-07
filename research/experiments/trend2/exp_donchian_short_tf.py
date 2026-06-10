"""exp_donchian_short_tf — ドンチャン・ブレイクアウト(close ベース)を短い時間足で検証。

固定グリッド(追い込みなし):
  H1 : (entry, exit) in {(55,20), (100,50), (200,50)}
  M30: (entry, exit) in {(55,20), (100,50), (200,50)}
  M15: (entry, exit) in {(100,50), (200,50), (400,100)}
計 9 構成 / side=both / 対象 = FX19(XAUUSD 除く)。

各構成について net(通常コスト)と gross(コスト0)のプールを構築し、
「シグナル自体にエッジがあるがコストで死ぬのか、シグナル自体が逆向きなのか」を切り分ける。

先読み禁止: rolling max/min は .shift(1) で直前バーまでの確定極値のみ使用。
close ベース(クロス12は close 複製 OHLC のため high/low は使わない)。

実行: PYTHONPATH=. uv run python research/experiments/trend2/exp_donchian_short_tf.py
"""

import sys

sys.path.insert(0, "/Users/yutootsuka/Documents/economy/.claude/worktrees/friendly-meitner-8d533c/research/lab")

import pandas as pd  # noqa: E402

import trend_lab as tl  # noqa: E402

OUT = tl.ROOT / "research" / "outputs" / "trend2_donchian_short_tf.csv"

GRID = [
    ("H1", 55, 20),
    ("H1", 100, 50),
    ("H1", 200, 50),
    ("M30", 55, 20),
    ("M30", 100, 50),
    ("M30", 200, 50),
    ("M15", 100, 50),
    ("M15", 200, 50),
    ("M15", 400, 100),
]

FX19 = [i for i in tl.default_instruments() if i != "XAUUSD"]


def donchian_close(data: pd.DataFrame, entry: int = 55, exit: int = 20):
    """close ベースのドンチャン・ブレイクアウト(両建て)。

    直前バーまでの rolling 極値(.shift(1))を確定値として使う = 先読みなし。
    """
    close = data["close"]
    upper = close.rolling(entry).max().shift(1)
    lower = close.rolling(entry).min().shift(1)
    exit_upper = close.rolling(exit).max().shift(1)
    exit_lower = close.rolling(exit).min().shift(1)

    long_entries = close > upper
    long_exits = close < exit_lower
    short_entries = close < lower
    short_exits = close > exit_upper
    return long_entries, long_exits, short_entries, short_exits


def run_grid(mode: str) -> list[dict]:
    rows = []
    for tf, e, x in GRID:
        pool = tl.build_pool(donchian_close, {"entry": e, "exit": x},
                             tf=tf, side="both", instruments=FX19)
        st = tl.pool_stats(pool)
        st.update({"mode": mode, "tf": tf, "entry": e, "exit": x})
        rows.append(st)
        print(f"[{mode}] {tf} ({e},{x}): n={st.get('n')} sum={st.get('sum_ret')} "
              f"pf={st.get('pool_pf')} is/oos={st.get('is_pf')}/{st.get('oos_pf')} "
              f"mean={st.get('mean_bps')}bps yearly={st.get('yearly_pos')}", flush=True)
    return rows


def main() -> None:
    print(f"instruments ({len(FX19)}): {FX19}", flush=True)

    # --- net(通常コスト) ---
    rows = run_grid("net")

    # --- gross(コスト0)診断 ---
    tl.register_spreads()  # キーを揃えてから全部 0 に
    from fxlab import config
    for k in list(config.SPREADS_PIPS):
        config.SPREADS_PIPS[k] = 0.0
    _orig = tl.register_spreads
    tl.register_spreads = lambda: None  # build_pool 内のリセット無効化
    try:
        rows += run_grid("gross")
    finally:
        tl.register_spreads = _orig

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print("\nsaved:", OUT)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
