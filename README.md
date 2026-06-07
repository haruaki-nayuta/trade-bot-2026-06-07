# fx-lab — FX 手法リサーチ&バックテスト環境

FX の手法を **考える(リサーチ)→ コード化 → バックテスト検証** まで一気通貫で AI に任せる環境。

- バックテスト: **vectorbt**(numba ベクトル化・並列・高速)
- データ: **Dukascopy** の 7 大メジャー M1(1分足)・過去 10 年・UTC
- 環境: **uv** + Python 3.12

## クイックスタート

```bash
# データ取得(初回のみ。10年×7ペア)
uv run python scripts/download_data.py

# 取得状況の確認
uv run python -c "from fxlab import summary; print(summary())"

# バックテスト
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1            # 単発
uv run python run_backtest.py ma_cross --pair EURUSD --tf H1 --sweep    # 並列パラメータ探索
uv run python run_backtest.py ma_cross --all-pairs --tf H1              # 7ペア横断
```

新しい手法は `strategies/_template.py` をコピーして作る。
**詳しい使い方・設計・評価指針は [CLAUDE.md](CLAUDE.md) を参照。**
