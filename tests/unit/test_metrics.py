import pandas as pd

from backtest import metrics


def test_compute_equity_metrics_basic():
    equity = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=5, freq="D"), "nav": [100, 101, 102, 101, 103]})
    result = metrics.compute_equity_metrics(equity)
    assert "sharpe" in result
    assert result["max_drawdown"] <= 0


def test_compute_trade_metrics_empty():
    empty = pd.DataFrame()
    result = metrics.compute_trade_metrics(empty)
    assert result == {}
