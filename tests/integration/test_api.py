"""Integration tests for the API via FastAPI TestClient.

No network calls — backtest uses injected price fixtures.
"""
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def _mock_prices(n: int = 10) -> pd.DataFrame:
    """Deterministic price fixture matching n business days."""
    dates = pd.bdate_range("2020-01-02", periods=n)
    closes = [100.0 + i * 2.0 for i in range(n)]
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * n,
    })


def _sma_prices() -> pd.DataFrame:
    """120 bars: 60 down + 60 up — enough for SMA crossover with default params."""
    closes = []
    for i in range(60):
        closes.append(round(100.0 - i * 0.3, 4))
    for i in range(60):
        closes.append(round(closes[-1] + (i + 1) * 1.0, 4))
    dates = pd.bdate_range("2020-01-02", periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    })


def _volatile_prices() -> pd.DataFrame:
    """100 bars of oscillating prices for RSI testing."""
    closes = []
    price = 100.0
    for i in range(100):
        price *= 0.97 if (i // 8) % 2 == 0 else 1.04
        closes.append(round(price, 4))
    dates = pd.bdate_range("2020-01-02", periods=len(closes))
    return pd.DataFrame({
        "date": dates,
        "open": closes,
        "high": [c * 1.01 for c in closes],
        "low": [c * 0.99 for c in closes],
        "close": closes,
        "volume": [1_000_000] * len(closes),
    })


# ── Health ───────────────────────────────────────────────────

class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "time" in body


# ── POST /backtests/ — Buy & Hold ────────────────────────────

class TestCreateBacktest:
    def test_buy_and_hold_success(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_mock_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "buy_and_hold",
                "start_date": "2020-01-02",
                "end_date": "2020-01-15",
            })
        assert r.status_code == 200
        body = r.json()

        # Required fields present
        assert "bt_id" in body
        assert body["symbol"] == "AAPL"
        assert body["strategy"] == "buy_and_hold"
        assert body["initial_cash"] == 100_000.0

        # Summary metrics
        summary = body["summary"]
        for key in ["total_return", "sharpe", "max_drawdown", "cagr", "volatility", "sortino", "calmar"]:
            assert key in summary

        # Equity curve
        assert len(body["equity_curve"]) == 10
        assert "date" in body["equity_curve"][0]
        assert "nav" in body["equity_curve"][0]

        # Trades
        assert len(body["trades"]) >= 1
        assert body["trades"][0]["action"] == "BUY"

        # Params
        assert "params" in body

    def test_no_nan_in_response(self, client):
        """JSON must never contain NaN or Infinity."""
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_mock_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "buy_and_hold",
                "start_date": "2020-01-02",
                "end_date": "2020-01-15",
            })
        text = r.text
        assert "NaN" not in text
        assert "Infinity" not in text
        assert "-Infinity" not in text

    def test_deterministic_output(self, client):
        """Two identical requests produce identical results (minus bt_id)."""
        payload = {
            "symbol": "AAPL",
            "strategy": "buy_and_hold",
            "start_date": "2020-01-02",
            "end_date": "2020-01-15",
        }
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_mock_prices()):
            a = client.post("/api/v1/backtests/", json=payload).json()
            b = client.post("/api/v1/backtests/", json=payload).json()

        assert a["summary"] == b["summary"]
        assert a["equity_curve"] == b["equity_curve"]
        assert a["trades"] == b["trades"]

    def test_custom_initial_cash(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_mock_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "buy_and_hold",
                "start_date": "2020-01-02",
                "end_date": "2020-01-15",
                "initial_cash": 50_000,
            })
        assert r.status_code == 200
        assert r.json()["initial_cash"] == 50_000.0


# ── POST /backtests/ — SMA Crossover ────────────────────────

class TestSMACrossoverAPI:
    def test_sma_crossover_success(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_sma_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "sma_crossover",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "sma_crossover"
        assert len(body["equity_curve"]) == 120
        assert body["params"]["fast_period"] == 20
        assert body["params"]["slow_period"] == 50

    def test_sma_custom_params(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_sma_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "sma_crossover",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
                "params": {"fast_period": 5, "slow_period": 10},
            })
        assert r.status_code == 200
        body = r.json()
        assert body["params"]["fast_period"] == 5
        assert body["params"]["slow_period"] == 10

    def test_sma_no_nan(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_sma_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "sma_crossover",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
            })
        assert "NaN" not in r.text
        assert "Infinity" not in r.text


# ── POST /backtests/ — RSI Mean Reversion ────────────────────

class TestRSIMeanReversionAPI:
    def test_rsi_success(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_volatile_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "rsi_mean_reversion",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
            })
        assert r.status_code == 200
        body = r.json()
        assert body["strategy"] == "rsi_mean_reversion"
        assert len(body["equity_curve"]) == 100
        assert body["params"]["rsi_period"] == 14

    def test_rsi_custom_params(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_volatile_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "rsi_mean_reversion",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
                "params": {"rsi_period": 7, "oversold": 25, "overbought": 75},
            })
        assert r.status_code == 200
        body = r.json()
        assert body["params"]["rsi_period"] == 7
        assert body["params"]["oversold"] == 25
        assert body["params"]["overbought"] == 75

    def test_rsi_no_nan(self, client):
        with patch("backtest.underlying.fetch_underlying_prices", return_value=_volatile_prices()):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "rsi_mean_reversion",
                "start_date": "2020-01-02",
                "end_date": "2020-06-30",
            })
        assert "NaN" not in r.text
        assert "Infinity" not in r.text


# ── Error Cases ──────────────────────────────────────────────

class TestBacktestErrors:
    def test_options_strategy_returns_409(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "straddle",
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
        })
        assert r.status_code == 409
        body = r.json()
        assert body["error_code"] == "UNSUPPORTED_STRATEGY"
        assert "supported_strategies" in body["details"]

    def test_unknown_strategy_returns_422(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "magic_money",
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
        })
        assert r.status_code == 422

    def test_bad_date_range_returns_422(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "buy_and_hold",
            "start_date": "2020-12-31",
            "end_date": "2020-01-01",
        })
        assert r.status_code == 422

    def test_missing_required_fields_returns_422(self, client):
        r = client.post("/api/v1/backtests/", json={"symbol": "AAPL"})
        assert r.status_code == 422

    def test_negative_initial_cash_returns_422(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "buy_and_hold",
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
            "initial_cash": -100,
        })
        assert r.status_code == 422

    def test_credit_spread_returns_409(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "credit_spread",
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
        })
        assert r.status_code == 409

    def test_covered_call_returns_409(self, client):
        r = client.post("/api/v1/backtests/", json={
            "symbol": "AAPL",
            "strategy": "covered_call",
            "start_date": "2020-01-02",
            "end_date": "2020-12-31",
        })
        assert r.status_code == 409

    def test_sma_bad_params_returns_422(self, client):
        """SMA with too few bars → should return 422."""
        prices = _mock_prices(10)
        with patch("backtest.underlying.fetch_underlying_prices", return_value=prices):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "sma_crossover",
                "start_date": "2020-01-02",
                "end_date": "2020-01-15",
            })
        assert r.status_code == 422

    def test_rsi_bad_params_returns_422(self, client):
        """RSI with too few bars → should return 422."""
        prices = _mock_prices(5)
        with patch("backtest.underlying.fetch_underlying_prices", return_value=prices):
            r = client.post("/api/v1/backtests/", json={
                "symbol": "AAPL",
                "strategy": "rsi_mean_reversion",
                "start_date": "2020-01-02",
                "end_date": "2020-01-08",
            })
        assert r.status_code == 422
