"""Unit tests for forecast.engine."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from forecast.engine import (
    ForecastResult,
    _compute_residuals,
    _grid_search_holt,
    _holt_smooth,
    _next_business_days,
    run_forecast,
)


# ── Helpers ──────────────────────────────────────────────────

def _make_prices(n: int = 100, start_price: float = 100.0, trend: float = 0.1) -> pd.DataFrame:
    """Generate synthetic daily prices with trend + noise."""
    rng = np.random.RandomState(42)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + trend + rng.normal(0, 0.5))
    dates = pd.bdate_range(start="2024-01-02", periods=n)
    return pd.DataFrame({"date": dates, "open": closes, "high": closes, "low": closes, "close": closes, "volume": [1_000_000] * n})


# ── _holt_smooth ─────────────────────────────────────────────

class TestHoltSmooth:
    def test_output_shapes(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        level, trend = _holt_smooth(values, 0.3, 0.1)
        assert len(level) == 5
        assert len(trend) == 5

    def test_constant_series(self):
        values = np.full(20, 50.0)
        level, trend = _holt_smooth(values, 0.3, 0.1)
        # trend should converge toward 0
        assert abs(trend[-1]) < 0.5

    def test_linear_series(self):
        values = np.arange(1.0, 51.0)
        level, trend = _holt_smooth(values, 0.5, 0.5)
        # trend should be close to 1
        assert abs(trend[-1] - 1.0) < 0.3

    def test_single_value(self):
        values = np.array([42.0])
        level, trend = _holt_smooth(values, 0.3, 0.1)
        assert level[0] == 42.0


# ── _compute_residuals ───────────────────────────────────────

class TestComputeResiduals:
    def test_perfect_fit_constant(self):
        values = np.full(30, 10.0)
        level, trend = _holt_smooth(values, 0.9, 0.1)
        resid = _compute_residuals(values, level, trend)
        assert len(resid) == 29
        # Residuals should be very small for a constant series
        assert np.max(np.abs(resid)) < 2.0

    def test_residual_length(self):
        values = np.arange(1.0, 51.0)
        level, trend = _holt_smooth(values, 0.3, 0.1)
        resid = _compute_residuals(values, level, trend)
        assert len(resid) == 49  # n-1


# ── _grid_search_holt ────────────────────────────────────────

class TestGridSearchHolt:
    def test_returns_tuple(self):
        values = np.arange(1.0, 101.0) + np.random.RandomState(0).normal(0, 0.5, 100)
        alpha, beta = _grid_search_holt(values)
        assert 0 < alpha <= 1
        assert 0 < beta <= 1

    def test_deterministic(self):
        values = np.arange(1.0, 101.0) + np.random.RandomState(0).normal(0, 0.5, 100)
        a1, b1 = _grid_search_holt(values)
        a2, b2 = _grid_search_holt(values)
        assert a1 == a2
        assert b1 == b2


# ── _next_business_days ──────────────────────────────────────

class TestNextBusinessDays:
    def test_basic(self):
        d = date(2024, 1, 5)  # Friday
        result = _next_business_days(d, 3)
        assert len(result) == 3
        assert result[0] == date(2024, 1, 8)  # Monday
        assert result[1] == date(2024, 1, 9)  # Tuesday
        assert result[2] == date(2024, 1, 10)  # Wednesday

    def test_skips_weekends(self):
        d = date(2024, 1, 5)  # Friday
        result = _next_business_days(d, 5)
        for r in result:
            assert r.weekday() < 5

    def test_empty(self):
        result = _next_business_days(date(2024, 1, 1), 0)
        assert result == []


# ── run_forecast ─────────────────────────────────────────────

class TestRunForecast:
    def test_holt_basic(self):
        prices = _make_prices(100)
        result = run_forecast(
            symbol="TEST",
            horizon_days=10,
            method="holt",
            prices=prices,
        )
        assert isinstance(result, ForecastResult)
        assert result.symbol == "TEST"
        assert result.method == "holt"
        assert len(result.forecast) == 10
        assert len(result.historical_tail) > 0
        assert "mae" in result.diagnostics
        assert "rmse" in result.diagnostics

    def test_drift_basic(self):
        prices = _make_prices(100)
        result = run_forecast(
            symbol="TEST",
            horizon_days=5,
            method="drift",
            prices=prices,
        )
        assert result.method == "drift"
        assert len(result.forecast) == 5
        assert "drift_per_day" in result.params_used

    def test_forecast_has_confidence_bands(self):
        prices = _make_prices(100)
        result = run_forecast(symbol="TEST", horizon_days=10, prices=prices)
        for pt in result.forecast:
            assert "date" in pt
            assert "price" in pt
            assert "lower" in pt
            assert "upper" in pt
            assert pt["lower"] <= pt["price"] <= pt["upper"]

    def test_bands_widen_with_horizon(self):
        prices = _make_prices(200)
        result = run_forecast(symbol="TEST", horizon_days=20, prices=prices)
        fc = result.forecast
        # width should generally increase
        first_width = fc[0]["upper"] - fc[0]["lower"]
        last_width = fc[-1]["upper"] - fc[-1]["lower"]
        assert last_width > first_width

    def test_to_dict_roundtrip(self):
        prices = _make_prices(100)
        result = run_forecast(symbol="TEST", horizon_days=5, prices=prices)
        d = result.to_dict()
        assert d["symbol"] == "TEST"
        assert len(d["forecast"]) == 5
        assert isinstance(d["diagnostics"], dict)

    def test_unknown_method_raises(self):
        prices = _make_prices(100)
        with pytest.raises(ValueError, match="Unknown forecast method"):
            run_forecast(symbol="TEST", method="arima", prices=prices)

    def test_horizon_too_large_raises(self):
        prices = _make_prices(100)
        with pytest.raises(ValueError, match="horizon_days must be 1-120"):
            run_forecast(symbol="TEST", horizon_days=200, prices=prices)

    def test_too_few_bars_raises(self):
        prices = _make_prices(10)
        with pytest.raises(ValueError, match="Not enough"):
            run_forecast(symbol="TEST", prices=prices)

    def test_no_nan_in_forecast(self):
        prices = _make_prices(100)
        result = run_forecast(symbol="TEST", horizon_days=30, prices=prices)
        for pt in result.forecast:
            assert not math.isnan(pt["price"])
            assert not math.isnan(pt["lower"])
            assert not math.isnan(pt["upper"])

    def test_with_custom_params(self):
        prices = _make_prices(100)
        result = run_forecast(
            symbol="TEST",
            method="holt",
            horizon_days=5,
            prices=prices,
            params={"alpha": 0.5, "beta": 0.2},
        )
        assert result.params_used["alpha"] == 0.5
        assert result.params_used["beta"] == 0.2

    def test_deterministic(self):
        prices = _make_prices(100)
        r1 = run_forecast(symbol="TEST", horizon_days=10, prices=prices)
        r2 = run_forecast(symbol="TEST", horizon_days=10, prices=prices)
        for p1, p2 in zip(r1.forecast, r2.forecast):
            assert p1["price"] == p2["price"]
