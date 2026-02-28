"""Forecasting engine — Holt double exponential smoothing + volatility bands.

Honest, simple, interpretable.  No pretending to be an ML model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.data_provider import fetch_underlying_prices


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    symbol: str
    method: str
    horizon_days: int
    training_start: str
    training_end: str
    last_close: float
    forecast: List[Dict[str, Any]]          # [{date, price, lower, upper}]
    historical_tail: List[Dict[str, Any]]   # last N days for chart context
    diagnostics: Dict[str, float]           # MAE, RMSE on in-sample fit
    params_used: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "method": self.method,
            "horizon_days": self.horizon_days,
            "training_start": self.training_start,
            "training_end": self.training_end,
            "last_close": self.last_close,
            "forecast": self.forecast,
            "historical_tail": self.historical_tail,
            "diagnostics": self.diagnostics,
            "params_used": self.params_used,
        }


# ---------------------------------------------------------------------------
# Holt's double exponential smoothing
# ---------------------------------------------------------------------------

def _holt_smooth(
    values: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Holt's linear trend method.

    Returns (level, trend) arrays same length as input.
    """
    n = len(values)
    level = np.zeros(n)
    trend = np.zeros(n)

    # Initialise: level = first value, trend = average of first few diffs
    level[0] = values[0]
    init_len = min(5, n - 1)
    if init_len > 0:
        trend[0] = float(np.mean(np.diff(values[: init_len + 1])))
    else:
        trend[0] = 0.0

    for t in range(1, n):
        level[t] = alpha * values[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]

    return level, trend


def _compute_residuals(values: np.ndarray, level: np.ndarray, trend: np.ndarray) -> np.ndarray:
    """One-step-ahead residuals for fitted Holt model."""
    n = len(values)
    residuals = np.zeros(n)
    for t in range(1, n):
        predicted = level[t - 1] + trend[t - 1]
        residuals[t] = values[t] - predicted
    return residuals[1:]  # skip first (no prediction)


def _grid_search_holt(values: np.ndarray) -> tuple[float, float]:
    """Find best (alpha, beta) by minimising RMSE on one-step-ahead forecasts."""
    best_rmse = float("inf")
    best_ab = (0.3, 0.1)

    for alpha in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        for beta in [0.01, 0.05, 0.1, 0.2, 0.3]:
            level, trend = _holt_smooth(values, alpha, beta)
            resid = _compute_residuals(values, level, trend)
            rmse = float(np.sqrt(np.mean(resid**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_ab = (alpha, beta)

    return best_ab


# ---------------------------------------------------------------------------
# Business-day date generation
# ---------------------------------------------------------------------------

def _next_business_days(start: date, n: int) -> List[date]:
    """Generate next n business days starting from start (exclusive)."""
    result = []
    current = start
    while len(result) < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            result.append(current)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

FORECAST_METHODS = {"holt", "drift"}

DEFAULT_FORECAST_PARAMS = {
    "holt": {"alpha": None, "beta": None},  # None = auto-tune
    "drift": {},
}


def run_forecast(
    *,
    symbol: str,
    horizon_days: int = 30,
    method: str = "holt",
    lookback_days: int = 252,
    prices: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
) -> ForecastResult:
    """Run a price forecast for *symbol*.

    Parameters
    ----------
    symbol : ticker symbol
    horizon_days : number of trading days to forecast (1-120)
    method : "holt" (double exponential smoothing) or "drift" (random walk with drift)
    lookback_days : trading days of history to use for fitting
    prices : inject prices directly (for testing)
    params : override smoothing parameters

    Returns
    -------
    ForecastResult with forecast prices + confidence bands.
    """
    if method not in FORECAST_METHODS:
        raise ValueError(f"Unknown forecast method: {method}. Use one of: {sorted(FORECAST_METHODS)}")
    if horizon_days < 1 or horizon_days > 120:
        raise ValueError(f"horizon_days must be 1-120, got {horizon_days}")
    if lookback_days < 30:
        raise ValueError(f"lookback_days must be >= 30, got {lookback_days}")

    # Fetch data
    if prices is None:
        end = date.today()
        start = end - timedelta(days=int(lookback_days * 1.6))  # overshoot to cover weekends
        prices = fetch_underlying_prices(symbol, start, end)

    if len(prices) < 30:
        raise ValueError(f"Not enough price data for forecast: got {len(prices)} bars, need >= 30")

    # Trim to lookback window
    prices = prices.tail(lookback_days).reset_index(drop=True)
    closes = prices["close"].astype(float).values
    dates = prices["date"].values

    user_params = params or {}

    if method == "holt":
        result = _forecast_holt(symbol, closes, dates, horizon_days, user_params)
    else:
        result = _forecast_drift(symbol, closes, dates, horizon_days)

    return result


def _forecast_holt(
    symbol: str,
    closes: np.ndarray,
    dates: np.ndarray,
    horizon: int,
    params: Dict[str, Any],
) -> ForecastResult:
    """Holt's double exponential smoothing forecast."""
    alpha = params.get("alpha")
    beta = params.get("beta")

    if alpha is None or beta is None:
        auto_alpha, auto_beta = _grid_search_holt(closes)
        alpha = alpha or auto_alpha
        beta = beta or auto_beta

    level, trend = _holt_smooth(closes, alpha, beta)
    residuals = _compute_residuals(closes, level, trend)
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

    # In-sample diagnostics
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # Forecast
    last_level = level[-1]
    last_trend = trend[-1]
    last_date = pd.Timestamp(dates[-1]).date()
    last_close = float(closes[-1])

    forecast_dates = _next_business_days(last_date, horizon)
    forecast_points = []
    for i, d in enumerate(forecast_dates, start=1):
        predicted = last_level + i * last_trend
        # Confidence interval widens with sqrt(h)
        margin = 1.96 * residual_std * math.sqrt(i)
        forecast_points.append({
            "date": d.isoformat(),
            "price": round(max(predicted, 0.01), 2),
            "lower": round(max(predicted - margin, 0.01), 2),
            "upper": round(predicted + margin, 2),
        })

    # Historical tail for chart
    tail_n = min(60, len(closes))
    historical_tail = [
        {"date": pd.Timestamp(dates[-tail_n + i]).date().isoformat(), "price": round(float(closes[-tail_n + i]), 2)}
        for i in range(tail_n)
    ]

    return ForecastResult(
        symbol=symbol,
        method="holt",
        horizon_days=horizon,
        training_start=pd.Timestamp(dates[0]).date().isoformat(),
        training_end=last_date.isoformat(),
        last_close=round(last_close, 2),
        forecast=forecast_points,
        historical_tail=historical_tail,
        diagnostics={"mae": round(mae, 4), "rmse": round(rmse, 4)},
        params_used={"alpha": round(alpha, 4), "beta": round(beta, 4)},
    )


def _forecast_drift(
    symbol: str,
    closes: np.ndarray,
    dates: np.ndarray,
    horizon: int,
) -> ForecastResult:
    """Random walk with drift (naive but honest baseline)."""
    n = len(closes)
    drift = (closes[-1] - closes[0]) / (n - 1)
    residuals = np.diff(closes) - drift
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 1.0

    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    last_close = float(closes[-1])
    last_date = pd.Timestamp(dates[-1]).date()

    forecast_dates = _next_business_days(last_date, horizon)
    forecast_points = []
    for i, d in enumerate(forecast_dates, start=1):
        predicted = last_close + i * drift
        margin = 1.96 * residual_std * math.sqrt(i)
        forecast_points.append({
            "date": d.isoformat(),
            "price": round(max(predicted, 0.01), 2),
            "lower": round(max(predicted - margin, 0.01), 2),
            "upper": round(predicted + margin, 2),
        })

    tail_n = min(60, len(closes))
    historical_tail = [
        {"date": pd.Timestamp(dates[-tail_n + i]).date().isoformat(), "price": round(float(closes[-tail_n + i]), 2)}
        for i in range(tail_n)
    ]

    return ForecastResult(
        symbol=symbol,
        method="drift",
        horizon_days=horizon,
        training_start=pd.Timestamp(dates[0]).date().isoformat(),
        training_end=last_date.isoformat(),
        last_close=round(last_close, 2),
        forecast=forecast_points,
        historical_tail=historical_tail,
        diagnostics={"mae": round(mae, 4), "rmse": round(rmse, 4)},
        params_used={"drift_per_day": round(drift, 4)},
    )
