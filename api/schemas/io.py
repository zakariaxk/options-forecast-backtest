"""Pydantic schemas for API request / response validation."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    time: datetime


# ── Backtest ──────────────────────────────────────────────────

class BacktestSubmitRequest(BaseModel):
    """POST /backtests/ request body."""
    symbol: str
    strategy: str = "buy_and_hold"
    start_date: date
    end_date: date
    initial_cash: float = Field(default=100_000.0, gt=0)
    params: Dict[str, Any] = Field(default_factory=dict)


class BacktestSubmitResponse(BaseModel):
    """Full inline backtest result."""
    bt_id: str
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_cash: float
    summary: Dict[str, float]
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    params: Dict[str, Any] = Field(default_factory=dict)


# ── Forecast ──────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    """POST /forecast/ request body."""
    symbol: str
    method: str = "holt"
    horizon_days: int = Field(default=30, ge=1, le=120)
    lookback_days: int = Field(default=252, ge=30)
    params: Dict[str, Any] = Field(default_factory=dict)


class ForecastResponse(BaseModel):
    symbol: str
    method: str
    horizon_days: int
    training_start: str
    training_end: str
    last_close: float
    forecast: List[Dict[str, Any]]
    historical_tail: List[Dict[str, Any]]
    diagnostics: Dict[str, float]
    params_used: Dict[str, Any] = Field(default_factory=dict)


# ── Options ───────────────────────────────────────────────────

class OptionsChainRequest(BaseModel):
    """POST /options/chain request body."""
    symbol: str
    expiry: Optional[str] = None  # YYYY-MM-DD or None for nearest


class OptionsChainResponse(BaseModel):
    symbol: str
    expiry: str
    underlying_price: float
    calls: List[Dict[str, Any]]
    puts: List[Dict[str, Any]]
    available_expiries: List[str]
