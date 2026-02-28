"""Pydantic schemas for API request / response validation."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    time: datetime


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
