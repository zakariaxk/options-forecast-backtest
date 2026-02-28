"""Forecast API router."""
from __future__ import annotations

from fastapi import APIRouter

from api.core.errors import ApiError
from api.core.logging import get_logger
from api.schemas.io import ForecastRequest, ForecastResponse
from forecast.engine import FORECAST_METHODS, run_forecast

router = APIRouter(prefix="/forecast", tags=["forecast"])
logger = get_logger("forecast")


@router.post("/", response_model=ForecastResponse)
def create_forecast(request: ForecastRequest) -> ForecastResponse:
    """Run a price forecast and return results inline."""

    if request.method not in FORECAST_METHODS:
        raise ApiError(
            error_code="UNKNOWN_METHOD",
            message=f"Unknown method '{request.method}'. Supported: {sorted(FORECAST_METHODS)}",
            status_code=422,
            details={"supported_methods": sorted(FORECAST_METHODS)},
        )

    try:
        result = run_forecast(
            symbol=request.symbol.upper(),
            horizon_days=request.horizon_days,
            method=request.method,
            lookback_days=request.lookback_days,
            params=request.params or None,
        )
    except ValueError as exc:
        raise ApiError(
            error_code="FORECAST_FAILED",
            message=str(exc),
            status_code=422,
        )

    logger.info(
        "forecast_run",
        symbol=result.symbol,
        method=result.method,
        horizon=result.horizon_days,
    )
    return ForecastResponse(**result.to_dict())
