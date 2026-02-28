"""Backtest API router."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from api.core.errors import ApiError
from api.core.logging import get_logger
from api.schemas.io import BacktestSubmitRequest, BacktestSubmitResponse
from backtest.underlying import UNDERLYING_STRATEGIES, run_underlying_backtest

router = APIRouter(prefix="/backtests", tags=["backtests"])
logger = get_logger("backtest")

# Options strategies — rejected with 409 until real data exists
_OPTIONS_STRATEGIES = {
    "straddle", "straddle_buy",
    "credit_spread", "credit_spread_sell",
    "covered_call",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


@router.post("/", response_model=BacktestSubmitResponse)
def create_backtest(request: BacktestSubmitRequest) -> BacktestSubmitResponse:
    """Run a backtest and return the full result inline."""

    # 409 — options strategy without options data
    if request.strategy in _OPTIONS_STRATEGIES:
        raise ApiError(
            error_code="UNSUPPORTED_STRATEGY",
            message=(
                f"Strategy '{request.strategy}' requires historical options data "
                f"which is not available. Use one of: {sorted(UNDERLYING_STRATEGIES)}"
            ),
            status_code=409,
            details={
                "strategy": request.strategy,
                "supported_strategies": sorted(UNDERLYING_STRATEGIES),
            },
        )

    # 422 — unknown strategy
    if request.strategy not in UNDERLYING_STRATEGIES:
        raise ApiError(
            error_code="UNKNOWN_STRATEGY",
            message=f"Unknown strategy '{request.strategy}'. Supported: {sorted(UNDERLYING_STRATEGIES)}",
            status_code=422,
            details={"supported_strategies": sorted(UNDERLYING_STRATEGIES)},
        )

    # 422 — bad date range
    if request.end_date <= request.start_date:
        raise ApiError(
            error_code="BAD_DATE_RANGE",
            message="end_date must be after start_date",
            status_code=422,
        )

    bt_id = f"bt_{_timestamp()}"

    try:
        result = run_underlying_backtest(
            symbol=request.symbol,
            strategy=request.strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_cash=request.initial_cash,
            bt_id=bt_id,
            params=request.params or None,
        )
    except ValueError as exc:
        raise ApiError(
            error_code="BACKTEST_FAILED",
            message=str(exc),
            status_code=422,
        )

    logger.info(
        "backtest_run",
        bt_id=result.bt_id,
        strategy=result.strategy,
        symbol=result.symbol,
    )
    return BacktestSubmitResponse(**result.to_dict())
