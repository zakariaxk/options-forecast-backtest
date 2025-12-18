from __future__ import annotations

from fastapi import APIRouter, Query

from api.core.logging import get_logger
from api.core.errors import not_found
from api.services.backtester import get_backtest, list_backtest_runs, submit_backtest
from common.schema import BacktestRequest, BacktestResponse

router = APIRouter(prefix="/backtests", tags=["backtests"])
logger = get_logger("backtest")


@router.post("/", response_model=BacktestResponse)
def create_backtest(request: BacktestRequest) -> BacktestResponse:
    response = submit_backtest(request)
    logger.info("backtest_run", bt_id=response.bt_id, status=response.status)
    return response


@router.get("/{symbol}/{bt_id}")
def get_backtest_result(symbol: str, bt_id: str) -> dict:
    try:
        return get_backtest(bt_id=bt_id, symbol=symbol)
    except FileNotFoundError as exc:
        raise not_found(str(exc))


@router.get("/{symbol}/{bt_id}/data")
def get_backtest_data(
    symbol: str,
    bt_id: str,
    limit_trades: int = Query(default=500, ge=0, le=5000),
    limit_equity: int = Query(default=2000, ge=0, le=20000),
) -> dict:
    try:
        result = get_backtest(bt_id=bt_id, symbol=symbol)
    except FileNotFoundError as exc:
        raise not_found(str(exc))
    import pandas as pd

    trades_path = result["trades_uri"]
    equity_path = result["equity_uri"]
    trades_df = pd.read_parquet(trades_path) if limit_trades != 0 else pd.DataFrame()
    equity_df = pd.read_parquet(equity_path) if limit_equity != 0 else pd.DataFrame()
    if limit_trades:
        trades_df = trades_df.tail(limit_trades)
    if limit_equity:
        equity_df = equity_df.tail(limit_equity)
    return {
        "bt_id": bt_id,
        "symbol": symbol,
        "metrics": result["metrics"],
        "trades": trades_df.to_dict(orient="records"),
        "equity": equity_df.to_dict(orient="records"),
    }


@router.get("/runs")
def list_runs() -> list[dict]:
    return list_backtest_runs()
