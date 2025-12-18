from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from api.core.settings import get_settings
from backtest.engine import run_backtest
from common.schema import BacktestConfig, BacktestRequest, BacktestResponse


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d%H%M%S")


def submit_backtest(req: BacktestRequest) -> BacktestResponse:
    config = _resolve_config(req)
    settings = get_settings()
    run_id = f"bt_{_timestamp()}"
    result = run_backtest(config)
    base = Path(settings.backtests_uri) / config.symbol / run_id
    base.mkdir(parents=True, exist_ok=True)
    result.save(base.as_posix())
    meta_path = base / "config.json"
    meta_path.write_text(json.dumps(config.dict(), default=str, indent=2))
    return BacktestResponse(bt_id=run_id, status="done")


def get_backtest(bt_id: str, symbol: str) -> dict:
    settings = get_settings()
    base = Path(settings.backtests_uri) / symbol / bt_id
    metrics_path = base / "metrics.json"
    trades_path = base / "trades.parquet"
    equity_path = base / "equity.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Backtest {bt_id} not found for symbol {symbol}")
    metrics = json.loads(metrics_path.read_text())
    return {
        "bt_id": bt_id,
        "symbol": symbol,
        "metrics": metrics,
        "trades_uri": trades_path.as_posix(),
        "equity_uri": equity_path.as_posix(),
    }


def _resolve_config(req: BacktestRequest) -> BacktestConfig:
    if req.config:
        return req.config
    if req.config_yaml:
        payload = yaml.safe_load(req.config_yaml)
        return BacktestConfig(**payload)
    raise ValueError("Backtest configuration is required")


def list_backtest_runs() -> list[dict]:
    settings = get_settings()
    base = Path(settings.backtests_uri)
    items: list[dict] = []
    if not base.exists():
        return items
    for symbol_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for run_dir in sorted([p for p in symbol_dir.iterdir() if p.is_dir()]):
            items.append({"symbol": symbol_dir.name, "bt_id": run_dir.name})
    return items
