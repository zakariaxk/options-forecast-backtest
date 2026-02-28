# API Contracts

Base URL: `/api/v1`

---

## GET /health

**Response** `200 OK`
```json
{
  "status": "ok",
  "time": "2026-02-28T09:38:22.976620Z"
}
```

---

## POST /backtests/

Run a backtest over underlying price data.

**Request Body**
```json
{
  "symbol": "AAPL",
  "strategy": "sma_crossover",
  "start_date": "2020-01-02",
  "end_date": "2020-12-31",
  "initial_cash": 100000.0,
  "params": {"fast_period": 20, "slow_period": 50}
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| symbol | string | yes | — | Ticker symbol |
| strategy | string | no | buy_and_hold | `buy_and_hold`, `sma_crossover`, `rsi_mean_reversion` |
| start_date | date | yes | — | Backtest start date (YYYY-MM-DD) |
| end_date | date | yes | — | Backtest end date (YYYY-MM-DD) |
| initial_cash | float | no | 100000.0 | Starting portfolio cash (must be > 0) |
| params | object | no | {} | Strategy-specific parameters (see Backtest Engine docs) |

**Response** `200 OK`
```json
{
  "bt_id": "bt_20260228093829",
  "symbol": "AAPL",
  "strategy": "buy_and_hold",
  "start_date": "2020-01-02",
  "end_date": "2020-12-31",
  "initial_cash": 100000.0,
  "summary": {
    "total_return": 0.312,
    "cagr": 0.312,
    "sharpe": 1.23,
    "sortino": 1.85,
    "max_drawdown": -0.35,
    "volatility": 0.38,
    "calmar": 0.89
  },
  "equity_curve": [
    {"date": "2020-01-02", "nav": 100000.0},
    {"date": "2020-01-03", "nav": 100150.0}
  ],
  "trades": [
    {
      "date": "2020-01-02",
      "action": "BUY",
      "symbol": "AAPL",
      "qty": 1335,
      "price": 74.06,
      "value": 98870.10
    }
  ],
  "params": {"fast_period": 20, "slow_period": 50}
}
```

**Error Responses**:
- `422 Unprocessable Entity` — Invalid input (bad dates, unknown symbol)
- `409 Conflict` — Unsupported strategy (e.g., options strategy without options data)

**409 Response Shape**:
```json
{
  "error_code": "UNSUPPORTED_STRATEGY",
  "message": "Strategy 'straddle' requires historical options data which is not available. Use 'buy_and_hold' for underlying-only backtests.",
  "details": {
    "strategy": "straddle",
    "supported_strategies": ["buy_and_hold", "rsi_mean_reversion", "sma_crossover"]
  }
}
```


