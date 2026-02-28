# Options Forecast & Backtest

Deterministic backtesting platform for equity strategies. Supports buy & hold, SMA crossover, and RSI mean-reversion over real underlying price data (via yfinance). FastAPI backend + vanilla JS frontend.

## Quick Start

```bash
# Create virtual environment and install dependencies
make setup

# Start the server (serves API + UI on http://localhost:8000)
make api

# Run tests
make test
```

Or manually:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/backtests/` | Run a backtest |

### POST /backtests/

```bash
# Buy & Hold
curl -X POST http://localhost:8000/api/v1/backtests/ \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "AAPL", "strategy": "buy_and_hold", "start_date": "2020-01-02", "end_date": "2020-12-31"}'

# SMA Crossover (custom params)
curl -X POST http://localhost:8000/api/v1/backtests/ \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "AAPL", "strategy": "sma_crossover", "start_date": "2020-01-02", "end_date": "2020-12-31", "params": {"fast_period": 10, "slow_period": 30}}'

# RSI Mean Reversion
curl -X POST http://localhost:8000/api/v1/backtests/ \
  -H 'Content-Type: application/json' \
  -d '{"symbol": "AAPL", "strategy": "rsi_mean_reversion", "start_date": "2020-01-02", "end_date": "2020-12-31"}'
```

Returns: summary metrics, equity curve, trade list, and effective params.

## Project Structure

```
api/          FastAPI backend
backtest/     Backtest engine (buy & hold, SMA crossover, RSI mean-reversion)
common/       Data provider, shared utilities
web/          Static HTML/JS/CSS UI
tests/        Unit and integration tests
docs/         Architecture docs, decisions, task tracking
```

## Supported Strategies

| Strategy | Params | Data Source | Status |
|----------|--------|-------------|--------|
| `buy_and_hold` | — | yfinance OHLCV | **Supported** |
| `sma_crossover` | `fast_period`, `slow_period` | yfinance OHLCV | **Supported** |
| `rsi_mean_reversion` | `rsi_period`, `oversold`, `overbought` | yfinance OHLCV | **Supported** |
| Options strategies | — | Historical options quotes | Not available (returns 409) |

## Docs

See [docs/00_INDEX.md](docs/00_INDEX.md) for full documentation index.

## License

See [LICENSE](LICENSE).
