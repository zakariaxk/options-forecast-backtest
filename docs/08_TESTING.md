# Testing Strategy

## Test Categories

### Unit Tests (`tests/unit/`)
- `test_metrics.py` — Equity and trade metric computation
- `test_portfolio.py` — Portfolio state management
- `test_buy_and_hold.py` — Buy & hold strategy logic
- `test_api_shape.py` — API response shape validation
- `test_determinism.py` — Same inputs produce same outputs

### Integration Tests (`tests/integration/`)
- API endpoint integration tests using FastAPI TestClient
- No network calls — all data provided via fixtures

## Fixtures
- All test data is synthetic/deterministic and created in fixtures
- No yfinance calls in tests
- No file system side effects (use tmp_path)

## Running Tests
```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

## Coverage Expectations
- All backtest math functions
- All API endpoints (happy path + error cases)
- Determinism: run backtest twice → identical output
