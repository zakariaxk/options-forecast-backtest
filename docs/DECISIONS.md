# Decisions

## 2026-02-28 — Phase 0: MVP Strategy Direction

**Context**: The existing backtest engine only supports options strategies (straddle, credit_spread, covered_call) which require a predictions DataFrame with historical options quote data. The only data source is yfinance, which provides current options chains but NOT historical options quotes.

**Decision**: Implement an underlying-only MVP first. The `buy_and_hold` strategy over daily OHLCV data from yfinance is the golden path. Options strategies will return `409 Conflict` until a real historical options data provider is integrated.

**Rationale**: 
- Honest results over fake results
- yfinance daily OHLCV is real, deterministic (when cached), and free
- Going to market with fake options backtesting would undermine credibility
- Buy & hold is a well-understood benchmark strategy

---

## 2026-02-28 — Phase 0: Doc stubs created

**Context**: Required docs (`00_INDEX.md` through `10_TASKLIST.md`) did not exist.

**Decision**: Created all stubs per the read order checklist in `AGENT_RULES.md`.

**Files created**:
- `docs/00_INDEX.md`
- `docs/01_ARCHITECTURE.md` (with Phase 0 summary)
- `docs/04_DATA_REALITY.md`
- `docs/05_API_CONTRACTS.md`
- `docs/06_BACKTEST_ENGINE.md`
- `docs/07_UI_WIRING.md`
- `docs/08_TESTING.md`
- `docs/09_DEPLOYMENT.md`
- `docs/10_TASKLIST.md`
- `docs/DECISIONS.md` (this file)
- `docs/notes/failure_map.md`

---

## 2026-02-28 — API contract: flat response for POST /backtests/

**Context**: The existing API returns only `{bt_id, status}` from POST /backtests/ and requires separate GET calls for metrics/trades/equity. This creates unnecessary round-trips and the backtest data is already computed.

**Decision**: Change `POST /backtests/` to return the full result inline: `{bt_id, symbol, strategy, start_date, end_date, initial_cash, summary, equity_curve, trades}`. This simplifies the UI wiring and makes the API more useful.

**Tradeoff**: Larger response payloads, but for an MVP with daily data the equity curve is at most ~252 points/year, which is well under 1MB.

---

## 2026-02-28 — Full rebuild: delete dead code, slim dependencies

**Context**: After Phase 0 audit and initial Phase 1 implementation, the codebase still contained massive amounts of dead code: ML pipelines (`pipelines/`, `ml/`), a Streamlit dashboard (`dashboard/`), options strategies (`backtest/strategies/`), the old options-only engine (`backtest/engine.py`, `backtest/broker.py`), unused API routers (predict, models, pipelines), services layer, and 27 dependencies including torch, mlflow, celery, redis, and mongo.

**Decision**: Mass-delete all non-functional code and rebuild from the ground up:
- Deleted: `pipelines/`, `ml/`, `dashboard/`, `backtest/strategies/`, `backtest/engine.py`, `backtest/broker.py`, `backtest/metrics.py`, `api/services/`, `api/routers/{predict,models,pipelines}.py`, `common/{cache,db,time,io,schema}.py`, `data/` (fake data), old tests, `designdoc.txt`
- Slimmed `requirements.txt` from 27 → 16 dependencies
- Rewrote: `api/main.py`, `api/routers/backtest.py`, `api/schemas/io.py`, `api/core/settings.py`
- Rebuilt from scratch: `web/index.html`, `web/static/app.js`, `web/static/app.css`, `Makefile`, `README.md`
- Wrote 30 tests (16 unit + 14 integration), all passing

**Rationale**: Dead code creates confusion, false confidence, and maintenance burden. A clean 10-file codebase that works is better than a 40-file codebase where 30 files are broken.

---

## 2026-02-28 — Phase 3: Add SMA Crossover & RSI Mean Reversion strategies

**Context**: Phase 1 MVP had only `buy_and_hold`. Phase 3 calls for additional underlying strategies.

**Decision**: Added two strategies:
1. **SMA Crossover** (`sma_crossover`) — buys on golden cross (fast SMA > slow SMA), sells on death cross. Configurable `fast_period` (default 20) and `slow_period` (default 50).
2. **RSI Mean Reversion** (`rsi_mean_reversion`) — buys when RSI drops below oversold, sells when RSI rises above overbought. Configurable `rsi_period` (default 14), `oversold` (default 30), `overbought` (default 70).

**API Change**: Added optional `params` dict to both request and response schemas. Strategy-specific parameters are passed in `params`; missing keys fall back to per-strategy defaults.

**E2E Verified** with real AAPL 2020 data:
- SMA crossover: 72.6% return, 3 trades (golden cross in April, death cross in October, re-entry in November)
- RSI mean reversion: 21.5% return, 2 trades (buy on COVID crash RSI<30, sell on recovery RSI>70)
- Buy & hold: 78.2% return (benchmark)

**Tests**: 64 total (44 unit + 20 integration), all passing.
