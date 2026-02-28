# Task List

## Phase 0 — Boot & Failure Map
- [x] Identify backend entrypoint (`api/main.py` → `uvicorn api.main:app`)
- [x] Identify web entrypoint (`web/index.html` served at `/`)
- [x] Attempt boot — server starts, `/health` returns 200
- [x] Audit `/backtest` — returns empty results (options-only engine, no underlying support)
- [x] Create missing doc stubs
- [x] Write `docs/notes/failure_map.md`
- [x] Write `docs/DECISIONS.md`

## Phase 1 — Golden Path MVP
- [x] Create `DataProvider` abstraction for fetching/caching underlying price data
- [x] Implement `buy_and_hold` strategy
- [x] Create new `run_underlying_backtest()` engine function
- [x] Update `BacktestRequest` schema (symbol, strategy, start_date, end_date, initial_cash)
- [x] Update `BacktestResponse` schema (summary, equity_curve, trades)
- [x] Update `POST /backtests/` endpoint to use new engine + return full results
- [x] Return 409 for options strategies
- [x] Sanitize JSON output (no NaN/Inf)
- [x] Wire UI backtest form to new API contract
- [x] Render equity curve chart in UI
- [x] Render summary metrics table in UI
- [x] Unit tests: buy & hold math
- [x] Unit tests: equity metrics
- [x] Unit tests: API response shape
- [x] Unit tests: determinism
- [x] Integration test: full backtest via TestClient
- [x] All tests pass (30/30)

## Phase 2 — Remove Hallucinations
- [x] Delete/disable options strategy endpoints
- [x] Remove prediction-dependent backtest path
- [x] Clean up unused dependencies (27 → 16)
- [x] Audit all endpoints for fake data
- [x] Mass-delete dead code: pipelines/, ml/, dashboard/, strategies/, old engine
- [x] Rebuild UI from scratch (no fake data or dead wiring)

## Phase 3 — Expand
- [x] Add SMA crossover strategy (with configurable fast/slow periods)
- [x] Add RSI mean-reversion strategy (with configurable period/thresholds)
- [x] Add optional `params` field to request/response schemas
- [x] Update UI with strategy dropdown + dynamic parameter inputs
- [x] Unit tests for new strategies + indicators (44 unit tests)
- [x] Integration tests for new strategy API endpoints (20 integration tests)
- [x] All tests pass (64/64)
- [x] E2E verified with real AAPL data
- [ ] Consider options data providers (deferred — no real historical options data source identified)
