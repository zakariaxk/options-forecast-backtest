# Phase 0 — Failure Map

**Date**: 2026-02-28  
**Author**: Automated audit

---

## 1. Backend Entrypoint

| Item | Value |
|------|-------|
| Module | `api/main.py` |
| App factory | `create_app()` |
| ASGI object | `app = create_app()` |
| Run command | `uvicorn api.main:app --reload --port 8000` |
| Requires venv | Yes — `source .venv/bin/activate` |

**Boot result**: ✅ Server starts successfully. `/api/v1/health` returns 200.

---

## 2. Web Entrypoint

| Item | Value |
|------|-------|
| File | `web/index.html` |
| Served at | `/` (via FastAPI `FileResponse`) |
| Static files | `web/static/` mounted at `/static` |
| JS | `web/static/app.js` — vanilla JS, no build step |
| CSS | `web/static/app.css` |

**Wiring**: UI calls `/api/v1/*` endpoints. Health check works. Backtest form assumes prediction-based workflow.

---

## 3. Backtest Implementation Audit

### Current Flow
1. UI selects a prediction run → sends `POST /api/v1/backtests/` with `{config: {name, symbol, strategy, start_date, end_date, data: {predictions_uri}}}`
2. `api/services/backtester.py::submit_backtest()` resolves config → calls `backtest/engine.py::run_backtest()`
3. `run_backtest()` loads predictions parquet → filters by date range → runs options strategy → returns result
4. Response is `{bt_id, status}` — does NOT include metrics/equity/trades inline

### Failures

| # | Issue | Severity | Detail |
|---|-------|----------|--------|
| F1 | **No underlying-only strategy** | 🔴 Critical | Only options strategies (straddle, credit_spread, covered_call) exist. No buy & hold. |
| F2 | **Predictions data is model output, not historical quotes** | 🔴 Critical | Demo predictions parquet is ML model output, not real historical options bid/ask/mid. |
| F3 | **Date range mismatch → empty results** | 🔴 Critical | Demo data spans 2025-07-25 to 2025-10-27. Requesting 2020 dates returns empty metrics `{}`. No validation. |
| F4 | **No data provider abstraction** | 🟡 Medium | Data loading is hardcoded to predictions parquet path. No way to fetch underlying OHLCV. |
| F5 | **Response doesn't include results** | 🟡 Medium | `POST /backtests/` returns `{bt_id, status}` only. Requires separate GET + parquet reads. |
| F6 | **UI requires prediction run for backtest** | 🟡 Medium | Backtest form has a mandatory "Predictions Run" dropdown. Can't run underlying-only backtest. |
| F7 | **Heavy unused dependencies** | 🟢 Low | requirements.txt includes mongo, redis, celery, boto3, mlflow, dvc, torch. |
| F8 | **Pydantic V1 @validator deprecation** | 🟢 Low | Warning on import. Functional but should be updated. |

### Data Files Present

```
data/predictions/AAPL/xgb_reg/demo/predictions.parquet    — 924 rows, 16 cols
data/predictions/AAPL/xgb_reg/pred_demo/predictions.parquet
data/predictions/AAPL/xgb_reg/pred_demo_1766032567/predictions.parquet
data/predictions/AAPL/xgb_reg/pred_demo_1766032756/predictions.parquet
data/backtests/AAPL/bt_20251218043817/     — has config.json + metrics.json (empty)
data/backtests/AAPL/bt_20251218043929/     — has config.json + metrics.json (empty)
data/models/xgb_reg/demo/                  — model.json, params.json, metrics.json
data/raw/AAPL/*/                           — various date range partitions
data/processed/AAPL/v1/                    — processed features
```

### Existing Tests

| Test | Status |
|------|--------|
| `test_compute_equity_metrics_basic` | ✅ Pass |
| `test_compute_trade_metrics_empty` | ✅ Pass |
| `test_portfolio_can_open_and_close` | ✅ Pass |

3 tests pass. All test options-era code. No tests for underlying-only backtesting (doesn't exist yet).

---

## 4. Conclusion

The repo is a partially-working options backtesting platform that **cannot produce real results** because it lacks historical options quote data. The `/backtest` endpoint silently returns empty results instead of erroring. The MVP must be rebuilt around underlying-only strategies using real OHLCV data from yfinance.
