# Boot Errors

## 2026-02-28 — Initial Boot Attempt

### Attempt 1: `python -c "import api.main"`
```
ModuleNotFoundError: No module named 'fastapi'
```
**Cause**: Ran outside venv. System python3 (3.14) doesn't have dependencies.  
**Fix**: `source .venv/bin/activate` first. Venv uses Python 3.11.3.

### Attempt 2: With venv activated
```
import ok
```
**Result**: ✅ Clean import.

### Attempt 3: `uvicorn api.main:app --host 127.0.0.1 --port 8000`
**Result**: ✅ Server starts. `/health` returns 200.

### Attempt 4: `POST /backtests/` with 2020 date range
**Result**: Returns `{"bt_id": "bt_...", "status": "done"}` with empty metrics `{}`, empty equity curve, empty trades.  
**Cause**: Date filter against 2025 prediction data produces zero rows → empty backtest.

### Warning on test run
```
PydanticDeprecatedSince20: Pydantic V1 style `@validator` validators are deprecated
```
**Source**: `common/schema.py:22`  
**Impact**: Non-breaking warning. Should migrate to `@field_validator`.
