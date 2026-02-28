# UI Wiring

## Architecture

- `web/index.html` — Single-page app served by FastAPI at `/`
- `web/static/app.js` — Vanilla JS, calls `/api/v1/*` endpoints
- `web/static/app.css` — Styles
- No build step required

## API Base URL

Configured as `const API = "/api/v1"` in `app.js`. Since the UI is served by FastAPI, same-origin requests work without CORS issues.

## MVP Wiring

### Health Check
- On page load, `GET /api/v1/health` is called
- Status pill shows green/red based on response

### Backtest Form
- User enters: symbol, strategy, start_date, end_date
- Form submits `POST /api/v1/backtests/` with JSON body
- Response renders: summary metrics table + equity curve chart
- No prediction run selection needed for underlying-only strategies

### Error Display
- API errors shown in result panel as JSON
- 409 errors for unsupported strategies shown clearly
