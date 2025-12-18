# Options Forecast & Backtest Platform

A full-stack research environment for options trading that ingests market data, engineers predictive features, trains machine learning models, and validates strategies with an event-driven backtester. The project includes a FastAPI service for programmatic access and a Streamlit dashboard for interactive analysis.

---

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Environment Setup](#environment-setup)
5. [Data & ML Pipelines](#data--ml-pipelines)
6. [Backtesting](#backtesting)
7. [FastAPI Service](#fastapi-service)
8. [Streamlit Dashboard](#streamlit-dashboard)
9. [Testing & Tooling](#testing--tooling)
10. [Configuration](#configuration)
11. [Extending the Platform](#extending-the-platform)
12. [Known Limitations](#known-limitations)

---

## High-Level Architecture

```
Streamlit UI  ─────────┐
                       │    FastAPI ────> Services (ingest/train/predict/backtest)
External client ───────┘          │
                                   ├── S3 / local parquet storage (raw, processed, predictions, models)
                                   ├── MongoDB / Redis (optional integrations)
                                   └── MLflow / Airflow (integration points)
```

The build centers on deterministic pipelines (`pipelines/`), shared utilities (`common/`), machine learning modules (`ml/`), and a modular backtesting engine (`backtest/`). Artifacts are stored locally under `data/` by default but can be redirected to S3 using environment settings.

---

## Project Structure

```
options-forecast-backtest/
├── api/                # FastAPI app (routers, services, core helpers)
├── backtest/           # Engine, broker simulator, risk metrics, strategies
├── common/             # IO utilities, schema definitions, DB/cache helpers
├── dashboard/          # Streamlit app and UI components
├── web/                # Lightweight web UI (served by FastAPI)
├── ml/                 # Datasets, PyTorch/XGBoost utilities, training helpers
├── pipelines/          # CLI pipelines (ingest, features, train, predict)
├── tests/              # Unit tests
├── data/               # Generated artifacts (raw, processed, models, predictions, backtests)
├── Makefile            # Developer workflows
├── requirements.txt    # Python dependencies
└── README.md
```

Key modules referenced in this guide:
- Pipelines (ingestion/features/train/predict): `pipelines/*.py`
- Backtesting engine: `backtest/engine.py`
- Strategies: `backtest/strategies/*.py`
- API entrypoint: `api/main.py`
- Streamlit dashboard: `dashboard/app.py`

---

## Prerequisites

- macOS or Linux with Python 3.11 (recommended; 3.14 is not fully supported by all dependencies).
- [Homebrew](https://brew.sh/) (macOS) for system packages.
- Git (to clone the repository).
- Optional integrations:
  - MongoDB / Redis (if using metadata/caching services).
  - AWS CLI credentials (if targeting S3).

### Mandatory system dependencies

```bash
brew install libomp  # required by XGBoost on macOS
brew install cmake   # only if you plan to build from source; wheels cover most needs
```

---

## Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-repo>/options-forecast-backtest.git
   cd options-forecast-backtest
   ```

2. **Create and activate a Python 3.11 virtual environment**
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   make setup
   ```
   - If you see a `pyarrow` compile error, ensure you are using Python 3.11.
   - `make setup` installs the packages listed in `requirements.txt` and upgrades `pip`.

4. **Environment configuration**
   ```bash
   cp .env.example .env
   # Update values in .env as needed (e.g., S3 buckets, Mongo URI)
   ```

---

## Data & ML Pipelines

All pipelines are implemented as Python modules that can be executed via the Makefile or directly with `python -m`.

### 1. Ingest Market Data

Fetch Yahoo Finance equities and options data (with greeks) for a symbol/date range:

```bash
make ingest SYMBOL=AAPL START=2020-01-01 END=2020-12-31
```

Artifacts:
- `data/raw/AAPL/2020-01-01_2020-12-31/equity.parquet`
- `data/raw/AAPL/2020-01-01_2020-12-31/options.parquet`
- `metadata.json`

Code reference: `pipelines/ingest_yf.py`

### 2. Engineer Features & Targets

Join options with historical prices, compute technicals/greeks, and generate regression/classification targets. Fallback logic fills missing option targets with underlying returns.

```bash
make features \
  SYMBOL=AAPL \
  PARTITION="AAPL/2020-01-01_2020-12-31" \
  VERSION=v1
```

Artifacts:
- `data/processed/AAPL/v1/features.parquet`
- `schema.json`
- `scaler.pkl`

Code reference: `pipelines/features.py`

### 3. Train Machine Learning Models

Two baseline trainers are provided:
- `pipelines/train_xgb.py` – XGBoost regression/classification.
- `pipelines/train_torch.py` – LSTM sequence model (optional).

Example (XGBoost regression):
```bash
make train \
  SYMBOL=AAPL \
  MODEL=xgb_reg \
  RUN=demo \
  VERSION=v1
```

Artifacts:
- `data/models/xgb_reg/demo/model.json`
- `params.json`
- `metrics.json`

### 4. Generate Predictions

Run batch inference for the trained model:

```bash
make predict \
  SYMBOL=AAPL \
  MODEL=xgb_reg \
  RUN=demo \
  VERSION=v1
```

Artifacts:
- `data/predictions/AAPL/xgb_reg/demo/predictions.parquet`
- `metadata.json`

Code reference: `pipelines/predict.py`

---

## Backtesting

Once predictions exist, the backtesting engine can simulate strategies using configurable risk/execution settings.

```bash
make backtest \
  SYMBOL=AAPL \
  START=2020-01-01 \
  END=2020-12-31 \
  PREDICTIONS=data/predictions/AAPL/xgb_reg/demo/predictions.parquet
```

Artifacts:
- `data/backtests/AAPL/bt_<timestamp>/metrics.json`
- `.../trades.parquet`
- `.../equity.parquet`

Strategies implemented:
- `backtest/strategies/straddle.py`
- `.../credit_spread.py`
- `.../covered_call.py`

Engine and risk logic: `backtest/engine.py`, `backtest/broker.py`, `backtest/metrics.py`

Backtest configuration schema: `common/schema.py`

---

## FastAPI Service

The API wraps the pipelines/backtesting logic and exposes metadata.

### Launch
```bash
source .venv/bin/activate
make api           # default http://127.0.0.1:8000
make api PORT=8001 # custom port
```

### Built-in Web UI

When the API is running, a lightweight web UI is served at:
- `GET /` (e.g. `http://127.0.0.1:8000/`)

It can:
- list discovered models, prediction runs, and backtests
- trigger prediction generation (writes parquet under `data/predictions/`)
- run backtests (writes artifacts under `data/backtests/`)

### Key Endpoints
- `GET /api/v1/health` – service health check.
- `POST /api/v1/predictions` – generate predictions from a trained model run.
- `GET /api/v1/predictions/{symbol}/{run}` – summarize prediction runs.
- `GET /api/v1/predictions/{symbol}/{run}/data` – return prediction rows as JSON (table-friendly).
- `GET /api/v1/predictions/runs` – list discovered prediction runs.
- `POST /api/v1/backtests` – run backtests via JSON/YAML config.
- `GET /api/v1/backtests/{symbol}/{bt_id}` – fetch backtest metrics/artifacts.
- `GET /api/v1/backtests/{symbol}/{bt_id}/data` – return trades/equity as JSON (table-friendly).
- `GET /api/v1/backtests/runs` – list discovered backtest runs.
- `GET /api/v1/models` – list available models and runs.
- `GET /api/v1/models/{model}/latest` – retrieve latest metrics for a model.
- `POST /api/v1/pipelines/ingest` – ingest equity + option contract history from Yahoo Finance.
- `POST /api/v1/pipelines/features` – build features from a raw partition.
- `POST /api/v1/pipelines/train` – train XGBoost/LSTM models.

Route implementations: `api/routers/*.py`  
Service layer: `api/services/*.py`  
Settings & logging: `api/core/settings.py`, `api/core/logging.py`

---

## Streamlit Dashboard

Interactive UI over the generated artifacts.

### Launch

```bash
source .venv/bin/activate
make dashboard                      # defaults to port 8501
# Skip the optional onboarding prompt by pressing Enter.
```

### Pages
- **Overview** – Snapshot of models and available prediction runs.
- **Predictions** – Inspect scored options for a given symbol/model/run.
- **Backtests** – Visualize equity curves, drawdowns, trades, and metrics.
- **Reports** – Quick access to stored artifacts (parquet/JSON).

Streamlit code: `dashboard/app.py`  
Components: `dashboard/components/plots.py`, `dashboard/components/tables.py`

---

## Testing & Tooling

### Run Tests
```bash
make test   # executes pytest (unit tests under tests/unit/)
```

Illustrative tests:
- `tests/unit/test_metrics.py` – verifies portfolio statistics calculations.
- `tests/unit/test_portfolio.py` – ensures fills open/close positions correctly.

### Lint & Type Check
```bash
make lint   # runs ruff and mypy (warning-only by default)
```

Customize the linters by editing the target in `Makefile` or providing config files (`pyproject.toml`, `ruff.toml`, etc.).

---

## Configuration

Environment values are loaded via `api/core/settings.py` (Pydantic Settings). Default `.env.example` includes:

```
ENV=dev
MONGO_URI=mongodb://localhost:27017
MONGO_DB=options_forecast
REDIS_URL=redis://localhost:6379/0
MODELS_URI=data/models
PREDICTIONS_URI=data/predictions
BACKTESTS_URI=data/backtests
PROCESSED_URI=data/processed
RAW_URI=data/raw
S3_BUCKET=ofb
S3_ENDPOINT_URL=
AWS_REGION=us-east-1
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=change-me-please-32-characters
```

To integrate with S3, set `S3_BUCKET`, `S3_ENDPOINT_URL`, and AWS credentials (via environment variables or AWS profiles). The IO utilities (`common/io.py`) auto-detect the scheme and push/pull to S3 when URIs start with `s3://`.

---

## Extending the Platform

- **New Data Sources**: Implement alternative ingestion modules in `pipelines/` and reference them in the Makefile or API.
- **Feature Engineering**: Add transformations in `pipelines/features.py` or derive new scalers.
- **Models**: Extend `ml/models/` and add corresponding training scripts. Register new model types in `pipelines/train_*.py`.
- **Strategies**: Implement additional strategy modules under `backtest/strategies/` and register them in `backtest/engine.py`.
- **API**: Add routers/services to expose new capabilities (e.g., artifact downloads, experiment tracking).
- **Automation**: Integrate with Airflow/Prefect by scripting pipeline calls or reusing the Python modules as DAG tasks.

---

## Known Limitations

- **Online Data Dependence**: `yfinance` ingestion requires network access; rate limits or DNS issues may interrupt runs. Consider caching or alternative providers for production.
- **Historical Options Coverage**: Yahoo Finance does not provide a true “full historical options chain by date” feed; this project ingests a limited set of option contracts from the current chain and pulls each contract’s historical price series for the chosen date range, which works best for recent windows.
- **Naive Option Targets**: Some illiquid options lack future price observations; fallback to underlying returns mitigates missing labels but may reduce signal quality.
- **Baseline Strategies**: The packaged strategies are illustrative (straddles, credit spreads, covered calls). Tailor risk rules and sizing for real deployments.
- **No MLflow/Airflow integration out of the box**: Hooks exist (see `common/schema.py` and settings) but additional configuration is necessary.
- **Authentication/Authorization**: The FastAPI endpoints assume a trusted environment—implement proper auth for shared deployments.

---

## Getting Help

- **Artifacts missing from the dashboard**: Ensure the relevant `make predict` / `make backtest` commands completed and artifacts exist under `data/`.
- **Streamlit cannot import `api`**: The dashboard automatically adds the project root to `sys.path`, but if you restructure directories adjust `dashboard/app.py`.
- **XGBoost / pyarrow errors**: Confirm Python 3.11, reinstall `libomp`, and re-run `make setup`.

Feel free to open issues or submit pull requests to improve the pipelines, models, or strategy implementations. Happy researching!
