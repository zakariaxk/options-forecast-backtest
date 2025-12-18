.PHONY: setup lint test ingest features train predict backtest api dashboard

PYTHON ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check . || true
	mypy || true

test:
	pytest -q

ingest:
	$(PYTHON) -m pipelines.ingest_yf --symbol $(SYMBOL) --start-date $(START) --end-date $(END) --dest-uri data/raw

features:
	$(PYTHON) -m pipelines.features --symbol $(SYMBOL) --raw-uri data/raw --raw-partition $(PARTITION) --version $(VERSION) --processed-uri data/processed

train:
	$(PYTHON) -m pipelines.train_xgb --symbol $(SYMBOL) --feature-version $(VERSION) --run-name $(RUN) --processed-uri data/processed --output-uri data/models --model-name $(MODEL)

predict:
	$(PYTHON) -m pipelines.predict --symbol $(SYMBOL) --model-name $(MODEL) --run-id $(RUN) --feature-version $(VERSION) --processed-uri data/processed --models-uri data/models --output-uri data/predictions

backtest:
	$(PYTHON) -c "from common.schema import BacktestConfig; from backtest.engine import run_backtest; cfg = BacktestConfig(name='cli_backtest', symbol='$(SYMBOL)', strategy='straddle', start_date='$(START)', end_date='$(END)', data={'predictions_uri': '$(PREDICTIONS)'}); result = run_backtest(cfg); print(result.metrics)"

api:
	uvicorn api.main:app --reload --port $(if $(PORT),$(PORT),8000)

dashboard:
	streamlit run dashboard/app.py
