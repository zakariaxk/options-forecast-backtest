.PHONY: setup lint test api

PYTHON ?= $(shell command -v python3 >/dev/null 2>&1 && echo python3 || echo python)

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

lint:
	ruff check .

test:
	.venv/bin/python -m pytest tests/ -v

api:
	.venv/bin/uvicorn api.main:app --reload --port $(if $(PORT),$(PORT),8000)
