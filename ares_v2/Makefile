.PHONY: env test lint reproduce-M00 reproduce-M00-smoke ledger-summary clean

SHELL := /bin/bash

env:
	uv venv --python 3.11 .venv
	source .venv/bin/activate && uv pip install -e ".[dev]"

test:
	pytest -v tests/

lint:
	ruff check src/ tests/ scripts/

reproduce-M00:
	bash scripts/smoke/run_m00.sh

reproduce-M00-smoke:
	bash scripts/smoke/run_smoke.sh

ledger-summary:
	python scripts/ledger.py summary

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
