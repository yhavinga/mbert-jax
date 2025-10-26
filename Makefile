.PHONY: help install download test clean lint format

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin
ARTIFACTS := artifacts/reference-configs

help:
	@echo "Available targets:"
	@echo "  install   - Create venv and install dependencies"
	@echo "  download  - Download reference model artifacts"
	@echo "  test      - Run test suite"
	@echo "  lint      - Run code linting (ruff/flake8)"
	@echo "  format    - Format code with black"
	@echo "  clean     - Remove generated files and cache"

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

download:
	@mkdir -p $(ARTIFACTS)
	$(BIN)/python scripts/py_reference_dump.py --model-name jhu-clsp/mmBERT-small \
		--output-dir $(ARTIFACTS)
	$(BIN)/python scripts/py_reference_dump.py --model-name jhu-clsp/mmBERT-base \
		--output-dir $(ARTIFACTS)
	@echo "✓ Reference artifacts downloaded to $(ARTIFACTS)"

test:
	$(BIN)/pytest tests/ -v

lint:
	@if $(BIN)/pip show ruff >/dev/null 2>&1; then \
		$(BIN)/ruff check src/ tests/ scripts/; \
	elif $(BIN)/pip show flake8 >/dev/null 2>&1; then \
		$(BIN)/flake8 src/ tests/ scripts/; \
	else \
		echo "No linter found. Install ruff or flake8."; \
	fi

format:
	@if $(BIN)/pip show black >/dev/null 2>&1; then \
		$(BIN)/black src/ tests/ scripts/; \
	else \
		echo "Black not found. Install with: pip install black"; \
	fi

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -f pytest.out err.log
	@echo "✓ Cleaned cache and temporary files"
