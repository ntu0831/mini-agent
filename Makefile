.PHONY: install dev test lint format typecheck build run web clean

install:
	python -m pip install -e .

dev:
	python -m pip install -e ".[dev,web]"

test:
	python -m pytest

lint:
	python -m ruff check .
	python -m ruff format --check .

format:
	python -m ruff check --fix .
	python -m ruff format .

typecheck:
	python -m mypy

build:
	python -m build

run:
	mini-agent

web:
	streamlit run app.py

clean:
	rm -rf .coverage .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
