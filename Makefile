.PHONY: install format lint test

install:
	pip install --upgrade pip
	pip install -e ".[dev]"
	pre-commit install

format:
	isort src tests
	black src tests

lint:
	flake8 src tests
	black --check src tests
	isort --check-only src tests

test:
	pytest
