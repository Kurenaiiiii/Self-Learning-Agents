.PHONY: install lint type test cov build docker doctor eval clean

install:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"

lint:
	.venv/bin/python -m ruff check src tests
	.venv/bin/python -m ruff format --check src tests

type:
	.venv/bin/python -m mypy src

test:
	.venv/bin/python -m pytest tests -q

cov:
	.venv/bin/python -m pytest tests -q --cov=selflearn --cov-report=term-missing

doctor:
	.venv/bin/python -m selflearn.cli doctor

eval:
	.venv/bin/python -m selflearn.cli eval --strategies hybrid --runs 3 --out benchmarks/results/hybrid.json

build:
	.venv/bin/python -m build

docker:
	docker build -t selflearn:latest .

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache htmlcov
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
