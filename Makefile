.PHONY: venv install lint format test
venv:
	python -m venv .venv
install:
	. .venv/Scripts/activate && pip install -U pip && pip install -e .[dev]
lint:
	ruff check . && black --check . && isort --check-only .
format:
	ruff check . --fix && black . && isort .
test:
	pytest -q
