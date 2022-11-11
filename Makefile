SHELL := /bin/bash
# system python interpreter. used only to create virtual environment
PY = python3
VENV = .venv
BIN=$(VENV)/bin

# make it work on windows too
ifeq ($(OS), Windows_NT)
    BIN=$(VENV)/Scripts
    PY=python
endif

default: .venv/dependencies

.venv/dependencies: .venv requirements.txt
	$(BIN)/pip install -Ur requirements.txt && touch .venv/dependencies

.venv/dev_dependencies: .venv/dependencies requirements.dev.txt
	$(BIN)/pip install -Ur requirements.dev.txt && touch .venv/dev_dependencies

.venv:
	test -d .venv || $(PY) -m venv $(VENV)

run: default
	$(BIN)/python3 -m backend.server

run_and_visit: default
	xdg-open http://localhost:8080 &
	$(BIN)/python3 -m backend.server

# dev
test: .venv/dev_dependencies
	$(BIN)/pytest backend/

test_cov: .venv/dev_dependencies
	$(BIN)/coverage run -m pytest backend/ --cov-config=.coveragerc
	$(BIN)/coverage html
	$(BIN)/coverage report

black: .venv/dev_dependencies
	$(BIN)/black backend/

mypy: .venv/dev_dependencies
	$(BIN)/mypy backend/

flake8: .venv/dev_dependencies
	$(BIN)/flake8 backend/

isort: .venv/dev_dependencies
	$(BIN)/isort backend/

codespell: .venv/dev_dependencies
	shopt -s globstar; $(BIN)/codespell --ignore-words=.spellignore **/*.py frontend/*.html frontend/main.js

style_format: isort black codespell

style_check: flake8 mypy

clean:
	rm -rf .venv/dependencies .venv/dev_dependencies
	rm -rf client_files pipes
	shopt -s globstar; rm -rf backend/**/__pycache__
	rm -rf backend/tests/tempfiles htmlcov .coverage
	rm -rf .mypy_cache .pytest_cache
