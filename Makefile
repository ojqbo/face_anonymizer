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

default: install_dependencies

install_dependencies: ensurevenv requirements.txt
	$(BIN)/pip install -Ur requirements.txt

ensurevenv:
	test -d .venv || $(PY) -m venv $(VENV)

run:
	$(BIN)/python3 -m backend.server

run_and_visit:
	xdg-open http://localhost:8080 &
	$(BIN)/python3 -m backend.server

# dev
test:
	$(BIN)/pytest backend/

test_cov:
	$(BIN)/coverage run -m pytest backend/ --cov-config=.coveragerc
	$(BIN)/coverage html
	$(BIN)/coverage report

black:
	$(BIN)/black backend/

mypy:
	$(BIN)/mypy backend/

flake8:
	$(BIN)/flake8 backend/

isort:
	$(BIN)/isort backend/

codespell:
	shopt -s globstar; $(BIN)/codespell --ignore-words=.spellignore **/*.py frontend/*.html frontend/main.js

style_format: isort black codespell

style_check: flake8 mypy

clean:
	rm -rf client_files pipes
	rm -rf backend/**/__pycache__
	rm -rf backend/tests/tempfiles htmlcov .coverage
	rm -rf .mypy_cache .pytest_cache
