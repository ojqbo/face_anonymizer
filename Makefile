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

install_development_dependencies: install_dependencies requirements.dev.txt
	$(BIN)/pip install -Ur requirements.dev.txt

ensurevenv:
	test -d .venv || $(PY) -m venv $(VENV)

run: default
	$(BIN)/python3 -m backend.server

run_and_visit: default
	xdg-open http://localhost:8080 &
	$(BIN)/python3 -m backend.server

# dev
test: install_development_dependencies
	$(BIN)/pytest backend/

test_cov: install_development_dependencies
	$(BIN)/coverage run -m pytest backend/ --cov-config=.coveragerc
	$(BIN)/coverage html
	$(BIN)/coverage report

black: install_development_dependencies
	$(BIN)/black backend/

mypy: install_development_dependencies
	$(BIN)/mypy backend/

flake8: install_development_dependencies
	$(BIN)/flake8 backend/

isort: install_development_dependencies
	$(BIN)/isort backend/

codespell: install_development_dependencies
	shopt -s globstar; $(BIN)/codespell --ignore-words=.spellignore **/*.py frontend/*.html frontend/main.js

style_format: isort black codespell

style_check: flake8 mypy

clean:
	rm -rf client_files pipes
	rm -rf backend/**/__pycache__
	rm -rf backend/tests/tempfiles htmlcov .coverage
	rm -rf .mypy_cache .pytest_cache
