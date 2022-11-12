SHELL := /bin/bash

.venv: poetry.lock
	poetry install --no-root && touch .venv

run: .venv
	poetry run python -m face_anonymizer.server

run_and_visit: .venv
	xdg-open http://localhost:8080 &
	poetry run python -m face_anonymizer.server

# dev
test: .venv
	poetry run pytest face_anonymizer/

test_cov: .venv
	poetry run coverage run -m pytest face_anonymizer/ --cov-config=.coveragerc
	poetry run coverage html
	poetry run coverage report

black: .venv
	poetry run black face_anonymizer/

mypy: .venv
	poetry run mypy face_anonymizer/

flake8: .venv
	poetry run flake8 face_anonymizer/

isort: .venv
	poetry run isort face_anonymizer/

codespell: .venv
	shopt -s globstar; poetry run codespell --ignore-words=.spellignore **/*.py frontend/*.html frontend/main.js

style_format: isort black codespell

style_check: flake8 mypy

clean:
	rm -rf client_files pipes
	shopt -s globstar; rm -rf face_anonymizer/**/__pycache__
	rm -rf face_anonymizer/tests/tempfiles htmlcov .coverage
	rm -rf .mypy_cache .pytest_cache
