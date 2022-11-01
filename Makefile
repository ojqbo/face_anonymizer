# dev
test:
	pytest backend/

test_cov:
	coverage run -m pytest backend/ --cov-config=.coveragerc
	coverage html -i
	coverage report --show-missing -i

black:
	black backend/

mypy:
	mypy backend/

flake8:
	flake8 backend/

isort:
	isort backend/

style_format: isort black

style_check: flake8 mypy

clean:
	rm -r client_files pipes
	rm -r backend/**/__pycache__
	rm -r backend/tests/tempfiles htmlcov .coverage
	rm -r .mypy_cache .pytest_cache

