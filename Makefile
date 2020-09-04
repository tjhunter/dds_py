

pytest:
	pytest -o log_cli=true -o log_cli_level=debug

build:
	python setup.py bdist_wheel
