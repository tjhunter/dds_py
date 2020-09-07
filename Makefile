.PHONY: build pytest dbc

pytest:
	pytest -o log_cli=true -o log_cli_level=debug

build:
	rm -rf ./dist/*.whl
	python setup.py bdist_wheel

dbc:
	databricks fs cp $(PWD)/dist/dds_py-0.1.0-py3-none-any.whl dbfs:/libs/dds_py-0.1.1-py3-none-any.whl --overwrite --profile dds
