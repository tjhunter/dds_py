.PHONY: build pytest dbc

pytest:
	pytest -o log_cli=true -o log_cli_level=debug

build:
	rm -rf ./dist/*.whl
	python setup.py bdist_wheel

lint:
	mypy --show-error-context --show-error-codes --pretty --allow-untyped-defs --strict  --show-traceback dds
	black --check dds
	flake8 dds


dbc:
	databricks fs cp $(PWD)/dist/dds_py-0.1.0-py3-none-any.whl dbfs:/libs/dds_py-0.1.1-py3-none-any.whl --overwrite --profile dds

release:
	git remote update
	git checkout origin/master
	git git tag -a v$(python -c "import dds._version; print(dds._version.version)") -m "tag"
	git push --tags


doc:
	PYTHONPATH=$(PWD) jupyter nbconvert --execute --clear-output doc_source/test_sklearn.ipynb
	mkdocs build
