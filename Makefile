.PHONY: help install run run-pipeline viz docs-build docs-serve test test-cov

help:
	@echo "Targets disponibles:"
	@echo "  make install              - Instala dependencias con uv"
	@echo "  make run                  - Ejecuta el pipeline completo de Kedro"
	@echo "  make run-pipeline PIPELINE=preprocessing - Ejecuta un pipeline especifico"
	@echo "  make viz                  - Abre Kedro Viz"
	@echo "  make docs-build           - Construye la documentacion con MkDocs"
	@echo "  make docs-serve           - Levanta la documentacion local"
	@echo "  make test                 - Ejecuta pytest"
	@echo "  make test-cov             - Ejecuta pytest con cobertura"

install:
	uv sync --extra dev

run:
	uv run kedro run

run-pipeline:
	uv run kedro run --pipeline $(PIPELINE)

viz:
	uv run kedro viz

docs-build:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src --cov-report=term-missing
