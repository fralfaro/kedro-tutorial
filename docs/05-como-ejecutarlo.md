# Cómo Ejecutarlo

## Requisitos

- Python `>=3.10,<3.12`
- `uv`
- los archivos `train.csv` y `test.csv` ubicados en `data/01_raw/`

## Crear el entorno e instalar dependencias

Si todavía no tienes el entorno listo:

```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra dev
```

Eso instalará el proyecto junto con las dependencias de desarrollo, incluyendo herramientas como `pytest`, `ruff`, `jupyter`, `mkdocs-material` y `neoteroi-mkdocs`.

## Ejecutar el pipeline completo

```bash
uv run kedro run
```

Ese comando ejecuta el pipeline `__default__`, que encadena:

1. ingestión,
2. preprocesamiento,
3. feature engineering,
4. entrenamiento,
5. inferencia.

## Ejecutar una sola etapa

Esto es muy útil para aprender, depurar o iterar más rápido:

```bash
uv run kedro run --pipeline data_ingestion
uv run kedro run --pipeline preprocessing
uv run kedro run --pipeline feature_engineering
uv run kedro run --pipeline model_training
uv run kedro run --pipeline inference
```

## Ejecutar por tags

Los nodes también tienen tags. Eso permite lanzar subconjuntos del flujo:

```bash
uv run kedro run --tags train
uv run kedro run --tags test
uv run kedro run --tags evaluation
```

## Visualizar el grafo del proyecto

Una de las herramientas más útiles cuando estás aprendiendo Kedro es:

```bash
uv run kedro viz
```

Eso abre una visualización interactiva del pipeline y ayuda mucho a entender cómo se conectan los datasets.

## Correr tests

```bash
uv run pytest
```

Si quieres cobertura:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

## Construir esta documentación

```bash
uv run mkdocs build
```

Para verla localmente:

```bash
uv run mkdocs serve
```

## Qué outputs deberías esperar

Después de una ejecución completa, los artefactos más importantes son:

| Archivo | Significado |
| --- | --- |
| `data/02_intermediate/train_preprocessed.parquet` | Train limpio y transformado. |
| `data/03_primary/train_featured.parquet` | Features finales de entrenamiento. |
| `data/05_models/xgboost_model.pkl` | Modelo entrenado serializado con joblib. |
| `data/07_reporting/metrics.json` | Métricas sobre validación. |
| `data/07_reporting/submission.csv` | Predicciones finales para Kaggle. |

## Recomendación para aprender más rápido

Si estás empezando con Kedro, prueba este orden:

1. corre `uv run kedro run --pipeline data_ingestion`,
2. mira qué archivos aparecen o se reutilizan,
3. abre `uv run kedro viz`,
4. y luego corre el pipeline completo.

Entenderás mucho mejor el proyecto si lo recorres por etapas en lugar de verlo como una caja negra.
