# Recorrido Del Proyecto

## Vista general

El proyecto está organizado siguiendo la estructura habitual de Kedro: código fuente en `src/`, configuración en `conf/`, documentación en `docs/`, datos por capas en `data/` y pruebas en `tests/`.

## Estructura principal

```text
.
├── conf/
│   ├── base/
│   │   ├── catalog.yml
│   │   └── parameters.yml
│   └── local/
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_model_input/
│   ├── 05_models/
│   ├── 06_model_output/
│   └── 07_reporting/
├── docs/
├── src/titanic_kedro/
│   ├── pipeline_registry.py
│   └── pipelines/
└── tests/
```

## Dónde está la lógica importante

### `src/titanic_kedro/pipeline_registry.py`

Es el punto donde Kedro registra todos los pipelines disponibles. También define cuál es el pipeline por defecto cuando ejecutas:

```bash
uv run kedro run
```

En este repo, `__default__` es la suma de:

1. `data_ingestion`
2. `preprocessing`
3. `feature_engineering`
4. `model_training`
5. `inference`

### `src/titanic_kedro/pipelines/`

Aquí vive la lógica de negocio separada por etapas. Cada subcarpeta tiene dos piezas:

- `pipeline.py`: define los nodes y cómo se conectan.
- `nodes.py`: implementa las funciones reales.

Eso es muy típico en Kedro y hace que la lectura sea bastante limpia: primero entiendes el flujo, luego bajas al detalle del código.

## Qué guarda cada capa de datos

Kedro suele organizar el directorio `data/` en capas. En este proyecto se usan estas:

| Capa | Carpeta | Qué contiene |
| --- | --- | --- |
| Raw | `data/01_raw/` | Archivos originales `train.csv` y `test.csv`. |
| Intermediate | `data/02_intermediate/` | Datos ya preprocesados. |
| Primary | `data/03_primary/` | Features listas para modelar y target separado. |
| Model Input | `data/04_model_input/` | `train/validation split`. |
| Models | `data/05_models/` | Modelo entrenado serializado. |
| Model Output | `data/06_model_output/` | Predicciones crudas sobre test. |
| Reporting | `data/07_reporting/` | Métricas y archivo final de submission. |

Esta convención ayuda mucho porque el estado del proyecto se entiende mirando la carpeta.

## Qué hace cada carpeta de pipelines

| Carpeta | Responsabilidad |
| --- | --- |
| `data_ingestion` | Validar estructura mínima de los archivos de entrada. |
| `preprocessing` | Limpiar datos y dejar variables numéricas o listas para transformar. |
| `feature_engineering` | Crear variables derivadas y elegir el subconjunto final para modelar. |
| `model_training` | Dividir datos, entrenar el modelo y evaluar. |
| `inference` | Aplicar el modelo al set de test y construir la salida final. |

## Qué se configura fuera del código

### `conf/base/catalog.yml`

Le dice a Kedro cómo leer y guardar datasets como:

- `train_raw`
- `train_preprocessed`
- `train_featured`
- `xgboost_model`
- `submission`

### `conf/base/parameters.yml`

Centraliza decisiones de configuración como:

- columnas a eliminar,
- estrategias de imputación,
- features finales,
- tamaño del conjunto de validación,
- hiperparámetros de `XGBoost`,
- umbral de predicción.

## Qué hay en `tests/`

Las pruebas se concentran en nodos de:

- `preprocessing`
- `feature_engineering`

Eso es útil porque Kedro promueve funciones pequeñas y testeables. En vez de testear todo el pipeline de una vez, aquí se validan transformaciones concretas.

## Dos detalles útiles para entender el repo

### Los datasets validados viven en memoria

Los outputs `train_raw_validated` y `test_raw_validated` no están definidos en el catálogo. Por eso Kedro los trata como `MemoryDataset`: existen durante la ejecución, pero no se escriben a disco.

### `Title` se crea, pero no se usa en el modelo final

Durante `preprocessing` se extrae la columna `Title` a partir del nombre del pasajero. Sin embargo, las `feature_columns` actuales no la incluyen. Eso convierte a `Title` en un buen ejemplo pedagógico:

- muestra cómo crear una variable derivada,
- pero también permite discutir por qué una feature puede quedar fuera del modelo final.

## Qué mirar después

El siguiente paso natural es `Pipelines Y Flujo`, porque ahí aparece el recorrido completo del dato desde el CSV crudo hasta el `submission.csv`.
