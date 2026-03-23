# Catálogo Y Parámetros

## Por qué estas dos piezas importan tanto

Si tuvieras que resumir Kedro en dos archivos de configuración, probablemente serían estos:

- `conf/base/catalog.yml`
- `conf/base/parameters.yml`

El primero responde a "dónde están las cosas". El segundo responde a "con qué reglas trabaja el proyecto".

## El catálogo de datos

En Kedro, el catálogo asigna un nombre lógico a cada dataset. Después, los pipelines usan ese nombre y no necesitan saber si el dato vive en:

- un CSV,
- un parquet,
- un pickle,
- o memoria temporal.

## Ejemplos de datasets de este proyecto

| Nombre lógico | Tipo | Ruta |
| --- | --- | --- |
| `train_raw` | `pandas.CSVDataset` | `data/01_raw/train.csv` |
| `train_preprocessed` | `pandas.ParquetDataset` | `data/02_intermediate/train_preprocessed.parquet` |
| `train_featured` | `pandas.ParquetDataset` | `data/03_primary/train_featured.parquet` |
| `xgboost_model` | `pickle.PickleDataset` | `data/05_models/xgboost_model.pkl` |
| `submission` | `pandas.CSVDataset` | `data/07_reporting/submission.csv` |
| `metrics` | `tracking.MetricsDataset` | `data/07_reporting/metrics.json` |

## Una idea potente del catálogo

En el código, un node declara algo como:

```python
inputs="train_raw"
outputs="train_raw_validated"
```

Ese node no necesita conocer rutas como `data/01_raw/train.csv`. Kedro las resuelve usando el catálogo.

Eso mejora mucho la mantenibilidad porque:

- el código queda más limpio,
- mover rutas o cambiar formatos exige menos cambios,
- y la lógica se mantiene separada de la infraestructura.

## Datasets en memoria

No todo se persiste a disco. En este proyecto, `train_raw_validated` y `test_raw_validated` no están declarados en el catálogo.

Consecuencia:

- Kedro los trata como `MemoryDataset`,
- viven solo durante la ejecución,
- y sirven como artefactos intermedios livianos entre ingestión y preprocesamiento.

## Las capas de datos

El catálogo también refleja una idea muy usada en Kedro: mover el dato por capas.

| Capa | Intención |
| --- | --- |
| `01_raw` | Datos de entrada originales. |
| `02_intermediate` | Datos ya limpios y transformados. |
| `03_primary` | Datos listos para modelar. |
| `04_model_input` | Splits usados por entrenamiento. |
| `05_models` | Modelos serializados. |
| `06_model_output` | Predicciones del modelo. |
| `07_reporting` | Artefactos finales y métricas. |

Esta convención hace que el proyecto sea más fácil de leer incluso sin abrir el código.

## Los parámetros del proyecto

`conf/base/parameters.yml` centraliza las decisiones configurables.

En este repo, se usan cuatro grupos principales:

- `preprocessing`
- `feature_engineering`
- `model_training`
- `inference`

## Qué controla cada bloque

### `preprocessing`

Define:

- columnas a eliminar,
- reglas de imputación,
- y, aunque aparece una lista de categóricas, el encoding real se hace con mapeos fijos en código.

### `feature_engineering`

Define:

- columnas finales que entran al modelo,
- nombre del target,
- nombre de la columna identificadora.

Aquí hay un detalle muy útil para enseñar: `Title` se genera en preprocessing, pero no aparece en `feature_columns`, así que no se usa en el modelo actual.

### `model_training`

Define:

- `random_state`,
- tamaño del set de validación,
- hiperparámetros de `XGBoost`,
- `early_stopping_rounds`.

### `inference`

Define:

- el umbral para transformar una probabilidad en clase `0` o `1`.

## Qué ventaja pedagógica tiene esto

Mover estas decisiones fuera del código permite enseñar mejor el proyecto porque queda claro:

- qué cosas son lógica estable,
- y qué cosas son configuración que puede cambiar.

Por ejemplo, cambiar el umbral de inferencia o el `max_depth` del modelo no obliga a reescribir nodos.

## Un detalle honesto sobre el catálogo actual

El catálogo define `evaluation_report`, pero el pipeline actual no lo produce. No es un problema para ejecutar el proyecto, pero sí es una buena observación para quien esté aprendiendo:

- el catálogo puede contener datasets futuros o planeados,
- pero conviene mantenerlo alineado con los pipelines reales para evitar confusión.

## Idea práctica para leer Kedro

Si alguna vez no entiendes un proyecto Kedro, una secuencia muy útil es esta:

1. mirar `pipeline_registry.py`,
2. abrir `pipeline.py`,
3. revisar `catalog.yml`,
4. y después bajar al detalle de `nodes.py`.

Con eso normalmente ya puedes reconstruir el flujo entero.
