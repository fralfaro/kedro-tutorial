# Kedro Desde Cero

## Qué es Kedro

Kedro es un framework para construir proyectos de datos y machine learning de forma ordenada, reproducible y mantenible. Su idea principal es separar con claridad:

- la lógica de negocio,
- la configuración,
- las rutas de datos,
- y el orden de ejecución.

Si vienes de notebooks o scripts sueltos, Kedro te ayuda a transformar ese trabajo en un proyecto más parecido a una aplicación.

## La idea mental más importante

Piensa en Kedro como una fábrica:

- los datos entran por un extremo,
- pasan por varias estaciones,
- cada estación hace una tarea concreta,
- y al final sale un resultado reproducible.

En Kedro, esas estaciones se llaman `nodes`, y el plano de cómo se conectan se llama `pipeline`.

## Conceptos base

### Node

Un `node` envuelve una función de Python. Define:

- qué entradas recibe,
- qué salida produce,
- y cómo se llama dentro del pipeline.

Ejemplo mental:

- entra un DataFrame crudo,
- el node lo limpia,
- sale un DataFrame listo para el siguiente paso.

### Pipeline

Un `pipeline` es una colección ordenada de `nodes`. Sirve para agrupar una etapa lógica del proyecto, por ejemplo:

- ingestión de datos,
- preprocesamiento,
- entrenamiento,
- inferencia.

### Dataset

En Kedro, un `dataset` es un nombre lógico para una pieza de información. Puede ser:

- un CSV,
- un parquet,
- un modelo serializado,
- un diccionario de métricas,
- o incluso un objeto en memoria.

Lo importante es que el código usa nombres como `train_raw` o `xgboost_model`, no rutas hardcodeadas.

### Data Catalog

El catálogo le dice a Kedro:

- dónde está cada dataset,
- cómo cargarlo,
- y cómo guardarlo.

En este proyecto eso vive en `conf/base/catalog.yml`.

### Parameters

Los parámetros son configuraciones externas al código, por ejemplo:

- columnas a eliminar,
- estrategias de imputación,
- hiperparámetros del modelo,
- umbral de clasificación.

En este repo están en `conf/base/parameters.yml`.

## Qué problema resuelve Kedro

Sin Kedro, es común terminar con:

- rutas repetidas en muchos archivos,
- notebooks difíciles de reproducir,
- pasos ejecutados en distinto orden,
- lógica mezclada con configuración,
- y poca claridad sobre qué artefacto produce cada etapa.

Kedro ordena eso haciendo explícitas las dependencias entre pasos.

## Cómo se ve eso en este proyecto

Este proyecto registra cinco pipelines:

- `data_ingestion`
- `preprocessing`
- `feature_engineering`
- `model_training`
- `inference`

Además, crea un pipeline `__default__` que concatena todos esos pasos para ejecutar el flujo completo.

Eso permite correr:

```bash
uv run kedro run
```

o bien una sola etapa:

```bash
uv run kedro run --pipeline preprocessing
```

## Una ventaja muy práctica

Kedro no solo te ayuda a correr el proyecto. También te obliga a pensar bien:

- qué entra a cada etapa,
- qué sale de cada etapa,
- y qué dependencia tiene cada paso del resto.

Ese cambio de mentalidad suele hacer que los proyectos sean mucho más fáciles de explicar, testear y mantener.

## Qué mirar después

Una vez que entiendas estos conceptos, lo más útil es pasar a `Recorrido Del Proyecto`, donde se muestra cómo aterrizan estas ideas en el código de este repositorio.
