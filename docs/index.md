# Tutorial de Kedro con Titanic

Este sitio explica, paso a paso, un proyecto real de [Kedro](https://kedro.org/) usando el dataset del Titanic y un modelo `XGBoost`. La idea no es solo mostrar comandos: también busca ayudarte a entender por qué el proyecto está dividido en pipelines, cómo fluye la información y qué archivos hay que mirar para no perderse.

## Qué hace este proyecto

El repositorio toma los archivos `train.csv` y `test.csv` del desafío Titanic, los valida, los transforma, crea algunas variables derivadas, entrena un modelo de clasificación y finalmente genera un `submission.csv` listo para Kaggle.

En otras palabras:

1. Carga datos crudos.
2. Limpia e imputa faltantes.
3. Construye variables útiles para el modelo.
4. Separa entrenamiento y validación.
5. Entrena `XGBoost`.
6. Evalúa el modelo.
7. Predice sobre el conjunto de test.

## Qué vas a encontrar en esta documentación

| Página | Para qué sirve |
| --- | --- |
| `Kedro Desde Cero` | Introducción a los conceptos base de Kedro para alguien que nunca lo ha usado. |
| `Recorrido Del Proyecto` | Explica la estructura del repositorio y los archivos más importantes. |
| `Pipelines Y Flujo` | Muestra el flujo completo del proyecto, incluyendo un diagrama Mermaid. |
| `Catálogo Y Parámetros` | Explica cómo Kedro sabe dónde están los datos y cómo se parametriza el proyecto. |
| `Cómo Ejecutarlo` | Comandos concretos para correr el proyecto, inspeccionarlo y probarlo. |

## Ideas clave antes de empezar

- En Kedro, un proyecto no se piensa como un script largo, sino como una secuencia de pasos pequeños y bien conectados.
- Cada paso importante suele representarse como un `node`.
- Los `nodes` se agrupan en `pipelines`.
- Los datos y modelos se nombran en un catálogo, para que el código no dependa de rutas escritas a mano en cada función.
- La configuración vive fuera del código, normalmente en `conf/base/`.

## Mapa rápido del caso Titanic

| Pipeline | Rol en el proyecto |
| --- | --- |
| `data_ingestion` | Carga y valida `train.csv` y `test.csv`. |
| `preprocessing` | Extrae `Title`, elimina columnas, imputa faltantes y codifica variables categóricas. |
| `feature_engineering` | Crea `FamilySize` e `IsAlone`, y selecciona las variables finales. |
| `model_training` | Hace `train/validation split`, entrena `XGBoost` y calcula métricas. |
| `inference` | Genera predicciones en test y construye `submission.csv`. |

## Qué aprenderás mirando este repo

- Cómo se registra un pipeline principal en Kedro.
- Cómo separar responsabilidades por etapas.
- Cómo evitar mezclar rutas de archivos con lógica de negocio.
- Cómo reutilizar el mismo flujo para entrenamiento e inferencia.
- Cómo documentar el paso de datos entre capas como `raw`, `intermediate`, `primary` y `model_output`.

## Punto de partida recomendado

Si nunca has usado Kedro, sigue este orden:

1. Lee `Kedro Desde Cero`.
2. Revisa `Recorrido Del Proyecto`.
3. Mira `Pipelines Y Flujo`.
4. Finalmente ejecuta el proyecto siguiendo `Cómo Ejecutarlo`.
