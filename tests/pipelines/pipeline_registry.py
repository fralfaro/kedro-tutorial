"""Registro de pipelines del proyecto.

Kedro llama a `register_pipelines()` al ejecutar `kedro run`.
Cada pipeline puede correrse de forma independiente:

    kedro run --pipeline data_ingestion
    kedro run --pipeline preprocessing
    kedro run --pipeline feature_engineering
    kedro run --pipeline model_training
    kedro run --pipeline inference
    kedro run  # corre el pipeline __default__ completo
"""

from kedro.pipeline import Pipeline

from titanic_kedro.pipelines.data_ingestion import pipeline as di
from titanic_kedro.pipelines.preprocessing import pipeline as pp
from titanic_kedro.pipelines.feature_engineering import pipeline as fe
from titanic_kedro.pipelines.model_training import pipeline as mt
from titanic_kedro.pipelines.inference import pipeline as inf


def register_pipelines() -> dict[str, Pipeline]:
    """Retorna el diccionario de pipelines registrados."""

    data_ingestion_pipeline = di.create_pipeline()
    preprocessing_pipeline = pp.create_pipeline()
    feature_engineering_pipeline = fe.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    inference_pipeline = inf.create_pipeline()

    return {
        "data_ingestion": data_ingestion_pipeline,
        "preprocessing": preprocessing_pipeline,
        "feature_engineering": feature_engineering_pipeline,
        "model_training": model_training_pipeline,
        "inference": inference_pipeline,
        "__default__": (
            data_ingestion_pipeline
            + preprocessing_pipeline
            + feature_engineering_pipeline
            + model_training_pipeline
            + inference_pipeline
        ),
    }
