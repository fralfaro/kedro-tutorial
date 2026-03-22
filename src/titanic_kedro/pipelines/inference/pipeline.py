"""Definición del pipeline inference."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import predict, build_submission


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Generar probabilidades y clase predicha
            node(
                func=predict,
                inputs=[
                    "xgboost_model",
                    "test_featured",
                    "params:inference.threshold",
                    "params:feature_engineering.id_column",
                    "params:feature_engineering.feature_columns",
                ],
                outputs="test_predictions",
                name="predict_node",
                tags=["inference"],
            ),
            # Construir archivo de submission
            node(
                func=build_submission,
                inputs=[
                    "test_predictions",
                    "params:feature_engineering.id_column",
                ],
                outputs="submission",
                name="build_submission_node",
                tags=["inference"],
            ),
        ]
    )
