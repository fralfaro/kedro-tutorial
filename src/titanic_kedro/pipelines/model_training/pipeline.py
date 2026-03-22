"""Definición del pipeline model_training."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import split_data, train_xgboost, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Split estratificado train / validación
            node(
                func=split_data,
                inputs=[
                    "train_featured",
                    "train_target",
                    "params:model_training.test_size",
                    "params:model_training.random_state",
                ],
                outputs=["X_train", "X_val", "y_train", "y_val"],
                name="split_data_node",
                tags=["model_training"],
            ),
            # Entrenamiento XGBoost con early stopping
            node(
                func=train_xgboost,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_val",
                    "y_val",
                    "params:model_training.xgboost",
                    "params:model_training.early_stopping_rounds",
                ],
                outputs="xgboost_model",
                name="train_xgboost_node",
                tags=["model_training"],
            ),
            # Evaluación sobre el set de validación
            node(
                func=evaluate_model,
                inputs=[
                    "xgboost_model",
                    "X_val",
                    "y_val",
                    "params:inference.threshold",
                ],
                outputs="metrics",
                name="evaluate_model_node",
                tags=["model_training", "evaluation"],
            ),
        ]
    )
