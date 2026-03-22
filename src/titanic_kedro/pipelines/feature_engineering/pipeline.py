"""Definición del pipeline feature_engineering."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import create_features, select_features_train, select_features_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            # Crear features en train
            node(
                func=create_features,
                inputs="train_preprocessed",
                outputs="train_with_features",
                name="create_features_train_node",
                tags=["feature_engineering", "train"],
            ),
            # Crear features en test
            node(
                func=create_features,
                inputs="test_preprocessed",
                outputs="test_with_features",
                name="create_features_test_node",
                tags=["feature_engineering", "test"],
            ),
            # Separar X e y de train
            node(
                func=select_features_train,
                inputs=[
                    "train_with_features",
                    "params:feature_engineering.feature_columns",
                    "params:feature_engineering.target_column",
                ],
                outputs=["train_featured", "train_target"],
                name="select_features_train_node",
                tags=["feature_engineering", "train"],
            ),
            # Seleccionar features de test (incluye PassengerId)
            node(
                func=select_features_test,
                inputs=[
                    "test_with_features",
                    "params:feature_engineering.feature_columns",
                    "params:feature_engineering.id_column",
                ],
                outputs="test_featured",
                name="select_features_test_node",
                tags=["feature_engineering", "test"],
            ),
        ]
    )
