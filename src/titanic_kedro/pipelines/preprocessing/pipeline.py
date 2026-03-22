"""Definición del pipeline preprocessing.

Aplica el mismo conjunto de transformaciones a train y test
de forma independiente, evitando data leakage.
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_dataset,
                inputs=[
                    "train_raw_validated",
                    "params:preprocessing.columns_to_drop",
                    "params:preprocessing.imputation",
                ],
                outputs="train_preprocessed",
                name="preprocess_train_node",
                tags=["preprocessing", "train"],
            ),
            node(
                func=preprocess_dataset,
                inputs=[
                    "test_raw_validated",
                    "params:preprocessing.columns_to_drop",
                    "params:preprocessing.imputation",
                ],
                outputs="test_preprocessed",
                name="preprocess_test_node",
                tags=["preprocessing", "test"],
            ),
        ]
    )
