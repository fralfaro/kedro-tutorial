"""Definición del pipeline data_ingestion."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import load_train_data, load_test_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=load_train_data,
                inputs="train_raw",
                outputs="train_raw_validated",
                name="load_train_data_node",
                tags=["ingestion", "train"],
            ),
            node(
                func=load_test_data,
                inputs="test_raw",
                outputs="test_raw_validated",
                name="load_test_data_node",
                tags=["ingestion", "test"],
            ),
        ]
    )
