"""Nodos del pipeline data_ingestion.

Responsabilidad:
  1. Validar que la carga sea correcta.
  2. Loggear información básica del dataset.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def load_train_data(train_raw: pd.DataFrame) -> pd.DataFrame:
    """Valida y loggea el dataset de entrenamiento.

    Args:
        train_raw: DataFrame cargado desde data/01_raw/train.csv.

    Returns:
        El mismo DataFrame sin modificar (pass-through).
    """
    logger.info("Dataset TRAIN cargado: %d filas x %d columnas", *train_raw.shape)
    logger.info("Columnas: %s", train_raw.columns.tolist())

    _validate_train(train_raw)

    missing = train_raw.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Valores faltantes en TRAIN:\n%s", missing.to_string())

    return train_raw


def load_test_data(test_raw: pd.DataFrame) -> pd.DataFrame:
    """Valida y loggea el dataset de inferencia.

    Args:
        test_raw: DataFrame cargado desde data/01_raw/test.csv.

    Returns:
        El mismo DataFrame sin modificar (pass-through).
    """
    logger.info("Dataset TEST cargado: %d filas x %d columnas", *test_raw.shape)
    logger.info("Columnas: %s", test_raw.columns.tolist())

    _validate_test(test_raw)

    missing = test_raw.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        logger.info("Valores faltantes en TEST:\n%s", missing.to_string())

    return test_raw


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _validate_train(df: pd.DataFrame) -> None:
    """Verifica columnas mínimas requeridas en train."""
    required = {"PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Embarked"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"train.csv le faltan columnas: {missing_cols}")

    if df["Survived"].isnull().any():
        raise ValueError("La columna 'Survived' tiene valores nulos en train.")

    if not df["Survived"].isin([0, 1]).all():
        raise ValueError("La columna 'Survived' debe contener solo 0 y 1.")

    logger.debug("Validación TRAIN: OK")


def _validate_test(df: pd.DataFrame) -> None:
    """Verifica columnas mínimas requeridas en test."""
    required = {"PassengerId", "Pclass", "Name", "Sex", "Age",
                "SibSp", "Parch", "Ticket", "Fare", "Embarked"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"test.csv le faltan columnas: {missing_cols}")

    if "Survived" in df.columns:
        logger.warning(
            "test.csv contiene la columna 'Survived' — se ignorará en inferencia."
        )

    logger.debug("Validación TEST: OK")
