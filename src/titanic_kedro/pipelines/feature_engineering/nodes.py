"""Nodos del pipeline feature_engineering.

Responsabilidad:
  1. Crear features derivadas a partir de columnas existentes.
  2. Seleccionar el subconjunto final de columnas que entra al modelo.
  3. Separar el target (Survived) del dataset de entrenamiento.

Todas las operaciones son deterministas (sin fit), por lo que
aplican igual a train y test.

Features creadas:
  - FamilySize : SibSp + Parch + 1  (grupo total incluyendo al pasajero)
  - IsAlone    : 1 si FamilySize == 1, 0 si no
  (Title ya fue creado en preprocessing)
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodo 1: crear features
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea features derivadas a partir de columnas ya preprocesadas.

    Args:
        df: DataFrame preprocesado (output de preprocessing).

    Returns:
        DataFrame con las columnas nuevas añadidas.
    """
    df = df.copy()

    # FamilySize: grupo total (pasajero + acompañantes)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    logger.info(
        "Feature 'FamilySize' creada. Distribución:\n%s",
        df["FamilySize"].value_counts().sort_index().to_string(),
    )

    # IsAlone: viaja sin acompañantes
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    alone_pct = df["IsAlone"].mean() * 100
    logger.info("Feature 'IsAlone' creada. %.1f%% viajan solos.", alone_pct)

    return df


# ---------------------------------------------------------------------------
# Nodo 2: seleccionar features finales para train
# (separa X de y)
# ---------------------------------------------------------------------------

def select_features_train(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selecciona las features del modelo y el target para train.

    Args:
        df: DataFrame con features creadas.
        feature_columns: lista de columnas que entran al modelo.
        target_column: nombre de la columna objetivo ('Survived').

    Returns:
        Tupla (X_featured, y_featured) como DataFrames separados.
    """
    # Verificar que todas las features existen
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Features faltantes en train: {missing}")

    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' no encontrado en train.")

    X = df[feature_columns].copy()
    y = df[[target_column]].copy()

    logger.info(
        "Train — X: %s, y: %s. Features: %s",
        X.shape, y.shape, feature_columns,
    )
    logger.info(
        "Distribución del target:\n%s",
        y[target_column].value_counts().to_string(),
    )

    return X, y


# ---------------------------------------------------------------------------
# Nodo 3: seleccionar features finales para test
# (solo X, sin target)
# ---------------------------------------------------------------------------

def select_features_test(
    df: pd.DataFrame,
    feature_columns: list[str],
    id_column: str,
) -> pd.DataFrame:
    """Selecciona las features del modelo para el dataset de inferencia.

    El PassengerId se conserva aparte para reconstruir el submission.

    Args:
        df: DataFrame con features creadas.
        feature_columns: lista de columnas que entran al modelo.
        id_column: nombre de la columna de identificador ('PassengerId').

    Returns:
        DataFrame con las features seleccionadas + columna id.
    """
    missing = set(feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Features faltantes en test: {missing}")

    # Conservar id para reconstruir submission más adelante
    cols = [id_column] + feature_columns if id_column in df.columns else feature_columns
    X_test = df[cols].copy()

    logger.info(
        "Test — shape: %s. Features: %s",
        X_test.shape, feature_columns,
    )

    return X_test
