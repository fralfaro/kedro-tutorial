"""Nodos del pipeline preprocessing.

Responsabilidad:
  1. Extraer el título del nombre (antes de dropear 'Name').
  2. Eliminar columnas de alta cardinalidad / sin valor predictivo.
  3. Imputar valores faltantes por columna según strategy en parameters.
  4. Encodear variables categóricas con mapeo fijo (reproducible sin sklearn).

El mismo conjunto de nodos se aplica a train y test por separado,
garantizando que no haya data leakage (el test nunca informa al fit).

Nota sobre encoding: usamos mapeo fijo en lugar de OrdinalEncoder
para asegurar consistencia entre train y test sin necesidad de
serializar un transformador adicional.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Mapeos fijos de encoding
SEX_MAP = {"male": 0, "female": 1}
EMBARKED_MAP = {"S": 0, "C": 1, "Q": 2}


# ---------------------------------------------------------------------------
# Nodo 1: extraer título (debe correr ANTES de dropear 'Name')
# ---------------------------------------------------------------------------

def extract_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título del pasajero desde la columna 'Name'.

    Agrupa títulos poco frecuentes bajo la categoría 'Rare'.

    Args:
        df: DataFrame con la columna 'Name'.

    Returns:
        DataFrame con la columna 'Title' añadida.
    """
    df = df.copy()

    # Extraer título con regex: "Apellido, Título. Nombre"
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].str.strip()

    # Normalizar variantes comunes
    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    df["Title"] = df["Title"].replace(title_map)

    # Cualquier título no contemplado → Rare
    known = {"Mr", "Miss", "Mrs", "Master", "Rare"}
    df["Title"] = df["Title"].apply(lambda t: t if t in known else "Rare")

    # Encodear a numérico
    title_encoding = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
    df["Title"] = df["Title"].map(title_encoding).astype(int)

    logger.info("Títulos extraídos y encodeados. Distribución:\n%s",
                df["Title"].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# Nodo 2: eliminar columnas
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """Elimina columnas de alta cardinalidad o sin valor predictivo.

    Args:
        df: DataFrame de entrada.
        columns_to_drop: lista de columnas a eliminar (desde parameters.yml).

    Returns:
        DataFrame sin las columnas especificadas.
    """
    existing = [c for c in columns_to_drop if c in df.columns]
    missing = set(columns_to_drop) - set(existing)

    if missing:
        logger.warning("Columnas a dropear no encontradas (se ignoran): %s", missing)

    df = df.drop(columns=existing)
    logger.info("Columnas eliminadas: %s. Shape resultante: %s", existing, df.shape)
    return df


# ---------------------------------------------------------------------------
# Nodo 3: imputar valores faltantes
# ---------------------------------------------------------------------------

def impute_missing(
    df: pd.DataFrame,
    imputation: dict[str, Any],
) -> pd.DataFrame:
    """Imputa valores faltantes según la estrategia definida en parameters.

    Estrategias soportadas: 'median', 'mean', 'most_frequent', 'constant'.

    Args:
        df: DataFrame con posibles valores nulos.
        imputation: dict con columna → {strategy: ..., [fill_value: ...]}.

    Returns:
        DataFrame sin valores nulos en las columnas especificadas.
    """
    df = df.copy()

    for col, config in imputation.items():
        if col not in df.columns:
            logger.warning("Columna '%s' no encontrada para imputar, se omite.", col)
            continue

        n_missing = df[col].isnull().sum()
        if n_missing == 0:
            logger.debug("Columna '%s': sin valores faltantes.", col)
            continue

        strategy = config["strategy"]

        if strategy == "median":
            fill_value = df[col].median()
        elif strategy == "mean":
            fill_value = df[col].mean()
        elif strategy == "most_frequent":
            fill_value = df[col].mode()[0]
        elif strategy == "constant":
            fill_value = config.get("fill_value", 0)
        else:
            raise ValueError(f"Estrategia desconocida: '{strategy}' para '{col}'")

        df[col] = df[col].fillna(fill_value)
        logger.info(
            "Columna '%s': %d nulos imputados con %s (valor=%.4s).",
            col, n_missing, strategy, str(fill_value),
        )

    return df


# ---------------------------------------------------------------------------
# Nodo 4: encodear variables categóricas
# ---------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica encoding fijo a variables categóricas.

    Usa mapeos deterministas (no fit de sklearn) para garantizar
    consistencia entre train y test sin serializar transformadores.

    Mapeos:
        Sex:      male=0, female=1
        Embarked: S=0, C=1, Q=2

    Args:
        df: DataFrame con columnas 'Sex' y 'Embarked'.

    Returns:
        DataFrame con las columnas encodeadas como int.
    """
    df = df.copy()

    if "Sex" in df.columns:
        unmapped = set(df["Sex"].dropna().unique()) - set(SEX_MAP)
        if unmapped:
            logger.warning("Valores desconocidos en 'Sex': %s → se mapean a -1", unmapped)
        df["Sex"] = df["Sex"].map(SEX_MAP).fillna(-1).astype(int)

    if "Embarked" in df.columns:
        unmapped = set(df["Embarked"].dropna().unique()) - set(EMBARKED_MAP)
        if unmapped:
            logger.warning("Valores desconocidos en 'Embarked': %s → se mapean a -1", unmapped)
        df["Embarked"] = df["Embarked"].map(EMBARKED_MAP).fillna(-1).astype(int)

    logger.info("Encoding aplicado: Sex y Embarked convertidos a int.")
    return df


# ---------------------------------------------------------------------------
# Nodo wrapper: aplica todo el preproceso en orden correcto
# ---------------------------------------------------------------------------

def preprocess_dataset(
    df: pd.DataFrame,
    columns_to_drop: list[str],
    imputation: dict[str, Any],
) -> pd.DataFrame:
    """Orquesta los pasos de preprocesamiento en el orden correcto.

    Orden:
        1. extract_title     (necesita 'Name', antes del drop)
        2. drop_columns      (elimina Name, Ticket, Cabin)
        3. impute_missing    (imputa Age, Embarked, Fare)
        4. encode_categoricals (Sex, Embarked → int)

    Args:
        df: DataFrame raw (train o test).
        columns_to_drop: desde params['preprocessing']['columns_to_drop'].
        imputation: desde params['preprocessing']['imputation'].

    Returns:
        DataFrame limpio y listo para feature engineering.
    """
    logger.info("Iniciando preprocesamiento. Shape inicial: %s", df.shape)

    df = extract_title(df)
    df = drop_columns(df, columns_to_drop)
    df = impute_missing(df, imputation)
    df = encode_categoricals(df)

    # Verificación final: no deben quedar nulos en columnas usadas por el modelo
    n_nulls = df.isnull().sum().sum()
    if n_nulls > 0:
        cols_with_nulls = df.columns[df.isnull().any()].tolist()
        logger.warning("Aún quedan %d nulos en: %s", n_nulls, cols_with_nulls)
    else:
        logger.info("Preprocesamiento completo. Shape final: %s. Sin nulos.", df.shape)

    return df
