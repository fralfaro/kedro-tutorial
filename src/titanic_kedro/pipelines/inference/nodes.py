"""Nodos del pipeline inference.

Responsabilidad:
  1. Cargar el modelo entrenado y generar predicciones sobre test.
  2. Aplicar el umbral de clasificación definido en parameters.
  3. Construir el archivo de submission en el formato Kaggle:
       PassengerId, Survived

El dataset de test nunca tuvo contacto con el fit del modelo
ni con el target — toda su transformación fue determinista.
"""

import logging

import pandas as pd
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodo 1: generar predicciones
# ---------------------------------------------------------------------------

def predict(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    threshold: float,
    id_column: str,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Genera predicciones de probabilidad y clase sobre el dataset de test.

    Args:
        model: modelo XGBClassifier entrenado.
        X_test: DataFrame con PassengerId + features (output de feature_engineering).
        threshold: umbral para convertir probabilidad → clase binaria.
        id_column: nombre de la columna de identificador ('PassengerId').
        feature_columns: columnas que el modelo espera como input.

    Returns:
        DataFrame con columnas: PassengerId, Survived_prob, Survived.
    """
    # Separar id de features
    ids = X_test[id_column] if id_column in X_test.columns else None
    X = X_test[feature_columns]

    # Verificar que las features coincidan con las del entrenamiento
    model_features = model.get_booster().feature_names
    if model_features is not None:
        missing = set(model_features) - set(X.columns)
        extra = set(X.columns) - set(model_features)
        if missing:
            raise ValueError(f"Features faltantes en test: {missing}")
        if extra:
            logger.warning("Features extra en test (se ignoran): %s", extra)
        X = X[model_features]

    # Predicción
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    logger.info(
        "Predicciones generadas: %d muestras | threshold=%.2f",
        len(y_pred), threshold,
    )
    logger.info(
        "Distribución predicha — 0: %d | 1: %d (%.1f%% supervivientes)",
        (y_pred == 0).sum(),
        (y_pred == 1).sum(),
        y_pred.mean() * 100,
    )

    # Construir resultado
    result = pd.DataFrame({
        id_column: ids.values if ids is not None else range(len(y_pred)),
        "Survived_prob": y_prob.round(4),
        "Survived": y_pred,
    })

    return result


# ---------------------------------------------------------------------------
# Nodo 2: construir submission
# ---------------------------------------------------------------------------

def build_submission(
    predictions: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    """Construye el CSV de submission en el formato Kaggle.

    Formato esperado:
        PassengerId,Survived
        892,0
        893,1
        ...

    Args:
        predictions: output del nodo predict.
        id_column: nombre de la columna de identificador.

    Returns:
        DataFrame con exactamente dos columnas: PassengerId y Survived.
    """
    submission = predictions[[id_column, "Survived"]].copy()
    submission["Survived"] = submission["Survived"].astype(int)

    logger.info(
        "Submission construido: %d filas | supervivientes: %d (%.1f%%)",
        len(submission),
        submission["Survived"].sum(),
        submission["Survived"].mean() * 100,
    )
    logger.info("Preview:\n%s", submission.head(10).to_string(index=False))

    return submission
