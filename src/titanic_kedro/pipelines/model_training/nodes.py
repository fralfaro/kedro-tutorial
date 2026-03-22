"""Nodos del pipeline model_training.

Responsabilidad:
  1. Dividir el dataset de train en train/validación (split temporal no,
     split estratificado sí — el target está desbalanceado ~38/62).
  2. Entrenar XGBoost 1.5.8 con early stopping sobre el set de validación.
  3. Evaluar el modelo y loggear métricas clave.
  4. Retornar el modelo entrenado para que inference lo consuma.

Nota XGBoost 1.5.8:
  - `use_label_encoder=False` es obligatorio para silenciar el warning.
  - `eval_metric` va en el constructor, NO en fit().
  - Early stopping se configura con `early_stopping_rounds` en fit()
    y requiere `eval_set`.
  - La API sklearn (`XGBClassifier`) es más cómoda que la nativa aquí
    porque Kedro ya maneja la serialización con joblib.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nodo 1: split train / validación
# ---------------------------------------------------------------------------

def split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide los datos en conjuntos de entrenamiento y validación.

    Usa split estratificado para respetar la distribución del target.

    Args:
        X: features de entrenamiento.
        y: target de entrenamiento (DataFrame de una columna).
        test_size: fracción del dataset para validación.
        random_state: semilla de reproducibilidad.

    Returns:
        Tupla (X_train, X_val, y_train, y_val).
    """
    y_series = y.iloc[:, 0]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y_series,
        test_size=test_size,
        random_state=random_state,
        stratify=y_series,
    )

    logger.info(
        "Split completado — Train: %d muestras | Val: %d muestras",
        len(X_train), len(X_val),
    )
    logger.info(
        "Distribución target en train:\n%s",
        y_train.value_counts(normalize=True).round(3).to_string(),
    )
    logger.info(
        "Distribución target en val:\n%s",
        y_val.value_counts(normalize=True).round(3).to_string(),
    )

    # Devolver y como DataFrame para que Kedro los serialice correctamente
    return (
        X_train,
        X_val,
        y_train.to_frame(),
        y_val.to_frame(),
    )


# ---------------------------------------------------------------------------
# Nodo 2: entrenar XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    xgboost_params: dict[str, Any],
    early_stopping_rounds: int,
) -> XGBClassifier:
    """Entrena un XGBClassifier con early stopping sobre el set de validación.

    Args:
        X_train: features de entrenamiento.
        y_train: target de entrenamiento.
        X_val: features de validación.
        y_val: target de validación.
        xgboost_params: hiperparámetros desde parameters.yml.
        early_stopping_rounds: rondas sin mejora antes de detener.

    Returns:
        Modelo XGBClassifier entrenado.
    """
    y_train_arr = y_train.iloc[:, 0].values
    y_val_arr = y_val.iloc[:, 0].values

    # XGBoost 1.5.8: use_label_encoder y eval_metric van en el constructor
    model = XGBClassifier(
        n_estimators=xgboost_params["n_estimators"],
        max_depth=xgboost_params["max_depth"],
        learning_rate=xgboost_params["learning_rate"],
        subsample=xgboost_params["subsample"],
        colsample_bytree=xgboost_params["colsample_bytree"],
        min_child_weight=xgboost_params["min_child_weight"],
        gamma=xgboost_params["gamma"],
        reg_alpha=xgboost_params["reg_alpha"],
        reg_lambda=xgboost_params["reg_lambda"],
        objective=xgboost_params["objective"],
        eval_metric=xgboost_params["eval_metric"],
        use_label_encoder=xgboost_params.get("use_label_encoder", False),
        verbosity=xgboost_params.get("verbosity", 0),
        random_state=xgboost_params.get("random_state", 42),
    )

    logger.info("Iniciando entrenamiento XGBoost 1.5.8...")
    logger.info("Hiperparámetros: %s", xgboost_params)

    model.fit(
        X_train,
        y_train_arr,
        eval_set=[(X_val, y_val_arr)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    best_iter = model.best_iteration
    best_score = model.best_score
    logger.info(
        "Entrenamiento finalizado. Mejor iteración: %d | Mejor logloss val: %.5f",
        best_iter, best_score,
    )

    # Log importancia de features (top 5)
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns,
    ).sort_values(ascending=False)
    logger.info("Top 5 feature importances:\n%s", importances.head(5).to_string())

    return model


# ---------------------------------------------------------------------------
# Nodo 3: evaluar modelo
# ---------------------------------------------------------------------------

def evaluate_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcula métricas de clasificación sobre el set de validación.

    Args:
        model: modelo XGBClassifier entrenado.
        X_val: features de validación.
        y_val: target de validación.
        threshold: umbral para convertir probabilidad en clase.

    Returns:
        Diccionario con métricas: accuracy, precision, recall, f1, roc_auc.
    """
    y_true = y_val.iloc[:, 0].values
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }

    cm = confusion_matrix(y_true, y_pred)

    logger.info("=" * 50)
    logger.info("MÉTRICAS DE VALIDACIÓN")
    logger.info("=" * 50)
    for name, value in metrics.items():
        logger.info("  %-12s: %.4f", name.upper(), value)
    logger.info("Matriz de confusión:\n%s", cm)
    logger.info("=" * 50)

    return metrics
