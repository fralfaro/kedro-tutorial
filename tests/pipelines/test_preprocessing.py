"""Tests para el pipeline preprocessing."""

import pandas as pd
import pytest

from titanic_kedro.pipelines.preprocessing.nodes import (
    extract_title,
    drop_columns,
    impute_missing,
    encode_categoricals,
    preprocess_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """DataFrame mínimo con la estructura de train.csv."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "Survived": [0, 1, 1, 0],
        "Pclass": [3, 1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
        ],
        "Sex": ["male", "female", "female", "male"],
        "Age": [22.0, 38.0, None, 8.0],
        "SibSp": [1, 1, 0, 3],
        "Parch": [0, 0, 0, 1],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2", "349909"],
        "Fare": [7.25, 71.28, 7.92, None],
        "Cabin": [None, "C85", None, None],
        "Embarked": ["S", "C", "S", None],
    })


# ---------------------------------------------------------------------------
# Tests: extract_title
# ---------------------------------------------------------------------------

class TestExtractTitle:
    def test_extrae_titulos_conocidos(self, sample_df):
        result = extract_title(sample_df)
        assert "Title" in result.columns

    def test_titulos_encodeados_como_int(self, sample_df):
        result = extract_title(sample_df)
        assert result["Title"].dtype in ["int32", "int64"]

    def test_mr_es_cero(self, sample_df):
        result = extract_title(sample_df)
        # PassengerId=1 es "Mr."
        assert result.loc[result["PassengerId"] == 1, "Title"].values[0] == 0

    def test_miss_es_uno(self, sample_df):
        result = extract_title(sample_df)
        # PassengerId=3 es "Miss."
        assert result.loc[result["PassengerId"] == 3, "Title"].values[0] == 1

    def test_master_es_tres(self, sample_df):
        result = extract_title(sample_df)
        # PassengerId=4 es "Master."
        assert result.loc[result["PassengerId"] == 4, "Title"].values[0] == 3

    def test_no_modifica_otras_columnas(self, sample_df):
        result = extract_title(sample_df)
        assert list(sample_df["Name"]) == list(result["Name"])


# ---------------------------------------------------------------------------
# Tests: drop_columns
# ---------------------------------------------------------------------------

class TestDropColumns:
    def test_elimina_columnas_existentes(self, sample_df):
        result = drop_columns(sample_df, ["Cabin", "Ticket"])
        assert "Cabin" not in result.columns
        assert "Ticket" not in result.columns

    def test_ignora_columnas_inexistentes(self, sample_df):
        # No debe lanzar error si la columna no existe
        result = drop_columns(sample_df, ["ColQueNoExiste", "Cabin"])
        assert "Cabin" not in result.columns

    def test_conserva_columnas_no_especificadas(self, sample_df):
        result = drop_columns(sample_df, ["Cabin"])
        assert "PassengerId" in result.columns
        assert "Survived" in result.columns


# ---------------------------------------------------------------------------
# Tests: impute_missing
# ---------------------------------------------------------------------------

class TestImputeMissing:
    def test_imputa_age_con_mediana(self, sample_df):
        imputation = {"Age": {"strategy": "median"}}
        result = impute_missing(sample_df, imputation)
        assert result["Age"].isnull().sum() == 0

    def test_imputa_embarked_con_moda(self, sample_df):
        imputation = {"Embarked": {"strategy": "most_frequent"}}
        result = impute_missing(sample_df, imputation)
        assert result["Embarked"].isnull().sum() == 0

    def test_imputa_fare_con_mediana(self, sample_df):
        imputation = {"Fare": {"strategy": "median"}}
        result = impute_missing(sample_df, imputation)
        assert result["Fare"].isnull().sum() == 0

    def test_estrategia_desconocida_lanza_error(self, sample_df):
        with pytest.raises(ValueError, match="Estrategia desconocida"):
            impute_missing(sample_df, {"Age": {"strategy": "interpolacion_magica"}})

    def test_columna_inexistente_no_lanza_error(self, sample_df):
        # Solo loggea warning
        result = impute_missing(sample_df, {"ColQueNoExiste": {"strategy": "median"}})
        assert result.shape == sample_df.shape


# ---------------------------------------------------------------------------
# Tests: encode_categoricals
# ---------------------------------------------------------------------------

class TestEncodeCategoricals:
    def test_sex_encodeado_como_int(self, sample_df):
        result = encode_categoricals(sample_df)
        assert result["Sex"].dtype in ["int32", "int64"]

    def test_male_es_cero(self, sample_df):
        result = encode_categoricals(sample_df)
        assert result.loc[result["PassengerId"] == 1, "Sex"].values[0] == 0

    def test_female_es_uno(self, sample_df):
        result = encode_categoricals(sample_df)
        assert result.loc[result["PassengerId"] == 2, "Sex"].values[0] == 1

    def test_embarked_encodeado_como_int(self, sample_df):
        # Primero imputa el nulo
        df = impute_missing(sample_df, {"Embarked": {"strategy": "most_frequent"}})
        result = encode_categoricals(df)
        assert result["Embarked"].dtype in ["int32", "int64"]


# ---------------------------------------------------------------------------
# Tests: preprocess_dataset (integración)
# ---------------------------------------------------------------------------

class TestPreprocessDataset:
    def test_pipeline_completo_sin_nulos(self, sample_df):
        columns_to_drop = ["Cabin", "Ticket", "Name"]
        imputation = {
            "Age": {"strategy": "median"},
            "Embarked": {"strategy": "most_frequent"},
            "Fare": {"strategy": "median"},
        }
        result = preprocess_dataset(sample_df, columns_to_drop, imputation)

        # No deben quedar nulos en las columnas usadas por el modelo
        model_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
        assert result[model_cols].isnull().sum().sum() == 0

    def test_columnas_dropeadas_no_existen(self, sample_df):
        columns_to_drop = ["Cabin", "Ticket", "Name"]
        imputation = {"Age": {"strategy": "median"}}
        result = preprocess_dataset(sample_df, columns_to_drop, imputation)
        for col in columns_to_drop:
            assert col not in result.columns

    def test_title_fue_creado(self, sample_df):
        result = preprocess_dataset(sample_df, ["Cabin", "Ticket", "Name"], {})
        assert "Title" in result.columns

    def test_no_modifica_input_original(self, sample_df):
        original_shape = sample_df.shape
        columns_to_drop = ["Cabin", "Ticket", "Name"]
        imputation = {"Age": {"strategy": "median"}}
        preprocess_dataset(sample_df, columns_to_drop, imputation)
        # El DataFrame original no debe modificarse (todos los nodos hacen .copy())
        assert sample_df.shape == original_shape
        assert "Title" not in sample_df.columns
