"""Tests para el pipeline feature_engineering."""

import pandas as pd
import pytest

from titanic_kedro.pipelines.feature_engineering.nodes import (
    create_features,
    select_features_train,
    select_features_test,
)


@pytest.fixture
def sample_preprocessed():
    """DataFrame que simula el output de preprocessing."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 1, 0, 1],
        "Pclass": [3, 1, 2, 3, 1],
        "Sex": [0, 1, 1, 0, 1],
        "Age": [22.0, 38.0, 26.0, 35.0, 28.0],
        "SibSp": [1, 1, 0, 0, 0],
        "Parch": [0, 0, 0, 0, 0],
        "Fare": [7.25, 71.28, 7.92, 8.05, 86.5],
        "Embarked": [0, 1, 0, 0, 2],
        "Title": [0, 2, 1, 0, 1],
    })


FEATURE_COLUMNS = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "FamilySize", "IsAlone",
]


class TestCreateFeatures:
    def test_family_size_correcto(self, sample_preprocessed):
        result = create_features(sample_preprocessed)
        # PassengerId=1: SibSp=1, Parch=0 → FamilySize=2
        assert result.loc[result["PassengerId"] == 1, "FamilySize"].values[0] == 2

    def test_viajero_solo(self, sample_preprocessed):
        result = create_features(sample_preprocessed)
        # PassengerId=3: SibSp=0, Parch=0 → FamilySize=1 → IsAlone=1
        assert result.loc[result["PassengerId"] == 3, "IsAlone"].values[0] == 1

    def test_viajero_acompanado(self, sample_preprocessed):
        result = create_features(sample_preprocessed)
        # PassengerId=1: FamilySize=2 → IsAlone=0
        assert result.loc[result["PassengerId"] == 1, "IsAlone"].values[0] == 0

    def test_is_alone_binario(self, sample_preprocessed):
        result = create_features(sample_preprocessed)
        assert set(result["IsAlone"].unique()).issubset({0, 1})

    def test_no_modifica_columnas_originales(self, sample_preprocessed):
        original_cols = set(sample_preprocessed.columns)
        result = create_features(sample_preprocessed)
        assert original_cols.issubset(set(result.columns))


class TestSelectFeaturesTrain:
    def test_retorna_x_e_y_separados(self, sample_preprocessed):
        df = create_features(sample_preprocessed)
        X, y = select_features_train(df, FEATURE_COLUMNS, "Survived")
        assert set(X.columns) == set(FEATURE_COLUMNS)
        assert "Survived" in y.columns

    def test_x_no_contiene_target(self, sample_preprocessed):
        df = create_features(sample_preprocessed)
        X, _ = select_features_train(df, FEATURE_COLUMNS, "Survived")
        assert "Survived" not in X.columns

    def test_mismo_numero_de_filas(self, sample_preprocessed):
        df = create_features(sample_preprocessed)
        X, y = select_features_train(df, FEATURE_COLUMNS, "Survived")
        assert len(X) == len(y) == len(df)

    def test_feature_faltante_lanza_error(self, sample_preprocessed):
        df = create_features(sample_preprocessed)
        with pytest.raises(ValueError, match="Features faltantes en train"):
            select_features_train(df, FEATURE_COLUMNS + ["FeatureQueNoExiste"], "Survived")

    def test_target_faltante_lanza_error(self, sample_preprocessed):
        df = create_features(sample_preprocessed)
        with pytest.raises(ValueError, match="Target"):
            select_features_train(df, FEATURE_COLUMNS, "TargetQueNoExiste")


class TestSelectFeaturesTest:
    def test_conserva_passenger_id(self, sample_preprocessed):
        df = create_features(sample_preprocessed.drop(columns=["Survived"]))
        result = select_features_test(df, FEATURE_COLUMNS, "PassengerId")
        assert "PassengerId" in result.columns

    def test_contiene_todas_las_features(self, sample_preprocessed):
        df = create_features(sample_preprocessed.drop(columns=["Survived"]))
        result = select_features_test(df, FEATURE_COLUMNS, "PassengerId")
        for col in FEATURE_COLUMNS:
            assert col in result.columns
