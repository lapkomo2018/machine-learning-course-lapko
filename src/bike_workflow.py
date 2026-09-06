from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "hour.csv"
TARGET = "cnt"
DROP_COLUMNS = ["instant", "dteday", "casual", "registered"]
CATEGORICAL_FEATURES = [
    "season",
    "mnth",
    "hr",
    "holiday",
    "weekday",
    "workingday",
    "weathersit",
]
NUMERIC_FEATURES = ["yr", "temp", "atemp", "hum", "windspeed"]


def load_bike_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Завантажити й хронологічно впорядкувати погодинні дані."""
    frame = pd.read_csv(path)
    frame["dteday"] = pd.to_datetime(frame["dteday"])
    return frame.sort_values(["dteday", "hr"]).reset_index(drop=True)


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Відокремити ціль і вилучити службові та витокові поля."""
    features = frame.drop(columns=[TARGET, *DROP_COLUMNS])
    return features, frame[TARGET]


def chronological_split(
    features: pd.DataFrame, target: pd.Series, train_fraction: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Поділити дані без перемішування, залишивши останній період для тесту."""
    split_index = int(len(features) * train_fraction)
    return (
        features.iloc[:split_index].copy(),
        features.iloc[split_index:].copy(),
        target.iloc[:split_index].copy(),
        target.iloc[split_index:].copy(),
    )


def make_preprocessor() -> ColumnTransformer:
    """Створити preprocessing для числових і категоріальних ознак."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_polynomial_preprocessor() -> ColumnTransformer:
    """Розширити числові ознаки поліномами другого степеня."""
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric_polynomial", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def make_models() -> dict[str, Pipeline]:
    """Побудувати baseline і три кандидатні регресійні моделі."""
    return {
        "Baseline": Pipeline(
            [("preprocessor", make_preprocessor()), ("model", DummyRegressor())]
        ),
        "Linear Regression": Pipeline(
            [("preprocessor", make_preprocessor()), ("model", LinearRegression())]
        ),
        "Polynomial d=2 (Ridge alpha=1)": Pipeline(
            [
                ("preprocessor", make_polynomial_preprocessor()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "HistGradientBoosting": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.08,
                        max_iter=300,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

