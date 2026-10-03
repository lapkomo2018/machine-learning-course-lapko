from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "titanic.csv"
RANDOM_STATE = 42
TARGET = "AtRisk"
NUMERIC_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# PassengerId, Name, Ticket — ідентифікатори; Cabin має 77% пропусків.
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]


def load_titanic(path: Path = DATA_PATH) -> pd.DataFrame:
    """Завантажити Titanic і додати позитивний клас AtRisk = 1 - Survived."""
    frame = pd.read_csv(path)
    frame[TARGET] = 1 - frame["Survived"]
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Залишити змістовні ознаки та цільову змінну; індекс = PassengerId."""
    data = frame.set_index("PassengerId")
    return data[FEATURES].copy(), data[TARGET].copy()


def stratified_split(
    features: pd.DataFrame, target: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Єдиний для ЛР2–ЛР4 і ПР4 стратифікований поділ 80/20."""
    return train_test_split(
        features, target, test_size=test_size, stratify=target, random_state=RANDOM_STATE
    )


def make_cv(n_splits: int = 5, random_state: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_preprocessor(scale: str | None = "standard") -> ColumnTransformer:
    """Числові: медіана (+ індикатор пропуску Age) і масштабування; категоріальні: мода + one-hot.

    scale: "standard", "minmax" або None (для дерев).
    """
    from sklearn.preprocessing import MinMaxScaler

    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale == "standard":
        numeric_steps.append(("scaler", StandardScaler()))
    elif scale == "minmax":
        numeric_steps.append(("scaler", MinMaxScaler()))
    elif scale is not None:
        raise ValueError(f"Unknown scale: {scale}")
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
