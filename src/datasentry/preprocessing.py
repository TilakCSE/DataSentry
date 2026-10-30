"""Training-only cleaning pipelines and a transparent selection policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip to Tukey fences learned exclusively from the fit partition."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(values, [25, 75], axis=0)
        spread = q3 - q1
        self.lower_ = q1 - self.multiplier * spread
        self.upper_ = q3 + self.multiplier * spread
        self.lower_ = np.where(np.isfinite(self.lower_), self.lower_, -np.inf)
        self.upper_ = np.where(np.isfinite(self.upper_), self.upper_, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)


class CategoricalModeImputer(BaseEstimator, TransformerMixin):
    """Fill each categorical column with its training mode.

    An entirely missing training column has no mode, so it receives a stable
    sentinel. String normalization also allows mixed CSV category types to be
    encoded consistently by OneHotEncoder.
    """

    def __init__(self, empty_value: str = "<MISSING>"):
        self.empty_value = empty_value

    def fit(self, X, y=None):
        values = pd.DataFrame(X)
        self.n_features_in_ = values.shape[1]
        self.fill_values_ = []
        for column in values.columns:
            observed = values[column].dropna().astype(str)
            mode = observed.value_counts().index[0] if len(observed) else self.empty_value
            self.fill_values_.append(mode)
        return self

    def transform(self, X):
        values = pd.DataFrame(X).copy()
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Categorical column count changed after fit")
        for index, fill_value in enumerate(self.fill_values_):
            values.iloc[:, index] = values.iloc[:, index].fillna(fill_value).astype(str)
        return values.to_numpy(dtype=object)


def select_strategy(train_features: pd.DataFrame) -> str:
    """Frozen heuristic using only the available training features.

    It is deliberately simple, and its merits must be measured against each
    fixed recipe rather than assumed.
    """
    numeric = train_features.select_dtypes(include="number")
    missing_fraction = float(train_features.isna().to_numpy().mean())
    if len(train_features) >= 400 and numeric.shape[1] >= 2 and 0.10 <= missing_fraction <= 0.25:
        return "knn"
    if not numeric.empty:
        q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
        spread = q3 - q1
        flagged = ((numeric.lt(q1 - 1.5 * spread)) | (numeric.gt(q3 + 1.5 * spread))).fillna(False)
        if float(flagged.to_numpy().mean()) > 0.05:
            return "iqr"
    return "median"


def build_preprocessor(train_features: pd.DataFrame, strategy: str) -> ColumnTransformer:
    if strategy not in {"median", "knn", "iqr"}:
        raise ValueError(f"Unsupported cleaning strategy: {strategy}")
    numeric = train_features.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in train_features.columns if column not in numeric]
    transformers = []
    if numeric:
        if strategy == "knn":
            steps = [("scale", StandardScaler()), ("impute", KNNImputer(n_neighbors=5,
                                                                           keep_empty_features=True))]
        else:
            steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
            if strategy == "iqr":
                steps.append(("clip", IQRClipper()))
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", CategoricalModeImputer()),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, sparse_threshold=0)
