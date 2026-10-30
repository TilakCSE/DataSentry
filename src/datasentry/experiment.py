"""Paired held-out comparisons of training-data cleaning strategies.

The public entry point accepts a DataFrame and returns one auditable result row
for every split, corruption condition, cleaning strategy, and estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .corruption import corrupt_training_features
from .preprocessing import build_preprocessor, select_strategy


@dataclass(frozen=True)
class ExperimentConfig:
    """Frozen parameters for a regression comparison.

    Defaults are short enough for a local demo. The paper runner supplies the
    prespecified five seeds and full factor grid explicitly.
    """

    seeds: tuple[int, ...] = (11,)
    levels: tuple[float, ...] = (0.0, 0.15, 0.30)
    mechanisms: tuple[str, ...] = ("mcar", "extreme")
    strategies: tuple[str, ...] = ("median", "knn", "iqr", "adaptive")
    models: tuple[str, ...] = ("ridge", "forest", "mlp")
    test_size: float = 0.2
    drop_columns: tuple[str, ...] = ()
    forest_estimators: int = 100
    mlp_max_iter: int = 250


def _validate(frame: pd.DataFrame, target: str, config: ExperimentConfig) -> None:
    if target not in frame.columns:
        raise ValueError(f"Target column not found: {target}")
    if target in config.drop_columns:
        raise ValueError("Target cannot appear in drop_columns")
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required")
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are unsupported")
    missing = set(config.drop_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"drop_columns not found: {sorted(missing)}")
    if frame.drop(columns=[target, *config.drop_columns]).shape[1] < 1:
        raise ValueError("At least one feature column is required")
    if pd.to_numeric(frame[target], errors="coerce").isna().any():
        raise ValueError("Regression target must contain finite numeric values only")
    if not np.isfinite(pd.to_numeric(frame[target]).to_numpy()).all():
        raise ValueError("Regression target must contain finite values only")
    if not config.seeds or not config.levels or not config.mechanisms or not config.strategies or not config.models:
        raise ValueError("Experiment factors cannot be empty")
    if any(level < 0 or level > 1 for level in config.levels):
        raise ValueError("Corruption levels must be between 0 and 1")
    for values, supported, label in [
        (config.mechanisms, {"mcar", "mar", "extreme"}, "mechanism"),
        (config.strategies, {"median", "knn", "iqr", "adaptive"}, "strategy"),
        (config.models, {"ridge", "forest", "mlp"}, "model"),
    ]:
        bad = set(values) - supported
        if bad:
            raise ValueError(f"Unsupported {label}: {sorted(bad)}")
    if config.forest_estimators < 1 or config.mlp_max_iter < 1:
        raise ValueError("Model iteration counts must be positive")


def make_group_split(features: pd.DataFrame, target: pd.Series, seed: int,
                     test_size: float) -> tuple[np.ndarray, np.ndarray]:
    """Split identical feature rows as indivisible groups."""
    groups = features.groupby(list(features.columns), dropna=False, sort=False).ngroup().to_numpy()
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct feature rows are required")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                         random_state=seed).split(features, target, groups))
    if len(train) < 2 or len(test) < 2:
        raise ValueError("Train and test partitions must each contain at least two rows")
    assert not set(groups[train]).intersection(groups[test])
    return train, test


def normalize_feature_missing(features: pd.DataFrame) -> pd.DataFrame:
    """Treat blank and whitespace-only feature strings as missing, row by row.

    This fixed normalization has no learned parameters, so applying it before
    grouping and splitting cannot use held-out statistics.
    """
    return features.replace(r"^\s*$", np.nan, regex=True)


def _estimator(model: str, seed: int, config: ExperimentConfig):
    if model == "ridge":
        return Ridge(alpha=1.0)
    if model == "forest":
        return RandomForestRegressor(n_estimators=config.forest_estimators,
                                     min_samples_leaf=3, random_state=seed, n_jobs=1)
    return MLPRegressor(hidden_layer_sizes=(32,), max_iter=config.mlp_max_iter,
                        early_stopping=True, random_state=seed)


def run_experiment(frame: pd.DataFrame, target: str,
                   config: ExperimentConfig | None = None) -> pd.DataFrame:
    """Evaluate every arm on identical, untouched held-out rows per seed.

    Synthetic corruption touches only copied training features. Each cleaning
    transformer and estimator is fitted only on that corrupted training copy.
    The target and held-out features are never used to choose a recipe. The
    adaptive policy is a fixed heuristic, not an optimized winner.
    """
    config = config or ExperimentConfig()
    _validate(frame, target, config)
    features = normalize_feature_missing(frame.drop(columns=[target, *config.drop_columns]))
    y = pd.to_numeric(frame[target]).astype(float)
    # CSVs may contain boolean values; sklearn handles them as categories here.
    features = features.replace([np.inf, -np.inf], np.nan)
    rows: list[dict] = []
    for seed in config.seeds:
        train_index, test_index = make_group_split(features, y, seed, config.test_size)
        x_train = features.iloc[train_index].reset_index(drop=True)
        y_train = y.iloc[train_index].reset_index(drop=True)
        x_test = features.iloc[test_index].reset_index(drop=True)
        y_test = y.iloc[test_index].reset_index(drop=True)
        split_sha = hashlib.sha256(np.asarray(sorted(test_index), dtype=np.int64).tobytes()).hexdigest()
        for mechanism in config.mechanisms:
            for level in config.levels:
                # The seed is independent of strategy/model and stable across runs.
                corruption_seed = seed * 1_000_003 + ("mcar", "mar", "extreme").index(mechanism) * 10_007 + round(level * 10_000)
                condition = corrupt_training_features(x_train, mechanism, level, corruption_seed)
                selected = select_strategy(condition.frame)
                for strategy in config.strategies:
                    actual = selected if strategy == "adaptive" else strategy
                    for model in config.models:
                        pipeline = Pipeline([
                            ("preprocess", build_preprocessor(condition.frame, actual)),
                            ("model", _estimator(model, seed, config)),
                        ])
                        started = time.perf_counter()
                        pipeline.fit(condition.frame, y_train)
                        prediction = pipeline.predict(x_test)
                        elapsed = time.perf_counter() - started
                        rows.append({
                            "seed": seed,
                            "mechanism": mechanism,
                            "level": level,
                            "strategy": strategy,
                            "selected_strategy": actual,
                            "comparison_role": ("original_reference" if level == 0 and strategy == "median"
                                                else "corrupted_minimal_repair" if level > 0 and strategy == "median"
                                                else "cleaning_alternative"),
                            "model": model,
                            "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                            "mae": float(mean_absolute_error(y_test, prediction)),
                            "r2": float(r2_score(y_test, prediction)),
                            "runtime_seconds": float(elapsed),
                            "affected_cells": condition.affected_cells,
                            "mask_sha256": condition.mask_sha256,
                            "split_sha256": split_sha,
                            "train_rows": len(x_train),
                            "test_rows": len(x_test),
                            "missing_fraction": float(condition.frame.isna().to_numpy().mean()),
                        })
    return pd.DataFrame.from_records(rows)
