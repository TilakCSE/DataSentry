"""Small, leakage-safe pilot for the paper; not the final benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SEEDS = (11, 23, 37, 53, 71)
LEVELS = (0.0, 0.05, 0.15, 0.30)


def make_pipeline(numeric: list[str], categorical: list[str], recipe: str, model: str, seed: int):
    if recipe == "knn":
        numeric_steps = [("scale", StandardScaler()), ("impute", KNNImputer(n_neighbors=5))]
    else:
        numeric_steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scale", StandardScaler())]
    categorical_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                         ("encode", OneHotEncoder(handle_unknown="ignore"))]
    prep = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline(categorical_steps), categorical),
    ])
    estimator = (Ridge(alpha=1.0) if model == "ridge" else
                 RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                       random_state=seed, n_jobs=1))
    return Pipeline([("preprocess", prep), ("model", estimator)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.csv)
    assert frame.shape == (649, 33), frame.shape
    # G1/G2 are later-period grades. Absences are excluded for an early forecast.
    features = frame.drop(columns=["G3", "G1", "G2", "absences"])
    target = frame["G3"]
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    groups = features.groupby(list(features.columns), dropna=False).ngroup().to_numpy()
    rows = []
    for seed in SEEDS:
        train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                                     random_state=seed).split(features, target, groups))
        x_train = features.iloc[train_idx].reset_index(drop=True)
        y_train = target.iloc[train_idx].reset_index(drop=True)
        x_test = features.iloc[test_idx].reset_index(drop=True)
        y_test = target.iloc[test_idx].reset_index(drop=True)
        assert set(train_idx).isdisjoint(test_idx)
        assert set(groups[train_idx]).isdisjoint(groups[test_idx])
        for level in LEVELS:
            dirty = x_train.copy()
            if level:
                rng = np.random.default_rng(seed * 1000 + int(level * 100))
                mask = rng.random(dirty.shape) < level
                for j, column in enumerate(dirty.columns):
                    dirty.loc[mask[:, j], column] = np.nan
            else:
                mask = np.zeros(dirty.shape, dtype=bool)
            observed_missing = dirty.isna().to_numpy().mean()
            # Pilot rule, frozen before running. A final policy needs broader pilots.
            adaptive_recipe = ("knn" if len(dirty) >= 400 and
                               0.10 <= observed_missing <= 0.25 else "median")
            for recipe in ("median", "knn", "adaptive"):
                chosen = adaptive_recipe if recipe == "adaptive" else recipe
                for model in ("ridge", "forest"):
                    pipe = make_pipeline(numeric, categorical, chosen, model, seed)
                    pipe.fit(dirty, y_train)
                    prediction = pipe.predict(x_test)
                    rows.append({
                        "seed": seed, "corruption": "MCAR_train_only",
                        "missing_level": level, "masked_cells": int(mask.sum()),
                        "recipe": recipe, "chosen_recipe": chosen, "model": model,
                        "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                        "mae": float(mean_absolute_error(y_test, prediction)),
                        "r2": float(r2_score(y_test, prediction)),
                        "train_rows": len(x_train), "test_rows": len(x_test),
                        "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(json.dumps({"rows": len(rows), "output": str(args.output),
                      "mean_rmse": pd.DataFrame(rows).groupby(["missing_level", "recipe", "model"])
                      ["rmse"].mean().round(3).to_dict().__str__()}, indent=2))


if __name__ == "__main__":
    main()
