"""Deterministic, descriptive quality checks for tabular CSV data.

Flags are observations, not claims that a value is wrong. In particular, an
IQR outlier may be a valid rare observation. This module does not clean data or
fit any statistics that should later be used to transform a held-out split.
"""

from __future__ import annotations

from collections import defaultdict
import re
import unicodedata

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype


def _percent(count: int, denominator: int) -> float:
    return round(100.0 * count / denominator, 2) if denominator else 0.0


def _canonical(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", text.strip()).casefold()


def profile_dataframe(df: pd.DataFrame, target: str | None = None) -> dict:
    """Return a JSON-compatible quality profile without changing ``df``.

    Missing counts include nulls and whitespace-only strings. Duplicate counts
    mean exact duplicate *rows* (beyond their first occurrence), including the
    target if present. Numeric flags use Tukey's 1.5 IQR rule on observed finite
    values. String variants differ only after Unicode normalization, trimming,
    whitespace collapse and case folding.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("profile_dataframe expects a pandas DataFrame")
    if target is not None and target not in df.columns:
        raise ValueError(f"Target column not found: {target}")
    if not df.columns.is_unique:
        raise ValueError("Column names must be unique for an unambiguous profile")

    rows, width = df.shape
    profiles: list[dict] = []
    outliers: list[dict] = []
    patterns: list[dict] = []
    missing_cells = 0
    notes = [
        "IQR flags identify potential numeric extremes, not verified errors.",
        "Exact duplicate rows may be legitimate repeated observations.",
        "String pattern flags cover case, surrounding/internal whitespace, and Unicode compatibility variants only.",
    ]

    for column in df.columns:
        series = df[column]
        blank = series.map(lambda value: isinstance(value, str) and not value.strip())
        blank_count = int((blank & ~series.isna()).sum())
        missing_count = int(series.isna().sum()) + blank_count
        missing_cells += missing_count
        if is_bool_dtype(series):
            kind = "categorical"
        elif is_numeric_dtype(series):
            kind = "numeric"
        elif is_datetime64_any_dtype(series):
            kind = "datetime"
        else:
            kind = "categorical"
        profiles.append(
            {
                "name": str(column),
                "dtype": str(series.dtype),
                "kind": kind,
                "missing_count": missing_count,
                "missing_pct": _percent(missing_count, rows),
                "blank_string_count": blank_count,
                "unique_count": int(series.nunique(dropna=True)),
                "is_target": bool(column == target),
            }
        )

        if kind == "numeric":
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            if len(finite) >= 4:
                q1, q3 = finite.quantile([0.25, 0.75])
                iqr = float(q3 - q1)
                if iqr > 0:
                    lower = float(q1 - 1.5 * iqr)
                    upper = float(q3 + 1.5 * iqr)
                    count = int(((finite < lower) | (finite > upper)).sum())
                    if count:
                        outliers.append(
                            {
                                "column": str(column),
                                "count": count,
                                "lower_bound": lower,
                                "upper_bound": upper,
                                "method": "Tukey 1.5 IQR",
                                "is_target": bool(column == target),
                            }
                        )
            continue

        if kind != "categorical" or is_bool_dtype(series):
            continue
        strings = series[series.map(lambda value: isinstance(value, str) and bool(value.strip()))]
        if strings.nunique() > 10000:
            notes.append(f"String variant check skipped for {column}: over 10,000 distinct strings.")
            continue
        counts = strings.value_counts()
        groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for value, count in counts.items():
            groups[_canonical(value)].append((value, int(count)))
        for canonical, variants in groups.items():
            if len(variants) < 2:
                continue
            ordered = sorted(variants, key=lambda item: (-item[1], item[0]))
            patterns.append(
                {
                    "column": str(column),
                    "canonical": canonical,
                    "variants": [{"value": value, "count": count} for value, count in ordered],
                    "affected_rows": sum(count for _, count in ordered[1:]),
                }
            )

    duplicates = int(df.duplicated().sum()) if rows and width else 0
    return {
        "overview": {
            "rows": rows,
            "columns": width,
            "missing_cells": missing_cells,
            "missing_pct": _percent(missing_cells, rows * width),
            "duplicate_rows": duplicates,
            "duplicate_pct": _percent(duplicates, rows),
        },
        "columns": profiles,
        "outliers": outliers,
        "inconsistent_patterns": patterns,
        "notes": notes,
    }
