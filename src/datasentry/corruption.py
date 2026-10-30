"""Deterministic, training-feature-only corruption generators."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorruptionResult:
    frame: pd.DataFrame
    mask: np.ndarray
    mask_sha256: str

    @property
    def affected_cells(self) -> int:
        return int(self.mask.sum())


def _observed_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame.notna().to_numpy(dtype=bool)


def _choose_mcar(eligible: np.ndarray, level: float, rng: np.random.Generator) -> np.ndarray:
    flat = np.flatnonzero(eligible.ravel())
    count = min(len(flat), round(level * len(flat)))
    mask = np.zeros(eligible.size, dtype=bool)
    if count:
        mask[rng.choice(flat, size=count, replace=False)] = True
    return mask.reshape(eligible.shape)


def _choose_mar(frame: pd.DataFrame, eligible: np.ndarray, level: float,
                rng: np.random.Generator) -> np.ndarray:
    """Mask other columns preferentially for high values of a visible driver.

    The driver is selected from training X, is never itself masked, and no y/test
    information participates. This is a feature-dependent MAR simulation.
    """
    if frame.shape[1] < 2:
        raise ValueError("MAR corruption requires at least two feature columns")
    numeric = frame.select_dtypes(include="number").columns.tolist()
    if numeric:
        variances = frame[numeric].var(numeric_only=True).fillna(0)
        driver = str(variances.idxmax())
        values = pd.to_numeric(frame[driver], errors="coerce")
        values = values.fillna(values.median() if values.notna().any() else 0)
        rank = values.rank(pct=True).to_numpy(dtype=float)
    else:
        driver = str(frame.columns[0])
        values = frame[driver].astype("string").fillna("<NA>")
        rank = values.map(values.value_counts(normalize=True)).rank(pct=True).to_numpy(dtype=float)
    driver_index = frame.columns.get_loc(driver)
    eligible = eligible.copy()
    eligible[:, driver_index] = False
    flat = np.flatnonzero(eligible.ravel())
    count = min(len(flat), round(level * len(flat)))
    mask = np.zeros(eligible.size, dtype=bool)
    if count:
        weights = 0.1 + 0.9 * rank[np.unravel_index(flat, eligible.shape)[0]]
        weights = weights / weights.sum()
        mask[rng.choice(flat, size=count, replace=False, p=weights)] = True
    return mask.reshape(eligible.shape)


def corrupt_training_features(frame: pd.DataFrame, mechanism: str, level: float,
                              seed: int) -> CorruptionResult:
    """Return a copy and its exact synthetic corruption mask.

    ``level`` is the fraction of eligible observed feature cells, except for
    MAR where the always-observed driver is excluded. Nothing is fitted or
    inferred from test rows or the target.
    """
    if not 0 <= level <= 1:
        raise ValueError("Corruption level must be between 0 and 1")
    if mechanism not in {"mcar", "mar", "extreme"}:
        raise ValueError(f"Unsupported corruption mechanism: {mechanism}")
    dirty = frame.copy(deep=True)
    rng = np.random.default_rng(seed)
    eligible = _observed_mask(dirty)
    if mechanism == "mar":
        mask = _choose_mar(dirty, eligible, level, rng)
    elif mechanism == "mcar":
        mask = _choose_mcar(eligible, level, rng)
    else:
        numeric = dirty.select_dtypes(include="number").columns.tolist()
        if not numeric:
            raise ValueError("Extreme-value corruption requires numeric features")
        allowed = np.zeros(eligible.shape, dtype=bool)
        for column in numeric:
            allowed[:, dirty.columns.get_loc(column)] = True
        mask = _choose_mcar(eligible & allowed, level, rng)
    if mechanism in {"mcar", "mar"}:
        dirty = dirty.mask(mask)
    else:
        for column in numeric:
            # Synthetic extremes may be fractional even when CSV values are ints.
            dirty[column] = dirty[column].astype(float)
            affected = mask[:, dirty.columns.get_loc(column)]
            if not affected.any():
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            q1, q3 = values.quantile([0.25, 0.75])
            median = values.median()
            spread = q3 - q1
            if not np.isfinite(spread) or spread <= 0:
                spread = values.std()
            if not np.isfinite(spread) or spread <= 0:
                spread = max(abs(median), 1.0)
            direction = rng.choice([-1.0, 1.0], size=int(affected.sum()))
            dirty.loc[affected, column] = median + direction * 10.0 * spread
    digest = hashlib.sha256(np.packbits(mask.ravel()).tobytes()).hexdigest()
    return CorruptionResult(dirty, mask, digest)
