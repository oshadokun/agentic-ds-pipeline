"""
split_service.py

Clean, strategy-aware splitting logic.

Strategies:
  stratified_holdout   : classification — preserves class distribution in all splits
  standard_holdout     : regression — random split
  time_ordered_holdout : time series — chronological split, no shuffling

Called from splitting.py (the agent) which orchestrates the full sequence:
  split_service.split() → preprocessing_service.fit_transform()

Rules:
  - Split strategy comes from RunSpec, not inferred locally.
  - No fitting or transformation happens here — pure index manipulation.
  - Returns raw (unprocessed) X/y DataFrames.
  - Val/test labels are never mutated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from contracts.schemas import RunSpec


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class SplitResult:
    """Holds the six raw split DataFrames plus metadata."""

    def __init__(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        strategy: str,
        sizes: dict[str, int],
        warnings: list[str],
    ):
        self.X_train = X_train
        self.X_val   = X_val
        self.X_test  = X_test
        self.y_train = y_train
        self.y_val   = y_val
        self.y_test  = y_test
        self.strategy = strategy
        self.sizes    = sizes      # {"train": int, "val": int, "test": int}
        self.warnings = warnings

    def save_raw(self, splits_dir: str | Path) -> None:
        """
        Save raw (pre-preprocessing) splits to parquet files.
        These are overwritten by preprocessing_service with the
        transformed versions.
        """
        p = Path(splits_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.X_train.to_parquet(p / "X_train_raw.parquet", index=False)
        self.X_val.to_parquet(p / "X_val_raw.parquet",     index=False)
        self.X_test.to_parquet(p / "X_test_raw.parquet",   index=False)
        self.y_train.to_frame().to_parquet(p / "y_train.parquet", index=False)
        self.y_val.to_frame().to_parquet(p / "y_val.parquet",     index=False)
        self.y_test.to_frame().to_parquet(p / "y_test.parquet",   index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def split(
    df: pd.DataFrame,
    run_spec: RunSpec,
) -> SplitResult:
    """
    Split df into train / val / test according to run_spec.split_strategy.

    Parameters
    ----------
    df       : the cleaned (structural-only) DataFrame; includes target column
    run_spec : RunSpec — provides target_column, time_column, split_strategy,
               test_size, val_size, random_seed

    Returns
    -------
    SplitResult with X_train, X_val, X_test, y_train, y_val, y_test
    """
    target_col     = run_spec.target_column
    time_col       = run_spec.time_column
    strategy       = run_spec.split_strategy
    test_size      = run_spec.test_size
    val_size       = run_spec.val_size
    random_seed    = run_spec.random_seed
    feature_cols   = run_spec.feature_columns

    if not target_col or target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Feature columns: use RunSpec list if populated, else everything except target
    if feature_cols:
        X_cols = [c for c in feature_cols if c in df.columns]
    else:
        X_cols = [c for c in df.columns if c != target_col]

    X = df[X_cols].copy()
    y = df[target_col].copy()

    warnings: list[str] = []

    if strategy == "stratified_holdout":
        result = _stratified_holdout(X, y, test_size, val_size, random_seed, warnings)

    elif strategy == "time_ordered_holdout":
        result = _time_ordered_holdout(df, X_cols, target_col, time_col, test_size, val_size, warnings)

    else:  # standard_holdout
        result = _standard_holdout(X, y, test_size, val_size, random_seed, warnings)

    # Post-split validation
    _validate(result, run_spec.task_family, warnings)

    return result


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def _stratified_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_seed: int,
    warnings: list[str],
) -> SplitResult:
    """Stratified split preserving class proportions in train / val / test."""
    # Check if stratification is feasible
    min_class_count = y.value_counts().min()
    n_splits = 3  # train, val, test

    stratify = y if min_class_count >= n_splits else None
    if stratify is None:
        warnings.append(
            f"Stratified split requested, but the minority class has only "
            f"{min_class_count} samples (need ≥ {n_splits}). "
            f"Falling back to random split."
        )

    # First: carve out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
        stratify=stratify,
    )

    # Then: split remainder into train / val
    # val_size is relative to the full dataset, so adjust relative to remaining
    adjusted_val_size = val_size / (1.0 - test_size)

    stratify_val = y_temp if (stratify is not None and y_temp.value_counts().min() >= 2) else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_seed,
        stratify=stratify_val,
    )

    sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test, "stratified_holdout", sizes, warnings)


def _standard_holdout(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_seed: int,
    warnings: list[str],
) -> SplitResult:
    """Random split without stratification — for regression."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_seed,
    )
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val_size,
        random_state=random_seed,
    )
    sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test, "standard_holdout", sizes, warnings)


def _time_ordered_holdout(
    df: pd.DataFrame,
    X_cols: list[str],
    target_col: str,
    time_col: str | None,
    test_size: float,
    val_size: float,
    warnings: list[str],
) -> SplitResult:
    """
    Chronological split. No shuffling. No stratification.

    If time_col is provided, sort by it first.
    Otherwise assume the DataFrame is already in chronological order.
    """
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        warnings.append(
            f"Data sorted chronologically by '{time_col}' before splitting."
        )
    else:
        warnings.append(
            "No time column found — assuming DataFrame is already in chronological order."
        )

    n = len(df)
    test_n  = max(1, int(n * test_size))
    val_n   = max(1, int(n * val_size))
    train_n = n - val_n - test_n

    if train_n < 10:
        raise ValueError(
            f"Not enough data for a time-ordered split: "
            f"{n} rows → train={train_n}, val={val_n}, test={test_n}. "
            f"Need at least 10 training rows."
        )

    train_df = df.iloc[:train_n]
    val_df   = df.iloc[train_n: train_n + val_n]
    test_df  = df.iloc[train_n + val_n:]

    X_train = train_df[X_cols].reset_index(drop=True)
    X_val   = val_df[X_cols].reset_index(drop=True)
    X_test  = test_df[X_cols].reset_index(drop=True)
    y_train = train_df[target_col].reset_index(drop=True)
    y_val   = val_df[target_col].reset_index(drop=True)
    y_test  = test_df[target_col].reset_index(drop=True)

    sizes = {"train": len(X_train), "val": len(X_val), "test": len(X_test)}
    return SplitResult(X_train, X_val, X_test, y_train, y_val, y_test, "time_ordered_holdout", sizes, warnings)


# ---------------------------------------------------------------------------
# Post-split validation
# ---------------------------------------------------------------------------

def _validate(result: SplitResult, task_family: str, warnings: list[str]) -> None:
    """Warn on small splits or missing classes after the split."""
    if len(result.X_train) < 30:
        warnings.append(
            f"Training split is very small ({len(result.X_train)} rows). "
            f"Model quality may be limited."
        )
    if len(result.X_val) < 10:
        warnings.append(
            f"Validation split is very small ({len(result.X_val)} rows)."
        )
    if len(result.X_test) < 10:
        warnings.append(
            f"Test split is very small ({len(result.X_test)} rows)."
        )

    if "classification" in task_family:
        train_classes = set(result.y_train.unique())
        val_classes   = set(result.y_val.unique())
        test_classes  = set(result.y_test.unique())

        if val_classes - train_classes:
            warnings.append(
                f"Validation set contains classes not seen in training: "
                f"{val_classes - train_classes}."
            )
        if test_classes - train_classes:
            warnings.append(
                f"Test set contains classes not seen in training: "
                f"{test_classes - train_classes}."
            )
