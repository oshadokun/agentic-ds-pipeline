"""
manifest_builder.py

Inspects the raw uploaded dataset and returns a DatasetManifest.

Called during the validation stage — before any split or preprocessing.
The manifest is saved to sessions/{id}/artifacts/manifest.json and is
used by task_router and pipeline_compiler.

This module is READ-ONLY with respect to the dataset — it never modifies
any row or column.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.schemas import DatasetManifest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HIGH_CARDINALITY_THRESHOLD = 0.95   # fraction of unique values relative to row count
_DATETIME_NAME_HINTS = {
    "date", "time", "timestamp", "datetime", "dt", "period",
    "week", "quarter", "year", "month", "day", "hour", "minute"
}
_DATETIME_VALUE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}",           # YYYY-MM-DD
    r"^\d{1,2}/\d{1,2}/\d{4}",       # M/D/YYYY
    r"^\d{1,2}-\d{1,2}-\d{4}",       # D-M-YYYY
    r"^\d{4}/\d{2}/\d{2}",           # YYYY/MM/DD
]
_LEAKAGE_HINTS = {
    "id", "_id", "uuid", "guid", "row_number", "index",
    "record_id", "sample_id", "customer_id", "user_id",
    "transaction_id", "order_id", "case_id",
}
_ID_LIKE_PATTERN = re.compile(r"(_id|_key|_num|_no|_code|^id$|^uuid$)", re.I)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_column(col: str, series: pd.Series) -> str:
    """Return one of: 'numeric', 'categorical', 'datetime', 'binary', 'text'."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({0, 1, 0.0, 1.0, True, False}):
            return "binary"
        return "numeric"

    if series.dtype == object:
        # Try to detect datetime-like string columns
        col_norm = col.lower().replace(" ", "_").replace("-", "_")
        if any(hint in col_norm for hint in _DATETIME_NAME_HINTS):
            sample = series.dropna().astype(str).head(50)
            if any(sample.str.match(p).mean() > 0.7 for p in _DATETIME_VALUE_PATTERNS):
                return "datetime"

        # High unique ratio → probably a text or ID column
        n_unique = series.nunique()
        n_total = len(series.dropna())
        if n_total > 0 and n_unique / n_total > _HIGH_CARDINALITY_THRESHOLD:
            return "text"  # will be caught as high-cardinality too

        # Short strings → categorical
        try:
            median_len = series.dropna().astype(str).str.len().median()
            if median_len > 50:
                return "text"
        except Exception:
            pass
        return "categorical"

    return "categorical"


def _detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that look like datetime, including string columns with date patterns."""
    result = []
    for col in df.columns:
        kind = _classify_column(col, df[col])
        if kind == "datetime":
            result.append(col)
    return result


def _detect_leakage_candidates(df: pd.DataFrame, target_col: str | None) -> list[str]:
    """
    Flag columns that are likely identifiers or post-event labels
    that should not be used as features.
    """
    suspects = []
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower().strip()
        # Exact match or pattern match
        if col_lower in _LEAKAGE_HINTS or _ID_LIKE_PATTERN.search(col_lower):
            suspects.append(col)
            continue
        # Integer-typed column with near-unique values → likely a row-level ID
        # (exclude continuous floats: those can be near-unique by nature)
        if pd.api.types.is_integer_dtype(df[col]):
            n_unique = df[col].nunique()
            n_rows = len(df)
            if n_rows > 0 and n_unique / n_rows > 0.98:
                suspects.append(col)
    return suspects


def _analyse_target(series: pd.Series, task_hint: str) -> dict:
    """Return distribution stats appropriate for the guessed task type."""
    if series is None:
        return {}
    result: dict = {"dtype": str(series.dtype), "null_count": int(series.isna().sum())}

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.nunique()
        result.update({
            "n_unique": int(n_unique),
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        })
        if n_unique <= 20:
            vc = series.value_counts(normalize=True)
            result["class_distribution"] = {str(k): round(float(v), 4) for k, v in vc.items()}
    else:
        vc = series.value_counts(normalize=True)
        result["n_unique"] = int(series.nunique())
        result["class_distribution"] = {str(k): round(float(v), 4) for k, v in vc.head(20).items()}

    return result


def _task_hypotheses(
    df: pd.DataFrame,
    target_col: str | None,
    datetime_cols: list[str],
) -> list[str]:
    """
    Return ordered list of task_family hypotheses based on data characteristics.
    Most likely first.
    """
    if target_col is None or target_col not in df.columns:
        return ["clustering"]

    target = df[target_col].dropna()
    n_unique = target.nunique()
    is_numeric_target = pd.api.types.is_numeric_dtype(target)
    has_time_col = len(datetime_cols) > 0

    hypotheses = []

    # Time series: time column present + numeric target
    if has_time_col and is_numeric_target:
        hypotheses.append("time_series")

    if n_unique == 2:
        hypotheses.append("binary_classification")
    elif 2 < n_unique <= 20:
        hypotheses.append("multiclass_classification")
    elif is_numeric_target and n_unique > 20:
        hypotheses.append("regression")
    elif not is_numeric_target:
        hypotheses.append("multiclass_classification")

    # Fallback
    if not hypotheses:
        hypotheses.append("regression" if is_numeric_target else "multiclass_classification")

    return list(dict.fromkeys(hypotheses))  # preserve order, deduplicate


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build(
    df: pd.DataFrame,
    dataset_id: str | None = None,
    target_column: str | None = None,
) -> DatasetManifest:
    """
    Inspect a raw DataFrame and return a DatasetManifest.

    Parameters
    ----------
    df            : raw DataFrame (already loaded, not modified)
    dataset_id    : optional identifier (defaults to a new UUID)
    target_column : user-declared target column, if known

    Returns
    -------
    DatasetManifest with all fields populated.
    """
    if dataset_id is None:
        dataset_id = str(uuid.uuid4())

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    datetime_cols: list[str] = []
    binary_cols: list[str] = []
    text_cols: list[str] = []

    for col in df.columns:
        kind = _classify_column(col, df[col])
        if kind == "numeric":
            numeric_cols.append(col)
        elif kind == "categorical":
            categorical_cols.append(col)
        elif kind == "datetime":
            datetime_cols.append(col)
        elif kind == "binary":
            binary_cols.append(col)
        else:
            text_cols.append(col)

    # Missingness
    missingness = {
        col: round(float(df[col].isna().mean()), 4)
        for col in df.columns
        if df[col].isna().any()
    }

    # Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # Constant columns
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]

    # High-cardinality columns (excluding target)
    high_card = []
    n_rows = len(df)
    for col in df.columns:
        if col == target_column:
            continue
        n_unique = df[col].nunique()
        if n_rows > 0 and n_unique / n_rows > _HIGH_CARDINALITY_THRESHOLD:
            high_card.append(col)

    # Leakage candidates
    leakage_candidates = _detect_leakage_candidates(df, target_column)

    # Candidate target columns
    candidate_targets = []
    for col in df.columns:
        if col in leakage_candidates:
            continue
        series = df[col].dropna()
        n_u = series.nunique()
        if n_u >= 2 and col not in datetime_cols:
            candidate_targets.append(col)

    # Candidate time columns
    candidate_time = list(datetime_cols)
    # Also look for integer sequence columns that might be temporal
    for col in numeric_cols:
        if col == target_column:
            continue
        col_lower = col.lower()
        if any(hint in col_lower for hint in ("year", "month", "day", "week", "quarter", "period", "date")):
            if col not in candidate_time:
                candidate_time.append(col)

    # Task hypotheses
    hypotheses = _task_hypotheses(df, target_column, datetime_cols + candidate_time)

    # Target distribution
    target_dist = None
    if target_column and target_column in df.columns:
        target_dist = _analyse_target(df[target_column], hypotheses[0] if hypotheses else "")

    # Warnings
    warnings: list[str] = []
    if n_rows < 100:
        warnings.append(f"Dataset has only {n_rows} rows — model quality may be limited.")
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate rows detected.")
    if constant_cols:
        warnings.append(f"Constant columns (zero variance): {', '.join(constant_cols)}.")
    if leakage_candidates:
        warnings.append(
            f"Possible ID/leakage columns detected: {', '.join(leakage_candidates)}. "
            "Review before training."
        )
    if target_column and target_column in missingness:
        pct = missingness[target_column] * 100
        warnings.append(f"Target column '{target_column}' has {pct:.1f}% missing values.")

    return DatasetManifest(
        dataset_id=dataset_id,
        row_count=n_rows,
        column_count=len(df.columns),
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        binary_columns=binary_cols,
        text_columns=text_cols,
        candidate_target_columns=candidate_targets,
        candidate_time_columns=candidate_time,
        missingness_summary=missingness,
        duplicate_row_count=duplicate_count,
        constant_columns=constant_cols,
        high_cardinality_columns=high_card,
        possible_leakage_columns=leakage_candidates,
        target_distribution=target_dist,
        task_hypotheses=hypotheses,
        warnings=warnings,
    )


def build_from_csv(
    csv_path: str | Path,
    target_column: str | None = None,
    dataset_id: str | None = None,
) -> DatasetManifest:
    """Convenience wrapper: load CSV and build manifest."""
    df = pd.read_csv(csv_path, low_memory=False)
    return build(df, dataset_id=dataset_id, target_column=target_column)
