"""
task_router.py

Validates the user-declared task type against the DatasetManifest and
returns a single, authoritative task_family string.

Rules:
  - numeric target + many unique values               → regression
  - target with 2 distinct values                     → binary_classification
  - target with 3–20 distinct values                  → multiclass_classification
  - time column present + numeric target              → time_series
  - no target column                                  → clustering (not yet supported)
  - mismatch between user-declared task and data      → ValueError with plain English message

This module must run before any split or preprocessing.
After task_router.resolve() returns, the task_family is locked and
must not be re-inferred by any downstream stage.
"""

from __future__ import annotations

from contracts.schemas import DatasetManifest


# ---------------------------------------------------------------------------
# Supported task families
# ---------------------------------------------------------------------------

SUPPORTED_TASK_FAMILIES = {
    "binary_classification",
    "multiclass_classification",
    "regression",
    "time_series",
}

# Aliases the frontend / user might send
_ALIASES: dict[str, str] = {
    "classification":         "binary_classification",
    "binary":                 "binary_classification",
    "binary_class":           "binary_classification",
    "multiclass":             "multiclass_classification",
    "multi_class":            "multiclass_classification",
    "multi_classification":   "multiclass_classification",
    "multiclassification":    "multiclass_classification",
    "reg":                    "regression",
    "linear_regression":      "regression",
    "ts":                     "time_series",
    "time_series_forecast":   "time_series",
    "forecast":               "time_series",
    "timeseries":             "time_series",
    "time-series":            "time_series",
}

# Metrics locked per task family — no stage may override these
_METRICS_MAP: dict[str, tuple[str, list[str]]] = {
    "binary_classification": (
        "roc_auc",
        ["accuracy", "precision", "recall", "f1", "pr_auc", "mcc", "log_loss"],
    ),
    "multiclass_classification": (
        "f1",
        ["accuracy", "precision", "recall", "mcc"],
    ),
    "regression": (
        "r2",
        ["mae", "rmse", "mape"],
    ),
    "time_series": (
        "mae",
        ["rmse"],
    ),
}

# Split strategies locked per task family
_SPLIT_STRATEGY_MAP: dict[str, str] = {
    "binary_classification":   "stratified_holdout",
    "multiclass_classification": "stratified_holdout",
    "regression":              "standard_holdout",
    "time_series":             "time_ordered_holdout",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TaskRoutingError(ValueError):
    """Raised when the declared task type contradicts the data."""


def resolve(
    declared_task_type: str,
    manifest: DatasetManifest,
    target_column: str | None = None,
    time_column: str | None = None,
) -> dict:
    """
    Validate declared_task_type against the manifest.

    Returns a dict with:
      task_family     : str   — authoritative task family
      primary_metric  : str
      secondary_metrics: list[str]
      split_strategy  : str
      warnings        : list[str]

    Raises TaskRoutingError with a plain English message if the declared
    task type clearly contradicts the data.
    """
    declared_norm = _normalise(declared_task_type)
    warnings: list[str] = []

    # Determine the data-driven task family
    data_family = _infer_from_data(manifest, target_column, time_column)

    # Resolve: prefer user declaration if it is consistent with data
    task_family = _reconcile(declared_norm, data_family, manifest, target_column, warnings)

    # Lock metrics and split strategy
    primary_metric, secondary_metrics = _METRICS_MAP[task_family]
    split_strategy = _SPLIT_STRATEGY_MAP[task_family]

    return {
        "task_family": task_family,
        "primary_metric": primary_metric,
        "secondary_metrics": secondary_metrics,
        "split_strategy": split_strategy,
        "warnings": warnings,
    }


def infer_from_manifest(
    manifest: DatasetManifest,
    target_column: str | None = None,
    time_column: str | None = None,
) -> str:
    """
    Return the most likely task_family purely from manifest data.
    Does not raise — used when no user declaration is available.
    """
    return _infer_from_data(manifest, target_column, time_column)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalise(raw: str) -> str:
    """Normalise a user-supplied task type string to a canonical family name."""
    if not raw:
        return ""
    cleaned = raw.strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(cleaned, cleaned)


def _infer_from_data(
    manifest: DatasetManifest,
    target_column: str | None,
    time_column: str | None,
) -> str:
    """Infer task_family from manifest characteristics."""
    # No target → clustering (not yet fully supported, but valid routing result)
    if not target_column:
        return "clustering"

    is_target_numeric = target_column in manifest.numeric_columns
    is_target_binary  = target_column in manifest.binary_columns

    # Count distinct values in target from target_distribution
    n_unique = _target_n_unique(manifest)
    has_time = bool(time_column or manifest.candidate_time_columns)

    # Time series: time column + numeric target + task_hypotheses agrees
    if has_time and is_target_numeric and "time_series" in manifest.task_hypotheses:
        return "time_series"

    # Binary
    if is_target_binary or n_unique == 2:
        return "binary_classification"

    # Multiclass
    if n_unique is not None and 2 < n_unique <= 20:
        if not is_target_numeric:
            return "multiclass_classification"
        # Numeric with few uniques — could be ordinal classification
        return "multiclass_classification"

    # Regression: numeric with many distinct values
    if is_target_numeric:
        return "regression"

    # Categorical target → multiclass
    if target_column in manifest.categorical_columns:
        return "multiclass_classification"

    # Fallback
    return manifest.task_hypotheses[0] if manifest.task_hypotheses else "regression"


def _target_n_unique(manifest: DatasetManifest) -> int | None:
    if manifest.target_distribution is None:
        return None
    return manifest.target_distribution.get("n_unique")


def _reconcile(
    declared: str,
    data_family: str,
    manifest: DatasetManifest,
    target_column: str | None,
    warnings: list[str],
) -> str:
    """
    Reconcile user declaration with data-driven inference.

    Raises TaskRoutingError on hard mismatches.
    Adds warnings for soft mismatches.
    """
    if not declared:
        # No user declaration — use data-driven result
        return data_family

    if declared not in SUPPORTED_TASK_FAMILIES:
        raise TaskRoutingError(
            f"Unknown task type '{declared}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_TASK_FAMILIES))}."
        )

    # Hard mismatch checks
    n_unique = _target_n_unique(manifest)
    is_target_numeric = target_column in manifest.numeric_columns if target_column else False

    if declared == "regression":
        if target_column and target_column in manifest.categorical_columns:
            raise TaskRoutingError(
                f"You declared a regression task, but the target column '{target_column}' "
                f"contains text/categorical values. Regression requires a numeric target. "
                f"Did you mean classification?"
            )
        if n_unique is not None and n_unique == 2:
            warnings.append(
                f"You declared regression, but the target column has only 2 unique values. "
                f"This looks like binary classification. "
                f"Proceeding as regression as requested, but consider switching."
            )

    elif declared == "binary_classification":
        if target_column and is_target_numeric:
            if n_unique is not None and n_unique > 20:
                raise TaskRoutingError(
                    f"You declared binary classification, but the target column "
                    f"'{target_column}' has {n_unique} unique numeric values. "
                    f"This looks like a regression problem. "
                    f"Binary classification requires exactly 2 distinct class values."
                )
        if n_unique is not None and n_unique > 2:
            raise TaskRoutingError(
                f"You declared binary classification, but the target column "
                f"'{target_column}' has {n_unique} distinct values. "
                f"Use multiclass_classification instead."
            )

    elif declared == "multiclass_classification":
        if n_unique is not None and n_unique > 50:
            warnings.append(
                f"The target column has {n_unique} distinct classes — this is a very high "
                f"cardinality classification problem and may perform poorly. "
                f"Consider whether this should be a regression task."
            )

    elif declared == "time_series":
        if not target_column:
            raise TaskRoutingError(
                "Time series forecasting requires a target column. "
                "Please specify the column you want to forecast."
            )
        if not manifest.candidate_time_columns:
            warnings.append(
                "You declared time series, but no datetime or sequential column was detected. "
                "Proceeding as requested — ensure your data is ordered chronologically."
            )

    # Soft mismatch: declared differs from inferred but is not a hard error
    if declared != data_family:
        warnings.append(
            f"The data suggests this might be a '{data_family}' task, "
            f"but you declared '{declared}'. "
            f"Proceeding with your choice."
        )

    return declared
