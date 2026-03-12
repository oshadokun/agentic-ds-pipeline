"""
evaluation_service.py

Single function that produces a frozen EvaluationPayload.

Rules:
  - Called from evaluation.py and tuning.py only.
  - All metrics computed ONCE from one frozen array set.
  - No y.round() or y.astype(int) — labels from the split are always correct dtype.
  - A continuous classification target is a hard error, not a silent coercion.
  - No chart generation here — charts consume y_true/y_pred from the payload paths.
  - Task-aware: correct metric set per task_family, no cross-contamination.
"""

from __future__ import annotations

import datetime
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)

from contracts.schemas import RunSpec, EvaluationPayload


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    run_spec: RunSpec,
    split_name: str,
    eval_dir: str | Path,
    class_imbalance: bool = False,
) -> EvaluationPayload:
    """
    Evaluate model on (X, y) and return a frozen EvaluationPayload.

    Parameters
    ----------
    model        : fitted sklearn-compatible model
    X            : preprocessed features (already transformed by preprocessor)
    y            : true labels — NEVER mutated here
    run_spec     : RunSpec for task_family, primary_metric, etc.
    split_name   : "val" or "test"
    eval_dir     : directory for saving prediction arrays (parquet)
    class_imbalance : hint for binary classification metric selection

    Returns
    -------
    EvaluationPayload — saved to eval_dir/evaluation_payload_{split_name}.json
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    task_family  = run_spec.task_family
    model_id     = run_spec.selected_model_id or "unknown"
    primary_met  = run_spec.primary_metric

    # ── Cast features to float64 ──────────────────────────────────────────
    X_arr = _to_float(X)

    _CLASSIFICATION_FAMILIES = {"binary_classification", "multiclass_classification"}

    # ── Guard: classification labels must not be continuous floats ────────
    if task_family in _CLASSIFICATION_FAMILIES:
        _assert_integer_labels(y, split_name)

    # ── Compute predictions ───────────────────────────────────────────────
    y_true = np.array(y)
    y_pred = np.array(model.predict(X_arr))

    y_score: np.ndarray | None = None
    if task_family in _CLASSIFICATION_FAMILIES and hasattr(model, "predict_proba"):
        try:
            y_score = model.predict_proba(X_arr)
        except Exception:
            pass

    # ── Save raw arrays ───────────────────────────────────────────────────
    y_true_path = _save_array(y_true,  eval_dir / f"y_true_{split_name}.parquet",  "y_true")
    y_pred_path = _save_array(y_pred,  eval_dir / f"y_pred_{split_name}.parquet",  "y_pred")
    y_score_path: str | None = None
    if y_score is not None:
        y_score_path = str(
            _save_matrix(y_score, eval_dir / f"y_score_{split_name}.parquet")
        )

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics: dict[str, float]
    cm: list[list[int]] | None
    label_order: list | None
    class_mapping: dict | None
    threshold_used: float | None

    if task_family == "binary_classification":
        metrics, cm, label_order, class_mapping, threshold_used = _eval_binary(
            y_true, y_pred, y_score, class_imbalance
        )
    elif task_family == "multiclass_classification":
        metrics, cm, label_order, class_mapping, threshold_used = _eval_multiclass(
            y_true, y_pred, y_score
        )
    elif task_family == "time_series":
        metrics, cm, label_order, class_mapping, threshold_used = _eval_timeseries(
            y_true, y_pred
        )
        # Override primary metric for time series
        primary_met = "mae"
    else:  # regression
        metrics, cm, label_order, class_mapping, threshold_used = _eval_regression(
            y_true, y_pred
        )

    # ── Verdict ───────────────────────────────────────────────────────────
    verdict, verdict_msg = _verdict(metrics, task_family, primary_met, class_imbalance)

    # ── Assemble payload ──────────────────────────────────────────────────
    payload = EvaluationPayload(
        task_family=task_family,
        model_id=model_id,
        split_name=split_name,
        y_true_path=str(y_true_path),
        y_pred_path=str(y_pred_path),
        y_score_path=y_score_path,
        threshold_used=threshold_used,
        label_order=label_order,
        class_mapping=class_mapping,
        confusion_matrix=cm,
        metrics=metrics,
        primary_metric=primary_met,
        verdict=verdict,
        verdict_message=verdict_msg,
        timestamp=datetime.datetime.utcnow().isoformat(),
        run_id=run_spec.run_id,
    )

    # Save payload
    payload.save(eval_dir / f"evaluation_payload_{split_name}.json")

    return payload


# ---------------------------------------------------------------------------
# Task-specific evaluation functions
# ---------------------------------------------------------------------------

def _eval_binary(
    y_true, y_pred, y_score, class_imbalance: bool
) -> tuple[dict, list, list, dict, float]:
    """Binary classification metrics."""
    labels = sorted(np.unique(y_true).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    threshold_used = 0.5

    metrics: dict[str, float] = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="binary", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="binary", zero_division=0)), 4),
    }

    try:
        metrics["mcc"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    except Exception:
        pass

    if y_score is not None and y_score.ndim == 2 and y_score.shape[1] >= 2:
        y_prob_pos = y_score[:, 1]
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob_pos)), 4)
        except Exception:
            pass
        try:
            metrics["pr_auc"] = round(float(average_precision_score(y_true, y_prob_pos)), 4)
        except Exception:
            pass
        try:
            metrics["log_loss"] = round(float(log_loss(y_true, y_score)), 4)
        except Exception:
            pass
        # Specificity
        if len(cm) == 2:
            tn = cm[0][0]; fp = cm[0][1]
            if (tn + fp) > 0:
                metrics["specificity"] = round(float(tn / (tn + fp)), 4)

    label_order   = [str(l) for l in labels]
    class_mapping = {str(i): str(l) for i, l in enumerate(labels)}
    return metrics, cm, label_order, class_mapping, threshold_used


def _eval_multiclass(
    y_true, y_pred, y_score
) -> tuple[dict, list, list, dict, float | None]:
    labels = sorted(np.unique(y_true).tolist())
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    metrics: dict[str, float] = {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    try:
        metrics["mcc"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    except Exception:
        pass

    label_order   = [str(l) for l in labels]
    class_mapping = {str(i): str(l) for i, l in enumerate(labels)}
    return metrics, cm, label_order, class_mapping, None


def _eval_regression(
    y_true, y_pred
) -> tuple[dict, None, None, None, None]:
    metrics: dict[str, float] = {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2":   round(float(r2_score(y_true, y_pred)), 4),
    }
    # MAPE — skip zero-target rows
    try:
        mask = np.array(y_true) != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100
            metrics["mape"] = round(mape, 4)
    except Exception:
        pass
    return metrics, None, None, None, None


def _eval_timeseries(
    y_true, y_pred
) -> tuple[dict, None, None, None, None]:
    """Time series: MAE and RMSE only — R2 is misleading for sequential forecast models."""
    metrics: dict[str, float] = {
        "mae":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
    }
    return metrics, None, None, None, None


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def _verdict(
    metrics: dict,
    task_family: str,
    primary_metric: str,
    class_imbalance: bool,
) -> tuple[str, str]:
    if task_family == "binary_classification":
        # For imbalanced data prefer PR-AUC; otherwise ROC-AUC
        score_key = "pr_auc" if class_imbalance else "roc_auc"
        score = metrics.get(score_key, metrics.get("f1", 0))
        if score >= 0.90:
            return "strong", "The model is performing strongly. Proceed to tuning."
        elif score >= 0.75:
            return "good",   "Good performance. Tuning may improve it further."
        elif score >= 0.60:
            return "fair",   "Learning but room for improvement. Tuning recommended."
        else:
            return "poor",   "Not performing well enough. Review data and features."

    elif task_family == "multiclass_classification":
        score = metrics.get("f1", 0)
        if score >= 0.85:
            return "strong", "Strong classification performance."
        elif score >= 0.65:
            return "good",   "Good. Tuning may improve further."
        else:
            return "fair",   "Moderate performance. Tuning recommended."

    elif task_family == "regression":
        r2 = metrics.get("r2", 0)
        if r2 >= 0.85:
            return "strong", "Explains most variation in the target. Strong."
        elif r2 >= 0.65:
            return "good",   "Captures main patterns. Tuning may improve."
        elif r2 >= 0.40:
            return "fair",   "Some patterns captured. Tuning recommended."
        else:
            return "poor",   "Not explaining variation well. Review features."

    elif task_family == "time_series":
        return "unknown", "Evaluated on MAE/RMSE — compare against baseline."

    return "unknown", "Unable to determine verdict."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_integer_labels(y: pd.Series, split_name: str) -> None:
    """
    Raise ValueError if classification labels are continuous floats.
    This is a hard error — not silently rounded.
    """
    y_arr = np.array(y)
    if np.issubdtype(y_arr.dtype, np.floating):
        unique_vals = np.unique(y_arr)
        non_integer = unique_vals[unique_vals != np.round(unique_vals)]
        if len(non_integer) > 0:
            raise ValueError(
                f"Classification labels in '{split_name}' split contain non-integer float values: "
                f"{non_integer[:5].tolist()}. "
                "This is a data pipeline error — labels should be integers or strings. "
                "Check that SMOTE was not applied to the label column and that the "
                "target column was not accidentally scaled."
            )


def _to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Cast feature DataFrame to float64."""
    try:
        return X.astype(np.float64)
    except Exception:
        result = X.copy()
        for col in result.columns:
            try:
                result[col] = result[col].astype(np.float64)
            except Exception:
                pass
        return result


def _save_array(arr: np.ndarray, path: Path, col_name: str) -> Path:
    """Save a 1D array as a single-column parquet file."""
    pd.DataFrame({col_name: arr}).to_parquet(path, index=False)
    return path


def _save_matrix(arr: np.ndarray, path: Path) -> Path:
    """Save a 2D array (e.g. predict_proba output) as parquet."""
    cols = [f"class_{i}" for i in range(arr.shape[1])]
    pd.DataFrame(arr, columns=cols).to_parquet(path, index=False)
    return path
