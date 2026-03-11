"""
Evaluation Agent
Measures model performance on the validation set (development) or test set (final).
Test set is evaluated exactly once — at the final evaluation after tuning.
"""

import json
import pickle
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score,
    matthews_corrcoef, log_loss,
    RocCurveDisplay
)


# ---------------------------------------------------------------------------
# Metric computation  (from evaluation SKILL)
# ---------------------------------------------------------------------------

def _evaluate_classifier(model, X, y, task_type: str) -> dict:
    # Cast to float64 to avoid dtype errors from mixed-type feature DataFrames
    try:
        X = X.astype(np.float64)
    except Exception:
        pass

    # Round to integers — SMOTE can produce continuous float target values
    y      = np.array(y).round().astype(int)
    y_pred = np.array(model.predict(X)).round().astype(int)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            pass

    results = {
        "accuracy":  round(float(accuracy_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "f1":        round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }

    # MCC — works for both binary and multiclass
    try:
        results["mcc"] = round(float(matthews_corrcoef(y, y_pred)), 4)
    except Exception:
        pass

    if y_proba is not None and task_type == "binary_classification":
        y_prob_pos = y_proba[:, 1]
        try:
            results["roc_auc"] = round(float(roc_auc_score(y, y_prob_pos)), 4)
            results["pr_auc"]  = round(float(average_precision_score(y, y_prob_pos)), 4)
        except Exception:
            pass
        try:
            results["log_loss"] = round(float(log_loss(y, y_proba)), 4)
        except Exception:
            pass
        # Specificity (binary only): TN / (TN + FP)
        try:
            cm = confusion_matrix(y, y_pred)
            if cm.shape == (2, 2):
                tn, fp = int(cm[0][0]), int(cm[0][1])
                results["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4)
        except Exception:
            pass

    return results


def _evaluate_regressor(model, X, y) -> dict:
    try:
        X = X.astype(np.float64)
    except Exception:
        pass
    y_pred = model.predict(X)
    result = {
        "mae":  round(float(mean_absolute_error(y, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
        "r2":   round(float(r2_score(y, y_pred)), 4),
    }
    # MAPE — skip rows where actual is zero to avoid division by zero
    try:
        y_arr  = np.array(y, dtype=float)
        yp_arr = np.array(y_pred, dtype=float)
        mask   = y_arr != 0
        if mask.sum() > 0:
            mape = float(np.mean(np.abs((y_arr[mask] - yp_arr[mask]) / y_arr[mask]))) * 100
            result["mape"] = round(mape, 4)
    except Exception:
        pass
    return result


def _evaluate_time_series_model(model, X, y) -> dict:
    """For ARIMA / Prophet: MAE and RMSE only — no R², no raw predictions array."""
    try:
        X = X.astype(np.float64)
    except Exception:
        pass
    y_vals = y.values if hasattr(y, "values") else np.array(y)
    y_pred = model.predict(X)
    return {
        "mae":  round(float(mean_absolute_error(y_vals, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_vals, y_pred))), 4),
    }


def _interpret_metrics(metrics: dict, task_type: str, target_col: str,
                        class_names: list = None, y_range: float = None,
                        is_ts: bool = False) -> list:
    interps = []

    if task_type == "binary_classification":
        acc = metrics.get("accuracy", 0)
        interps.append(
            f"The model correctly predicted the outcome for {acc:.1%} of rows it had not seen before."
        )
        if "roc_auc" in metrics:
            auc = metrics["roc_auc"]
            verdict = ("excellent" if auc >= 0.90
                       else "good" if auc >= 0.80
                       else "fair — there is room for improvement" if auc >= 0.70
                       else "poor — the model is struggling to distinguish between outcomes")
            interps.append(
                f"The model's ability to distinguish between outcomes scores "
                f"{auc:.2f} out of 1.0 — this is {verdict}."
            )
        interps.append(
            f"When the model predicts a positive outcome, it is right "
            f"{metrics.get('precision', 0):.1%} of the time."
        )
        interps.append(
            f"Out of all actual positive cases, the model correctly identified "
            f"{metrics.get('recall', 0):.1%} of them."
        )

    elif task_type in ["multiclass_classification"]:
        acc = metrics.get("accuracy", 0)
        f1  = metrics.get("f1", 0)
        interps.append(f"The model correctly predicted the outcome for {acc:.1%} of rows.")
        interps.append(f"The weighted F1 score is {f1:.2f} (1.0 = perfect).")

    elif is_ts:
        # Time series: MAE only — no R² language
        mae     = metrics.get("mae", 0)
        rmse    = metrics.get("rmse", 0)
        mae_pct = (mae / y_range * 100) if y_range else None
        interps.append(
            f"On average, its forecasts are off by {mae:.4g} "
            f"(in the same units as '{target_col}'). "
            f"RMSE: {rmse:.4g}. "
            f"We'll try to improve this in the tuning step."
        )
        if mae_pct is not None:
            if mae_pct < 5:
                interps.append(
                    f"This is a strong result — the average error is only {mae_pct:.1f}% of the value range."
                )
            elif mae_pct < 15:
                interps.append(
                    f"This is reasonable — the average error is {mae_pct:.1f}% of the value range."
                )
            else:
                interps.append(
                    f"There is room for improvement — the average error is {mae_pct:.1f}% of the value range."
                )

    elif task_type == "regression":
        mae = metrics.get("mae", 0)
        r2  = metrics.get("r2", 0)
        interps.append(
            f"On average, the model's predictions are off by about {mae:.4f} "
            f"(in the same units as '{target_col}')."
        )
        quality = ("strong performance" if r2 > 0.8
                   else "room for improvement" if r2 > 0.5
                   else "the model is struggling to explain the variation")
        interps.append(
            f"The model explains {r2:.1%} of the variation in '{target_col}'. "
            f"This is {quality}."
        )

    return interps


def _performance_verdict(metrics: dict, task_type: str,
                          class_imbalance: bool = False) -> tuple[str, str]:
    if task_type == "binary_classification":
        score = metrics.get("pr_auc" if class_imbalance else "roc_auc", 0)
        if score >= 0.90:
            return "strong", "The model is performing strongly. We recommend proceeding to tuning to see if we can improve it further."
        elif score >= 0.75:
            return "good",   "The model is performing well. Tuning may improve it further."
        elif score >= 0.60:
            return "fair",   "The model is learning but there is meaningful room for improvement. We recommend tuning before deployment."
        else:
            return "poor",   "The model is not performing well enough to be reliable. We recommend reviewing the data and feature engineering steps before proceeding."

    elif task_type in ["multiclass_classification"]:
        score = metrics.get("f1", 0)
        if score >= 0.85:
            return "strong", "Strong classification performance across all outcomes."
        elif score >= 0.65:
            return "good",   "Good performance. Tuning may improve it further."
        else:
            return "fair",   "Performance is moderate. Tuning is recommended."

    elif task_type == "regression":
        r2 = metrics.get("r2", 0)
        if r2 >= 0.85:
            return "strong", "The model explains most of the variation in your target. Strong performance."
        elif r2 >= 0.65:
            return "good",   "The model captures the main patterns. Tuning may improve it further."
        elif r2 >= 0.40:
            return "fair",   "The model captures some patterns but misses others. Tuning is recommended."
        else:
            return "poor",   "The model is not explaining the variation well. Review features and data quality."

    return "unknown", "Unable to determine verdict."


def _performance_verdict_ts(metrics: dict, y_range: float) -> tuple[str, str]:
    """Verdict for time series models based on MAE as a % of the target value range."""
    mae     = metrics.get("mae", 0)
    mae_pct = (mae / y_range * 100) if y_range else 50
    if mae_pct < 5:
        return "strong", "The model is forecasting with strong accuracy. We recommend proceeding to tuning to see if we can improve it further."
    elif mae_pct < 15:
        return "good",   "The model is capturing the main patterns. Tuning may improve accuracy further."
    elif mae_pct < 30:
        return "fair",   "The model is learning but the error is noticeable. Tuning is recommended before deployment."
    else:
        return "poor",   "Forecast errors are large relative to the value range. Review feature engineering and data quality before proceeding."


def _interpret_confusion_matrix(cm: list, class_names: list) -> str:
    if not cm or not class_names or len(class_names) != 2:
        return ""
    try:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        return (
            f"Out of {total} predictions:\n"
            f"  \u2713  {tp} were correctly predicted as '{class_names[1]}'\n"
            f"  \u2713  {tn} were correctly predicted as '{class_names[0]}'\n"
            f"  \u2717  {fp} were predicted as '{class_names[1]}' but were actually '{class_names[0]}' (false alarms)\n"
            f"  \u2717  {fn} were predicted as '{class_names[0]}' but were actually '{class_names[1]}' (missed cases)"
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------


def _plot_roc(model, X, y, output_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        RocCurveDisplay.from_estimator(model, X, y, ax=ax, color="#2E75B6")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set_title(
            "ROC Curve — How well the model separates outcomes\n"
            "Closer to the top-left corner = better performance"
        )
    except Exception:
        ax.text(0.5, 0.5, "ROC curve not available for this model.",
                ha="center", va="center")
    plt.tight_layout()
    path = Path(output_dir) / "roc_curve.png"
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(path)


def _plot_time_series_predictions(y_true, y_pred, output_dir: str) -> str:
    """Line chart: actual vs predicted over time (index)."""
    n = len(y_true)
    idx = list(range(n))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(idx, y_true,  color="#1B3A5C", linewidth=1.5, label="Actual",    alpha=0.9)
    ax.plot(idx, y_pred,  color="#D97706", linewidth=1.5, label="Predicted", alpha=0.8, linestyle="--")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.set_title("Actual vs Predicted over time")
    ax.legend()
    plt.tight_layout()
    path = Path(output_dir) / "time_series_predictions.png"
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(path)


def _plot_residuals(y_true, y_pred, output_dir: str) -> str:
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#2E75B6")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Error (actual minus predicted)")
    axes[0].set_title("Prediction errors — ideally scattered randomly around zero")

    axes[1].scatter(y_true, y_pred, alpha=0.4, color="#2E75B6")
    min_v = min(min(y_true), min(y_pred))
    max_v = max(max(y_true), max(y_pred))
    axes[1].plot([min_v, max_v], [min_v, max_v], "r--")
    axes[1].set_xlabel("Actual value")
    axes[1].set_ylabel("Predicted value")
    axes[1].set_title("Predicted vs actual — dots close to the red line = accurate predictions")
    plt.tight_layout()
    path = Path(output_dir) / "residual_plot.png"
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(path)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    try:
        return _run(session, decisions)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[evaluation] UNHANDLED ERROR:\n{tb}")
        return {
            "stage":                 "evaluation",
            "status":                "failed",
            "plain_english_summary": f"Evaluation failed: {exc}",
            "error_detail":          tb,
        }


def _run(session: dict, decisions: dict) -> dict:
    session_id        = session["session_id"]
    target_col        = session["goal"].get("target_column")
    task_type         = session["goal"].get("task_type", "binary_classification")
    class_imbalance   = session.get("config", {}).get("class_imbalance_detected", False)
    is_ts             = session.get("config", {}).get("is_time_series", False)
    is_final          = decisions.get("is_final_evaluation", False)
    sessions_dir      = Path("sessions")
    session_dir       = sessions_dir / session_id

    output_dir = session_dir / "reports" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the best model (tuned if available, otherwise trained)
    tuned_path  = session_dir / "models" / "tuned_model.pkl"
    best_json   = session_dir / "models" / "best_model.json"
    model_path  = None

    if is_final and tuned_path.exists():
        model_path = tuned_path
    elif best_json.exists():
        with open(best_json) as f:
            model_info = json.load(f)
        model_path = Path(model_info["model_path"])

    if model_path is None or not Path(model_path).exists():
        return {
            "stage":                 "evaluation",
            "status":                "failed",
            "plain_english_summary": "No trained model found. Please run the training stage first."
        }

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    splits_dir = session_dir / "data" / "processed" / "splits"
    split_name = "test" if is_final else "validation"
    suffix     = "test" if is_final else "val"

    try:
        X_eval = pd.read_csv(splits_dir / f"X_{suffix}.csv")
        y_eval = pd.read_csv(splits_dir / f"y_{suffix}.csv").squeeze()
    except FileNotFoundError:
        return {
            "stage":                 "evaluation",
            "status":                "failed",
            "plain_english_summary": f"No {split_name} split found. Please run the splitting stage first."
        }

    # Get class names for classification
    class_names = None
    if "classification" in task_type:
        class_names = sorted([str(c) for c in y_eval.unique()])

    # Compute metrics
    time_series_data = None
    high_r2_warning  = False
    high_auc_warning = False
    is_time_series_model = (
        is_ts or
        model.__class__.__name__ in ("ARIMAWrapper", "ProphetWrapper") or
        "time_series" in task_type or
        "forecast" in task_type
    )
    y_range = None

    if "classification" in task_type:
        metrics = _evaluate_classifier(model, X_eval, y_eval, task_type)
        # Warn on suspiciously high AUC-ROC for imbalanced datasets
        if class_imbalance and metrics.get("roc_auc", 0) > 0.99:
            high_auc_warning = True
        charts = []
        if task_type == "binary_classification":
            charts.append(_plot_roc(model, X_eval, y_eval, str(output_dir)))

    elif is_time_series_model:
        metrics = _evaluate_time_series_model(model, X_eval, y_eval)
        y_pred  = model.predict(X_eval)
        y_true_list = y_eval.tolist()
        y_pred_list = [float(v) for v in y_pred]
        y_range = float(y_eval.max() - y_eval.min()) if len(y_eval) > 0 else None
        charts  = [_plot_time_series_predictions(y_true_list, y_pred_list, str(output_dir))]
        # Data for interactive frontend chart (sample to ≤200 points for performance)
        step = max(1, len(y_true_list) // 200)
        time_series_data = [
            {"index": i, "actual": round(y_true_list[i], 4), "predicted": round(y_pred_list[i], 4)}
            for i in range(0, len(y_true_list), step)
        ]

    else:
        metrics = _evaluate_regressor(model, X_eval, y_eval)
        if metrics.get("r2", 0) > 0.98:
            high_r2_warning = bool(True)
        y_pred  = model.predict(X_eval)
        y_true_list = y_eval.tolist()
        y_pred_list = [float(v) for v in y_pred]
        charts  = [_plot_residuals(y_true_list, y_pred_list, str(output_dir))]

    interpretations = _interpret_metrics(
        metrics, task_type, target_col, class_names,
        y_range=y_range, is_ts=is_time_series_model
    )

    if is_time_series_model:
        verdict, verdict_msg = _performance_verdict_ts(metrics, y_range)
    else:
        verdict, verdict_msg = _performance_verdict(metrics, task_type, class_imbalance)

    cm_text = _interpret_confusion_matrix(
        metrics.get("confusion_matrix", []), class_names
    ) if is_final and "classification" in task_type else None

    summary = " ".join(interpretations[:2]) + f" Our assessment: {verdict_msg}"

    # Primary metric value for config
    if is_time_series_model:
        primary_metric       = "mae"
        primary_metric_value = metrics.get("mae", 0)
    elif task_type == "binary_classification":
        primary_metric       = "pr_auc" if class_imbalance else "roc_auc"
        primary_metric_value = metrics.get(primary_metric, metrics.get("accuracy", 0))
    elif task_type == "regression":
        primary_metric       = "r2"
        primary_metric_value = metrics.get("r2", 0)
    else:
        primary_metric       = "f1"
        primary_metric_value = metrics.get("f1", 0)

    return {
        "stage":                     "evaluation",
        "status":                    "success",
        "split_evaluated":           split_name,
        "metrics":                   metrics,
        "is_time_series":            is_time_series_model,
        "high_r2_warning":           bool(high_r2_warning),
        "high_auc_warning":          bool(high_auc_warning),
        "interpretations":           interpretations,
        "verdict":                   verdict,
        "verdict_message":           verdict_msg,
        "primary_metric_name":       primary_metric,
        "primary_metric_value":      primary_metric_value,
        "confusion_matrix_text":     cm_text,
        "charts":                    charts,
        "class_names":               class_names,
        "time_series_data":          time_series_data,
        "decisions_required":        [],
        "decisions_made":            [],
        "plain_english_summary":     summary,
        "report_section": {
            "stage":   "evaluation",
            "title":   "How Well the Model Performs",
            "summary": summary,
            "decision_made": f"Evaluated on {split_name} set.",
            "why_this_matters": (
                "Evaluation tells us whether the model has actually learned something useful, "
                "or whether it is just guessing. These numbers tell the story of what the model "
                "can and cannot do."
            )
        },
        "config_updates": {
            "primary_metric":       primary_metric,
            "primary_metric_value": primary_metric_value,
            "evaluation_metric":    primary_metric,
            "verdict":              verdict
        }
    }
