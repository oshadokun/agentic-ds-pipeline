---
name: evaluation
description: >
  Responsible for thoroughly measuring how well the trained model performs and
  translating every metric into plain English that a non-technical user can
  understand and act on. Always called by the Orchestrator after Training completes.
  Evaluates on the validation set during development and the test set only once at
  the final stage. Selects appropriate metrics based on task type. Produces charts,
  a confusion matrix, and a plain English verdict on whether the model is good enough
  to proceed to tuning and deployment. Never evaluates on the test set more than once.
  Trigger when any of the following are mentioned: "evaluate model", "model performance",
  "accuracy", "ROC-AUC", "confusion matrix", "precision", "recall", "F1", "RMSE",
  "R squared", "model results", "how good is the model", or any request to measure
  or understand model performance.
---

# Evaluation Skill

The Evaluation agent measures how well the model performs and — critically — explains
what those numbers mean in terms the user actually cares about. A ROC-AUC of 0.87
means nothing to a non-technical user. "The model correctly identifies 87 out of
every 100 at-risk customers" means something.

The evaluation agent has one strict rule:

**The test set is evaluated exactly once — at the very end, after all tuning is
complete. Before that, only the validation set is used.**

---

## Responsibilities

1. Select appropriate metrics based on task type
2. Evaluate the model on the validation set during development
3. Produce all relevant charts — confusion matrix, ROC curve, residual plots
4. Translate every metric into a plain English interpretation
5. Give a clear verdict — is the model good enough to proceed?
6. Evaluate on the test set exactly once at the end of the full pipeline
7. Flag any concerns about reliability or bias in the results

---

## Metric Selection by Task Type

```python
def select_metrics(task_type, class_imbalance=False):
    if task_type == "binary_classification":
        primary = "roc_auc" if not class_imbalance else "pr_auc"
        return {
            "primary":    primary,
            "supporting": ["accuracy", "precision", "recall", "f1"],
            "chart":      ["confusion_matrix", "roc_curve", "pr_curve"],
            "plain_english_primary": (
                "ROC-AUC" if primary == "roc_auc"
                else "PR-AUC (Precision-Recall area under curve)"
            )
        }

    elif task_type == "multiclass_classification":
        return {
            "primary":    "f1_weighted",
            "supporting": ["accuracy", "precision_macro", "recall_macro"],
            "chart":      ["confusion_matrix"],
            "plain_english_primary": "Weighted F1 Score"
        }

    elif task_type == "regression":
        return {
            "primary":    "rmse",
            "supporting": ["mae", "r2"],
            "chart":      ["residual_plot", "predicted_vs_actual"],
            "plain_english_primary": "RMSE (Root Mean Squared Error)"
        }
```

---

## Computing Metrics

### Classification
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import numpy as np

def evaluate_classifier(model, X, y, task_type, threshold=0.5):
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None

    results = {
        "accuracy":  round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, average="weighted",
                                           zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, average="weighted",
                                        zero_division=0), 4),
        "f1":        round(f1_score(y, y_pred, average="weighted",
                                    zero_division=0), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist()
    }

    if y_proba is not None and task_type == "binary_classification":
        y_prob_pos = y_proba[:, 1]
        results["roc_auc"] = round(roc_auc_score(y, y_prob_pos), 4)
        results["pr_auc"]  = round(average_precision_score(y, y_prob_pos), 4)

    return results
```

### Regression
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regressor(model, X, y):
    y_pred = model.predict(X)
    rmse   = round(float(np.sqrt(mean_squared_error(y, y_pred))), 4)
    mae    = round(float(mean_absolute_error(y, y_pred)), 4)
    r2     = round(float(r2_score(y, y_pred)), 4)

    return {
        "rmse": rmse,
        "mae":  mae,
        "r2":   r2,
        "predictions": y_pred.tolist()
    }
```

---

## Plain English Interpretations

```python
def interpret_metrics(metrics, task_type, target_col, class_names=None):
    interpretations = []

    if task_type == "binary_classification":
        acc = metrics["accuracy"]
        interpretations.append(
            f"The model correctly predicted the outcome for "
            f"{acc:.1%} of rows it had not seen before."
        )

        if "roc_auc" in metrics:
            auc = metrics["roc_auc"]
            if auc >= 0.90:
                verdict = "excellent"
            elif auc >= 0.80:
                verdict = "good"
            elif auc >= 0.70:
                verdict = "fair — there is room for improvement"
            else:
                verdict = "poor — the model is struggling to distinguish between outcomes"

            interpretations.append(
                f"The model's ability to distinguish between outcomes scores "
                f"{auc:.2f} out of 1.0 — this is {verdict}."
            )

        prec = metrics["precision"]
        rec  = metrics["recall"]
        interpretations.append(
            f"When the model predicts a positive outcome, it is right "
            f"{prec:.1%} of the time (precision)."
        )
        interpretations.append(
            f"Out of all actual positive cases, the model correctly "
            f"identified {rec:.1%} of them (recall)."
        )

    elif task_type == "regression":
        rmse = metrics["rmse"]
        mae  = metrics["mae"]
        r2   = metrics["r2"]

        interpretations.append(
            f"On average, the model's predictions are off by about {mae:.2f} "
            f"(in the same units as '{target_col}')."
        )
        interpretations.append(
            f"The model explains {r2:.1%} of the variation in '{target_col}'. "
            + ("This is strong performance." if r2 > 0.8
               else "There is room for improvement." if r2 > 0.5
               else "The model is struggling to explain the variation.")
        )

    return interpretations
```

---

## Confusion Matrix Interpretation

Always explain the confusion matrix in plain English.

```python
def interpret_confusion_matrix(cm, class_names):
    if len(class_names) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        return (
            f"Out of {total} predictions:\n"
            f"  ✓  {tp} were correctly predicted as '{class_names[1]}'\n"
            f"  ✓  {tn} were correctly predicted as '{class_names[0]}'\n"
            f"  ✗  {fp} were predicted as '{class_names[1]}' but were actually '{class_names[0]}' "
            f"(false alarms)\n"
            f"  ✗  {fn} were predicted as '{class_names[0]}' but were actually '{class_names[1]}' "
            f"(missed cases)"
        )
```

---

## Charts

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
from pathlib import Path

def plot_confusion_matrix(model, X, y, class_names, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=class_names,
        cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix — What the model predicted vs what actually happened")
    plt.tight_layout()
    path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)

def plot_roc_curve(model, X, y, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(model, X, y, ax=ax, color="#2E75B6")
    ax.plot([0, 1], [0, 1], "k--", label="Random guess")
    ax.set_title("ROC Curve — How well the model separates outcomes\n"
                 "Closer to the top-left corner = better performance")
    plt.tight_layout()
    path = Path(output_dir) / "roc_curve.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)

def plot_residuals(y_true, y_pred, output_dir):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.4, color="#2E75B6")
    axes[0].axhline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Predicted value")
    axes[0].set_ylabel("Error (actual minus predicted)")
    axes[0].set_title("Prediction errors — ideally scattered randomly around zero")

    # Predicted vs actual
    axes[1].scatter(y_true, y_pred, alpha=0.4, color="#2E75B6")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--")
    axes[1].set_xlabel("Actual value")
    axes[1].set_ylabel("Predicted value")
    axes[1].set_title("Predicted vs actual — dots close to the red line = accurate predictions")

    plt.tight_layout()
    path = Path(output_dir) / "residual_plot.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)
```

---

## Performance Verdict

Always give the user a clear verdict before proceeding.

```python
def performance_verdict(metrics, task_type, class_imbalance=False):
    if task_type == "binary_classification":
        score = metrics.get("pr_auc" if class_imbalance else "roc_auc", 0)
        if score >= 0.90:
            return "strong",  "The model is performing strongly. We recommend proceeding to tuning to see if we can improve it further."
        elif score >= 0.75:
            return "good",    "The model is performing well. Tuning may improve it further."
        elif score >= 0.60:
            return "fair",    "The model is learning but there is meaningful room for improvement. We recommend tuning before deployment."
        else:
            return "poor",    "The model is not performing well enough to be reliable. We recommend reviewing the data and feature engineering steps before proceeding."

    elif task_type == "regression":
        r2 = metrics.get("r2", 0)
        if r2 >= 0.85:
            return "strong",  "The model explains most of the variation in your target. Strong performance."
        elif r2 >= 0.65:
            return "good",    "The model captures the main patterns in your data. Tuning may improve it further."
        elif r2 >= 0.40:
            return "fair",    "The model captures some patterns but misses others. Tuning is recommended."
        else:
            return "poor",    "The model is not explaining the variation well. Review features and data quality before proceeding."
```

---

## Running the Full Evaluation Pipeline

```python
def run_evaluation(model, X_val, X_test, y_val, y_test,
                   task_type, target_col, class_names,
                   class_imbalance, session_id,
                   is_final_evaluation=False):

    output_dir = f"sessions/{session_id}/reports/evaluation"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Use validation set during development, test set only for final evaluation
    X_eval = X_test if is_final_evaluation else X_val
    y_eval = y_test if is_final_evaluation else y_val
    split_name = "test" if is_final_evaluation else "validation"

    # Compute metrics
    if "classification" in task_type:
        metrics = evaluate_classifier(model, X_eval, y_eval, task_type)
        charts  = [
            plot_confusion_matrix(model, X_eval, y_eval, class_names, output_dir),
            plot_roc_curve(model, X_eval, y_eval, output_dir)
        ]
    else:
        metrics = evaluate_regressor(model, X_eval, y_eval)
        charts  = [plot_residuals(y_eval,
                                   model.predict(X_eval), output_dir)]

    interpretations = interpret_metrics(metrics, task_type, target_col, class_names)
    verdict, verdict_msg = performance_verdict(metrics, task_type, class_imbalance)

    if is_final_evaluation:
        cm_text = interpret_confusion_matrix(
            metrics.get("confusion_matrix", []), class_names
        ) if "classification" in task_type else None
    else:
        cm_text = None

    return {
        "metrics":          metrics,
        "interpretations":  interpretations,
        "verdict":          verdict,
        "verdict_message":  verdict_msg,
        "charts":           charts,
        "split_evaluated":  split_name,
        "confusion_matrix_text": cm_text
    }
```

---

## Output Written to Session

**Charts:**
`sessions/{session_id}/reports/evaluation/`

**Result JSON:**
`sessions/{session_id}/outputs/evaluation/result.json`

```json
{
  "stage": "evaluation",
  "status": "success",
  "split_evaluated": "validation",
  "metrics": { "roc_auc": 0.891, "accuracy": 0.847, "precision": 0.831, "recall": 0.862, "f1": 0.846 },
  "verdict": "good",
  "verdict_message": "The model is performing well. Tuning may improve it further.",
  "interpretations": [ "...", "..." ],
  "charts": [ "...confusion_matrix.png", "...roc_curve.png" ],
  "decisions_required": [],
  "plain_english_summary": "The model correctly predicted the outcome for 84.7% of cases it had not seen before. Its ability to distinguish between outcomes scores 0.89 out of 1.0 — this is good performance. We recommend proceeding to tuning.",
  "report_section": {
    "stage": "evaluation",
    "title": "How Well the Model Performs",
    "summary": "...",
    "decision_made": "...",
    "why_this_matters": "Evaluation tells us whether the model has actually learned something useful, or whether it is just guessing. These numbers tell the story of what the model can and cannot do."
  },
  "config_updates": {
    "primary_metric": "roc_auc",
    "primary_metric_value": 0.891,
    "verdict": "good"
  }
}
```

---

## What to Tell the User

Present results as a story, not a table of numbers:

"Here is how your model performed on data it had never seen before:

{plain_english_interpretation_1}
{plain_english_interpretation_2}

[Show confusion matrix with interpretation]

[Show ROC curve with caption]

Our assessment: {verdict_message}

{proceed_message}"

Always end with a clear recommendation — proceed to tuning, go back and review
features, or the model is ready to deploy.

**Never present raw metric names without explanation.** Always pair every number
with a plain English sentence about what it means in the context of the user's goal.

---

## Reference Files

- `references/metrics-guide.md` — full guide to every metric, when to use it, and how to interpret it
