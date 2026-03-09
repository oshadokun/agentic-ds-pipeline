---
name: monitoring
description: >
  Responsible for detecting when a deployed model's performance or input data
  starts to degrade over time. Always called by the Orchestrator after Deployment
  succeeds and periodically thereafter. Monitors for data drift, prediction drift,
  and performance decay. Generates plain English alerts and a monitoring dashboard.
  Recommends retraining when degradation is detected. Saves all monitoring reports
  to the session reports directory. Trigger when any of the following are mentioned:
  "monitor model", "data drift", "model drift", "model decay", "retraining",
  "model performance over time", "production monitoring", "model health", "drift
  detection", "when to retrain", or any request to track how a deployed model
  is performing over time.
---

# Monitoring Skill

The Monitoring agent watches the deployed model over time and raises the alarm
when something changes. Models do not stay accurate forever — the world changes,
customer behaviour shifts, new patterns emerge. A model trained on last year's
data may make poor predictions on today's data without anyone noticing.

Monitoring catches this before it causes problems.

There are three things to monitor:

1. **Data drift** — are the inputs changing compared to what the model was trained on?
2. **Prediction drift** — is the distribution of predictions shifting?
3. **Performance decay** — is the model getting less accurate over time?

---

## Responsibilities

1. Establish a baseline from the training data at deployment time
2. Compare new incoming data against the baseline to detect drift
3. Monitor prediction distributions for unexpected shifts
4. Monitor performance metrics if ground truth labels become available
5. Generate plain English alerts when drift or decay is detected
6. Produce a monitoring report saved to the session directory
7. Recommend retraining when thresholds are crossed
8. Explain what drift means and why it matters in plain English

---

## What Drift Means — Plain English

Always explain this before presenting monitoring results:

"Your model learned patterns from data collected up to a certain point in time.
If the world changes after that — customer behaviour shifts, prices change, new
products are introduced — the patterns the model learned may no longer apply.

Drift detection watches for these changes. If your input data starts to look
noticeably different from what the model was trained on, or if the model's
predictions start shifting in unexpected ways, we raise an alert.

This does not necessarily mean the model is wrong — but it is a signal worth
investigating before it becomes a problem."

---

## Establishing the Baseline

Run at deployment time. The baseline is the reference point for all future comparisons.

```python
import pandas as pd
import numpy as np
import json
from pathlib import Path

def establish_baseline(X_train, feature_columns, model,
                        scaler, session_id):
    """
    Compute and save baseline statistics from training data.
    Called once at deployment time.
    """
    baseline = {}

    for col in feature_columns:
        if X_train[col].dtype in ["int64", "float64"]:
            baseline[col] = {
                "type":   "numeric",
                "mean":   float(X_train[col].mean()),
                "std":    float(X_train[col].std()),
                "min":    float(X_train[col].min()),
                "max":    float(X_train[col].max()),
                "q25":    float(X_train[col].quantile(0.25)),
                "median": float(X_train[col].median()),
                "q75":    float(X_train[col].quantile(0.75)),
                "null_pct": float(X_train[col].isna().mean())
            }
        else:
            value_counts = X_train[col].value_counts(normalize=True)
            baseline[col] = {
                "type":         "categorical",
                "value_counts": value_counts.round(4).to_dict(),
                "n_unique":     int(X_train[col].nunique()),
                "null_pct":     float(X_train[col].isna().mean())
            }

    # Baseline prediction distribution
    if scaler:
        numeric_cols = X_train.select_dtypes("number").columns.tolist()
        X_scaled = X_train.copy()
        X_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    else:
        X_scaled = X_train

    predictions = model.predict(X_scaled)
    baseline["_predictions"] = {
        "mean":   float(np.mean(predictions)),
        "std":    float(np.std(predictions)),
        "min":    float(np.min(predictions)),
        "max":    float(np.max(predictions)),
        "distribution": np.unique(predictions,
                                   return_counts=True)[1].tolist()
                    if len(np.unique(predictions)) <= 20
                    else "continuous"
    }

    baseline["_established_at"] = pd.Timestamp.now().isoformat()
    baseline["_n_training_rows"] = len(X_train)

    # Save baseline
    baseline_path = f"sessions/{session_id}/monitoring/baseline.json"
    Path(f"sessions/{session_id}/monitoring").mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    return baseline, baseline_path
```

---

## Data Drift Detection

```python
from scipy import stats

def detect_data_drift(current_df, baseline, feature_columns,
                       significance_level=0.05):
    """
    Compare current data distribution against baseline.
    Uses KS test for numeric, chi-squared for categorical.
    """
    drift_results = []

    for col in feature_columns:
        if col not in baseline:
            continue

        col_baseline = baseline[col]
        result = {"feature": col, "type": col_baseline["type"]}

        if col_baseline["type"] == "numeric":
            # Kolmogorov-Smirnov test
            # Generates synthetic baseline sample from stored statistics
            baseline_sample = np.random.normal(
                loc=col_baseline["mean"],
                scale=max(col_baseline["std"], 1e-6),
                size=1000
            )
            current_vals = current_df[col].dropna().values

            if len(current_vals) < 10:
                result["status"] = "insufficient_data"
                drift_results.append(result)
                continue

            ks_stat, p_value = stats.ks_2samp(baseline_sample, current_vals)

            current_mean = float(current_vals.mean())
            mean_shift   = abs(current_mean - col_baseline["mean"])
            std_baseline = max(col_baseline["std"], 1e-6)
            mean_shift_normalised = mean_shift / std_baseline

            result.update({
                "ks_statistic":          round(float(ks_stat), 4),
                "p_value":               round(float(p_value), 4),
                "drift_detected":        p_value < significance_level,
                "current_mean":          round(current_mean, 4),
                "baseline_mean":         round(col_baseline["mean"], 4),
                "mean_shift_normalised": round(mean_shift_normalised, 4),
                "severity":              _drift_severity(p_value,
                                                         mean_shift_normalised)
            })

        else:
            # Categorical — compare value distributions
            current_counts = current_df[col].value_counts(normalize=True)
            baseline_counts = pd.Series(col_baseline["value_counts"])

            # New categories not seen during training
            new_categories = set(current_counts.index) - set(baseline_counts.index)
            missing_categories = set(baseline_counts.index) - set(current_counts.index)

            # Population Stability Index (PSI)
            psi = _compute_psi(baseline_counts, current_counts)

            result.update({
                "psi":                 round(float(psi), 4),
                "drift_detected":      psi > 0.2,
                "new_categories":      list(new_categories),
                "missing_categories":  list(missing_categories),
                "severity":            "high" if psi > 0.25
                                       else "medium" if psi > 0.1
                                       else "low"
            })

        result["plain_english"] = _drift_plain_english(result)
        drift_results.append(result)

    return drift_results


def _drift_severity(p_value, mean_shift_normalised):
    if p_value < 0.01 and mean_shift_normalised > 1.0:
        return "high"
    elif p_value < 0.05 and mean_shift_normalised > 0.5:
        return "medium"
    elif p_value < 0.05:
        return "low"
    else:
        return "none"


def _compute_psi(expected, actual, bins=10):
    """Population Stability Index for categorical drift."""
    all_cats = set(expected.index) | set(actual.index)
    psi = 0.0
    for cat in all_cats:
        e = max(expected.get(cat, 0), 1e-6)
        a = max(actual.get(cat, 0), 1e-6)
        psi += (a - e) * np.log(a / e)
    return psi


def _drift_plain_english(result):
    col = result["feature"]
    if not result.get("drift_detected"):
        return f"'{col}' looks similar to what the model was trained on — no drift detected."

    severity = result.get("severity", "low")
    if result["type"] == "numeric":
        shift = result.get("mean_shift_normalised", 0)
        direction = ("higher" if result["current_mean"] > result["baseline_mean"]
                     else "lower")
        return (
            f"'{col}' has shifted — values are now {direction} on average than "
            f"when the model was trained. This is a {severity}-severity drift."
        )
    else:
        new_cats = result.get("new_categories", [])
        psi = result.get("psi", 0)
        msg = f"'{col}' distribution has changed (PSI: {psi:.2f}) — {severity} severity."
        if new_cats:
            msg += f" New categories seen: {new_cats}. The model has never seen these values."
        return msg
```

---

## Prediction Drift Detection

```python
def detect_prediction_drift(current_predictions, baseline):
    """
    Check if the distribution of predictions has shifted.
    """
    pred_baseline = baseline.get("_predictions", {})
    if not pred_baseline:
        return {"status": "no_baseline"}

    current_mean = float(np.mean(current_predictions))
    current_std  = float(np.std(current_predictions))
    baseline_mean = pred_baseline.get("mean", current_mean)
    baseline_std  = pred_baseline.get("std", current_std)

    mean_shift = abs(current_mean - baseline_mean)
    std_shift  = abs(current_std - baseline_std)

    drift_detected = (
        mean_shift > 0.1 * abs(baseline_mean) or
        std_shift  > 0.2 * abs(baseline_std)
    )

    return {
        "drift_detected":  drift_detected,
        "current_mean":    round(current_mean, 4),
        "baseline_mean":   round(baseline_mean, 4),
        "current_std":     round(current_std, 4),
        "baseline_std":    round(baseline_std, 4),
        "mean_shift_pct":  round(mean_shift / max(abs(baseline_mean), 1e-6) * 100, 2),
        "plain_english": (
            f"The model's predictions have shifted — the average prediction has "
            f"changed by {mean_shift:.3f} compared to when the model was deployed. "
            f"This may indicate a change in the underlying data."
        ) if drift_detected else (
            "The model's predictions are in a similar range to when it was deployed."
        )
    }
```

---

## Performance Decay Detection

Only possible when ground truth labels are available for recent predictions.

```python
def detect_performance_decay(recent_predictions, recent_labels,
                               baseline_metric_value, task_type,
                               metric_name):
    """
    Compare recent performance against baseline.
    Requires ground truth labels for recent predictions.
    """
    from sklearn.metrics import roc_auc_score, mean_squared_error
    import numpy as np

    if len(recent_predictions) < 30:
        return {
            "status": "insufficient_data",
            "plain_english": (
                "We do not yet have enough labelled recent data to measure "
                "performance decay reliably. We need at least 30 labelled examples."
            )
        }

    if task_type == "binary_classification":
        current_score = roc_auc_score(recent_labels, recent_predictions)
    else:
        current_score = float(np.sqrt(mean_squared_error(
            recent_labels, recent_predictions
        )))

    decay     = baseline_metric_value - current_score
    decay_pct = abs(decay) / max(baseline_metric_value, 1e-6) * 100

    if decay_pct > 10:
        severity = "high"
        recommendation = "retraining_recommended"
    elif decay_pct > 5:
        severity = "medium"
        recommendation = "monitor_closely"
    else:
        severity = "low"
        recommendation = "no_action_needed"

    return {
        "baseline_score":    round(baseline_metric_value, 4),
        "current_score":     round(current_score, 4),
        "decay":             round(decay, 4),
        "decay_pct":         round(decay_pct, 2),
        "severity":          severity,
        "recommendation":    recommendation,
        "plain_english": (
            f"The model's performance has dropped by {decay_pct:.1f}% compared "
            f"to when it was deployed ({baseline_metric_value:.3f} → "
            f"{current_score:.3f}). "
            + ("We recommend retraining the model on more recent data."
               if recommendation == "retraining_recommended"
               else "We recommend monitoring this closely."
               if recommendation == "monitor_closely"
               else "This is within an acceptable range — no action needed yet.")
        )
    }
```

---

## Generating the Monitoring Report

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_monitoring_report(drift_results, prediction_drift,
                                 performance_decay, session_id,
                                 report_number=1):

    output_dir = Path(f"sessions/{session_id}/reports/monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)

    drifted_features = [r for r in drift_results if r.get("drift_detected")]
    total_features   = len(drift_results)
    drift_pct        = len(drifted_features) / max(total_features, 1)

    # Summary chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Drift summary
    labels  = ["No drift", "Drift detected"]
    sizes   = [total_features - len(drifted_features), len(drifted_features)]
    colours = ["#2E75B6", "#C0392B"]
    axes[0].pie(sizes, labels=labels, colors=colours,
                autopct="%1.0f%%", startangle=90)
    axes[0].set_title(f"Data Drift Summary\n{len(drifted_features)} of "
                       f"{total_features} features have drifted")

    # Severity breakdown
    severities = [r.get("severity", "none") for r in drift_results]
    sev_counts = {s: severities.count(s)
                  for s in ["high", "medium", "low", "none"]}
    sev_colours = {"high": "#C0392B", "medium": "#E67E22",
                   "low": "#F1C40F", "none": "#2E75B6"}
    axes[1].bar(sev_counts.keys(), sev_counts.values(),
                color=[sev_colours[s] for s in sev_counts.keys()])
    axes[1].set_title("Drift Severity by Feature")
    axes[1].set_ylabel("Number of features")

    plt.suptitle(f"Monitoring Report #{report_number} — "
                  f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                  fontsize=12)
    plt.tight_layout()
    chart_path = output_dir / f"monitoring_report_{report_number}.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()

    # Retraining recommendation
    high_drift   = sum(1 for r in drift_results
                       if r.get("severity") == "high")
    retrain_recommended = (
        high_drift > 0 or
        prediction_drift.get("drift_detected") or
        performance_decay.get("recommendation") == "retraining_recommended"
    )

    return {
        "report_number":        report_number,
        "total_features":       total_features,
        "drifted_features":     len(drifted_features),
        "drift_pct":            round(drift_pct, 4),
        "high_severity":        high_drift,
        "prediction_drift":     prediction_drift,
        "performance_decay":    performance_decay,
        "retrain_recommended":  retrain_recommended,
        "chart_path":           str(chart_path),
        "drifted_feature_list": drifted_features
    }
```

---

## Running the Full Monitoring Pipeline

```python
def run_monitoring(current_df, current_predictions,
                    baseline, feature_columns,
                    task_type, session_id,
                    current_labels=None,
                    baseline_metric_value=None,
                    metric_name=None,
                    report_number=1):

    # 1. Data drift
    drift_results = detect_data_drift(
        current_df, baseline, feature_columns
    )

    # 2. Prediction drift
    prediction_drift = detect_prediction_drift(
        current_predictions, baseline
    )

    # 3. Performance decay (only if labels available)
    if current_labels is not None and baseline_metric_value:
        performance_decay = detect_performance_decay(
            current_predictions, current_labels,
            baseline_metric_value, task_type, metric_name
        )
    else:
        performance_decay = {
            "status": "no_labels",
            "plain_english": (
                "Performance decay monitoring requires labelled recent data. "
                "Once ground truth labels are available for recent predictions, "
                "this check will run automatically."
            )
        }

    # 4. Generate report
    report = generate_monitoring_report(
        drift_results, prediction_drift,
        performance_decay, session_id, report_number
    )

    # 5. Save report
    report_path = (f"sessions/{session_id}/monitoring/"
                   f"report_{report_number}.json")
    with open(report_path, "w") as f:
        json.dump({k: v for k, v in report.items()
                   if k != "chart_path"}, f, indent=2, default=str)

    return report, drift_results, prediction_drift, performance_decay
```

---

## Output Written to Session

**Baseline (saved at deployment):**
`sessions/{session_id}/monitoring/baseline.json`

**Monitoring reports:**
`sessions/{session_id}/monitoring/report_{n}.json`

**Charts:**
`sessions/{session_id}/reports/monitoring/monitoring_report_{n}.png`

**Result JSON:**
`sessions/{session_id}/outputs/monitoring/result.json`

```json
{
  "stage": "monitoring",
  "status": "success",
  "report_number": 1,
  "drifted_features": 2,
  "total_features": 14,
  "high_severity_drift": 1,
  "prediction_drift_detected": false,
  "retrain_recommended": true,
  "plain_english_summary": "We checked your model's health. 2 out of 14 input features have drifted from what the model was trained on. One feature has drifted significantly. We recommend investigating and considering retraining.",
  "report_section": {
    "stage": "monitoring",
    "title": "Monitoring Your Model's Health",
    "summary": "...",
    "why_this_matters": "Models do not stay accurate forever. Monitoring catches when the real world has changed enough that the model's training data is no longer representative — before it starts making poor predictions."
  }
}
```

---

## What to Tell the User

Opening:
"We have run a health check on your deployed model.
Here is what we found:"

Drift summary:
"{drifted_features} out of {total_features} input features have changed
compared to what the model was trained on.

Features with notable changes:
{list each drifted feature with its plain English message}"

Prediction drift:
"{prediction_drift.plain_english}"

Performance:
"{performance_decay.plain_english}"

Retraining recommendation:
If recommended:
"Based on what we found, we recommend retraining the model on more recent
data. The patterns it learned may no longer fully reflect the current state
of your data. We can walk you through the process — it follows the same
pipeline as the original training."

If not recommended:
"The model appears healthy. No action is needed at this time. We recommend
running this check again in [time period] or whenever you notice unexpected
predictions."

---

## Reference Files

- `references/drift-guide.md` — detailed guide to drift types and retraining strategies
