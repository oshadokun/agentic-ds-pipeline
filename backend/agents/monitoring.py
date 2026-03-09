"""
Monitoring Agent
Watches the deployed model over time for data drift, prediction drift,
and performance decay. Establishes a baseline at deployment time and
compares subsequent data against it. Recommends retraining when thresholds
are crossed.
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def _establish_baseline(X_train: pd.DataFrame, feature_columns: list,
                          model, session_id: str) -> tuple:
    baseline = {}

    for col in feature_columns:
        if col not in X_train.columns:
            continue
        series = X_train[col].astype(np.float64) if pd.api.types.is_numeric_dtype(X_train[col]) else X_train[col]
        if pd.api.types.is_numeric_dtype(series):
            baseline[col] = {
                "type":     "numeric",
                "mean":     float(series.mean()),
                "std":      float(series.std()),
                "min":      float(series.min()),
                "max":      float(series.max()),
                "q25":      float(series.quantile(0.25)),
                "median":   float(series.median()),
                "q75":      float(series.quantile(0.75)),
                "null_pct": float(series.isna().mean())
            }
        else:
            value_counts = series.value_counts(normalize=True)
            baseline[col] = {
                "type":         "categorical",
                "value_counts": value_counts.round(4).to_dict(),
                "n_unique":     int(series.nunique()),
                "null_pct":     float(series.isna().mean())
            }

    # Baseline prediction distribution — splits are already scaled, no scaler needed
    X_pred = X_train.astype(np.float64)
    predictions = model.predict(X_pred)
    unique_preds = np.unique(predictions)
    baseline["_predictions"] = {
        "mean":         float(np.mean(predictions)),
        "std":          float(np.std(predictions)),
        "min":          float(np.min(predictions)),
        "max":          float(np.max(predictions)),
        "distribution": (
            np.unique(predictions, return_counts=True)[1].tolist()
            if len(unique_preds) <= 20 else "continuous"
        )
    }

    baseline["_established_at"]   = pd.Timestamp.now().isoformat()
    baseline["_n_training_rows"]  = len(X_train)

    monitoring_dir = Path("sessions") / session_id / "monitoring"
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = monitoring_dir / "baseline.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline, f, indent=2)

    return baseline, str(baseline_path)


# ---------------------------------------------------------------------------
# Drift detection helpers
# ---------------------------------------------------------------------------

def _drift_severity(p_value: float, mean_shift_norm: float) -> str:
    if p_value < 0.01 and mean_shift_norm > 1.0:
        return "high"
    elif p_value < 0.05 and mean_shift_norm > 0.5:
        return "medium"
    elif p_value < 0.05:
        return "low"
    return "none"


def _compute_psi(expected: pd.Series, actual: pd.Series) -> float:
    """Population Stability Index for categorical drift."""
    all_cats = set(expected.index) | set(actual.index)
    psi = 0.0
    for cat in all_cats:
        e = max(float(expected.get(cat, 0)), 1e-6)
        a = max(float(actual.get(cat, 0)), 1e-6)
        psi += (a - e) * np.log(a / e)
    return psi


def _drift_plain_english(result: dict) -> str:
    col = result["feature"]
    if not result.get("drift_detected"):
        return f"'{col}' looks similar to what the model was trained on — no drift detected."

    severity = result.get("severity", "low")
    if result["type"] == "numeric":
        direction = (
            "higher" if result.get("current_mean", 0) > result.get("baseline_mean", 0)
            else "lower"
        )
        return (
            f"'{col}' has shifted — values are now {direction} on average than "
            f"when the model was trained. This is a {severity}-severity drift."
        )
    else:
        new_cats = result.get("new_categories", [])
        psi      = result.get("psi", 0)
        msg      = f"'{col}' distribution has changed (PSI: {psi:.2f}) — {severity} severity."
        if new_cats:
            msg += f" New categories seen: {new_cats}. The model has never seen these values."
        return msg


# ---------------------------------------------------------------------------
# Data drift
# ---------------------------------------------------------------------------

def _detect_data_drift(current_df: pd.DataFrame, baseline: dict,
                        feature_columns: list,
                        significance_level: float = 0.05) -> list:
    drift_results = []

    for col in feature_columns:
        if col not in baseline or col not in current_df.columns:
            continue

        col_baseline = baseline[col]
        result = {"feature": col, "type": col_baseline["type"]}

        if col_baseline["type"] == "numeric":
            rng = np.random.default_rng(42)
            baseline_sample = rng.normal(
                loc=col_baseline["mean"],
                scale=max(col_baseline["std"], 1e-6),
                size=1000
            )
            current_vals = current_df[col].dropna().astype(np.float64).values

            if len(current_vals) < 10:
                result["status"] = "insufficient_data"
                drift_results.append(result)
                continue

            ks_stat, p_value = stats.ks_2samp(baseline_sample, current_vals)

            current_mean      = float(current_vals.mean())
            mean_shift        = abs(current_mean - col_baseline["mean"])
            std_baseline      = max(col_baseline["std"], 1e-6)
            mean_shift_norm   = mean_shift / std_baseline

            result.update({
                "ks_statistic":          round(float(ks_stat), 4),
                "p_value":               round(float(p_value), 4),
                "drift_detected":        bool(p_value < significance_level),
                "current_mean":          round(current_mean, 4),
                "baseline_mean":         round(col_baseline["mean"], 4),
                "mean_shift_normalised": round(mean_shift_norm, 4),
                "severity":              _drift_severity(p_value, mean_shift_norm)
            })

        else:
            current_counts  = current_df[col].value_counts(normalize=True)
            baseline_counts = pd.Series(col_baseline["value_counts"])
            new_cats        = set(current_counts.index) - set(baseline_counts.index)
            missing_cats    = set(baseline_counts.index) - set(current_counts.index)
            psi             = _compute_psi(baseline_counts, current_counts)

            result.update({
                "psi":                 round(float(psi), 4),
                "drift_detected":      bool(psi > 0.2),
                "new_categories":      list(new_cats),
                "missing_categories":  list(missing_cats),
                "severity":            (
                    "high" if psi > 0.25
                    else "medium" if psi > 0.1
                    else "low"
                )
            })

        result["plain_english"] = _drift_plain_english(result)
        drift_results.append(result)

    return drift_results


# ---------------------------------------------------------------------------
# Prediction drift
# ---------------------------------------------------------------------------

def _detect_prediction_drift(current_predictions, baseline: dict) -> dict:
    pred_baseline = baseline.get("_predictions", {})
    if not pred_baseline:
        return {"status": "no_baseline"}

    current_mean  = float(np.mean(current_predictions))
    current_std   = float(np.std(current_predictions))
    baseline_mean = pred_baseline.get("mean", current_mean)
    baseline_std  = pred_baseline.get("std", current_std)

    mean_shift     = abs(current_mean - baseline_mean)
    std_shift      = abs(current_std - baseline_std)
    drift_detected = bool(
        mean_shift > 0.1 * abs(baseline_mean) or
        std_shift  > 0.2 * max(abs(baseline_std), 1e-6)
    )

    return {
        "drift_detected":  drift_detected,
        "current_mean":    round(current_mean, 4),
        "baseline_mean":   round(baseline_mean, 4),
        "current_std":     round(current_std, 4),
        "baseline_std":    round(baseline_std, 4),
        "mean_shift_pct":  round(
            mean_shift / max(abs(baseline_mean), 1e-6) * 100, 2
        ),
        "plain_english": (
            f"The model's predictions have shifted — the average prediction has "
            f"changed by {mean_shift:.3f} compared to when the model was deployed. "
            f"This may indicate a change in the underlying data."
        ) if drift_detected else (
            "The model's predictions are in a similar range to when it was deployed."
        )
    }


# ---------------------------------------------------------------------------
# Performance decay
# ---------------------------------------------------------------------------

def _detect_performance_decay(recent_predictions, recent_labels,
                                baseline_metric_value: float,
                                task_type: str, metric_name: str) -> dict:
    from sklearn.metrics import roc_auc_score, r2_score, f1_score

    if len(recent_predictions) < 30:
        return {
            "status": "insufficient_data",
            "plain_english": (
                "We do not yet have enough labelled recent data to measure "
                "performance decay reliably. We need at least 30 labelled examples."
            )
        }

    try:
        if task_type == "binary_classification":
            current_score = float(roc_auc_score(recent_labels, recent_predictions))
        elif task_type == "multiclass_classification":
            current_score = float(f1_score(recent_labels, recent_predictions,
                                           average="weighted", zero_division=0))
        else:
            current_score = float(r2_score(recent_labels, recent_predictions))
    except Exception as exc:
        return {"status": "error", "plain_english": f"Could not compute performance: {exc}"}

    decay     = baseline_metric_value - current_score
    decay_pct = abs(decay) / max(abs(baseline_metric_value), 1e-6) * 100

    if decay_pct > 10:
        severity       = "high"
        recommendation = "retraining_recommended"
    elif decay_pct > 5:
        severity       = "medium"
        recommendation = "monitor_closely"
    else:
        severity       = "low"
        recommendation = "no_action_needed"

    return {
        "baseline_score": round(baseline_metric_value, 4),
        "current_score":  round(current_score, 4),
        "decay":          round(decay, 4),
        "decay_pct":      round(decay_pct, 2),
        "severity":       severity,
        "recommendation": recommendation,
        "plain_english": (
            f"The model's performance has dropped by {decay_pct:.1f}% compared "
            f"to when it was deployed ({baseline_metric_value:.3f} \u2192 "
            f"{current_score:.3f}). "
            + (
                "We recommend retraining the model on more recent data."
                if recommendation == "retraining_recommended"
                else "We recommend monitoring this closely."
                if recommendation == "monitor_closely"
                else "This is within an acceptable range — no action needed yet."
            )
        )
    }


# ---------------------------------------------------------------------------
# Monitoring report chart
# ---------------------------------------------------------------------------

def _generate_monitoring_chart(drift_results: list, session_id: str,
                                report_number: int) -> str:
    output_dir = Path("sessions") / session_id / "reports" / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_features   = len(drift_results)
    drifted_features = [r for r in drift_results if r.get("drift_detected")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Pie: drifted vs healthy
    labels  = ["No drift", "Drift detected"]
    sizes   = [total_features - len(drifted_features), len(drifted_features)]
    colours = ["#2E75B6", "#C0392B"]
    axes[0].pie(sizes, labels=labels, colors=colours,
                autopct="%1.0f%%", startangle=90)
    axes[0].set_title(
        f"Data Drift Summary\n"
        f"{len(drifted_features)} of {total_features} features have drifted"
    )

    # Bar: severity breakdown
    severities = [r.get("severity", "none") for r in drift_results]
    sev_counts = {s: severities.count(s)
                  for s in ["high", "medium", "low", "none"]}
    sev_colours = {
        "high": "#C0392B", "medium": "#E67E22",
        "low":  "#F1C40F", "none":   "#2E75B6"
    }
    axes[1].bar(
        list(sev_counts.keys()),
        list(sev_counts.values()),
        color=[sev_colours[s] for s in sev_counts.keys()]
    )
    axes[1].set_title("Drift Severity by Feature")
    axes[1].set_ylabel("Number of features")

    plt.suptitle(
        f"Monitoring Report #{report_number} — "
        f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=12
    )
    plt.tight_layout()
    chart_path = output_dir / f"monitoring_report_{report_number}.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(chart_path)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id          = session["session_id"]
    task_type           = session["goal"].get("task_type", "binary_classification")
    sessions_dir        = Path("sessions")
    session_dir         = sessions_dir / session_id
    splits_dir          = session_dir / "data" / "processed" / "splits"
    monitoring_dir      = session_dir / "monitoring"
    baseline_path       = monitoring_dir / "baseline.json"

    # Sub-action: "establish_baseline" is called right after deployment
    action = decisions.get("action", "run_monitoring")

    # Load model and scaler
    tuned_path = session_dir / "models" / "tuned_model.pkl"
    best_json  = session_dir / "models" / "best_model.json"

    if tuned_path.exists():
        model_pkl_path = tuned_path
    elif best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        model_pkl_path = Path(info["model_path"])
    else:
        return {
            "stage":                 "monitoring",
            "status":                "failed",
            "plain_english_summary": "No trained model found. Please run training first."
        }

    with open(model_pkl_path, "rb") as f:
        model = pickle.load(f)

    scaler_pkl = session_dir / "models" / "scaler.pkl"
    scaler = None
    if scaler_pkl.exists():
        with open(scaler_pkl, "rb") as f:
            scaler = pickle.load(f)

    # Load training data
    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
    except FileNotFoundError:
        return {
            "stage":                 "monitoring",
            "status":                "failed",
            "plain_english_summary": "No training data found. Please run splitting first."
        }

    feature_columns = X_train.columns.tolist()

    # Establish baseline on first run (or re-establish if explicitly requested)
    if action == "establish_baseline" or not baseline_path.exists():
        baseline, b_path = _establish_baseline(
            X_train, feature_columns, model, session_id
        )
    else:
        b_path = str(baseline_path)

    # Load baseline (just written above on first run, or loaded from disk on re-runs)
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Load current data — use validation set as proxy for "new" data
    # (in production this would be live inference data)
    try:
        X_current = pd.read_csv(splits_dir / "X_val.csv")
    except FileNotFoundError:
        X_current = X_train.sample(min(200, len(X_train)), random_state=0)

    # Generate predictions on current data — splits are already scaled
    current_predictions = model.predict(X_current.astype(np.float64))

    # Drift detection
    drift_results    = _detect_data_drift(X_current, baseline, feature_columns)
    prediction_drift = _detect_prediction_drift(current_predictions, baseline)

    # Performance decay (only if labels available)
    baseline_metric = session["config"].get("tuned_score") or \
                      session["config"].get("primary_metric_value", 0.0)
    metric_name     = session["config"].get("primary_metric", "roc_auc")

    try:
        y_val = pd.read_csv(splits_dir / "y_val.csv").squeeze()
        performance_decay = _detect_performance_decay(
            current_predictions, y_val.tolist(),
            baseline_metric, task_type, metric_name
        )
    except Exception:
        performance_decay = {
            "status": "no_labels",
            "plain_english": (
                "Performance decay monitoring requires labelled recent data. "
                "Once ground truth labels are available for recent predictions, "
                "this check will run automatically."
            )
        }

    # Report number
    existing_reports = list(monitoring_dir.glob("report_*.json")) if monitoring_dir.exists() else []
    report_number    = len(existing_reports) + 1

    # Chart
    chart_path = _generate_monitoring_chart(drift_results, session_id, report_number)

    # Retraining recommendation
    high_drift_count    = sum(1 for r in drift_results if r.get("severity") == "high")
    retrain_recommended = (
        high_drift_count > 0 or
        prediction_drift.get("drift_detected") or
        performance_decay.get("recommendation") == "retraining_recommended"
    )

    drifted_features = [r for r in drift_results if r.get("drift_detected")]

    # Save report JSON
    monitoring_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "report_number":       report_number,
        "total_features":      len(drift_results),
        "drifted_features":    len(drifted_features),
        "high_severity":       high_drift_count,
        "prediction_drift":    prediction_drift,
        "performance_decay":   performance_decay,
        "retrain_recommended": retrain_recommended
    }
    with open(monitoring_dir / f"report_{report_number}.json", "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    if drifted_features:
        drift_detail = " ".join([r["plain_english"] for r in drifted_features[:3]])
    else:
        drift_detail = "No drift detected in any features."

    retrain_msg = (
        "Based on what we found, we recommend retraining the model on more recent data. "
        "The patterns it learned may no longer fully reflect the current state of your data."
        if retrain_recommended else
        "The model appears healthy. No action is needed at this time."
    )

    summary = (
        f"We checked your model's health. "
        f"{len(drifted_features)} out of {len(drift_results)} input features have drifted "
        f"from what the model was trained on. {drift_detail} {retrain_msg}"
    )

    return {
        "stage":                       "monitoring",
        "status":                      "success",
        "report_number":               report_number,
        "total_features":              len(drift_results),
        "drifted_features_count":      len(drifted_features),
        "high_severity_drift":         high_drift_count,
        "drift_results":               drift_results,
        "prediction_drift":            prediction_drift,
        "performance_decay":           performance_decay,
        "retrain_recommended":         retrain_recommended,
        "chart":                       chart_path,
        "decisions_required":          [],
        "decisions_made":              [],
        "plain_english_summary":       summary,
        "report_section": {
            "stage":   "monitoring",
            "title":   "Monitoring Your Model's Health",
            "summary": summary,
            "decision_made": f"Ran monitoring report #{report_number}.",
            "why_this_matters": (
                "Models do not stay accurate forever. Monitoring catches when the real "
                "world has changed enough that the model's training data is no longer "
                "representative — before it starts making poor predictions."
            )
        },
        "config_updates": {
            "last_monitoring_report": report_number,
            "retrain_recommended":    retrain_recommended
        }
    }
