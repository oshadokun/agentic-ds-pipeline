"""
Explainability Agent
Uses SHAP to explain why the model makes the predictions it does.
Produces global feature importance, a beeswarm summary plot, and
individual prediction waterfall charts. Flags potential bias if
sensitive columns are driving predictions.
"""

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Explainer selection
# ---------------------------------------------------------------------------

def _get_explainer(model, X_train, model_id: str):
    import shap

    if model_id in ["random_forest", "random_forest_regressor",
                    "xgboost", "xgboost_regressor"]:
        explainer      = shap.TreeExplainer(model)
        explainer_type = "tree"
    elif model_id in ["logistic_regression", "logistic_regression_multi", "ridge"]:
        explainer      = shap.LinearExplainer(model, X_train)
        explainer_type = "linear"
    else:
        background     = shap.sample(X_train, min(100, len(X_train)))
        explainer      = shap.KernelExplainer(model.predict, background)
        explainer_type = "kernel"

    return explainer, explainer_type


# ---------------------------------------------------------------------------
# Global importance
# ---------------------------------------------------------------------------

def _compute_global_importance(explainer, X_sample, feature_names: list,
                                output_dir: str):
    import shap

    shap_values = explainer.shap_values(X_sample)

    # For binary classification shap_values may be a list — use class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_n = min(20, len(importance_df))
    top   = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    ax.barh(top["feature"][::-1], top["importance"][::-1],
            color="#2E75B6", edgecolor="white")
    ax.set_xlabel("Average influence on predictions (SHAP value)")
    ax.set_title(
        "What drives predictions — top features\n"
        "Longer bar = greater influence on the model's decisions",
        fontsize=11
    )
    plt.tight_layout()
    chart_path = Path(output_dir) / "feature_importance_shap.png"
    plt.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close()

    return shap_vals, importance_df, str(chart_path)


def _interpret_global_importance(importance_df: pd.DataFrame,
                                  target_col: str, n: int = 5) -> str:
    top = importance_df.head(n)
    lines = ["The model's predictions are most influenced by these factors:"]
    for i, row in top.iterrows():
        lines.append(
            f"  {i + 1}. '{row['feature']}' — this column has the biggest "
            f"average influence on whether the model predicts one outcome or another."
        )
    bottom = importance_df.tail(3)
    lines.append(
        "\nThese columns had the least influence: "
        + ", ".join([f"'{r['feature']}'" for _, r in bottom.iterrows()])
        + ". They may not be contributing much to the model."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SHAP beeswarm summary plot
# ---------------------------------------------------------------------------

def _plot_shap_summary(shap_vals, X_sample, feature_names: list,
                        output_dir: str) -> str:
    import shap

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_vals,
        X_sample,
        feature_names=feature_names,
        max_display=15,
        show=False,
        plot_type="dot"
    )
    plt.title(
        "How each feature influences predictions\n"
        "Red = high value, Blue = low value  |  "
        "Right = pushes prediction up, Left = pushes prediction down",
        fontsize=10
    )
    plt.tight_layout()
    path = Path(output_dir) / "shap_summary.png"
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(path)


# ---------------------------------------------------------------------------
# Individual prediction waterfall charts
# ---------------------------------------------------------------------------

def _explain_single_prediction(explainer, row, model_id: str,
                                output_dir: str, row_index: int) -> str:
    import shap

    try:
        shap_values = explainer(row)

        if hasattr(shap_values, "values"):
            if len(shap_values.values.shape) == 3:
                sv = shap_values[0, :, 1]   # binary classification — class 1
            else:
                sv = shap_values[0]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(sv, max_display=12, show=False)
        plt.title(
            f"Why the model made this prediction — example row {row_index + 1}\n"
            "Each bar shows how much a feature pushed the prediction up (red) or down (blue)",
            fontsize=10
        )
        plt.tight_layout()
        path = Path(output_dir) / f"explanation_row_{row_index}.png"
        plt.savefig(path, bbox_inches="tight", dpi=100)
        plt.close()
        return str(path)
    except Exception:
        plt.close("all")
        return ""


def _narrate_single_prediction(shap_vals_row, feature_names: list,
                                 feature_values, prediction,
                                 prediction_proba=None) -> str:
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap":    shap_vals_row,
        "value":   feature_values
    }).sort_values("shap", key=abs, ascending=False)

    top_positive = contributions[contributions["shap"] > 0].head(3)
    top_negative = contributions[contributions["shap"] < 0].head(3)

    lines = [f"Prediction: {prediction}"]
    if prediction_proba is not None:
        lines.append(f"Confidence: {prediction_proba:.1%}")

    lines.append("\nMain reasons this outcome was predicted:")
    for _, row in top_positive.iterrows():
        lines.append(
            f"  \u2191 '{row['feature']}' = {row['value']} "
            f"pushed the prediction toward this outcome"
        )

    if not top_negative.empty:
        lines.append("\nFactors that pushed against this outcome:")
        for _, row in top_negative.iterrows():
            lines.append(
                f"  \u2193 '{row['feature']}' = {row['value']} "
                f"pushed away from this outcome"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bias check
# ---------------------------------------------------------------------------

def _check_for_bias(importance_df: pd.DataFrame, sensitive_columns: list,
                    threshold: float = 0.05) -> list:
    warnings  = []
    total_imp = importance_df["importance"].sum()

    for col in sensitive_columns:
        matches = importance_df[
            importance_df["feature"].str.contains(col, case=False, na=False)
        ]
        if not matches.empty:
            col_importance = float(matches["importance"].sum())
            pct = col_importance / max(total_imp, 1e-9)

            if pct > threshold:
                warnings.append({
                    "column":         col,
                    "importance":     round(col_importance, 4),
                    "pct_of_total":   round(pct, 4),
                    "plain_english": (
                        f"'{col}' appears to be influencing the model's predictions "
                        f"({pct:.1%} of total influence). If this column contains "
                        f"sensitive information such as age, gender, or ethnicity, "
                        f"you may want to consider whether this is appropriate for "
                        f"your use case before deploying."
                    )
                })

    return warnings


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id       = session["session_id"]
    target_col       = session["goal"].get("target_column")
    task_type        = session["goal"].get("task_type", "binary_classification")
    model_id         = session["config"].get("model_id")
    sensitive_cols   = session["config"].get("sensitive_columns", [])
    sessions_dir     = Path("sessions")
    session_dir      = sessions_dir / session_id
    splits_dir       = session_dir / "data" / "processed" / "splits"
    output_dir       = session_dir / "reports" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model — prefer tuned
    tuned_path = session_dir / "models" / "tuned_model.pkl"
    best_json  = session_dir / "models" / "best_model.json"

    if tuned_path.exists():
        model_path = tuned_path
    elif best_json.exists():
        with open(best_json) as f:
            model_info = json.load(f)
        model_path = Path(model_info["model_path"])
    else:
        return {
            "stage":                 "explainability",
            "status":                "failed",
            "plain_english_summary": "No trained model found. Please run training first."
        }

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load data splits
    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        X_val   = pd.read_csv(splits_dir / "X_val.csv")
        y_val   = pd.read_csv(splits_dir / "y_val.csv").squeeze()
    except FileNotFoundError:
        return {
            "stage":                 "explainability",
            "status":                "failed",
            "plain_english_summary": "No split data found. Please run splitting first."
        }

    feature_names = X_train.columns.tolist()

    # Cast to float64 — SHAP ufunc requires numpy-native floats, not Python floats
    X_train  = X_train.astype(np.float64)
    X_val    = X_val.astype(np.float64)

    # Sample for speed on large datasets
    sample_size = min(500, len(X_val))
    X_sample    = X_val.iloc[:sample_size]

    try:
        explainer, explainer_type = _get_explainer(model, X_train, model_id)

        # Global importance
        shap_vals, importance_df, importance_chart = _compute_global_importance(
            explainer, X_sample, feature_names, str(output_dir)
        )

        # Beeswarm summary
        summary_chart = _plot_shap_summary(
            shap_vals, X_sample, feature_names, str(output_dir)
        )

        # Individual prediction waterfall charts (up to 3)
        individual_explanations = []
        for i in range(min(3, len(X_sample))):
            row   = X_sample.iloc[[i]]
            chart = _explain_single_prediction(
                explainer, row, model_id, str(output_dir), row_index=i
            )

            # Plain-English narration
            row_shap = shap_vals[i] if len(shap_vals.shape) == 2 else shap_vals
            prediction = model.predict(row)[0]
            proba = None
            if hasattr(model, "predict_proba") and "classification" in task_type:
                try:
                    proba = float(model.predict_proba(row)[0].max())
                except Exception:
                    pass

            narration = _narrate_single_prediction(
                row_shap, feature_names,
                row.iloc[0].tolist(), prediction, proba
            )

            individual_explanations.append({
                "row_index": i,
                "chart":     chart,
                "narration": narration
            })

        # Bias check
        bias_warnings = _check_for_bias(importance_df, sensitive_cols)

        # Global narration
        global_narration = _interpret_global_importance(importance_df, target_col)

        top_features = importance_df["feature"].head(10).tolist()

    except Exception as exc:
        return {
            "stage":                 "explainability",
            "status":                "failed",
            "plain_english_summary": f"SHAP analysis failed: {str(exc)}"
        }

    charts = [importance_chart, summary_chart]
    charts += [e["chart"] for e in individual_explanations if e.get("chart")]

    bias_text = (
        "\n\nBias check: " + "; ".join(
            w["plain_english"] for w in bias_warnings
        ) if bias_warnings else
        "\n\nWe checked whether any sensitive columns are driving predictions — "
        "we did not find any significant concerns."
    )

    summary = (
        f"We analysed why the model makes the predictions it does. "
        f"The three most influential factors are: "
        + ", ".join([f"'{f}'" for f in top_features[:3]])
        + f". {bias_text}"
    )

    return {
        "stage":                   "explainability",
        "status":                  "success",
        "explainer_type":          explainer_type,
        "top_features":            top_features,
        "importance_table":        importance_df.head(20).to_dict(orient="records"),
        "global_narration":        global_narration,
        "individual_explanations": individual_explanations,
        "bias_warnings":           bias_warnings,
        "charts":                  charts,
        "decisions_required":      [],
        "decisions_made":          [],
        "plain_english_summary":   summary,
        "report_section": {
            "stage":   "explainability",
            "title":   "Why the Model Makes Its Predictions",
            "summary": summary,
            "decision_made": "No decisions required — this stage is for transparency.",
            "why_this_matters": (
                "Understanding what the model pays attention to builds trust, "
                "catches unexpected patterns, and is often required before a model "
                "can be deployed in regulated environments."
            )
        },
        "config_updates": {
            "top_features":        top_features,
            "bias_warnings_count": len(bias_warnings)
        }
    }
