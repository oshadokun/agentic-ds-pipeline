---
name: explainability
description: >
  Responsible for explaining why the model makes the predictions it does — both
  globally across all predictions and locally for individual ones. Always called
  by the Orchestrator after Tuning completes. Uses SHAP to measure the contribution
  of each feature to each prediction. Translates every finding into plain English
  that a non-technical user can understand. Produces a feature importance summary,
  individual prediction explanations, and flags any potential bias or unexpected
  patterns. Saves all outputs to the session reports directory. Trigger when any
  of the following are mentioned: "explain model", "why did the model predict",
  "feature importance", "SHAP", "model interpretability", "what drives predictions",
  "model transparency", "which features matter", or any request to understand the
  reasoning behind model predictions.
---

# Explainability Skill

The Explainability agent answers the most important question a user can ask about
their model: **why does it make the predictions it does?**

A model that cannot be explained cannot be trusted. For non-technical users especially,
understanding what the model is paying attention to is essential — both to build
confidence and to catch any unexpected or problematic patterns before deployment.

This skill uses SHAP (SHapley Additive exPlanations) — the gold standard for model
explainability. Every number it produces has a precise meaning: how much did this
feature push the prediction up or down for this specific row.

---

## Responsibilities

1. Compute SHAP values for the tuned model
2. Produce a global feature importance summary — what matters most overall
3. Produce individual prediction explanations — why did the model predict this for this row
4. Identify any unexpected or potentially problematic patterns
5. Flag potential bias — if a sensitive column is driving predictions
6. Translate everything into plain English
7. Save all charts and explanations to the session reports directory

---

## Choosing the Right SHAP Explainer

```python
import shap
import numpy as np
import pandas as pd

def get_explainer(model, X_train, model_id):
    """
    Select the appropriate SHAP explainer based on model type.
    Each explainer is optimised for its model family.
    """

    if model_id in ["random_forest", "random_forest_regressor",
                    "xgboost", "xgboost_regressor"]:
        # TreeExplainer is fast and exact for tree-based models
        explainer = shap.TreeExplainer(model)
        explainer_type = "tree"

    elif model_id in ["logistic_regression", "logistic_regression_multi",
                      "ridge"]:
        # LinearExplainer is exact for linear models
        explainer = shap.LinearExplainer(model, X_train)
        explainer_type = "linear"

    else:
        # KernelExplainer works for any model — slower but universal
        background = shap.sample(X_train, 100)
        explainer = shap.KernelExplainer(model.predict, background)
        explainer_type = "kernel"

    return explainer, explainer_type
```

---

## Global Feature Importance

Answers: **What does the model pay attention to overall?**

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def compute_global_importance(explainer, X_sample, feature_names,
                               model_id, output_dir):
    """
    Compute and plot global SHAP feature importance.
    Use a sample for speed on large datasets.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list — take class 1
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_vals).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs
    }).sort_values("importance", ascending=False)

    # Plot
    top_n = min(20, len(importance_df))
    top   = importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    bars = ax.barh(top["feature"][::-1], top["importance"][::-1],
                   color="#2E75B6", edgecolor="white")
    ax.set_xlabel("Average influence on predictions (SHAP value)")
    ax.set_title(
        "What drives predictions — top features\n"
        "Longer bar = greater influence on the model's decisions",
        fontsize=11
    )
    plt.tight_layout()
    path = Path(output_dir) / "feature_importance_shap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    return shap_vals, importance_df, str(path)


def interpret_global_importance(importance_df, target_col, n=5):
    top = importance_df.head(n)
    lines = [
        f"The model's predictions are most influenced by these factors:"
    ]
    for i, row in top.iterrows():
        lines.append(
            f"  {i+1}. '{row['feature']}' — this column has the biggest "
            f"average influence on whether the model predicts one outcome "
            f"or another."
        )
    bottom = importance_df.tail(3)
    lines.append(
        f"\nThese columns had the least influence: "
        + ", ".join([f"'{r['feature']}'" for _, r in bottom.iterrows()])
        + ". They may not be contributing much to the model."
    )
    return "\n".join(lines)
```

---

## SHAP Summary Plot (Beeswarm)

Shows both importance and direction — does high values push predictions up or down?

```python
def plot_shap_summary(shap_vals, X_sample, feature_names, output_dir):
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
        "Red = high value, Blue = low value | Right = pushes prediction up, Left = pushes down",
        fontsize=10
    )
    plt.tight_layout()
    path = Path(output_dir) / "shap_summary.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)
```

**Plain English caption:**
"Each dot is one row from your data. Red dots mean the feature had a high value
for that row, blue means low. Dots to the right mean that feature pushed the
prediction toward a positive outcome for that row. Dots to the left mean it
pushed toward a negative outcome."

---

## Individual Prediction Explanation

Answers: **Why did the model make this specific prediction?**

```python
def explain_single_prediction(explainer, row, feature_names,
                               base_value, model_id, output_dir,
                               row_index=0):
    """
    Explain one prediction with a waterfall chart.
    """
    shap_values = explainer(row)

    # Handle classification list format
    if hasattr(shap_values, "values"):
        if len(shap_values.values.shape) == 3:
            sv = shap_values[0, :, 1]  # binary classification — class 1
        else:
            sv = shap_values[0]
    else:
        sv = shap_values

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(sv, max_display=12, show=False)
    plt.title(
        f"Why the model made this prediction — row {row_index + 1}\n"
        "Each bar shows how much a feature pushed the prediction up (red) or down (blue)",
        fontsize=10
    )
    plt.tight_layout()
    path = Path(output_dir) / f"explanation_row_{row_index}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)


def narrate_single_prediction(shap_vals_row, feature_names,
                               feature_values, prediction,
                               prediction_proba=None):
    """
    Tell the story of one prediction in plain English.
    """
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
            f"  ↑ '{row['feature']}' = {row['value']} "
            f"pushed the prediction toward this outcome"
        )

    if not top_negative.empty:
        lines.append("\nFactors that pushed against this outcome:")
        for _, row in top_negative.iterrows():
            lines.append(
                f"  ↓ '{row['feature']}' = {row['value']} "
                f"pushed away from this outcome"
            )

    return "\n".join(lines)
```

---

## Bias and Fairness Check

```python
def check_for_bias(importance_df, sensitive_columns, threshold=0.05):
    """
    Flag if sensitive columns appear in the top drivers of predictions.
    Does not make a decision — surfaces for user awareness.
    """
    warnings = []
    total_importance = importance_df["importance"].sum()

    for col in sensitive_columns:
        matches = importance_df[importance_df["feature"].str.contains(
            col, case=False, na=False
        )]
        if not matches.empty:
            col_importance = matches["importance"].sum()
            pct = col_importance / total_importance if total_importance > 0 else 0

            if pct > threshold:
                warnings.append({
                    "column":     col,
                    "importance": round(float(col_importance), 4),
                    "pct_of_total": round(pct, 4),
                    "plain_english": (
                        f"'{col}' appears to be influencing the model's predictions "
                        f"({pct:.1%} of total influence). If this column contains "
                        f"sensitive information such as age, gender, or ethnicity, "
                        f"you may want to consider whether this is appropriate for "
                        f"your use case before deploying."
                    )
                })

    return warnings
```

---

## Running the Full Explainability Pipeline

```python
def run_explainability(model, model_id, X_train, X_test,
                        y_test, feature_names, target_col,
                        sensitive_columns, session_id,
                        n_sample=500):

    output_dir = f"sessions/{session_id}/reports/explainability"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Sample for speed on large datasets
    sample_size = min(n_sample, len(X_test))
    X_sample    = X_test.iloc[:sample_size]

    # Get explainer
    explainer, explainer_type = get_explainer(model, X_train, model_id)

    # Global importance
    shap_vals, importance_df, importance_chart = compute_global_importance(
        explainer, X_sample, feature_names, model_id, output_dir
    )

    # Summary plot
    summary_chart = plot_shap_summary(
        shap_vals, pd.DataFrame(X_sample, columns=feature_names),
        feature_names, output_dir
    )

    # Individual explanations — explain first 3 rows as examples
    individual_explanations = []
    for i in range(min(3, len(X_sample))):
        row   = X_sample.iloc[[i]]
        chart = explain_single_prediction(
            explainer, row, feature_names,
            None, model_id, output_dir, row_index=i
        )
        individual_explanations.append({
            "row_index": i,
            "chart":     chart
        })

    # Bias check
    bias_warnings = check_for_bias(importance_df, sensitive_columns)

    # Global interpretation
    global_narration = interpret_global_importance(importance_df, target_col)

    return {
        "importance_df":           importance_df,
        "importance_chart":        importance_chart,
        "summary_chart":           summary_chart,
        "individual_explanations": individual_explanations,
        "bias_warnings":           bias_warnings,
        "global_narration":        global_narration,
        "explainer_type":          explainer_type
    }
```

---

## Output Written to Session

**Charts:**
`sessions/{session_id}/reports/explainability/`
- `feature_importance_shap.png`
- `shap_summary.png`
- `explanation_row_0.png`
- `explanation_row_1.png`
- `explanation_row_2.png`

**Result JSON:**
`sessions/{session_id}/outputs/explainability/result.json`

```json
{
  "stage": "explainability",
  "status": "success",
  "top_features": [ ... ],
  "bias_warnings": [ ... ],
  "global_narration": "...",
  "charts": [ ... ],
  "decisions_required": [],
  "plain_english_summary": "We analysed why the model makes the predictions it does. The three most influential factors are: '{feature_1}', '{feature_2}', and '{feature_3}'. We have also checked whether any sensitive columns are driving predictions and will share what we found.",
  "report_section": {
    "stage": "explainability",
    "title": "Why the Model Makes Its Predictions",
    "summary": "...",
    "decision_made": "No decisions required — this stage is for transparency.",
    "why_this_matters": "Understanding what the model pays attention to builds trust, catches unexpected patterns, and is often required before a model can be deployed in regulated environments."
  },
  "config_updates": {
    "top_features": [ ... ],
    "bias_warnings_count": 0
  }
}
```

---

## What to Tell the User

Opening:
"Now we are going to look inside the model — not just at how accurate it is,
but at *why* it makes the predictions it does. This is important for two reasons:
it helps you trust the model, and it helps catch any unexpected or problematic
patterns before it goes live."

Global importance:
"{global_narration}

Here is a chart showing all features ranked by their influence on the model's
decisions. [show feature_importance_shap.png]"

Summary plot:
"This chart shows more detail — not just which features matter, but whether
high or low values push the prediction up or down.
[show shap_summary.png with plain English caption]"

Individual explanations:
"Here are three examples of individual predictions and the reasons behind them:
[show waterfall charts with narration for each]"

Bias check:
If warnings exist:
"We noticed that {col} — which may contain sensitive information — is
influencing the model's predictions. Here is what we found: {warning}.
You should consider whether this is appropriate for your use case
before deploying the model."

If no warnings:
"We checked whether any sensitive columns are driving predictions —
we did not find any significant concerns."

---

## Reference Files

- `references/shap-guide.md` — how SHAP works and how to interpret each chart type
