---
name: normalisation
description: >
  Responsible for scaling numeric features so they are on a comparable range before
  model training. Always called by the Orchestrator after Feature Engineering completes.
  Selects the appropriate scaling strategy based on the data's characteristics and the
  planned model type. Critically enforces the rule that scalers are fitted on training
  data only — never on the full dataset — to prevent data leakage. Saves the fitted
  scaler for use during deployment. Explains every decision in plain English. Trigger
  when any of the following are mentioned: "normalise", "scale features", "standardise
  data", "MinMaxScaler", "StandardScaler", "RobustScaler", "data leakage", "scaling",
  or any request to prepare numeric features for a model that is sensitive to value ranges.
---

# Normalisation Skill

The Normalisation agent scales numeric features so the model treats them fairly.
Without scaling, a column with values in the thousands would dominate a column with
values between 0 and 1 — not because it is more important, but simply because its
numbers are bigger.

This skill has one critical rule that must never be broken:

**The scaler is always fitted on training data only. Never on the full dataset.**

Fitting on the full dataset leaks information about the test set into the model —
making performance estimates falsely optimistic. This is called data leakage and
it is one of the most common and damaging mistakes in machine learning.

---

## Responsibilities

1. Identify which columns need scaling
2. Recommend the appropriate scaling strategy based on data characteristics and model type
3. Present the recommendation and alternatives to the user
4. Fit the scaler on training data only
5. Transform training, validation, and test sets using the fitted scaler
6. Save the fitted scaler for use during deployment
7. Explain every decision in plain English

---

## When Scaling is and is Not Needed

```python
MODELS_REQUIRING_SCALING = [
    "logistic_regression", "linear_regression", "ridge", "lasso",
    "elasticnet", "svm", "knn", "neural_network", "pca"
]

MODELS_NOT_REQUIRING_SCALING = [
    "random_forest", "xgboost", "lightgbm", "catboost",
    "decision_tree", "gradient_boosting"
]

def scaling_required(model_type):
    if model_type in MODELS_REQUIRING_SCALING:
        return True, "The model you are using is sensitive to the scale of your numbers — scaling is required."
    elif model_type in MODELS_NOT_REQUIRING_SCALING:
        return False, "The model you are using is not sensitive to scale — scaling is optional but will not cause harm."
    else:
        return True, "We recommend scaling as a safe default."
```

**Always scale** when the model is not yet chosen — it keeps all options open.

---

## Scaling Strategies

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)

SCALING_STRATEGIES = {
    "standard": {
        "label": "Standard scaling — centre around zero with consistent spread (recommended default)",
        "tradeoff": "Works well for most data. Sensitive to outliers — if your data has extreme values, consider Robust scaling instead.",
        "best_for": "Normally distributed data, linear models, PCA, neural networks",
        "scaler": StandardScaler()
    },
    "minmax": {
        "label": "Min-Max scaling — squeeze all values into the range 0 to 1",
        "tradeoff": "Bounded range is useful for neural networks. Very sensitive to outliers — one extreme value will compress everything else.",
        "best_for": "Neural networks, when a bounded range is required",
        "scaler": MinMaxScaler()
    },
    "robust": {
        "label": "Robust scaling — scales using the middle 50% of values, ignoring extremes",
        "tradeoff": "Much less affected by outliers than Standard or Min-Max. Slightly less intuitive.",
        "best_for": "Data with significant outliers that cannot be removed",
        "scaler": RobustScaler()
    },
    "power": {
        "label": "Power transformation — makes skewed distributions more symmetrical",
        "tradeoff": "Effective for heavily skewed data. More complex transformation — harder to interpret.",
        "best_for": "Heavily skewed numeric columns before linear models",
        "scaler": PowerTransformer(method="yeo-johnson")
    },
    "none": {
        "label": "No scaling — leave values as they are",
        "tradeoff": "Only appropriate for tree-based models (Random Forest, XGBoost) which do not need scaling.",
        "best_for": "Tree-based models only",
        "scaler": None
    }
}
```

---

## Recommending a Strategy

```python
import numpy as np

def recommend_scaling_strategy(df, numeric_cols, model_type=None):
    """
    Recommend a scaling strategy based on data characteristics.
    """

    # Check for outliers
    outlier_cols = []
    skewed_cols  = []

    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outlier_pct = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).mean()
        if outlier_pct > 0.05:
            outlier_cols.append(col)

        skew = abs(df[col].skew())
        if skew > 1.5:
            skewed_cols.append(col)

    # Decision logic
    if model_type in MODELS_NOT_REQUIRING_SCALING:
        recommended = "none"
        reason = "The model you are using handles different scales natively — scaling is not required."

    elif len(outlier_cols) > len(numeric_cols) * 0.3:
        recommended = "robust"
        reason = f"{len(outlier_cols)} of your columns have significant extreme values. Robust scaling handles these well."

    elif len(skewed_cols) > len(numeric_cols) * 0.3:
        recommended = "power"
        reason = f"{len(skewed_cols)} of your columns have skewed distributions. Power transformation will make them more symmetrical."

    elif model_type == "neural_network":
        recommended = "minmax"
        reason = "Neural networks often perform better when inputs are in the 0 to 1 range."

    else:
        recommended = "standard"
        reason = "Standard scaling is the most widely used approach and works well for your data."

    return recommended, reason, outlier_cols, skewed_cols
```

---

## Applying Scaling — The Critical Rule

```python
import pickle
from pathlib import Path

def apply_scaling(X_train, X_val, X_test, numeric_cols, strategy, session_id):
    """
    CRITICAL: Fit ONLY on X_train. Transform all three sets.
    Never fit on X_val, X_test, or the full dataset.
    """

    if strategy == "none":
        return X_train, X_val, X_test, None, "No scaling applied."

    scaler = SCALING_STRATEGIES[strategy]["scaler"]

    # Fit on training data ONLY
    scaler.fit(X_train[numeric_cols])

    # Transform all three splits using the SAME fitted scaler
    X_train = X_train.copy()
    X_val   = X_val.copy()
    X_test  = X_test.copy()

    X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val[numeric_cols]   = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols]  = scaler.transform(X_test[numeric_cols])

    # Save scaler — needed at deployment to scale new incoming data
    scaler_path = f"sessions/{session_id}/models/scaler.pkl"
    Path(f"sessions/{session_id}/models").mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    msg = (
        f"Scaled {len(numeric_cols)} numeric columns using {SCALING_STRATEGIES[strategy]['label']}. "
        f"The scaling was learned from your training data only and then applied consistently "
        f"to all data splits. The scaler has been saved for use when the model is deployed."
    )

    return X_train, X_val, X_test, scaler, msg
```

---

## Columns That Should NOT Be Scaled

```python
def identify_cols_to_skip(df, target_col):
    """
    Some columns should never be scaled.
    """
    skip = []

    for col in df.columns:
        if col == target_col:
            skip.append((col, "This is your target column — never scale the target."))
            continue

        # Binary columns (0/1) — already on a 0-1 scale
        if set(df[col].dropna().unique()).issubset({0, 1}):
            skip.append((col, f"'{col}' is already a 0/1 column — scaling would change its meaning."))
            continue

        # One-hot encoded columns
        if df[col].nunique() == 2 and df[col].min() == 0 and df[col].max() == 1:
            skip.append((col, f"'{col}' is a yes/no column — it does not need scaling."))

    return skip
```

---

## Before and After Visualisation

Always show the user a before/after comparison for the most important columns.

```python
import matplotlib.pyplot as plt

def plot_scaling_comparison(X_before, X_after, cols, output_dir, n=4):
    cols_to_plot = cols[:n]
    fig, axes = plt.subplots(2, len(cols_to_plot), figsize=(4*len(cols_to_plot), 6))

    for i, col in enumerate(cols_to_plot):
        # Before
        axes[0, i].hist(X_before[col].dropna(), bins=30, color="#AECDE8", edgecolor="white")
        axes[0, i].set_title(f"{col}\nBefore scaling", fontsize=9)

        # After
        axes[1, i].hist(X_after[col].dropna(), bins=30, color="#2E75B6", edgecolor="white")
        axes[1, i].set_title(f"After scaling", fontsize=9)

    plt.suptitle(
        "How scaling changed your numeric columns\n"
        "The shape stays the same — only the numbers on the axis change.",
        fontsize=11
    )
    plt.tight_layout()
    path = Path(output_dir) / "scaling_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return str(path)
```

**Plain English caption for the chart:**
"Scaling changes the numbers but not the shape. The model now sees all columns
on a comparable scale — none will dominate simply because its values happen to
be larger."

---

## Running the Full Normalisation Pipeline

```python
def run_normalisation(X_train, X_val, X_test, target_col,
                      model_type, user_decision, session_id):

    numeric_cols = X_train.select_dtypes("number").columns.tolist()
    cols_to_skip = [c for c, _ in identify_cols_to_skip(
        X_train, target_col
    )]
    numeric_cols = [c for c in numeric_cols if c not in cols_to_skip]

    strategy = user_decision.get("scaling_strategy", "standard")

    X_train_s, X_val_s, X_test_s, scaler, msg = apply_scaling(
        X_train, X_val, X_test, numeric_cols, strategy, session_id
    )

    output_dir = f"sessions/{session_id}/reports"
    chart_path = plot_scaling_comparison(X_train, X_train_s, numeric_cols, output_dir)

    # Save scaled splits
    base = f"sessions/{session_id}/data/processed"
    Path(base).mkdir(parents=True, exist_ok=True)
    X_train_s.to_csv(f"{base}/X_train.csv", index=False)
    X_val_s.to_csv(f"{base}/X_val.csv", index=False)
    X_test_s.to_csv(f"{base}/X_test.csv", index=False)

    return X_train_s, X_val_s, X_test_s, scaler, msg, chart_path
```

---

## Output Written to Session

**Scaled data splits:**
- `sessions/{session_id}/data/processed/X_train.csv`
- `sessions/{session_id}/data/processed/X_val.csv`
- `sessions/{session_id}/data/processed/X_test.csv`

**Fitted scaler:**
`sessions/{session_id}/models/scaler.pkl`

**Chart:**
`sessions/{session_id}/reports/scaling_comparison.png`

**Result JSON:**
`sessions/{session_id}/outputs/normalisation/result.json`

```json
{
  "stage": "normalisation",
  "status": "success",
  "scaling_strategy": "standard",
  "columns_scaled": [ ... ],
  "columns_skipped": [ ... ],
  "scaler_path": "sessions/{session_id}/models/scaler.pkl",
  "chart_path": "sessions/{session_id}/reports/scaling_comparison.png",
  "decisions_required": [ ... ],
  "plain_english_summary": "We scaled your 12 numeric columns using Standard scaling. The shape of your data has not changed — only the numbers on the axis. The scaler has been saved so new data can be scaled consistently when the model is deployed.",
  "report_section": {
    "stage": "normalisation",
    "title": "Scaling Your Data",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "Scaling ensures the model treats all columns fairly — a column with values in the thousands will not overshadow a column with values between 0 and 1 simply because its numbers are bigger."
  },
  "config_updates": {
    "scaling_strategy": "standard",
    "scaled_columns": [ ... ],
    "scaler_path": "..."
  }
}
```

---

## What to Tell the User

Before scaling:
"Some of your columns have very different value ranges — for example, one might
go from 0 to 1 while another goes from 0 to 1,000,000. We need to put them on
a comparable scale so the model treats them fairly. Here is what we recommend:"

After scaling:
"Your data has been scaled. The shape of your data has not changed — only the
numbers on the axis. Here is a before and after view of your most important columns."

Always explain the leakage rule in plain English if the user asks:
"We only learned the scaling from your training data. If we had used all your data
to learn the scaling, the model would have had an unfair peek at the test data —
making it look better than it really is. This is called data leakage and we always
avoid it."

---

## Reference Files

- `references/scaling-guide.md` — detailed guidance on each strategy with worked examples
