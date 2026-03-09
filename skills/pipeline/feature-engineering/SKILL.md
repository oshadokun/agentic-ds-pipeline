---
name: feature-engineering
description: >
  Responsible for transforming raw cleaned columns into informative features that
  the model can learn from effectively. Always called by the Orchestrator after
  Cleaning completes. Handles categorical encoding, datetime expansion, feature
  creation, redundant feature removal, and feature selection. Uses findings from
  EDA to make informed decisions. Every significant transformation is explained to
  the user in plain English before being applied. Never removes a feature silently.
  Trigger when any of the following are mentioned: "feature engineering", "encode
  categories", "create features", "feature selection", "transform columns",
  "handle categorical", "datetime features", "drop redundant columns", or any
  request to prepare features for model training.
---

# Feature Engineering Skill

The Feature Engineering agent transforms cleaned data into a form the model can
learn from most effectively. It turns raw columns into meaningful signals — encoding
categories, extracting information from dates, creating new combinations, and removing
columns that add noise rather than signal.

Every transformation is explained in plain English. The user understands what changed
and why before the pipeline proceeds.

---

## Responsibilities

1. Encode categorical columns into numeric form
2. Expand datetime columns into useful component features
3. Identify and remove redundant or uninformative columns
4. Create interaction features where relevant
5. Handle high cardinality columns appropriately
6. Apply feature selection to reduce noise
7. Present significant decisions to the user
8. Write the transformed dataset to the session interim directory

---

## Feature Engineering Modules

### 1. Categorical Encoding

Different encoding strategies suit different situations.
Always choose based on column cardinality and task type.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ENCODING_STRATEGIES = {
    "onehot": {
        "label": "Create a separate yes/no column for each category (recommended for small number of categories)",
        "tradeoff": "Simple and effective. Creates extra columns — can become unwieldy with many categories.",
        "best_for": "Columns with fewer than 15 unique values"
    },
    "label": {
        "label": "Replace each category with a number",
        "tradeoff": "Compact but implies an order that may not exist — e.g. it treats 'blue=1, red=2' as if blue < red.",
        "best_for": "Tree-based models only (Random Forest, XGBoost). Not suitable for linear models."
    },
    "frequency": {
        "label": "Replace each category with how often it appears in the data",
        "tradeoff": "Useful when frequency is meaningful. Loses category identity.",
        "best_for": "High cardinality columns where frequency matters"
    },
    "target": {
        "label": "Replace each category with the average outcome for that category",
        "tradeoff": "Very powerful but requires careful handling to prevent data leakage. Applied using cross-validation.",
        "best_for": "High cardinality columns with a strong relationship to the target"
    },
    "drop": {
        "label": "Remove this column",
        "tradeoff": "Loses all information in the column. Only appropriate for ID columns or columns with no predictive value.",
        "best_for": "ID columns, free text with no structure"
    }
}

def recommend_encoding(col, df, target_col, task_type):
    n_unique = df[col].nunique()
    unique_pct = n_unique / len(df)

    if unique_pct > 0.9:
        recommended = "drop"
        reason = f"'{col}' has almost as many unique values as rows — it looks like an ID column."
    elif n_unique <= 2:
        recommended = "label"
        reason = f"'{col}' only has 2 values — we will convert them to 0 and 1."
    elif n_unique <= 15:
        recommended = "onehot"
        reason = f"'{col}' has {n_unique} categories — we will create a separate yes/no column for each."
    elif n_unique <= 50:
        recommended = "frequency"
        reason = f"'{col}' has {n_unique} categories — we will replace each with how often it appears."
    else:
        recommended = "target"
        reason = f"'{col}' has {n_unique} categories — we will replace each with the average outcome for that category."

    return recommended, reason


def apply_encoding(df, col, strategy, target_col=None):
    if strategy == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        msg = f"Created {len(dummies.columns)} yes/no columns from '{col}'."

    elif strategy == "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        msg = f"Converted '{col}' categories to numbers."

    elif strategy == "frequency":
        freq_map = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq_map)
        df = df.drop(columns=[col])
        msg = f"Replaced '{col}' with how often each category appears."

    elif strategy == "target" and target_col:
        # Simple target encoding with smoothing to reduce leakage risk
        global_mean = df[target_col].mean()
        agg = df.groupby(col)[target_col].agg(["mean", "count"])
        smoothing = 10
        agg["smoothed"] = (agg["mean"] * agg["count"] + global_mean * smoothing) / (agg["count"] + smoothing)
        df[col + "_target_enc"] = df[col].map(agg["smoothed"])
        df = df.drop(columns=[col])
        msg = f"Replaced '{col}' with the average outcome for each category (smoothed target encoding)."

    elif strategy == "drop":
        df = df.drop(columns=[col])
        msg = f"Removed '{col}' — it did not appear useful for prediction."

    else:
        msg = f"No encoding applied to '{col}'."

    return df, msg
```

---

### 2. Datetime Feature Expansion

```python
def expand_datetime(df, col):
    dt = pd.to_datetime(df[col])
    new_cols = []

    df[f"{col}_year"]        = dt.dt.year
    df[f"{col}_month"]       = dt.dt.month
    df[f"{col}_day"]         = dt.dt.day
    df[f"{col}_dayofweek"]   = dt.dt.dayofweek
    df[f"{col}_is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    df[f"{col}_quarter"]     = dt.dt.quarter

    # Cyclical encoding for month and day of week
    import numpy as np
    df[f"{col}_month_sin"]   = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_month_cos"]   = np.cos(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_dow_sin"]     = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df[f"{col}_dow_cos"]     = np.cos(2 * np.pi * dt.dt.dayofweek / 7)

    new_cols = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek",
                f"{col}_is_weekend", f"{col}_quarter",
                f"{col}_month_sin", f"{col}_month_cos",
                f"{col}_dow_sin", f"{col}_dow_cos"]

    df = df.drop(columns=[col])

    return df, (
        f"Expanded '{col}' into {len(new_cols)} date-based features: "
        f"year, month, day, day of week, weekend flag, quarter, "
        f"and cyclical encodings for month and day of week."
    )
```

**Plain English explanation for cyclical encoding:**
"Months and days of the week wrap around — December is close to January,
Sunday is close to Saturday. We use a mathematical technique to represent
this circular relationship so the model understands it correctly."

---

### 3. Redundant Feature Removal

Uses EDA findings to remove columns that are nearly identical to another column.

```python
def remove_redundant_features(df, target_col, high_corr_pairs, threshold=0.95):
    cols_to_drop = set()
    actions = []

    for pair in high_corr_pairs:
        col_a = pair["col_a"]
        col_b = pair["col_b"]
        corr  = pair["correlation"]

        if corr >= threshold:
            # Drop the one with lower correlation to target (keep more informative)
            if col_a in df.columns and col_b in df.columns and target_col in df.columns:
                corr_a = abs(df[col_a].corr(df[target_col]))
                corr_b = abs(df[col_b].corr(df[target_col]))
                to_drop = col_b if corr_a >= corr_b else col_a
                cols_to_drop.add(to_drop)
                actions.append(
                    f"Removed '{to_drop}' because it is almost identical to "
                    f"'{col_a if to_drop == col_b else col_b}' "
                    f"(similarity: {corr:.2f}). Keeping the one more related to the outcome."
                )

    df = df.drop(columns=list(cols_to_drop), errors="ignore")
    return df, actions
```

---

### 4. Feature Selection

After encoding, reduce to the most informative features.

```python
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np

def select_features(df, target_col, task_type, max_features=30):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fill any remaining nulls for selection only
    X_filled = X.fillna(X.median(numeric_only=True))

    # Use mutual information — works for both classification and regression
    if "classification" in task_type:
        scores = mutual_info_classif(X_filled, y, random_state=42)
    else:
        scores = mutual_info_regression(X_filled, y, random_state=42)

    importance_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": scores
    }).sort_values("importance", ascending=False)

    # Keep top features — but always keep at least 5
    n_keep = min(max_features, max(5, len(importance_df)))
    selected = importance_df.head(n_keep)["feature"].tolist()
    dropped  = importance_df.tail(len(importance_df) - n_keep)["feature"].tolist()

    plain_english = (
        f"We measured how informative each column is for predicting '{target_col}'. "
        f"We kept the top {n_keep} most informative columns"
    )
    if dropped:
        plain_english += f" and removed {len(dropped)} columns that added little value."
    else:
        plain_english += "."

    return selected + [target_col], importance_df, plain_english
```

---

### 5. Decisions Required from User

| Situation | Present to user? |
|---|---|
| Encoding strategy for each categorical column | Yes — show recommendation and alternatives |
| Dropping a column identified as ID-like | Yes — confirm before dropping |
| Expanding datetime columns | No — apply automatically, inform in summary |
| Removing redundant correlated features | Yes — explain which pair, which is kept |
| Feature selection results | Yes — show importance ranking, ask for approval |

---

## Running Full Feature Engineering Pipeline

```python
def run_feature_engineering(df, target_col, task_type, eda_findings,
                             user_decisions, session_id):
    actions_log = []

    # 1. Remove redundant features using EDA correlation findings
    high_corr_pairs = eda_findings.get("correlations", {}).get("high_corr_pairs", [])
    df, redundant_actions = remove_redundant_features(df, target_col, high_corr_pairs)
    actions_log.extend(redundant_actions)

    # 2. Expand datetime columns
    datetime_cols = df.select_dtypes("datetime").columns.tolist()
    for col in datetime_cols:
        df, msg = expand_datetime(df, col)
        actions_log.append(msg)

    # 3. Apply user-confirmed encoding strategies
    cat_cols = df.select_dtypes("object").columns.tolist()
    for col in cat_cols:
        if col == target_col:
            continue
        strategy = user_decisions.get("encoding", {}).get(col, "onehot")
        df, msg = apply_encoding(df, col, strategy, target_col)
        actions_log.append(msg)

    # 4. Feature selection — present results to user before finalising
    selected_features, importance_df, selection_msg = select_features(
        df, target_col, task_type
    )
    actions_log.append(selection_msg)

    # Apply user-confirmed feature selection
    if user_decisions.get("feature_selection_approved"):
        df = df[selected_features]

    # Save
    output_path = f"sessions/{session_id}/data/interim/features.csv"
    df.to_csv(output_path, index=False)

    return df, actions_log, importance_df, output_path
```

---

## Output Written to Session

**Transformed data:**
`sessions/{session_id}/data/interim/features.csv`

**Feature importance chart:**
`sessions/{session_id}/reports/feature_importance.png`

**Result JSON:**
`sessions/{session_id}/outputs/feature_engineering/result.json`

```json
{
  "stage": "feature_engineering",
  "status": "success",
  "output_data_path": "sessions/{session_id}/data/interim/features.csv",
  "actions_taken": [ ... ],
  "feature_importance": { ... },
  "decisions_required": [ ... ],
  "plain_english_summary": "We transformed your data ready for modelling. We encoded 4 category columns, expanded 1 date column into 10 features, removed 2 redundant columns, and selected the 20 most informative features.",
  "report_section": {
    "stage": "feature_engineering",
    "title": "Preparing Your Features",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "Models can only learn from numbers — this stage turns all your data into a form the model can understand, while keeping the most useful information."
  },
  "config_updates": {
    "feature_columns": [ ... ],
    "encoding_strategies": { ... },
    "features_selected": 20,
    "features_dropped": 4
  }
}
```

---

## What to Tell the User

Before encoding decisions:
"Your data contains {n} text or category columns. The model cannot use text
directly — we need to convert them to numbers. Here is what we recommend for each:"

After feature selection:
"We measured how useful each column is for predicting '{target_col}'.
Here are the most important ones: [top 5 in plain English].
Here are the ones we recommend removing: [list with reason].
Does this look right to you?"

---

## Reference Files

- `references/encoding-guide.md` — detailed guidance on encoding strategies by scenario
- `references/feature-selection-guide.md` — when and how to apply feature selection
