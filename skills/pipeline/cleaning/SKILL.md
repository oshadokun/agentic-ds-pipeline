---
name: cleaning
description: >
  Responsible for fixing data quality issues identified during Validation and EDA.
  Always called by the Orchestrator after EDA completes. Handles missing values,
  duplicate rows, outliers, data type corrections, and inconsistent formatting.
  Every significant cleaning decision is presented to the user with alternatives
  and tradeoffs before being applied. Never silently drops or modifies data without
  user awareness. Writes a cleaned dataset to the session interim directory. Trigger
  when any of the following are mentioned: "clean data", "handle missing values",
  "remove duplicates", "fix outliers", "data cleaning", "imputation", "fix data
  types", or any request to prepare raw data for modelling.
---

# Cleaning Skill

The Cleaning agent takes the validated data and fixes the quality issues found during
Validation and EDA. It is the first agent that actually modifies the data. Because of
this, every significant decision must be shown to the user with a clear explanation of
what is being done, why, and what the alternatives are.

The guiding principle is: **never lose data silently**. If rows or columns are being
removed, the user knows exactly how many and why.

---

## Responsibilities

1. Remove duplicate rows
2. Handle missing values in each column
3. Fix data type inconsistencies
4. Handle outliers
5. Standardise text formatting in categorical columns
6. Present every significant decision to the user before applying it
7. Write the cleaned dataset to the session interim directory
8. Record every action taken for the report

---

## Cleaning Modules

### 1. Duplicate Removal
```python
def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    return df, {
        "rows_removed": removed,
        "rows_remaining": len(df),
        "plain_english": f"We removed {removed} duplicate rows. Your dataset now has {len(df)} rows."
    }
```

**When to present to user:**
- Always inform the user how many duplicates were removed
- If > 10% of rows are duplicates, ask for confirmation before removing

---

### 2. Missing Value Handling

This is the most consequential cleaning decision. Always present options to the user.

```python
from sklearn.impute import SimpleImputer, KNNImputer

IMPUTE_STRATEGIES = {
    "median": {
        "label": "Fill with the middle value (recommended for numbers with outliers)",
        "tradeoff": "Safe and reliable. Not thrown off by extreme values."
    },
    "mean": {
        "label": "Fill with the average value",
        "tradeoff": "Simple but can be distorted by very high or very low values."
    },
    "knn": {
        "label": "Fill using similar rows (most accurate, slower)",
        "tradeoff": "Uses patterns from nearby rows to estimate the missing value. Best quality but takes longer on large datasets."
    },
    "mode": {
        "label": "Fill with the most common value (for categories)",
        "tradeoff": "Standard approach for text columns. Assumes the most common value is the best guess."
    },
    "drop_rows": {
        "label": "Remove rows with missing values",
        "tradeoff": "Cleanest option but you lose data. Only recommended if fewer than 5% of rows are affected."
    },
    "drop_col": {
        "label": "Remove this column entirely",
        "tradeoff": "Appropriate when a column is more than 80% empty and unlikely to be useful."
    }
}

def recommend_impute_strategy(col, df, missing_pct):
    """
    Recommend the best strategy based on column type and missing percentage.
    Always return the recommendation AND the alternatives.
    """
    dtype = df[col].dtype

    if missing_pct > 0.8:
        recommended = "drop_col"
    elif missing_pct < 0.05:
        recommended = "drop_rows"
    elif dtype in ["int64", "float64"]:
        # Check for outliers — if present, median is safer
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        has_outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).any()
        recommended = "median" if has_outliers else "mean"
    else:
        recommended = "mode"

    # Build alternatives list — recommended first, others follow
    if dtype in ["int64", "float64"]:
        alternatives = ["median", "mean", "knn", "drop_rows", "drop_col"]
    else:
        alternatives = ["mode", "drop_rows", "drop_col"]

    return recommended, [IMPUTE_STRATEGIES[s] | {"id": s} for s in alternatives]


def apply_imputation(df, col, strategy):
    if strategy == "drop_rows":
        before = len(df)
        df = df.dropna(subset=[col])
        return df, f"Removed {before - len(df)} rows where '{col}' was missing."

    elif strategy == "drop_col":
        df = df.drop(columns=[col])
        return df, f"Removed column '{col}' — it was too empty to be useful."

    elif strategy == "median":
        fill_val = df[col].median()
        df[col] = df[col].fillna(fill_val)
        return df, f"Filled missing values in '{col}' with the middle value ({fill_val:.2f})."

    elif strategy == "mean":
        fill_val = df[col].mean()
        df[col] = df[col].fillna(fill_val)
        return df, f"Filled missing values in '{col}' with the average ({fill_val:.2f})."

    elif strategy == "mode":
        fill_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(fill_val)
        return df, f"Filled missing values in '{col}' with the most common value ('{fill_val}')."

    elif strategy == "knn":
        import numpy as np
        imputer = KNNImputer(n_neighbors=5)
        numeric_cols = df.select_dtypes("number").columns.tolist()
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df, f"Filled missing values in '{col}' using patterns from similar rows."

    return df, "No action taken."
```

---

### 3. Outlier Handling

```python
OUTLIER_STRATEGIES = {
    "cap": {
        "label": "Cap at the boundary values (recommended)",
        "tradeoff": "Keeps all rows but limits extreme values. Good when outliers might be genuine but disproportionate."
    },
    "remove": {
        "label": "Remove rows with extreme values",
        "tradeoff": "Cleaner data but you lose rows. Only recommended if outliers are clearly errors."
    },
    "keep": {
        "label": "Keep as-is",
        "tradeoff": "No data lost but extreme values may affect the model. Some models handle this better than others."
    }
}

def handle_outliers(df, col, strategy):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_mask = (df[col] < lower) | (df[col] > upper)
    outlier_count = outlier_mask.sum()

    if strategy == "cap":
        df[col] = df[col].clip(lower=lower, upper=upper)
        return df, f"Capped {outlier_count} extreme values in '{col}' at the boundary values ({lower:.2f} to {upper:.2f})."

    elif strategy == "remove":
        before = len(df)
        df = df[~outlier_mask]
        return df, f"Removed {before - len(df)} rows with extreme values in '{col}'."

    elif strategy == "keep":
        return df, f"Kept extreme values in '{col}' as-is ({outlier_count} values outside the normal range)."

    return df, "No action taken."
```

**When to present outlier decisions:**
Only for columns where outlier % > 1%. For very minor outlier counts, default to
cap and inform the user in the summary without asking.

---

### 4. Data Type Corrections
```python
def fix_dtypes(df):
    actions = []
    for col in df.columns:
        if df[col].dtype == object:
            # Try converting to numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().mean() > 0.9:
                df[col] = converted
                actions.append(f"Converted '{col}' from text to number.")

            # Try converting to datetime
            elif df[col].str.match(r"\d{4}-\d{2}-\d{2}").mean() > 0.8:
                try:
                    df[col] = pd.to_datetime(df[col])
                    actions.append(f"Converted '{col}' to a date column.")
                except Exception:
                    pass

    return df, actions
```

---

### 5. Categorical Text Standardisation
```python
def standardise_categoricals(df):
    actions = []
    for col in df.select_dtypes("object").columns:
        # Strip whitespace
        df[col] = df[col].str.strip()
        # Lowercase (preserves meaning, aids consistency)
        original_unique = df[col].nunique()
        df[col] = df[col].str.lower()
        new_unique = df[col].nunique()
        if new_unique < original_unique:
            actions.append(f"Standardised '{col}' — merged {original_unique - new_unique} variants caused by capitalisation differences.")

    return df, actions
```

---

## Decisions Required from User

For each of the following, the Orchestrator must pause and present options:

| Situation | Present to user? |
|---|---|
| Column > 80% missing | Yes — recommend drop, show alternatives |
| Column 5–80% missing | Yes — recommend strategy, show alternatives |
| Column < 5% missing | No — apply recommended strategy, inform in summary |
| Outliers > 1% of rows | Yes — recommend cap, show alternatives |
| Outliers < 1% of rows | No — cap automatically, inform in summary |
| Duplicates > 10% | Yes — confirm before removing |
| Duplicates < 10% | No — remove automatically, inform in summary |
| Data type correction | No — apply automatically, inform in summary |
| Text standardisation | No — apply automatically, inform in summary |

---

## Running Full Cleaning Pipeline

```python
def run_cleaning(df, target_col, eda_findings, user_decisions, session_id):
    """
    user_decisions: dict of {column: strategy} confirmed by user via UI
    """
    actions_log = []

    # 1. Duplicates
    df, dup_result = remove_duplicates(df)
    actions_log.append(dup_result)

    # 2. Data type fixes
    df, dtype_actions = fix_dtypes(df)
    actions_log.extend(dtype_actions)

    # 3. Text standardisation
    df, text_actions = standardise_categoricals(df)
    actions_log.extend(text_actions)

    # 4. Missing values — apply user decisions
    for col, strategy in user_decisions.get("missing", {}).items():
        if col in df.columns:
            df, msg = apply_imputation(df, col, strategy)
            actions_log.append(msg)

    # 5. Outliers — apply user decisions
    for col, strategy in user_decisions.get("outliers", {}).items():
        if col in df.columns:
            df, msg = handle_outliers(df, col, strategy)
            actions_log.append(msg)

    # Save
    output_path = f"sessions/{session_id}/data/interim/cleaned.csv"
    df.to_csv(output_path, index=False)

    return df, actions_log, output_path
```

---

## Output Written to Session

**Cleaned data:**
`sessions/{session_id}/data/interim/cleaned.csv`

**Result JSON:**
`sessions/{session_id}/outputs/cleaning/result.json`

```json
{
  "stage": "cleaning",
  "status": "success",
  "output_data_path": "sessions/{session_id}/data/interim/cleaned.csv",
  "actions_taken": [ ... ],
  "rows_before": 1204,
  "rows_after": 1187,
  "cols_before": 18,
  "cols_after": 16,
  "decisions_required": [ ... ],
  "plain_english_summary": "We cleaned your data. We removed 17 duplicate rows and filled in missing values across 3 columns. Your dataset now has 1,187 rows and 16 columns ready for the next stage.",
  "report_section": {
    "stage": "cleaning",
    "title": "Cleaning Your Data",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "Clean data means the model learns from reliable information rather than gaps and errors."
  },
  "config_updates": {
    "impute_strategies": { ... },
    "outlier_strategies": { ... },
    "dropped_columns": [ ... ],
    "rows_removed": 17
  }
}
```

---

## What to Tell the User

Before cleaning begins:
"Here is what we found that needs attention. For each item, we have a recommendation
— but you can choose a different approach if you prefer."

After cleaning completes:
"Your data has been cleaned. Here is a summary of everything we did:
[list each action in plain English]
Your dataset now has {rows_after} rows and {cols_after} columns."

Always end with:
"Nothing has been lost without your knowledge. Here is exactly what changed."

---

## Reference Files

- `references/imputation-guide.md` — when to use each imputation strategy in depth
- `references/outlier-guide.md` — when outliers should be kept vs removed vs capped
