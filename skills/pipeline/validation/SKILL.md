---
name: validation
description: >
  Responsible for checking that loaded data meets the minimum quality standards
  required for the pipeline to proceed reliably. Always called by the Orchestrator
  immediately after Ingestion. Validates schema, data types, target column presence,
  class distribution, missing value thresholds, and duplicate levels. Categorises
  every finding as a hard stop, a warning requiring user decision, or an advisory
  note. Never transforms data — only inspects and reports. Trigger when any of the
  following are mentioned: "validate data", "check data quality", "data checks",
  "schema validation", or any point in the pipeline where data quality needs to be
  confirmed before proceeding.
---

# Validation Skill

The Validation agent is the quality gate of the pipeline. Its job is to inspect the
loaded data thoroughly and determine whether it is safe to proceed. It does not fix
anything — that is the Cleaning agent's job. It finds problems, categorises them by
severity, and reports everything clearly to the Orchestrator.

Nothing proceeds past validation until the user has acknowledged every hard stop and
confirmed every warning that requires a decision.

---

## Responsibilities

1. Confirm the target column exists and contains usable values
2. Validate data types are appropriate for the task
3. Check missing value levels across all columns
4. Check for duplicate rows
5. Check class distribution (classification tasks)
6. Check dataset size is sufficient
7. Check for constant or near-constant columns
8. Categorise every finding by severity
9. Return a full validation report to the Orchestrator

---

## Validation Checks

### 1. Target Column
```python
def validate_target(df, target_col, task_type):
    results = []

    # Does it exist?
    if target_col not in df.columns:
        results.append({
            "check": "target_exists",
            "severity": "hard_stop",
            "message": f"The column '{target_col}' was not found in your data.",
            "action": "Please check the column name and try again."
        })
        return results  # Cannot continue without target

    target = df[target_col]

    # Is it entirely empty?
    if target.isna().all():
        results.append({
            "check": "target_not_empty",
            "severity": "hard_stop",
            "message": f"The column '{target_col}' appears to be completely empty.",
            "action": "Please check your data source and try again."
        })

    # Missing values in target
    missing_pct = target.isna().mean()
    if missing_pct > 0.05:
        results.append({
            "check": "target_missing_values",
            "severity": "hard_stop",
            "message": f"{missing_pct:.1%} of your target column values are missing.",
            "action": "Rows with a missing target value cannot be used for training. We recommend removing them."
        })
    elif missing_pct > 0:
        results.append({
            "check": "target_missing_values",
            "severity": "warning",
            "message": f"{target.isna().sum()} rows have a missing target value ({missing_pct:.1%}).",
            "action": "These rows will be dropped before training."
        })

    # Classification: check class distribution
    if task_type == "binary_classification":
        counts = target.value_counts(normalize=True)
        minority_pct = counts.min()
        if minority_pct < 0.05:
            results.append({
                "check": "class_imbalance",
                "severity": "warning",
                "message": f"Your data is heavily imbalanced — only {minority_pct:.1%} of rows belong to the minority class.",
                "action": "We will handle this during training. You will be asked to choose a strategy."
            })
        elif minority_pct < 0.20:
            results.append({
                "check": "class_imbalance",
                "severity": "advisory",
                "message": f"Your data has some imbalance — {minority_pct:.1%} minority class. We will keep an eye on this."
            })

    return results
```

---

### 2. Dataset Size
```python
def validate_size(df):
    results = []
    n_rows = len(df)

    if n_rows < 50:
        results.append({
            "check": "minimum_rows",
            "severity": "hard_stop",
            "message": f"Your dataset only has {n_rows} rows. This is too small to train a reliable model.",
            "action": "A minimum of 50 rows is needed. Ideally 500 or more."
        })
    elif n_rows < 200:
        results.append({
            "check": "minimum_rows",
            "severity": "warning",
            "message": f"Your dataset has {n_rows} rows. This is quite small — the model may not learn reliably.",
            "action": "We will use cross-validation to get the most out of the available data."
        })
    elif n_rows < 1000:
        results.append({
            "check": "minimum_rows",
            "severity": "advisory",
            "message": f"Your dataset has {n_rows} rows. This is workable but more data generally produces better results."
        })

    return results
```

---

### 3. Missing Values
```python
def validate_missing(df):
    results = []
    missing = df.isna().mean()

    for col, pct in missing[missing > 0].items():
        if pct > 0.8:
            results.append({
                "check": "missing_values",
                "severity": "warning",
                "column": col,
                "missing_pct": pct,
                "message": f"'{col}' is {pct:.1%} empty. This column may not be useful.",
                "action": "We recommend dropping this column. You will be asked to confirm."
            })
        elif pct > 0.3:
            results.append({
                "check": "missing_values",
                "severity": "advisory",
                "column": col,
                "missing_pct": pct,
                "message": f"'{col}' has {pct:.1%} missing values. We will handle this during cleaning."
            })

    return results
```

---

### 4. Duplicate Rows
```python
def validate_duplicates(df):
    results = []
    dup_count = df.duplicated().sum()
    dup_pct = dup_count / len(df)

    if dup_pct > 0.1:
        results.append({
            "check": "duplicates",
            "severity": "warning",
            "message": f"{dup_count} duplicate rows found ({dup_pct:.1%} of your data).",
            "action": "We recommend removing duplicates. You will be asked to confirm."
        })
    elif dup_count > 0:
        results.append({
            "check": "duplicates",
            "severity": "advisory",
            "message": f"{dup_count} duplicate rows found. These will be removed during cleaning."
        })

    return results
```

---

### 5. Constant and Near-Constant Columns
```python
def validate_variance(df):
    results = []

    for col in df.columns:
        n_unique = df[col].nunique()
        if n_unique == 1:
            results.append({
                "check": "constant_column",
                "severity": "warning",
                "column": col,
                "message": f"'{col}' has only one unique value — it contains no useful information.",
                "action": "We recommend dropping this column. You will be asked to confirm."
            })
        elif n_unique / len(df) > 0.95 and df[col].dtype == object:
            results.append({
                "check": "high_cardinality",
                "severity": "advisory",
                "column": col,
                "message": f"'{col}' has {n_unique} unique text values — it may be an ID column rather than a useful feature.",
                "action": "We will flag this during feature engineering."
            })

    return results
```

---

### 6. Data Type Consistency
```python
def validate_dtypes(df, target_col):
    results = []

    for col in df.columns:
        if col == target_col:
            continue
        # Numeric columns stored as text
        if df[col].dtype == object:
            sample = df[col].dropna().head(100)
            numeric_pct = sample.str.match(r"^-?\d+\.?\d*$").mean()
            if numeric_pct > 0.9:
                results.append({
                    "check": "dtype_mismatch",
                    "severity": "advisory",
                    "column": col,
                    "message": f"'{col}' looks like it contains numbers but is stored as text.",
                    "action": "We will convert this to a number during cleaning."
                })

    return results
```

---

## Severity Levels

| Severity | Meaning | Pipeline Action |
|---|---|---|
| `hard_stop` | Cannot proceed safely | Pipeline halts. User must resolve before continuing. |
| `warning` | Significant issue requiring a decision | Pipeline pauses. User is shown the issue and asked to choose an action. |
| `advisory` | Notable but not blocking | Shown to user as information. Pipeline continues automatically. |

---

## Running All Checks

```python
def run_validation(df, target_col, task_type):
    all_results = []
    all_results += validate_target(df, target_col, task_type)
    all_results += validate_size(df)
    all_results += validate_missing(df)
    all_results += validate_duplicates(df)
    all_results += validate_variance(df)
    all_results += validate_dtypes(df, target_col)

    hard_stops = [r for r in all_results if r["severity"] == "hard_stop"]
    warnings   = [r for r in all_results if r["severity"] == "warning"]
    advisories = [r for r in all_results if r["severity"] == "advisory"]

    overall_status = "hard_stop" if hard_stops else "warnings" if warnings else "passed"

    return {
        "overall_status": overall_status,
        "hard_stops": hard_stops,
        "warnings": warnings,
        "advisories": advisories,
        "total_checks_run": len(all_results)
    }
```

---

## Output Written to Session

**No data file is written** — Validation does not transform data.
The validated data path remains: `sessions/{session_id}/data/raw/ingested.csv`

**Result JSON:**
`sessions/{session_id}/outputs/validation/result.json`

```json
{
  "stage": "validation",
  "status": "warnings",
  "output_data_path": "sessions/{session_id}/data/raw/ingested.csv",
  "validation_report": {
    "overall_status": "warnings",
    "hard_stops": [],
    "warnings": [ ... ],
    "advisories": [ ... ],
    "total_checks_run": 14
  },
  "decisions_required": [ ... ],
  "plain_english_summary": "Your data passed all critical checks. We found 2 things worth your attention before we continue.",
  "report_section": {
    "stage": "validation",
    "title": "Checking Your Data Quality",
    "summary": "We ran 14 checks on your data.",
    "decision_made": "...",
    "alternatives_considered": null,
    "why_this_matters": "Catching data quality issues early means the model learns from reliable information rather than noise."
  },
  "config_updates": {
    "validation_passed": true,
    "class_imbalance_detected": false
  }
}
```

---

## What to Tell the User

Present findings grouped by severity using clear, non-alarming language:

**If hard stops exist:**
"Before we can continue, there is something that needs your attention:
[list each hard stop with its plain English message and required action]"

**If warnings exist:**
"We found a couple of things worth discussing before we go further:
[list each warning with its message and the options available]"

**If only advisories:**
"Your data looks good. A few small things to be aware of:
[list advisories briefly]
We can continue when you're ready."

**If everything passes:**
"Great news — your data passed all our quality checks. Everything looks good to
proceed to the next stage."

---

## Reference Files

- `references/validation-thresholds.md` — configurable thresholds for all checks
