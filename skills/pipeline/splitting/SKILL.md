---
name: splitting
description: >
  Responsible for dividing the dataset into training, validation, and test sets
  before model training. Always called by the Orchestrator after Feature Engineering
  and before Normalisation. Selects the appropriate splitting strategy based on task
  type, dataset size, and data characteristics. Enforces correct temporal splitting
  for time series data. Handles class imbalance through stratification. Saves all
  splits to the session processed directory. Explains the purpose of each split to
  the user in plain English. Trigger when any of the following are mentioned:
  "split data", "train test split", "validation set", "holdout set", "cross
  validation", "time series split", "stratified split", or any request to partition
  data before model training.
---

# Splitting Skill

The Splitting agent divides the data into three parts — training, validation, and
test — so the model can be trained on one portion, tuned on another, and evaluated
honestly on a third that it has never seen.

This is one of the most important steps for getting a reliable estimate of how the
model will perform in the real world. Done incorrectly, the model will appear to
perform better than it actually does.

---

## Responsibilities

1. Determine the appropriate splitting strategy based on task type and data
2. Explain the purpose of each split to the user in plain English
3. Apply stratification for classification tasks to preserve class balance
4. Apply temporal splitting for time series data — never random splits
5. Recommend split ratios based on dataset size
6. Save all splits to the session processed directory
7. Validate that each split is large enough to be meaningful

---

## Why Three Splits?

Always explain this to the user before splitting:

"We are going to divide your data into three groups:

- **Training set** — the data the model learns from. Think of this as the textbook.
- **Validation set** — the data we use to tune and adjust the model during development. Think of this as practice exams.
- **Test set** — the data we use for a final honest evaluation at the very end. The model never sees this during training or tuning. Think of this as the final exam.

This separation means the performance we report is a genuine measure of how the
model will perform on new, unseen data — not just a measure of how well it memorised
your existing data."

---

## Splitting Strategies

### Strategy 1 — Standard Random Split (default)
Used for: most classification and regression tasks

```python
from sklearn.model_selection import train_test_split

def standard_split(X, y, test_size=0.2, val_size=0.1,
                   stratify=True, random_state=42):
    stratify_col = y if stratify else None

    # First split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state
    )

    # Then split validation from the remaining training data
    val_ratio = val_size / (1 - test_size)
    stratify_col = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=stratify_col,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
```

---

### Strategy 2 — Temporal Split (time series only)
Used for: any data with a time dimension — never use random splits for time series

```python
def temporal_split(df, datetime_col, test_pct=0.2, val_pct=0.1):
    """
    For time series data, always split by time — not randomly.
    Random splits leak future information into the past.
    """
    df = df.sort_values(datetime_col)
    n = len(df)

    test_start = int(n * (1 - test_pct))
    val_start  = int(n * (1 - test_pct - val_pct))

    train = df.iloc[:val_start]
    val   = df.iloc[val_start:test_start]
    test  = df.iloc[test_start:]

    return train, val, test
```

**Why random splits are wrong for time series:**
"Imagine trying to predict tomorrow's weather using data that includes tomorrow's
temperature as a training example. That is what happens when we split time series
data randomly — future information ends up in the training set, and the model
learns to cheat. We always split time series by time — the model trains on the
past and is tested on the future."

---

### Strategy 3 — Cross Validation (small datasets)
Used for: datasets with fewer than 500 rows where a single split would leave too
little data for reliable training or evaluation

```python
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

def get_cv_strategy(task_type, n_splits=5, is_time_series=False):
    if is_time_series:
        return TimeSeriesSplit(n_splits=n_splits), (
            f"We will use {n_splits}-fold time series cross-validation. "
            "The model is trained and tested {n_splits} times on different time windows."
        )
    elif "classification" in task_type:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), (
            f"We will use {n_splits}-fold stratified cross-validation. "
            f"Your data is divided into {n_splits} equal groups. "
            "The model is trained and tested {n_splits} times, each time using a different group as the test set."
        )
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=42), (
            f"We will use {n_splits}-fold cross-validation. "
            f"Your data is divided into {n_splits} equal groups. "
            "The model is trained and tested {n_splits} times."
        )
```

---

## Recommended Split Ratios by Dataset Size

```python
def recommend_split_ratios(n_rows):
    if n_rows < 200:
        return {
            "strategy":  "cross_validation",
            "n_splits":  5,
            "reason":    f"Your dataset has {n_rows} rows. It is too small for a reliable holdout split — we recommend cross-validation to get the most out of your data."
        }
    elif n_rows < 1000:
        return {
            "strategy":  "standard",
            "test_size": 0.20,
            "val_size":  0.10,
            "reason":    f"Your dataset has {n_rows} rows. We recommend 70% for training, 10% for validation, and 20% for final testing."
        }
    elif n_rows < 10000:
        return {
            "strategy":  "standard",
            "test_size": 0.15,
            "val_size":  0.10,
            "reason":    f"Your dataset has {n_rows} rows. We recommend 75% for training, 10% for validation, and 15% for final testing."
        }
    else:
        return {
            "strategy":  "standard",
            "test_size": 0.10,
            "val_size":  0.10,
            "reason":    f"Your dataset has {n_rows} rows. We recommend 80% for training, 10% for validation, and 10% for final testing. With this much data, a smaller test set is still reliable."
        }
```

---

## Stratification — Preserving Class Balance

```python
def check_stratification_needed(y, task_type):
    if "classification" not in task_type:
        return False, "Stratification is not needed for regression tasks."

    class_pcts = y.value_counts(normalize=True)
    minority_pct = class_pcts.min()

    if minority_pct < 0.3:
        return True, (
            f"Your data is imbalanced — the minority class makes up {minority_pct:.1%} of rows. "
            "We will use stratified splitting to ensure each split has a representative proportion "
            "of each outcome. Without this, the test set might by chance contain very few examples "
            "of the minority class."
        )
    else:
        return True, (
            "We will use stratified splitting to ensure each split has a proportional "
            "representation of each outcome."
        )
```

---

## Validating Split Sizes

```python
def validate_splits(X_train, X_val, X_test, y_train, y_val, y_test, task_type):
    warnings = []
    MIN_ROWS = 30

    for name, X, y in [("Training", X_train, y_train),
                        ("Validation", X_val, y_val),
                        ("Test", X_test, y_test)]:
        if len(X) < MIN_ROWS:
            warnings.append({
                "severity": "warning",
                "message": f"The {name} set only has {len(X)} rows. This may be too small for reliable results.",
                "action": "Consider using cross-validation instead."
            })

        if "classification" in task_type:
            missing_classes = set(y_train.unique()) - set(y.unique())
            if missing_classes:
                warnings.append({
                    "severity": "hard_stop",
                    "message": f"The {name} set is missing some outcome classes: {missing_classes}. This will cause errors during evaluation.",
                    "action": "We will re-split with stratification enabled."
                })

    return warnings
```

---

## Running the Full Splitting Pipeline

```python
import pandas as pd
from pathlib import Path

def run_splitting(df, target_col, task_type, is_time_series,
                  datetime_col, user_decision, session_id):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    recommendation = recommend_split_ratios(len(df))
    strategy = user_decision.get("strategy", recommendation["strategy"])

    if strategy == "cross_validation":
        cv, cv_msg = get_cv_strategy(task_type, n_splits=5,
                                      is_time_series=is_time_series)
        # For cross-validation, save the full X and y for the training loop
        base = f"sessions/{session_id}/data/processed/splits"
        Path(base).mkdir(parents=True, exist_ok=True)
        X.to_csv(f"{base}/X_full.csv", index=False)
        y.to_csv(f"{base}/y_full.csv", index=False)
        return None, None, None, None, None, None, cv, cv_msg

    elif is_time_series and datetime_col:
        train, val, test = temporal_split(
            df, datetime_col,
            test_pct=user_decision.get("test_size", 0.2),
            val_pct=user_decision.get("val_size", 0.1)
        )
        X_train = train.drop(columns=[target_col])
        X_val   = val.drop(columns=[target_col])
        X_test  = test.drop(columns=[target_col])
        y_train, y_val, y_test = train[target_col], val[target_col], test[target_col]

    else:
        stratify, _ = check_stratification_needed(y, task_type)
        X_train, X_val, X_test, y_train, y_val, y_test = standard_split(
            X, y,
            test_size=user_decision.get("test_size", 0.2),
            val_size=user_decision.get("val_size", 0.1),
            stratify=stratify
        )

    # Validate
    warnings = validate_splits(X_train, X_val, X_test,
                                y_train, y_val, y_test, task_type)

    # Save splits
    base = f"sessions/{session_id}/data/processed/splits"
    Path(base).mkdir(parents=True, exist_ok=True)
    for name, X, y in [("train", X_train, y_train),
                        ("val",   X_val,   y_val),
                        ("test",  X_test,  y_test)]:
        X.to_csv(f"{base}/X_{name}.csv", index=False)
        y.to_csv(f"{base}/y_{name}.csv", index=False)

    return X_train, X_val, X_test, y_train, y_val, y_test, None, warnings
```

---

## Output Written to Session

**Split files:**
- `sessions/{session_id}/data/processed/splits/X_train.csv`
- `sessions/{session_id}/data/processed/splits/X_val.csv`
- `sessions/{session_id}/data/processed/splits/X_test.csv`
- `sessions/{session_id}/data/processed/splits/y_train.csv`
- `sessions/{session_id}/data/processed/splits/y_val.csv`
- `sessions/{session_id}/data/processed/splits/y_test.csv`

**Result JSON:**
`sessions/{session_id}/outputs/splitting/result.json`

```json
{
  "stage": "splitting",
  "status": "success",
  "strategy": "standard",
  "split_sizes": {
    "train": 843,
    "val":   120,
    "test":  241
  },
  "split_pcts": {
    "train": "70%",
    "val":   "10%",
    "test":  "20%"
  },
  "stratified": true,
  "is_time_series": false,
  "warnings": [],
  "plain_english_summary": "We divided your 1,204 rows into three groups: 843 for training (70%), 120 for validation (10%), and 241 for the final test (20%). We made sure each group has a proportional mix of outcomes.",
  "report_section": {
    "stage": "splitting",
    "title": "Dividing Your Data",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "Keeping a portion of data completely separate from training is the only way to get an honest measure of how the model will perform on new data it has never seen before."
  },
  "config_updates": {
    "split_strategy": "standard",
    "train_size": 843,
    "val_size": 120,
    "test_size": 241,
    "stratified": true,
    "cv_folds": null
  }
}
```

---

## What to Tell the User

Before splitting:
"We are going to divide your data into three groups — for training, tuning,
and final testing. Here is what we recommend based on the size of your dataset,
and why:"

After splitting:
"Your data has been divided:
- {train_size} rows for training ({train_pct}%)
- {val_size} rows for validation ({val_pct}%)
- {test_size} rows for the final test ({test_pct}%)

The test set will be set aside and not touched until the very end.
We will not look at its results until the model is fully trained and tuned."

---

## Reference Files

- `references/splitting-guide.md` — detailed guidance on splitting strategies and common mistakes
