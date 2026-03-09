"""
Splitting Agent
Divides the (scaled) dataset into train, validation, and test sets.
Reads from data/processed/scaled.csv (produced by normalisation agent).
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit


# ---------------------------------------------------------------------------
# Strategy helpers  (from splitting SKILL)
# ---------------------------------------------------------------------------

def _recommend_ratios(n_rows: int) -> dict:
    if n_rows < 200:
        return {
            "strategy": "cross_validation",
            "n_splits": 5,
            "reason":   f"Your dataset has {n_rows} rows. It is too small for a reliable holdout split — we recommend cross-validation to get the most out of your data."
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
            "reason":    f"Your dataset has {n_rows} rows. We recommend 80% for training, 10% for validation, and 10% for final testing."
        }


def _standard_split(X, y, test_size: float, val_size: float,
                     stratify: bool, random_state: int = 42):
    strat = y if stratify else None

    # Split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=strat,
        random_state=random_state
    )

    # Split val from remainder
    val_ratio = val_size / (1 - test_size)
    strat2    = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=strat2,
        random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _temporal_split(df: pd.DataFrame, datetime_col: str,
                     target_col: str, test_pct: float, val_pct: float):
    df   = df.sort_values(datetime_col)
    n    = len(df)
    t_st = int(n * (1 - test_pct))
    v_st = int(n * (1 - test_pct - val_pct))

    train = df.iloc[:v_st]
    val   = df.iloc[v_st:t_st]
    test  = df.iloc[t_st:]

    X_train = train.drop(columns=[target_col])
    X_val   = val.drop(columns=[target_col])
    X_test  = test.drop(columns=[target_col])
    return (X_train, X_val, X_test,
            train[target_col], val[target_col], test[target_col])


def _check_stratification(y, task_type: str) -> tuple[bool, str]:
    if "classification" not in task_type:
        return False, "Stratification is not needed for regression tasks."
    pcts        = y.value_counts(normalize=True)
    minority    = float(pcts.min())
    return True, (
        f"We will use stratified splitting to ensure each split has a proportional "
        f"representation of each outcome."
        + (f" Your data is imbalanced ({minority:.1%} minority class) — stratification is especially important here."
           if minority < 0.3 else "")
    )


def _validate_splits(X_train, X_val, X_test,
                      y_train, y_val, y_test, task_type: str) -> list:
    warnings = []
    for name, X, y in [("Training", X_train, y_train),
                        ("Validation", X_val, y_val),
                        ("Test", X_test, y_test)]:
        if len(X) < 30:
            warnings.append({
                "severity": "warning",
                "message":  f"The {name} set only has {len(X)} rows. This may be too small for reliable results.",
                "action":   "Consider using cross-validation instead."
            })
        if "classification" in task_type:
            missing = set(y_train.unique()) - set(y.unique())
            if missing:
                warnings.append({
                    "severity": "hard_stop",
                    "message":  f"The {name} set is missing some outcome classes: {missing}.",
                    "action":   "We will re-split with stratification enabled."
                })
    return warnings


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id = session["session_id"]
    target_col = session["goal"].get("target_column")
    task_type  = session["goal"].get("task_type", "binary_classification")
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    # Read from scaled.csv (normalisation output) or fall back to features.csv
    data_path = session_dir / "data" / "processed" / "scaled.csv"
    if not data_path.exists():
        data_path = session_dir / "data" / "interim" / "features.csv"
    if not data_path.exists():
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": "No processed data found. Please run normalisation first."
        }

    df = pd.read_csv(data_path, low_memory=False)

    if target_col not in df.columns:
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": f"Target column '{target_col}' not found in the data."
        }

    n_rows = len(df)
    rec    = _recommend_ratios(n_rows)

    # Return decision options if not yet provided
    if not decisions or decisions.get("phase") == "request_decisions":
        is_ts_goal = "time_series" in task_type or "forecast" in task_type
        if is_ts_goal:
            rec_strategy = "temporal"
            rec_reason   = "Your goal is time series analysis — a time-based split ensures the model is always trained on past data and tested on future data, which is the correct approach."
        else:
            rec_strategy = rec["strategy"]
            rec_reason   = rec["reason"]
        return {
            "stage":  "splitting",
            "status": "decisions_required",
            "decisions_required": [{
                "id":                   "split_strategy",
                "question":             "How would you like to divide your data for training and testing?",
                "recommendation":       rec_strategy,
                "recommendation_reason": rec_reason,
                "alternatives": [
                    {"id": "standard",          "label": "Random split into train, validation, and test",          "tradeoff": "Good for most datasets. Simple and reliable."},
                    {"id": "cross_validation",  "label": "Cross-validation (for small datasets)",                  "tradeoff": "Gets the most out of limited data. Slower to train."},
                    {"id": "temporal",          "label": "Time-based split (for time series data)",                "tradeoff": "Required for data with a time dimension — prevents future information leaking into training."}
                ]
            }],
            "plain_english_summary": (
                "We are going to divide your data into three groups — for training, tuning, and final testing. "
                "Here is what we recommend based on the size of your dataset."
            )
        }

    is_ts_goal = "time_series" in task_type or "forecast" in task_type
    strategy   = decisions.get("split_strategy", "temporal" if is_ts_goal else rec.get("strategy", "standard"))
    test_size  = float(decisions.get("test_size",  rec.get("test_size",  0.2)))
    val_size   = float(decisions.get("val_size",   rec.get("val_size",   0.1)))
    is_ts      = decisions.get("is_time_series", is_ts_goal)
    dt_col     = decisions.get("datetime_col")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    splits_dir = session_dir / "data" / "processed" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    cv_object  = None
    cv_message = None

    if strategy == "cross_validation":
        n_splits = int(decisions.get("n_splits", rec.get("n_splits", 5)))
        if is_ts:
            cv_object  = TimeSeriesSplit(n_splits=n_splits)
            cv_message = f"We will use {n_splits}-fold time series cross-validation."
        elif "classification" in task_type:
            cv_object  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_message = f"We will use {n_splits}-fold stratified cross-validation."
        else:
            cv_object  = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_message = f"We will use {n_splits}-fold cross-validation."

        X.to_csv(splits_dir / "X_full.csv", index=False)
        y.to_csv(splits_dir / "y_full.csv", index=False)

        split_sizes = {"full": n_rows}
        summary = (
            f"We set up {n_splits}-fold cross-validation for your {n_rows:,} rows. "
            f"The model will be trained and tested {n_splits} times on different portions of the data."
        )

    elif is_ts and dt_col:
        X_train, X_val, X_test, y_train, y_val, y_test = _temporal_split(
            df, dt_col, target_col, test_size, val_size
        )
        split_sizes = {
            "train": len(X_train),
            "val":   len(X_val),
            "test":  len(X_test)
        }
        for name, Xp, yp in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            Xp.to_csv(splits_dir / f"X_{name}.csv", index=False)
            yp.to_csv(splits_dir / f"y_{name}.csv", index=False)

        summary = (
            f"We split your data by time: {len(X_train):,} rows for training, "
            f"{len(X_val):,} for validation, {len(X_test):,} for final testing."
        )

    else:
        stratify, strat_msg = _check_stratification(y, task_type)
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = _standard_split(
                X, y, test_size, val_size, stratify
            )
        except ValueError:
            # Fallback without stratification if too few samples per class
            X_train, X_val, X_test, y_train, y_val, y_test = _standard_split(
                X, y, test_size, val_size, False
            )

        val_warnings = _validate_splits(X_train, X_val, X_test, y_train, y_val, y_test, task_type)

        hard_stops = [w for w in val_warnings if w.get("severity") == "hard_stop"]
        if hard_stops:
            return {
                "stage":                 "splitting",
                "status":                "hard_stop",
                "warnings":              hard_stops,
                "plain_english_summary": (
                    hard_stops[0]["message"] + " " + hard_stops[0]["action"]
                )
            }

        split_sizes = {
            "train": len(X_train),
            "val":   len(X_val),
            "test":  len(X_test)
        }
        for name, Xp, yp in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
            Xp.to_csv(splits_dir / f"X_{name}.csv", index=False)
            yp.to_csv(splits_dir / f"y_{name}.csv", index=False)

        train_pct = round(len(X_train) / n_rows * 100)
        val_pct   = round(len(X_val)   / n_rows * 100)
        test_pct  = round(len(X_test)  / n_rows * 100)
        summary = (
            f"We divided your {n_rows:,} rows into three groups: "
            f"{len(X_train):,} for training ({train_pct}%), "
            f"{len(X_val):,} for validation ({val_pct}%), "
            f"and {len(X_test):,} for the final test ({test_pct}%). "
            f"We made sure each group has a proportional mix of outcomes."
        )

    return {
        "stage":             "splitting",
        "status":            "success",
        "strategy":          strategy,
        "split_sizes":       split_sizes,
        "is_time_series":    is_ts,
        "cv_message":        cv_message,
        "output_data_path":  str(splits_dir),
        "decisions_required": [],
        "decisions_made":    [{"decision": k, "chosen": v} for k, v in decisions.items()],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "splitting",
            "title":   "Dividing Your Data",
            "summary": summary,
            "decision_made": f"Used {strategy} split strategy.",
            "alternatives_considered": "Cross-validation, temporal split, and random split were available.",
            "why_this_matters": (
                "Keeping a portion of data completely separate from training is the only way to get "
                "an honest measure of how the model will perform on new data it has never seen before."
            )
        },
        "config_updates": {
            "split_strategy": strategy,
            "split_sizes":    split_sizes,
            "stratified":     strategy == "standard",
            "cv_folds":       decisions.get("n_splits") if strategy == "cross_validation" else None
        }
    }
