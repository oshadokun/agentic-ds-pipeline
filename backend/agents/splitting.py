"""
Splitting Agent
Divides the cleaned dataset into train / validation / test sets,
then orchestrates the fit-on-train / transform-all preprocessing step.

Execution order (enforced here):
  1. split_service.split()          — pure index manipulation, no fitting
  2. pipeline_compiler.compile_run_spec() — build RunSpec from session config
  3. preprocessing_service.fit_transform() — fit on X_train only, transform all

This agent is ORCHESTRATION ONLY.
No fit/transform logic lives in this file.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit

# Task-family constants — always use set membership, never substring matching
CLASSIFICATION_FAMILIES = {"binary_classification", "multiclass_classification"}
TIME_SERIES_FAMILIES    = {"time_series", "time_series_forecast"}


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
    if task_type not in CLASSIFICATION_FAMILIES:
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
        if task_type in CLASSIFICATION_FAMILIES:
            missing = set(y_train.unique()) - set(y.unique())
            if missing:
                warnings.append({
                    "severity": "hard_stop",
                    "message":  f"The {name} set is missing some outcome classes: {missing}.",
                    "action":   "We will re-split with stratification enabled."
                })
    return warnings


# ---------------------------------------------------------------------------
# RunSpec update helper
# ---------------------------------------------------------------------------

def _update_run_spec_from_decisions(run_spec, all_decisions: dict,
                                     df, target_col: str) -> None:
    """
    Merge accumulated session decisions (scaling, imputation, outliers,
    feature columns, resampling) into a pre-loaded RunSpec in-place.
    Called by splitting._run() after loading the RunSpec compiled by validation.
    """
    pp = run_spec.preprocessing_plan or {}

    # Feature columns
    if "feature_columns" in all_decisions and all_decisions["feature_columns"]:
        pp["feature_columns"] = all_decisions["feature_columns"]
    elif not pp.get("feature_columns"):
        pp["feature_columns"] = [c for c in df.columns if c != target_col]

    # Scaling strategy
    if "scaling_strategy" in all_decisions:
        pp["scaling_strategy"] = all_decisions["scaling_strategy"]

    # Imputation strategies  (keys: missing_{col})
    imputation = {
        k.replace("missing_", "", 1): v
        for k, v in all_decisions.items()
        if k.startswith("missing_")
    }
    if imputation:
        pp["imputation_strategies"] = imputation

    # Outlier strategies  (keys: outlier_{col})
    outlier = {
        k.replace("outlier_", "", 1): v
        for k, v in all_decisions.items()
        if k.startswith("outlier_")
    }
    if outlier:
        pp["outlier_strategies"] = outlier

    # Encoding strategies  (keys: encode_{col})
    encoding = {
        k.replace("encode_", "", 1): v
        for k, v in all_decisions.items()
        if k.startswith("encode_")
    }
    if encoding:
        pp["encoding_strategies"] = encoding

    # Resampling
    imbalance = (
        all_decisions.get("imbalance_strategy") or
        all_decisions.get("balance_classes")
    )
    if imbalance:
        if run_spec.resampling_plan is not None:
            run_spec.resampling_plan["strategy"] = imbalance
        elif run_spec.is_classification:
            run_spec.resampling_plan = {"strategy": imbalance, "min_class_count": None}

    run_spec.preprocessing_plan = pp


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    import traceback
    try:
        return _run(session, decisions)
    except Exception as exc:
        tb = traceback.format_exc()
        print(f"[splitting] ERROR:\n{tb}")
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": f"Splitting failed: {exc}",
        }


def _run(session: dict, decisions: dict) -> dict:
    from contracts.schemas import RunSpec
    from services.split_service import split as do_split
    from services.preprocessing_service import fit_transform

    session_id   = session["session_id"]
    target_col   = session["goal"].get("target_column")
    session_cfg  = session.get("config", {})
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    # ── RunSpec is required — compiled by validation stage ────────────────
    run_spec_path = session_dir / "artifacts" / "run_spec.json"
    if not run_spec_path.exists():
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": (
                "No RunSpec found. The validation stage must complete successfully "
                "before splitting can run. Please run the pipeline from the validation stage."
            )
        }
    run_spec = RunSpec.load(run_spec_path)
    task_type = run_spec.task_family  # single source of truth

    # Read from cleaned.csv; fall back to features.csv only
    # NOTE: scaled.csv is NOT canonical — it is an unchanged copy of features.csv
    for candidate in [
        session_dir / "data" / "interim" / "cleaned.csv",
        session_dir / "data" / "interim"   / "features.csv",
    ]:
        if candidate.exists():
            data_path = candidate
            break
    else:
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": "No cleaned data found. Please run cleaning first."
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

    # ── Phase 1: Return decision options if not yet provided ──────────────
    if not decisions or decisions.get("phase") == "request_decisions":
        is_ts_goal = task_type in TIME_SERIES_FAMILIES
        if is_ts_goal:
            rec_strategy = "temporal"
            rec_reason   = (
                "Your goal is time series analysis — a time-based split ensures the model "
                "is always trained on past data and tested on future data."
            )
        else:
            rec_strategy = rec["strategy"]
            rec_reason   = rec["reason"]
        _test_pct  = round(rec.get("test_size", 0.2) * 100)
        _val_pct   = round(rec.get("val_size",  0.1) * 100)
        _train_pct = 100 - _test_pct - _val_pct
        return {
            "stage":  "splitting",
            "status": "decisions_required",
            "total_rows": n_rows,
            "recommended_ratios": {"train": _train_pct, "val": _val_pct, "test": _test_pct},
            "decisions_required": [{
                "id":                    "split_strategy",
                "question":              "How would you like to divide your data for training and testing?",
                "recommendation":        rec_strategy,
                "recommendation_reason": rec_reason,
                "alternatives": [
                    {"id": "standard", "label": "Random split into train, validation, and test",  "tradeoff": "Good for most datasets. Simple and reliable."},
                    {"id": "temporal", "label": "Time-based split (for time series data)",        "tradeoff": "Required for time series — prevents future leakage into training."}
                ]
            }],
            "plain_english_summary": (
                "We are going to divide your data into three groups — for training, tuning, "
                "and final testing. Here is what we recommend based on your dataset size."
            )
        }

    # ── Phase 2: Execute split + preprocessing ────────────────────────────
    is_ts_goal    = task_type in TIME_SERIES_FAMILIES
    strategy      = decisions.get("split_strategy", "temporal" if is_ts_goal else rec.get("strategy", "standard"))
    _split_ratios = decisions.get("split_ratios") or {}
    test_size     = float(_split_ratios.get("test") or decisions.get("test_size") or rec.get("test_size", 0.15))
    val_size      = float(_split_ratios.get("val")  or decisions.get("val_size")  or rec.get("val_size",  0.15))
    is_ts         = decisions.get("is_time_series", is_ts_goal)

    # ── Issue 4: Cross-validation not yet supported with new preprocessing ─
    if strategy == "cross_validation":
        return {
            "stage":                 "splitting",
            "status":                "failed",
            "plain_english_summary": (
                "Cross-validation is not yet supported with the new preprocessing pipeline. "
                "Please use standard or temporal split instead."
            )
        }

    splits_dir = session_dir / "data" / "processed" / "splits"
    models_dir = session_dir / "models"
    splits_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Update RunSpec with accumulated decisions from previous stages ─────
    all_decisions = {**session_cfg, **decisions}
    all_decisions["target_column"] = target_col
    _update_run_spec_from_decisions(run_spec, all_decisions, df, target_col)

    # Apply split parameters and strategy override
    run_spec.test_size   = test_size
    run_spec.val_size    = val_size
    run_spec.time_column = session_cfg.get("datetime_col") or decisions.get("datetime_col")

    _strategy_map = {
        "standard": "standard_holdout",
        "temporal": "time_ordered_holdout",
    }
    run_spec.split_strategy = _strategy_map.get(strategy, run_spec.split_strategy)

    # Save updated RunSpec before executing
    artifacts_dir = session_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_spec.save(artifacts_dir / "run_spec.json")

    # ── Step 1: Split (pure index manipulation, no fitting) ───────────────
    split_result = do_split(df, run_spec)

    # Save raw (pre-preprocessing) label splits — labels never mutated
    y_splits_dir = splits_dir
    split_result.y_train.to_frame().to_csv(y_splits_dir / "y_train.csv", index=False)
    split_result.y_val.to_frame().to_csv(y_splits_dir   / "y_val.csv",   index=False)
    split_result.y_test.to_frame().to_csv(y_splits_dir  / "y_test.csv",  index=False)

    # ── Step 2: Preprocessing — fit on X_train ONLY, transform all ───────
    prep_result = fit_transform(
        run_spec,
        split_result.X_train,
        split_result.X_val,
        split_result.X_test,
        splits_dir,
        models_dir,
    )

    split_sizes = split_result.sizes
    train_pct   = round(split_sizes["train"] / n_rows * 100)
    val_pct     = round(split_sizes["val"]   / n_rows * 100)
    test_pct    = round(split_sizes["test"]  / n_rows * 100)

    summary = (
        f"We divided your {n_rows:,} rows: "
        f"{split_sizes['train']:,} for training ({train_pct}%), "
        f"{split_sizes['val']:,} for validation ({val_pct}%), "
        f"{split_sizes['test']:,} for final testing ({test_pct}%). "
        f"Imputation and scaling were fitted on training data only."
    )

    prep_warnings = split_result.warnings + prep_result.warnings

    return _build_result(
        strategy, split_sizes, n_rows, is_ts,
        str(splits_dir), str(prep_result.preprocessor_path),
        decisions, summary, warnings=prep_warnings
    )


def _build_result(
    strategy, split_sizes, n_rows, is_ts,
    output_data_path, preprocessor_path, decisions, summary,
    warnings=None,
) -> dict:
    return {
        "stage":              "splitting",
        "status":             "success",
        "strategy":           strategy,
        "split_sizes":        split_sizes,
        "total_rows":         n_rows,
        "is_time_series":     is_ts,
        "output_data_path":   output_data_path,
        "preprocessor_path":  preprocessor_path,   # additive — frontend ignores
        "warnings":           warnings or [],
        "decisions_required": [],
        "decisions_made":     [{"decision": k, "chosen": v} for k, v in decisions.items()],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "splitting",
            "title":   "Dividing Your Data",
            "summary": summary,
            "decision_made": f"Used {strategy} split strategy.",
            "alternatives_considered": "Cross-validation, temporal split, and random split were available.",
            "why_this_matters": (
                "Keeping data separate from training is the only way to get an honest measure "
                "of how the model will perform on new data. "
                "Fitting preprocessing only on training data prevents test-set leakage."
            )
        },
        "config_updates": {
            "split_strategy":   strategy,
            "split_sizes":      split_sizes,
            "stratified":       strategy in ("standard", "stratified_holdout"),
            "cv_folds":         decisions.get("n_splits") if strategy == "cross_validation" else None,
            "preprocessor_path": preprocessor_path,
        }
    }
