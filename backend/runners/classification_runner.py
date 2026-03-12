"""
classification_runner.py

Trains a classification model on pre-split, pre-preprocessed data.

Rules:
  - Runs a DummyClassifier baseline FIRST.
  - No advanced model result is meaningful unless it beats the baseline.
  - SMOTE is applied ONLY here, ONLY on X_train, ONLY if resampling_plan allows it.
  - No SMOTE on X_val or X_test.
  - Val/test labels are never mutated.
  - Feature dtypes are cast to float64 before training and scoring.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.utils.class_weight import compute_class_weight

from contracts.schemas import RunSpec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    run_spec: RunSpec,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    model_id: str,
    models_dir: str | Path,
) -> dict:
    """
    Train a classification model and return a result dict compatible
    with the existing training agent response shape.

    Parameters
    ----------
    run_spec    : RunSpec (for resampling_plan, task_family, primary_metric)
    X_train     : preprocessed training features
    X_val       : preprocessed validation features
    y_train     : training labels (integer/string, never float from SMOTE)
    y_val       : validation labels — NEVER mutated here
    model_id    : canonical model identifier string
    models_dir  : directory to save model.pkl and best_model.json

    Returns
    -------
    dict with keys: model_id, model, train_score, val_score, baseline_score,
                    beats_baseline, class_weights_applied, actions_log,
                    model_path, metric_name
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_tr = _to_float(X_train)
    X_v  = _to_float(X_val)
    y_tr = y_train.copy()
    y_v  = y_val.copy()   # never mutated

    actions_log: list[str] = []

    # ── 1. Baseline: DummyClassifier (stratified) ─────────────────────────
    dummy = DummyClassifier(strategy="stratified", random_state=42)
    dummy.fit(X_tr, y_tr)
    baseline_score = float(dummy.score(X_v, y_v))
    actions_log.append(
        f"Baseline (stratified random guessing): {baseline_score:.4f} "
        f"on {run_spec.primary_metric or 'accuracy'}."
    )

    # ── 2. Optional: SMOTE on X_train only ───────────────────────────────
    resampling = run_spec.resampling_strategy
    class_weights = None

    if resampling == "smote":
        X_tr, y_tr = _apply_smote(X_tr, y_tr, actions_log)

    elif resampling == "undersample":
        X_tr, y_tr = _apply_undersample(X_tr, y_tr, actions_log)

    elif resampling == "class_weights":
        class_weights, cw_msg = _compute_class_weights(y_tr, run_spec.task_family)
        if cw_msg:
            actions_log.append(cw_msg)

    elif resampling == "none":
        minority_pct = float(y_tr.value_counts(normalize=True).min())
        if minority_pct < 0.2:
            actions_log.append(
                f"Note: minority class is only {minority_pct:.1%} of training data. "
                "No balancing applied as requested."
            )

    # ── 3. Train model ────────────────────────────────────────────────────
    model = _make_model(model_id, class_weights, run_spec.task_family)

    from sklearn.linear_model import LogisticRegression
    is_xgb = "xgboost" in model_id
    if is_xgb:
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    else:
        model.fit(X_tr, y_tr)

    train_score = float(model.score(X_tr, y_tr))
    val_score   = float(model.score(X_v, y_v))

    # ── 4. Baseline check ─────────────────────────────────────────────────
    beats_baseline = val_score > baseline_score
    if not beats_baseline:
        actions_log.append(
            f"Warning: the model ({val_score:.4f}) did not beat the baseline "
            f"({baseline_score:.4f}). Review feature quality or try a different model."
        )

    # ── 5. Save model ─────────────────────────────────────────────────────
    model_path = models_dir / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_id":             model_id,
        "model":                model,
        "train_score":          round(train_score, 4),
        "val_score":            round(val_score, 4),
        "baseline_score":       round(baseline_score, 4),
        "beats_baseline":       beats_baseline,
        "class_weights_applied": class_weights is not None,
        "actions_log":          actions_log,
        "model_path":           str(model_path),
        "metric_name":          "Accuracy",
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model(model_id: str, class_weights, task_family: str):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    if model_id == "logistic_regression":
        return LogisticRegression(
            C=1.0, penalty="l2", max_iter=1000,
            class_weight=class_weights, random_state=42, solver="lbfgs"
        )
    elif model_id == "logistic_regression_multi":
        return LogisticRegression(
            C=1.0, multi_class="multinomial", max_iter=1000,
            class_weight=class_weights, random_state=42, solver="lbfgs"
        )
    elif model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight=class_weights, n_jobs=-1, random_state=42
        )
    elif model_id == "xgboost":
        from xgboost import XGBClassifier
        spw = 1.0
        if class_weights and len(class_weights) == 2:
            classes = sorted(class_weights.keys())
            spw = class_weights[classes[1]] / max(class_weights[classes[0]], 1e-9)
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            scale_pos_weight=spw, eval_metric="logloss",
            random_state=42, verbosity=0
        )
    raise ValueError(f"Unknown classification model_id: {model_id}")


# ---------------------------------------------------------------------------
# Resampling helpers
# ---------------------------------------------------------------------------

def _apply_smote(X_train: pd.DataFrame, y_train: pd.Series, log: list[str]):
    """SMOTE on X_train only. Never called on val or test."""
    try:
        from imblearn.over_sampling import SMOTE
        min_count = int(y_train.value_counts().min())
        k = min(5, max(1, min_count - 1))
        sm = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        log.append(
            f"Applied SMOTE — training set grew from {len(y_train)} to {len(y_res)} rows. "
            f"Classes are now balanced."
        )
        return (
            pd.DataFrame(X_res, columns=X_train.columns),
            pd.Series(y_res, name=y_train.name),
        )
    except ImportError:
        log.append(
            "SMOTE requested but imbalanced-learn is not installed. "
            "Continuing without balancing. Run: pip install imbalanced-learn"
        )
        return X_train, y_train


def _apply_undersample(X_train: pd.DataFrame, y_train: pd.Series, log: list[str]):
    from sklearn.utils import resample as sk_resample
    combined = X_train.copy()
    combined["__target__"] = y_train.values
    min_count = int(combined["__target__"].value_counts().min())
    parts = []
    for cls, grp in combined.groupby("__target__"):
        if len(grp) > min_count:
            grp = sk_resample(grp, replace=False, n_samples=min_count, random_state=42)
        parts.append(grp)
    combined = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    y_out = combined.pop("__target__")
    log.append(
        f"Undersampled majority class — training set now {len(y_out)} rows, balanced."
    )
    return combined, y_out


def _compute_class_weights(y_train: pd.Series, task_family: str):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes.tolist(), weights.tolist()))
    minority = float(y_train.value_counts(normalize=True).min())
    if minority < 0.2:
        msg = (
            f"Applied class weights — minority class is {minority:.1%} of data. "
            f"Model will pay extra attention to the rarer outcome."
        )
        return cw_dict, msg
    return None, "Classes are balanced — no class weights needed."


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_float(X: pd.DataFrame) -> pd.DataFrame:
    """Cast all columns to float64 — prevents dtype errors from mixed types."""
    try:
        return X.astype(np.float64)
    except Exception:
        # Some columns might resist float64 — convert column by column
        result = X.copy()
        for col in result.columns:
            try:
                result[col] = result[col].astype(np.float64)
            except Exception:
                pass
        return result
