"""
regression_runner.py

Trains a regression model on pre-split, pre-preprocessed data.

Rules:
  - Runs a mean predictor baseline FIRST.
  - No advanced model is presented as meaningful unless it beats the baseline.
  - No SMOTE, no resampling of any kind.
  - Val/test labels are never mutated.
  - Feature dtypes are cast to float64 before training and scoring.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
    Train a regression model and return a result dict compatible
    with the existing training agent response shape.

    Returns
    -------
    dict with keys: model_id, model, train_score, val_score, baseline_score,
                    beats_baseline, actions_log, model_path, metric_name
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    X_tr = _to_float(X_train)
    X_v  = _to_float(X_val)
    y_tr = y_train.copy()
    y_v  = y_val.copy()   # never mutated

    actions_log: list[str] = []

    # ── 1. Baseline: mean predictor ───────────────────────────────────────
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_tr, y_tr)
    baseline_preds = dummy.predict(X_v)
    baseline_r2  = float(r2_score(y_v, baseline_preds))
    baseline_mae = float(mean_absolute_error(y_v, baseline_preds))
    actions_log.append(
        f"Baseline (mean predictor): R2={baseline_r2:.4f}, MAE={baseline_mae:.4f}."
    )

    # ── 2. Train model ────────────────────────────────────────────────────
    model = _make_model(model_id)

    if "xgboost" in model_id:
        model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
    else:
        model.fit(X_tr, y_tr)

    # Use R² as the primary score (higher is better)
    train_preds = model.predict(X_tr)
    val_preds   = model.predict(X_v)

    train_score = float(r2_score(y_tr, train_preds))
    val_score   = float(r2_score(y_v,  val_preds))
    val_mae     = float(mean_absolute_error(y_v, val_preds))
    val_rmse    = float(np.sqrt(mean_squared_error(y_v, val_preds)))

    # ── 3. Baseline check ─────────────────────────────────────────────────
    beats_baseline = val_score > baseline_r2
    if not beats_baseline:
        actions_log.append(
            f"Warning: the model (R2={val_score:.4f}) did not beat the mean baseline "
            f"(R2={baseline_r2:.4f}). Review feature quality or try a different model."
        )

    # ── 4. Save model ─────────────────────────────────────────────────────
    model_path = models_dir / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_id":       model_id,
        "model":          model,
        "train_score":    round(train_score, 4),
        "val_score":      round(val_score, 4),
        "val_mae":        round(val_mae, 4),
        "val_rmse":       round(val_rmse, 4),
        "baseline_score": round(baseline_r2, 4),
        "beats_baseline": beats_baseline,
        "actions_log":    actions_log,
        "model_path":     str(model_path),
        "metric_name":    "R2",
    }


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _make_model(model_id: str):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    if model_id == "ridge":
        return Ridge(alpha=1.0)
    elif model_id == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    elif model_id == "xgboost_regressor":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            random_state=42, verbosity=0
        )
    raise ValueError(f"Unknown regression model_id: {model_id}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_float(X: pd.DataFrame) -> pd.DataFrame:
    try:
        return X.astype(np.float64)
    except Exception:
        result = X.copy()
        for col in result.columns:
            try:
                result[col] = result[col].astype(np.float64)
            except Exception:
                pass
        return result
