"""
timeseries_runner.py

Trains a time series / forecasting model on pre-split, pre-preprocessed data.

Rules:
  - Chronological split only. No shuffling. Split must have been done by
    split_service with strategy="time_ordered_holdout" before this runs.
  - Runs a naive previous-value (persistence) baseline FIRST.
  - No SMOTE or resampling of any kind.
  - Val/test labels are never mutated.
  - Lag features must not cross the split boundary — this is enforced
    upstream by preprocessing_service and pipeline_compiler.
  - Feature dtypes are cast to float64.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    Train a time series model and return a result dict compatible
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

    # ── 1. Baseline: naive persistence (predict last known value) ─────────
    last_train_val = float(y_tr.iloc[-1])
    baseline_preds = np.full(len(y_v), last_train_val)
    baseline_mae  = float(mean_absolute_error(y_v, baseline_preds))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_v, baseline_preds)))
    actions_log.append(
        f"Naive baseline (repeat last value {last_train_val:.4g}): "
        f"MAE={baseline_mae:.4f}, RMSE={baseline_rmse:.4f}."
    )

    # ── 2. Train model ────────────────────────────────────────────────────
    is_arima   = model_id == "arima"
    is_prophet = model_id == "prophet"

    if is_arima or is_prophet:
        model = _make_statistical_model(model_id)
        model.fit(X_tr, y_tr)
        val_preds   = model.predict(X_v)
        train_preds = model.predict(X_tr)
    else:
        model = _make_ml_model(model_id)
        if "xgboost" in model_id:
            model.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        else:
            model.fit(X_tr, y_tr)
        train_preds = model.predict(X_tr)
        val_preds   = model.predict(X_v)

    train_mae  = float(mean_absolute_error(y_tr, train_preds))
    val_mae    = float(mean_absolute_error(y_v,  val_preds))
    val_rmse   = float(np.sqrt(mean_squared_error(y_v, val_preds)))

    # ── 3. Baseline check (lower MAE = better) ────────────────────────────
    beats_baseline = val_mae < baseline_mae
    if not beats_baseline:
        actions_log.append(
            f"Warning: the model (MAE={val_mae:.4f}) did not beat the naive baseline "
            f"(MAE={baseline_mae:.4f}). Review lag features or try a different model."
        )

    # ── 4. Save model ─────────────────────────────────────────────────────
    model_path = models_dir / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {
        "model_id":       model_id,
        "model":          model,
        "train_score":    round(train_mae, 4),    # MAE (lower is better)
        "val_score":      round(val_mae, 4),
        "val_rmse":       round(val_rmse, 4),
        "baseline_score": round(baseline_mae, 4),
        "beats_baseline": beats_baseline,
        "actions_log":    actions_log,
        "model_path":     str(model_path),
        "metric_name":    "MAE",
    }


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _make_statistical_model(model_id: str):
    """ARIMA and Prophet wrappers (sklearn-compatible interface from training.py)."""
    # Import from training.py to reuse existing wrappers
    import sys, importlib
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from agents.training import ARIMAWrapper, ProphetWrapper

    if model_id == "arima":
        return ARIMAWrapper(order=(1, 1, 1))
    elif model_id == "prophet":
        return ProphetWrapper()
    raise ValueError(f"Unknown statistical model_id: {model_id}")


def _make_ml_model(model_id: str):
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    if model_id == "ridge":
        return Ridge(alpha=1.0)
    elif model_id in ("random_forest_regressor", "random_forest"):
        return RandomForestRegressor(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    elif model_id in ("xgboost_regressor", "xgboost"):
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            random_state=42, verbosity=0
        )
    raise ValueError(f"Unknown time series ML model_id: {model_id}")


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
