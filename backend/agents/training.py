"""
Training Agent
Selects, configures, and trains a model. Applies regularisation and handles
class imbalance. Saves the trained model.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_number(n: float) -> str:
    """Format a number for user-facing messages — never scientific notation."""
    abs_n = abs(n)
    if abs_n >= 1_000_000:
        return f"{n / 1_000_000:.1f} million"
    if abs_n >= 1_000:
        return f"{n:,.0f}"
    if abs_n >= 1:
        return f"{n:.2f}"
    return f"{n:.4f}"


# ---------------------------------------------------------------------------
# Time series model wrappers (sklearn-compatible interface)
# ---------------------------------------------------------------------------

class ARIMAWrapper:
    """Wraps statsmodels ARIMA as a sklearn-compatible estimator."""

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self._model_fit = None

    def fit(self, X, y):
        from statsmodels.tsa.arima.model import ARIMA
        y_vals = y.values if hasattr(y, "values") else np.array(y)
        model = ARIMA(y_vals, order=self.order)
        self._model_fit = model.fit()
        self._n_train = len(y_vals)
        return self

    def predict(self, X):
        n = len(X)
        forecast = self._model_fit.forecast(steps=n)
        return np.array(forecast)

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))


class ProphetWrapper:
    """Wraps Facebook Prophet as a sklearn-compatible estimator."""

    def __init__(self):
        self._model = None
        self._last_date = None

    def fit(self, X, y):
        from prophet import Prophet
        y_vals = y.values if hasattr(y, "values") else np.array(y)
        n = len(y_vals)
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame({"ds": dates, "y": y_vals})
        self._model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        self._model.fit(df)
        self._last_date = dates[-1]
        return self

    def predict(self, X):
        n = len(X)
        future_dates = pd.date_range(
            self._last_date + pd.Timedelta(days=1), periods=n, freq="D"
        )
        future = pd.DataFrame({"ds": future_dates})
        forecast = self._model.predict(future)
        return forecast["yhat"].values

    def score(self, X, y):
        y_pred = self.predict(X)
        return float(r2_score(y, y_pred))


# ---------------------------------------------------------------------------
# Model registry  (from training SKILL)
# ---------------------------------------------------------------------------

def _make_model(model_id: str, class_weights):
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
    elif model_id == "ridge":
        return Ridge(alpha=1.0)
    elif model_id == "random_forest":
        return RandomForestClassifier(
            n_estimators=200, max_depth=10,
            class_weight=class_weights, n_jobs=-1, random_state=42
        )
    elif model_id == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=200, max_depth=10, n_jobs=-1, random_state=42
        )
    elif model_id == "xgboost":
        from xgboost import XGBClassifier
        spw = 1.0
        if class_weights and len(class_weights) == 2:
            classes = sorted(class_weights.keys())
            spw     = class_weights[classes[1]] / max(class_weights[classes[0]], 1e-9)
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42, verbosity=0
        )
    elif model_id == "xgboost_regressor":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, learning_rate=0.05,
            reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            random_state=42, verbosity=0
        )
    elif model_id == "arima":
        return ARIMAWrapper(order=(1, 1, 1))
    elif model_id == "prophet":
        return ProphetWrapper()
    raise ValueError(f"Unknown model_id: {model_id}")


def _recommend_models(task_type: str, n_rows: int,
                       interpretability_needed: bool = False) -> list:
    recs = []

    if task_type == "binary_classification":
        recs.append({
            "id":       "logistic_regression",
            "name":     "Logistic Regression",
            "role":     "baseline",
            "reason":   "Simple, fast, and highly interpretable. A great starting point.",
            "tradeoff": "May not capture complex patterns. Works best when relationships are roughly linear.",
            "interpretable": True
        })
        if n_rows > 500:
            recs.append({
                "id":       "random_forest",
                "name":     "Random Forest",
                "role":     "strong_candidate",
                "reason":   "Handles complex patterns well, robust to outliers.",
                "tradeoff": "Less interpretable than Logistic Regression.",
                "interpretable": False
            })
        if n_rows > 1000:
            recs.append({
                "id":       "xgboost",
                "name":     "XGBoost",
                "role":     "strong_candidate",
                "reason":   "Often the best performing model on tabular data.",
                "tradeoff": "Requires more tuning. Less interpretable without SHAP analysis.",
                "interpretable": False
            })

    elif task_type == "regression":
        recs.append({
            "id":       "ridge",
            "name":     "Ridge Regression",
            "role":     "baseline",
            "reason":   "Simple, fast, and interpretable. Good baseline for any regression task.",
            "tradeoff": "Assumes a linear relationship between features and outcome.",
            "interpretable": True
        })
        if n_rows > 500:
            recs.append({
                "id":       "random_forest_regressor",
                "name":     "Random Forest",
                "role":     "strong_candidate",
                "reason":   "Handles non-linear relationships automatically.",
                "tradeoff": "Less interpretable than Ridge Regression.",
                "interpretable": False
            })
        if n_rows > 1000:
            recs.append({
                "id":       "xgboost_regressor",
                "name":     "XGBoost",
                "role":     "strong_candidate",
                "reason":   "Strong performer on tabular regression tasks.",
                "tradeoff": "Requires more tuning. Less interpretable without SHAP.",
                "interpretable": False
            })

    elif task_type == "multiclass_classification":
        recs.append({
            "id":       "logistic_regression_multi",
            "name":     "Logistic Regression (multiclass)",
            "role":     "baseline",
            "reason":   "Extends naturally to multiple classes. Simple and interpretable.",
            "tradeoff": "May struggle with complex non-linear boundaries.",
            "interpretable": True
        })
        recs.append({
            "id":       "random_forest",
            "name":     "Random Forest",
            "role":     "strong_candidate",
            "reason":   "Handles multiple classes natively and well.",
            "tradeoff": "Less interpretable than logistic regression.",
            "interpretable": False
        })

    elif "time_series" in task_type or "forecast" in task_type:
        recs.append({
            "id":       "random_forest_regressor",
            "name":     "Random Forest",
            "role":     "recommended",
            "reason":   "Uses all lag and date features you created. Best accuracy for most time series datasets that have feature columns. Handles non-linear seasonal patterns automatically.",
            "tradeoff": "Cannot extrapolate beyond the range seen in training data. Less interpretable than linear models.",
            "interpretable": False
        })
        if n_rows > 1000:
            recs.append({
                "id":       "xgboost_regressor",
                "name":     "XGBoost",
                "role":     "strong_candidate",
                "reason":   "Strong performer on tabular time series data with engineered features. Often slightly more accurate than Random Forest after tuning.",
                "tradeoff": "Requires more tuning to get the best results. Less interpretable without SHAP.",
                "interpretable": False
            })
        recs.append({
            "id":       "ridge",
            "name":     "Ridge Regression",
            "role":     "strong_candidate",
            "reason":   "Fast and interpretable. Uses all engineered features including month, day of week, and lag columns. Good linear baseline.",
            "tradeoff": "Assumes linear relationships — may miss complex seasonal patterns.",
            "interpretable": True
        })
        recs.append({
            "id":       "arima",
            "name":     "ARIMA",
            "role":     "alternative",
            "reason":   "Classic statistical model. Best when you have a raw sequence with no extra feature columns and want an interpretable model.",
            "tradeoff": "Works on the target sequence only — does not use lag or date features. If you have feature columns, Random Forest will usually perform better.",
            "interpretable": True
        })
        try:
            import prophet  # noqa: F401
            recs.append({
                "id":       "prophet",
                "name":     "Prophet",
                "role":     "alternative",
                "reason":   "Best for data with strong seasonal patterns (daily, weekly, yearly cycles). Handles holidays and trend changes automatically.",
                "tradeoff": "Works on the target sequence only. Slower to train. Does not use lag or date features.",
                "interpretable": True
            })
        except ImportError:
            pass

    if interpretability_needed:
        for r in recs:
            if not r["interpretable"]:
                r["warning"] = "This model is harder to explain. We can use SHAP to explain individual predictions."

    return recs


def _get_class_weights(y_train, task_type: str):
    if "classification" not in task_type:
        return None, "Class weights are not applicable to regression tasks."

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw_dict = dict(zip(classes.tolist(), weights.tolist()))

    minority = float(y_train.value_counts(normalize=True).min())
    if minority < 0.2:
        msg = (
            f"Your data has an imbalance — the rarer outcome makes up only {minority:.1%} of rows. "
            f"We will tell the model to pay extra attention to the rarer outcome."
        )
        return cw_dict, msg
    return None, "Your classes are reasonably balanced — no adjustment needed."


def _apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """Apply SMOTE to generate synthetic minority-class examples."""
    from imblearn.over_sampling import SMOTE
    # k_neighbors must be < minority class count
    min_count = y_train.value_counts().min()
    k = min(5, max(1, min_count - 1))
    sm = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name=y_train.name)


def _apply_undersample(X_train: pd.DataFrame, y_train: pd.Series):
    """Downsample majority class(es) to match the minority class count."""
    combined = X_train.copy()
    combined["__target__"] = y_train.values
    minority_count = combined["__target__"].value_counts().min()
    parts = []
    for cls, grp in combined.groupby("__target__"):
        if len(grp) > minority_count:
            grp = resample(grp, replace=False, n_samples=minority_count, random_state=42)
        parts.append(grp)
    combined = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    y_out = combined.pop("__target__")
    return combined, y_out


def _detect_overfitting(train_score: float, val_score: float,
                         threshold: float = 0.10) -> dict:
    gap = train_score - val_score
    if gap > threshold:
        return {
            "overfitting_detected": True,
            "gap": round(gap, 4),
            "plain_english": (
                f"The model performs better on the training data ({train_score:.3f}) than on the "
                f"validation data ({val_score:.3f}). This suggests some memorisation. "
                f"We recommend tuning to address this."
            )
        }
    return {
        "overfitting_detected": False,
        "gap": round(gap, 4),
        "plain_english": (
            f"The model performs similarly on both training ({train_score:.3f}) and "
            f"validation data ({val_score:.3f}). This is a good sign — the model is generalising well."
        )
    }


def _train_model(model_id: str, X_train, y_train, X_val, y_val,
                  class_weights, session_id: str) -> tuple:
    sessions_dir = Path("sessions")
    model = _make_model(model_id, class_weights)

    if "xgboost" in model_id:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    train_score = float(model.score(X_train, y_train))
    val_score   = float(model.score(X_val, y_val))

    models_dir  = sessions_dir / session_id / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path  = models_dir / f"{model_id}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, train_score, val_score, str(model_path)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id   = session["session_id"]
    target_col   = session["goal"].get("target_column")
    task_type    = session["goal"].get("task_type", "binary_classification")
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id
    splits_dir   = session_dir / "data" / "processed" / "splits"

    # Read splits
    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        X_val   = pd.read_csv(splits_dir / "X_val.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
        y_val   = pd.read_csv(splits_dir / "y_val.csv").squeeze()
    except FileNotFoundError:
        return {
            "stage":                 "training",
            "status":                "failed",
            "plain_english_summary": "No split data found. Please run the splitting stage first."
        }

    n_rows  = len(X_train)
    n_feats = X_train.shape[1]
    interp  = decisions.get("interpretability_needed", False)
    # Treat as time series if either the task_type says so or the config flag was set by validation
    if session.get("config", {}).get("is_time_series", False) and "time_series" not in task_type:
        task_type = "time_series"
    recs    = _recommend_models(task_type, n_rows, interp)

    # Return model choices if not yet made
    if not decisions or decisions.get("phase") == "request_decisions":
        return {
            "stage":  "training",
            "status": "decisions_required",
            "decisions_required": [{
                "id":       "model_selection",
                "question": (
                    "Now we are ready to train a model. Which model would you like to use? "
                    "We recommend starting with the simplest option."
                ),
                "recommendation":       recs[0]["id"],
                "recommendation_reason": recs[0]["reason"],
                "alternatives": recs
            }, {
                "id":       "interpretability_needed",
                "question": "How important is it that you can explain exactly why the model makes each prediction?",
                "recommendation": False,
                "recommendation_reason": "Most use cases do not require strict explainability.",
                "alternatives": [
                    {"id": True,  "label": "High — I need to explain every prediction", "tradeoff": "Limits model choices to simpler models."},
                    {"id": False, "label": "Not required — accuracy is the priority",   "tradeoff": "Opens up more powerful model options."}
                ]
            }],
            "plain_english_summary": (
                "Now we are ready to train a model. A model is a mathematical pattern learned "
                "from your data that can predict new outcomes. Here are the options we recommend:"
            )
        }

    model_id = decisions.get("model_selection", recs[0]["id"])

    # Apply imbalance strategy chosen during validation
    imbalance_strategy = session.get("config", {}).get("imbalance_strategy") or "none"
    actions_log        = []
    class_weights      = None

    if "classification" in task_type and imbalance_strategy != "none":
        if imbalance_strategy == "class_weights":
            class_weights, cw_msg = _get_class_weights(y_train, task_type)
            if class_weights:
                actions_log.append(cw_msg)

        elif imbalance_strategy == "smote":
            try:
                X_train, y_train = _apply_smote(X_train, y_train)
                actions_log.append(
                    f"Applied SMOTE — created synthetic minority-class examples. "
                    f"Training set is now {len(y_train)} rows, balanced across classes."
                )
            except ImportError:
                actions_log.append(
                    "SMOTE was requested but imbalanced-learn is not installed. "
                    "Continuing without balancing. Run: pip install imbalanced-learn"
                )

        elif imbalance_strategy == "undersample":
            X_train, y_train = _apply_undersample(X_train, y_train)
            actions_log.append(
                f"Undersampled majority class — training set is now {len(y_train)} rows, balanced across classes."
            )
    elif "classification" in task_type and imbalance_strategy == "none":
        # Check if imbalance was detected and warn
        if session.get("config", {}).get("class_imbalance_detected"):
            actions_log.append(
                "No imbalance correction applied as requested. "
                "The model may favour the majority class."
            )

    # Train
    try:
        model, train_s, val_s, model_path = _train_model(
            model_id, X_train, y_train, X_val, y_val, class_weights, session_id
        )
    except Exception as exc:
        return {
            "stage":                 "training",
            "status":                "failed",
            "plain_english_summary": f"Training failed: {str(exc)}"
        }

    model_display_name = next(
        (r["name"] for r in recs if r["id"] == model_id),
        model_id.replace("_", " ").title()
    )

    is_ts_model = model_id in ("arima", "prophet")

    if is_ts_model:
        # For ARIMA/Prophet use MAE and RMSE on both train and val — never mix R² and MAE
        y_pred_train = model.predict(X_train)
        y_pred_val   = model.predict(X_val)
        mae_train    = float(mean_absolute_error(y_train, y_pred_train))
        mae_val      = float(mean_absolute_error(y_val,   y_pred_val))
        rmse_val     = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        metric_name       = "MAE"
        metric_value      = round(mae_val, 4)
        train_score_display = round(mae_train, 4)
        val_score_display   = round(mae_val, 4)
        overfitting_detected = False
        summary = (
            f"We trained a {model_display_name} model on your data. "
            f"On average, its predictions are off by {_fmt_number(mae_val)} "
            f"(RMSE: {_fmt_number(rmse_val)}). "
            f"We'll try to improve this in the tuning step."
        )
    else:
        overfit = _detect_overfitting(train_s, val_s)
        actions_log.append(overfit["plain_english"])
        metric_name         = "Accuracy" if "classification" in task_type else "R²"
        metric_value        = round(val_s, 4)
        train_score_display = round(train_s, 4)
        val_score_display   = round(val_s, 4)
        overfitting_detected = overfit["overfitting_detected"]
        summary = (
            f"We trained a {model_display_name} model on your data. "
            + overfit["plain_english"]
        )

    # Save best model reference
    best_model_info = {
        "model_id":    model_id,
        "model_path":  model_path,
        "val_score":   val_score_display,
        "metric_name": metric_name,
    }
    models_dir = session_dir / "models"
    with open(models_dir / "best_model.json", "w") as f:
        json.dump(best_model_info, f)

    return {
        "stage":                  "training",
        "status":                 "success",
        "model_id":               model_id,
        "model_type":             model_display_name,
        "metric_name":            metric_name,
        "metric_value":           metric_value,
        "model_path":             model_path,
        "train_score":            train_score_display,
        "val_score":              val_score_display,
        "overfitting_detected":   overfitting_detected,
        "class_weights_applied":  class_weights is not None,
        "decisions_required":     [],
        "decisions_made":         [{"decision": "model_selection", "chosen": model_id}],
        "plain_english_summary":  summary,
        "report_section": {
            "stage":   "training",
            "title":   "Training Your Model",
            "summary": summary,
            "decision_made": f"Trained {model_id.replace('_', ' ').title()} model.",
            "alternatives_considered": "; ".join([r["name"] for r in recs if r["id"] != model_id]),
            "why_this_matters": (
                "Training is where the model learns the patterns in your data. The choices made here "
                "directly affect how well it performs on new data."
            )
        },
        "config_updates": {
            "model_id":           model_id,
            "model_path":         model_path,
            "class_weights":      class_weights,
            "imbalance_strategy": imbalance_strategy,
            "framework":          "sklearn" if "xgboost" not in model_id else "xgboost"
        }
    }
