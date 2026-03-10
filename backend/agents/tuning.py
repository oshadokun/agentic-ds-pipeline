"""
Tuning Agent
Finds the best hyperparameter settings using Optuna.
Always tunes on the validation set, never the test set.
Retrains the final model on train + validation combined with best params.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Search spaces  (from tuning SKILL)
# ---------------------------------------------------------------------------

def _get_search_space(model_id: str, trial) -> dict:
    if model_id == "logistic_regression":
        return {
            "C":       trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver":  "saga",
            "max_iter": 1000
        }
    elif model_id == "logistic_regression_multi":
        return {
            "C":       trial.suggest_float("C", 1e-3, 10.0, log=True),
            "solver":  "lbfgs",
            "max_iter": 1000
        }
    elif model_id == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}

    elif model_id in ["random_forest", "random_forest_regressor"]:
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":      trial.suggest_categorical("max_features", ["sqrt", "log2"])
        }

    elif model_id in ["xgboost", "xgboost_regressor"]:
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":        trial.suggest_int("max_depth", 3, 10),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    return {}


def _make_model(model_id: str, params: dict, class_weights):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if model_id == "logistic_regression":
        return LogisticRegression(class_weight=class_weights, random_state=42, **params)
    elif model_id == "logistic_regression_multi":
        return LogisticRegression(class_weight=class_weights, random_state=42, multi_class="multinomial", **params)
    elif model_id == "ridge":
        return Ridge(**params)
    elif model_id == "random_forest":
        return RandomForestClassifier(class_weight=class_weights, n_jobs=-1, random_state=42, **params)
    elif model_id == "random_forest_regressor":
        return RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    elif model_id == "xgboost":
        from xgboost import XGBClassifier
        spw = 1.0
        if class_weights and len(class_weights) == 2:
            ks  = sorted(class_weights.keys())
            spw = class_weights[ks[1]] / max(class_weights[ks[0]], 1e-9)
        return XGBClassifier(scale_pos_weight=spw, eval_metric="logloss",
                              random_state=42, verbosity=0, **params)
    elif model_id == "xgboost_regressor":
        from xgboost import XGBRegressor
        return XGBRegressor(random_state=42, verbosity=0, **params)
    raise ValueError(f"Unknown model_id: {model_id}")


def _recommend_n_trials(n_rows: int, model_id: str) -> tuple[int, str]:
    if n_rows > 50000:
        n, est = 30,  "5–10 minutes"
    elif n_rows > 10000:
        n, est = 50,  "3–7 minutes"
    elif n_rows > 1000:
        n, est = 75,  "2–5 minutes"
    else:
        n, est = 100, "1–3 minutes"

    if "xgboost" in model_id:
        n = max(20, n // 2)

    return n, est


def _compare_performance(baseline: float, tuned: float) -> dict:
    imp    = tuned - baseline
    pct    = (imp / max(abs(baseline), 1e-9)) * 100
    if imp > 0.02:
        verdict = "meaningful improvement"
        rec     = "The tuned model is noticeably better — we will use it going forward."
    elif imp > 0.005:
        verdict = "small improvement"
        rec     = "The tuned model is slightly better. We recommend using it."
    elif abs(imp) <= 0.005:
        verdict = "no meaningful difference"
        rec     = "Tuning did not meaningfully change performance. The original settings were already good."
    else:
        verdict = "performance declined"
        rec     = "The tuned model performed slightly worse. We will keep the original trained model."

    return {
        "baseline_score": round(baseline, 4),
        "tuned_score":    round(tuned, 4),
        "improvement":    round(imp, 4),
        "pct_change":     round(pct, 2),
        "verdict":        verdict,
        "recommendation": rec,
        "plain_english":  (
            f"Before tuning: {baseline:.3f}\n"
            f"After tuning:  {tuned:.3f}\n"
            f"Change: {'+' if imp >= 0 else ''}{imp:.3f} ({'+' if pct >= 0 else ''}{pct:.1f}%)\n\n"
            f"{rec}"
        )
    }


PARAM_EXPLANATIONS = {
    "C":               lambda v: f"Regularisation strength set to {v:.3f}.",
    "alpha":           lambda v: f"Regularisation strength (alpha) set to {v:.3f}.",
    "n_estimators":    lambda v: f"The model builds {v} decision trees and combines their results.",
    "max_depth":       lambda v: f"Each tree can make up to {v} decisions before giving an answer.",
    "learning_rate":   lambda v: f"Learning rate set to {v:.4f} — {'slow, careful learning.' if v < 0.05 else 'moderate learning pace.' if v < 0.15 else 'fast learning.'}",
    "subsample":       lambda v: f"Each tree only sees {v:.0%} of the training rows — adds variety.",
    "min_samples_leaf": lambda v: f"Each decision leaf requires at least {v} samples.",
}


def _explain_params(params: dict) -> str:
    lines = []
    for k, v in params.items():
        if k in PARAM_EXPLANATIONS:
            lines.append(f"  \u2022 {PARAM_EXPLANATIONS[k](v)}")
    return "\n".join(lines) if lines else "Parameters updated."


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def _tune_arima(session_id, session_dir, splits_dir) -> dict:
    """Grid search over ARIMA (p, d, q) orders. Returns a full stage result dict."""
    from sklearn.metrics import mean_absolute_error as _mae
    # Import ARIMAWrapper from training agent
    import importlib
    _training = importlib.import_module("training")
    ARIMAWrapper = _training.ARIMAWrapper

    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        X_val   = pd.read_csv(splits_dir / "X_val.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
        y_val   = pd.read_csv(splits_dir / "y_val.csv").squeeze()
    except FileNotFoundError:
        return {
            "stage":                 "tuning",
            "status":                "failed",
            "plain_english_summary": "No split data found. Please run the splitting stage first."
        }

    # Baseline MAE from the existing trained model
    models_dir = session_dir / "models"
    baseline_mae = None
    best_json = models_dir / "best_model.json"
    if best_json.exists():
        with open(best_json) as f:
            info = json.load(f)
        with open(info["model_path"], "rb") as f:
            trained_model = pickle.load(f)
        try:
            y_pred_base  = trained_model.predict(X_val)
            baseline_mae = float(_mae(y_val, y_pred_base))
        except Exception:
            baseline_mae = None

    # Grid search
    best_order  = (1, 1, 1)
    best_mae    = float("inf")
    tried       = 0
    for p in [0, 1, 2, 3]:
        for d in [0, 1, 2]:
            for q in [0, 1, 2, 3]:
                try:
                    m = ARIMAWrapper(order=(p, d, q))
                    m.fit(X_train, y_train)
                    y_pred = m.predict(X_val)
                    mae    = float(_mae(y_val, y_pred))
                    tried += 1
                    if mae < best_mae:
                        best_mae   = mae
                        best_order = (p, d, q)
                except Exception:
                    continue

    best_params = {"p": best_order[0], "d": best_order[1], "q": best_order[2]}

    # Retrain on train + val combined with best order
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    final_model = ARIMAWrapper(order=best_order)
    final_model.fit(X_combined, y_combined)

    tuned_path  = models_dir / "tuned_model.pkl"
    with open(tuned_path, "wb") as f:
        pickle.dump(final_model, f)

    params_path = models_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Update best_model.json if MAE improved (lower is better)
    improved = baseline_mae is not None and best_mae < baseline_mae
    if improved:
        best_info = {}
        if best_json.exists():
            with open(best_json) as f:
                best_info = json.load(f)
        best_info["model_path"] = str(tuned_path)
        best_info["tuned"]      = True
        with open(best_json, "w") as f:
            json.dump(best_info, f, indent=2)

    if baseline_mae is not None:
        imp     = baseline_mae - best_mae          # positive = improvement (lower MAE)
        pct     = (imp / max(abs(baseline_mae), 1e-9)) * 100
        if imp > 0.01 * baseline_mae:
            verdict = "meaningful improvement"
            rec     = f"The tuned ARIMA{best_order} is noticeably more accurate — we will use it going forward."
        elif imp > 0:
            verdict = "small improvement"
            rec     = f"ARIMA{best_order} is slightly more accurate. We recommend using it."
        else:
            verdict = "no meaningful difference"
            rec     = f"The original model was already well-specified. Using ARIMA{best_order} as the final model."
        comparison_text = (
            f"Before tuning: MAE {baseline_mae:.4g}\n"
            f"After tuning:  MAE {best_mae:.4g}\n"
            f"Change: {'+' if -imp >= 0 else ''}{-imp:.4g} ({'+' if -pct >= 0 else ''}{-pct:.1f}%)\n\n"
            f"{rec}"
        )
        baseline_score = round(baseline_mae, 4)
        tuned_score    = round(best_mae, 4)
        improvement    = round(imp, 4)
    else:
        verdict         = "no meaningful difference"
        comparison_text = rec = f"Best ARIMA order found: {best_order}."
        baseline_score  = None
        tuned_score     = round(best_mae, 4)
        improvement     = 0

    summary = (
        f"We searched {tried} ARIMA(p,d,q) combinations. "
        f"Best order: ARIMA{best_order} with MAE {best_mae:.4g}. "
        f"{rec}"
    )

    return {
        "stage":              "tuning",
        "status":             "success",
        "model_id":           "arima",
        "n_trials_run":       tried,
        "best_params":        best_params,
        "baseline_score":     baseline_score,
        "tuned_score":        tuned_score,
        "improvement":        improvement,
        "verdict":            verdict,
        "param_explanations": f"Best ARIMA order: p={best_order[0]}, d={best_order[1]}, q={best_order[2]}.",
        "model_path":         str(tuned_path),
        "decisions_required": [],
        "decisions_made":     [],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "tuning",
            "title":   "Fine-Tuning the Model",
            "summary": summary,
            "decision_made": f"Grid search over ARIMA(p,d,q). Best order: {best_order}.",
            "why_this_matters": (
                "Choosing the right ARIMA order controls how many past values and error terms "
                "the model learns from. The best order minimises forecast error on held-out data."
            )
        },
        "config_updates": {
            "tuned_model_path": str(tuned_path),
            "best_params":      best_params,
            "tuned_score":      tuned_score
        }
    }


def run(session: dict, decisions: dict) -> dict:
    session_id    = session["session_id"]
    task_type     = session["goal"].get("task_type", "binary_classification")
    model_id      = session["config"].get("model_id")
    is_ts         = session.get("config", {}).get("is_time_series", False)
    class_weights = session["config"].get("class_weights")
    sessions_dir  = Path("sessions")
    session_dir   = sessions_dir / session_id
    splits_dir    = session_dir / "data" / "processed" / "splits"

    if not model_id:
        return {
            "stage":                 "tuning",
            "status":                "failed",
            "plain_english_summary": "No trained model found in session config. Please run training first."
        }

    # ---- ARIMA: dedicated grid search, no Optuna -------------------------
    if model_id == "arima" or (is_ts and model_id == "arima"):
        return _tune_arima(session_id, session_dir, splits_dir)

    # ---- Prophet: skip tuning entirely ------------------------------------
    if model_id == "prophet":
        # Point tuned_model.pkl at the existing trained model so downstream
        # stages (evaluation, explainability) continue to work unchanged.
        models_dir = session_dir / "models"
        best_json  = models_dir / "best_model.json"
        tuned_path = models_dir / "tuned_model.pkl"
        if best_json.exists():
            with open(best_json) as f:
                info = json.load(f)
            with open(info["model_path"], "rb") as f:
                existing_model = pickle.load(f)
            with open(tuned_path, "wb") as f:
                pickle.dump(existing_model, f)

        summary = (
            "Prophet works best with its default settings for most datasets. "
            "No hyperparameter tuning was applied — the model from the training stage will be used."
        )
        return {
            "stage":              "tuning",
            "status":             "success",
            "model_id":           "prophet",
            "n_trials_run":       0,
            "best_params":        {},
            "baseline_score":     None,
            "tuned_score":        None,
            "improvement":        0,
            "verdict":            "skipped",
            "param_explanations": "No tuning applied.",
            "model_path":         str(tuned_path) if best_json.exists() else None,
            "decisions_required": [],
            "decisions_made":     [],
            "plain_english_summary": summary,
            "report_section": {
                "stage":   "tuning",
                "title":   "Fine-Tuning the Model",
                "summary": summary,
                "decision_made": "Tuning skipped for Prophet.",
                "why_this_matters": (
                    "Prophet's built-in seasonality and trend components are already well-calibrated "
                    "for most time series problems. Additional tuning rarely improves accuracy."
                )
            },
            "config_updates": {
                "tuned_model_path": str(tuned_path) if best_json.exists() else None,
                "best_params":      {},
                "tuned_score":      None
            }
        }

    # ---- All other models: existing Optuna logic --------------------------
    try:
        X_train = pd.read_csv(splits_dir / "X_train.csv")
        X_val   = pd.read_csv(splits_dir / "X_val.csv")
        y_train = pd.read_csv(splits_dir / "y_train.csv").squeeze()
        y_val   = pd.read_csv(splits_dir / "y_val.csv").squeeze()
    except FileNotFoundError:
        return {
            "stage":                 "tuning",
            "status":                "failed",
            "plain_english_summary": "No split data found. Please run the splitting stage first."
        }

    n_rows          = len(X_train)
    n_trials_rec, time_est = _recommend_n_trials(n_rows, model_id)

    # Return decision options if phase=request or no decisions
    if not decisions or decisions.get("phase") == "request_decisions":
        return {
            "stage":  "tuning",
            "status": "decisions_required",
            "decisions_required": [{
                "id":       "n_trials",
                "question": "How many different combinations of settings should we try?",
                "recommendation":       n_trials_rec,
                "recommendation_reason": f"Based on your dataset size, {n_trials_rec} trials should take approximately {time_est}.",
                "alternatives": [
                    {"id": max(10, n_trials_rec // 2), "label": f"Quick ({max(10, n_trials_rec // 2)} trials)", "tradeoff": "Faster but less thorough."},
                    {"id": n_trials_rec,               "label": f"Recommended ({n_trials_rec} trials — ~{time_est})", "tradeoff": "Good balance of speed and thoroughness."},
                    {"id": n_trials_rec * 2,           "label": f"Thorough ({n_trials_rec * 2} trials)",            "tradeoff": "Most thorough but takes longer."}
                ]
            }],
            "plain_english_summary": (
                "We are going to fine-tune the model by trying different combinations of settings. "
                f"We recommend {n_trials_rec} trials which will take approximately {time_est}."
            )
        }

    n_trials = int(decisions.get("n_trials", n_trials_rec))

    # Get baseline val score from evaluation result (primary_metric_value is R² for regression,
    # roc_auc/pr_auc for classification, f1 for multiclass — all higher-is-better)
    eval_path = session_dir / "outputs" / "evaluation" / "result.json"
    baseline_score = None
    if eval_path.exists():
        with open(eval_path) as f:
            eval_result = json.load(f)
        baseline_score = eval_result.get("config_updates", {}).get("primary_metric_value")
    if baseline_score is None:
        best_json = session_dir / "models" / "best_model.json"
        if best_json.exists():
            with open(best_json) as f:
                info = json.load(f)
            with open(info["model_path"], "rb") as f:
                trained_model = pickle.load(f)
            baseline_score = float(trained_model.score(X_val, y_val))
        else:
            baseline_score = 0.0

    # Scoring metric
    if task_type == "binary_classification":
        scoring = "roc_auc"
    elif task_type == "multiclass_classification":
        scoring = "f1_weighted"
    else:
        scoring = "r2"

    # Run Optuna study
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = _get_search_space(model_id, trial)
        if not params:
            return 0.0
        model  = _make_model(model_id, params, class_weights)
        scores = cross_val_score(model, X_train, y_train, cv=3,
                                  scoring=scoring, n_jobs=-1, error_score=0.0)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params   = study.best_params
    best_cv_score = study.best_value

    # Compare
    comparison = _compare_performance(baseline_score, best_cv_score)

    # Retrain on train + val combined
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    final_model = _make_model(model_id, best_params, class_weights)
    final_model.fit(X_combined, y_combined)

    models_dir  = session_dir / "models"
    tuned_path  = models_dir / "tuned_model.pkl"
    with open(tuned_path, "wb") as f:
        pickle.dump(final_model, f)

    params_path = models_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    # Update best_model.json so downstream stages use the tuned model when it improved
    if comparison["improvement"] > 0:
        best_json_path = models_dir / "best_model.json"
        best_info = {}
        if best_json_path.exists():
            with open(best_json_path) as f:
                best_info = json.load(f)
        best_info["model_path"] = str(tuned_path)
        best_info["val_score"]  = comparison["tuned_score"]
        best_info["tuned"]      = True
        with open(best_json_path, "w") as f:
            json.dump(best_info, f, indent=2)

    param_explanation = _explain_params(best_params)

    summary = (
        f"We tried {n_trials} different combinations of settings. "
        f"{comparison['plain_english']}"
    )

    return {
        "stage":             "tuning",
        "status":            "success",
        "model_id":          model_id,
        "n_trials_run":      n_trials,
        "best_params":       best_params,
        "baseline_score":    comparison["baseline_score"],
        "tuned_score":       comparison["tuned_score"],
        "improvement":       comparison["improvement"],
        "verdict":           comparison["verdict"],
        "param_explanations": param_explanation,
        "model_path":        str(tuned_path),
        "decisions_required": [],
        "decisions_made":    [{"decision": "n_trials", "chosen": n_trials}],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "tuning",
            "title":   "Fine-Tuning the Model",
            "summary": summary,
            "decision_made": f"Ran {n_trials} Optuna trials. {comparison['verdict']}.",
            "alternatives_considered": "Different numbers of trials were available.",
            "why_this_matters": (
                "The right settings can meaningfully improve how well the model generalises to "
                "new data. Tuning is like calibrating an instrument — the tool is already working, "
                "we are just making it more precise."
            )
        },
        "config_updates": {
            "tuned_model_path": str(tuned_path),
            "best_params":      best_params,
            "tuned_score":      comparison["tuned_score"]
        }
    }
