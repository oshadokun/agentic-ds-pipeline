---
name: tuning
description: >
  Responsible for systematically finding the best hyperparameter settings for the
  trained model. Always called by the Orchestrator after Evaluation confirms the
  model is worth improving. Uses Optuna for efficient search — smarter than trying
  every combination. Always tunes on the validation set, never the test set. Compares
  tuned model against baseline to confirm genuine improvement. Explains what
  hyperparameters are and what changed in plain English. Saves the tuned model.
  Trigger when any of the following are mentioned: "tune model", "hyperparameter
  tuning", "optimise model", "improve model", "grid search", "random search",
  "Optuna", "best parameters", "model settings", or any request to improve a
  trained model's performance through parameter adjustment.
---

# Tuning Skill

The Tuning agent finds the best settings for the trained model through systematic
search. Think of it like adjusting the dials on a piece of equipment — the model
is already working, and now we are finding the settings that make it work best.

Tuning always happens on the validation set. The test set is never touched.
The baseline performance is always preserved so we can confirm tuning actually
helped rather than just changed things.

---

## Responsibilities

1. Explain what hyperparameters are in plain English before tuning begins
2. Define the search space for the chosen model
3. Run Optuna to find the best parameter combination efficiently
4. Compare tuned performance against baseline — confirm improvement is genuine
5. Retrain the final model with best parameters on train + validation combined
6. Save the tuned model
7. Report what changed and by how much in plain English

---

## What are Hyperparameters — Plain English

Always explain this to the user before tuning:

"Every model has settings that control how it learns. These are called
hyperparameters — think of them like the settings on a camera. The camera's
job is to take a photo, but whether you get a sharp or blurry result depends
on how the settings are configured.

During training, the model learned patterns from your data. But the settings
that control *how* it learned were chosen beforehand. Tuning is the process of
systematically trying different combinations of settings to find the ones that
produce the best results.

We use a smart search method that learns from each attempt — rather than blindly
trying every combination, it focuses its search on the most promising areas."

---

## Tuning with Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score
import numpy as np

optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_search_space(model_id, trial):
    """Define hyperparameter search space per model."""

    if model_id == "logistic_regression":
        return {
            "C":      trial.suggest_float("C", 1e-3, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver":  "saga",
            "max_iter": 1000
        }

    elif model_id == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)
        }

    elif model_id in ["random_forest", "random_forest_regressor"]:
        return {
            "n_estimators":    trial.suggest_int("n_estimators", 50, 500),
            "max_depth":       trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features":    trial.suggest_categorical(
                                   "max_features", ["sqrt", "log2", 0.5]),
        }

    elif model_id in ["xgboost", "xgboost_regressor"]:
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 600),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
```

---

## Running the Optuna Study

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

MODEL_CONSTRUCTORS = {
    "logistic_regression":     lambda p, cw: LogisticRegression(
                                    class_weight=cw, random_state=42, **p),
    "ridge":                   lambda p, _:  Ridge(**p),
    "random_forest":           lambda p, cw: RandomForestClassifier(
                                    class_weight=cw, n_jobs=-1, random_state=42, **p),
    "random_forest_regressor": lambda p, _:  RandomForestRegressor(
                                    n_jobs=-1, random_state=42, **p),
    "xgboost":                 lambda p, cw: XGBClassifier(
                                    scale_pos_weight=_spw(cw),
                                    use_label_encoder=False,
                                    eval_metric="logloss",
                                    random_state=42, **p),
    "xgboost_regressor":       lambda p, _:  XGBRegressor(random_state=42, **p),
}

def _spw(cw):
    if cw and len(cw) == 2:
        classes = sorted(cw.keys())
        return cw[classes[1]] / cw[classes[0]]
    return 1.0

def run_optuna_study(model_id, X_train, y_train, task_type,
                     class_weights, scoring, n_trials, cv_folds=3):

    def objective(trial):
        params = get_search_space(model_id, trial)
        model  = MODEL_CONSTRUCTORS[model_id](params, class_weights)
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring=scoring, n_jobs=-1
        )
        return scores.mean()

    direction = "maximize"
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials,
                   show_progress_bar=False)

    return study.best_params, study.best_value, study
```

---

## Recommending Number of Trials

```python
def recommend_n_trials(n_rows, model_id):
    """
    Balance search thoroughness against computation time.
    More trials = better search but longer runtime.
    """
    if n_rows > 50000:
        n_trials = 30
        time_est = "5–10 minutes"
    elif n_rows > 10000:
        n_trials = 50
        time_est = "3–7 minutes"
    elif n_rows > 1000:
        n_trials = 75
        time_est = "2–5 minutes"
    else:
        n_trials = 100
        time_est = "1–3 minutes"

    # XGBoost is slower — reduce trials
    if "xgboost" in model_id:
        n_trials = max(20, n_trials // 2)

    return n_trials, time_est
```

Always tell the user upfront how long tuning will take.

---

## Comparing Tuned vs Baseline

```python
def compare_performance(baseline_score, tuned_score, metric_name):
    improvement = tuned_score - baseline_score
    pct_change  = (improvement / baseline_score) * 100 if baseline_score > 0 else 0

    if improvement > 0.02:
        verdict = "meaningful improvement"
        recommendation = "The tuned model is noticeably better — we will use it going forward."
    elif improvement > 0.005:
        verdict = "small improvement"
        recommendation = (
            "The tuned model is slightly better. The improvement is modest — "
            "we recommend using it, but do not expect a dramatic difference."
        )
    elif abs(improvement) <= 0.005:
        verdict = "no meaningful difference"
        recommendation = (
            "Tuning did not meaningfully change performance. "
            "The original model settings were already good. "
            "We will use the tuned model but the difference is negligible."
        )
    else:
        verdict = "performance declined"
        recommendation = (
            "The tuned model performed slightly worse — this can happen occasionally. "
            "We will keep the original trained model."
        )

    return {
        "baseline_score":  round(baseline_score, 4),
        "tuned_score":     round(tuned_score, 4),
        "improvement":     round(improvement, 4),
        "pct_change":      round(pct_change, 2),
        "verdict":         verdict,
        "recommendation":  recommendation,
        "plain_english":   (
            f"Before tuning: {baseline_score:.3f}\n"
            f"After tuning:  {tuned_score:.3f}\n"
            f"Change:        {'+' if improvement >= 0 else ''}{improvement:.3f} "
            f"({'+' if pct_change >= 0 else ''}{pct_change:.1f}%)\n\n"
            f"{recommendation}"
        )
    }
```

---

## Retraining on Train + Validation Combined

After finding the best parameters, retrain on all available labelled data
(train + validation) for a stronger final model. The test set remains untouched.

```python
import pandas as pd
import pickle
from pathlib import Path

def retrain_final_model(model_id, best_params, X_train, X_val,
                         y_train, y_val, class_weights, session_id):
    """
    Combine train and validation for final model training.
    This gives the model more data to learn from.
    The test set is never used here.
    """
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    model = MODEL_CONSTRUCTORS[model_id](best_params, class_weights)
    model.fit(X_combined, y_combined)

    model_path = f"sessions/{session_id}/models/tuned_model.pkl"
    Path(f"sessions/{session_id}/models").mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, model_path
```

---

## Explaining Best Parameters in Plain English

```python
PARAM_EXPLANATIONS = {
    "C": {
        "name": "Regularisation strength",
        "explain": lambda v: (
            f"Set to {v:.3f} — "
            + ("strong regularisation, keeping the model simple." if v < 0.1
               else "moderate regularisation, balancing simplicity and fit." if v < 1.0
               else "light regularisation, allowing the model more flexibility.")
        )
    },
    "alpha": {
        "name": "Regularisation strength",
        "explain": lambda v: f"Set to {v:.3f} — controls how much we penalise complexity."
    },
    "n_estimators": {
        "name": "Number of trees",
        "explain": lambda v: f"The model builds {v} decision trees and combines their results."
    },
    "max_depth": {
        "name": "Tree depth",
        "explain": lambda v: (
            f"Each tree can make up to {v} decisions before giving an answer. "
            + ("Shallow trees — simpler, less risk of memorising data." if v <= 5
               else "Medium depth — balancing complexity and generalisation." if v <= 10
               else "Deep trees — more complex patterns captured.")
        )
    },
    "learning_rate": {
        "name": "Learning rate",
        "explain": lambda v: (
            f"Set to {v:.4f} — "
            + ("slow, careful learning — more trees needed but often more accurate." if v < 0.05
               else "moderate learning pace." if v < 0.15
               else "fast learning — fewer trees but may be less precise.")
        )
    },
    "subsample": {
        "name": "Row sampling rate",
        "explain": lambda v: (
            f"Each tree only sees {v:.0%} of the training rows — "
            "adds variety and reduces overfitting."
        )
    }
}

def explain_best_params(best_params):
    explanations = []
    for param, value in best_params.items():
        if param in PARAM_EXPLANATIONS:
            explanations.append(
                f"  • {PARAM_EXPLANATIONS[param]['name']}: "
                f"{PARAM_EXPLANATIONS[param]['explain'](value)}"
            )
    return "\n".join(explanations) if explanations else "Parameters updated."
```

---

## Running the Full Tuning Pipeline

```python
def run_tuning(model_id, X_train, X_val, y_train, y_val,
               task_type, class_weights, baseline_val_score,
               scoring, user_decisions, session_id):

    n_rows   = len(X_train)
    n_trials, time_est = recommend_n_trials(n_rows, model_id)

    # Override if user chose a different number of trials
    n_trials = user_decisions.get("n_trials", n_trials)

    # Run search
    best_params, best_cv_score, study = run_optuna_study(
        model_id, X_train, y_train,
        task_type, class_weights, scoring, n_trials
    )

    # Compare against baseline
    comparison = compare_performance(baseline_val_score, best_cv_score,
                                     scoring)

    # Retrain on train + val with best params
    tuned_model, model_path = retrain_final_model(
        model_id, best_params,
        X_train, X_val, y_train, y_val,
        class_weights, session_id
    )

    param_explanation = explain_best_params(best_params)

    return tuned_model, best_params, comparison, param_explanation, model_path
```

---

## Output Written to Session

**Tuned model:**
`sessions/{session_id}/models/tuned_model.pkl`

**Best parameters:**
`sessions/{session_id}/models/best_params.json`

**Result JSON:**
`sessions/{session_id}/outputs/tuning/result.json`

```json
{
  "stage": "tuning",
  "status": "success",
  "model_id": "xgboost",
  "n_trials_run": 50,
  "best_params": { ... },
  "baseline_score": 0.891,
  "tuned_score": 0.912,
  "improvement": 0.021,
  "verdict": "meaningful improvement",
  "param_explanations": "...",
  "model_path": "sessions/{session_id}/models/tuned_model.pkl",
  "plain_english_summary": "We tried 50 different combinations of settings and found a better configuration. The model improved from 0.891 to 0.912 — a meaningful gain. Here is what changed and why it helps.",
  "report_section": {
    "stage": "tuning",
    "title": "Fine-Tuning the Model",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "The right settings can meaningfully improve how well the model generalises to new data. Tuning is like calibrating an instrument — the tool is already working, we are just making it more precise."
  },
  "config_updates": {
    "tuned_model_path": "...",
    "best_params": { ... },
    "tuned_score": 0.912
  }
}
```

---

## What to Tell the User

Before tuning begins:
"We are going to fine-tune the model by trying different combinations of
settings to find the best configuration. Here is what we plan to do:
- Try {n_trials} different combinations of settings
- This will take approximately {time_est}
- We will compare the results against the current model to confirm any improvement is genuine

Shall we proceed?"

After tuning completes:
"Tuning is complete. Here is what we found:

{comparison.plain_english}

Here is what changed in the model settings:
{param_explanation}

{next_step_message}"

---

## Reference Files

- `references/tuning-guide.md` — how Optuna works and when to adjust the search
