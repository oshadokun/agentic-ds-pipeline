---
name: training
description: >
  Responsible for selecting, configuring, and training a machine learning model
  on the prepared data splits. Always called by the Orchestrator after Splitting
  and Normalisation complete. Recommends appropriate models based on task type,
  dataset size, and interpretability requirements. Applies appropriate regularisation
  to prevent overfitting. Handles class imbalance during training. Trains a baseline
  model first before any complexity is added. Saves the trained model to the session
  models directory. Explains every model choice and regularisation decision in plain
  English. Trigger when any of the following are mentioned: "train model", "fit model",
  "model selection", "choose algorithm", "logistic regression", "random forest",
  "XGBoost", "regularisation", "overfitting", "class imbalance", "baseline model",
  or any request to build or train a predictive model.
---

# Training Skill

The Training agent selects the right model for the task, applies appropriate
regularisation to prevent it from memorising the training data, trains it on the
prepared splits, and saves it for evaluation.

The guiding principle is: **start simple, add complexity only if needed.**

A simple model that the user understands and trusts is almost always preferable
to a complex model that is marginally more accurate but impossible to explain.
The user always has a say in this tradeoff.

---

## Responsibilities

1. Recommend models based on task type, data size, and interpretability needs
2. Always train a simple baseline model first
3. Apply appropriate regularisation to prevent overfitting
4. Handle class imbalance through class weights or resampling
5. Train the model on the training set
6. Monitor for overfitting using the validation set
7. Save the trained model to the session models directory
8. Explain every decision in plain English

---

## Model Selection

### Step 1 — Understand the requirements
Before recommending a model, establish:

```python
def gather_requirements(task_type, n_rows, n_features,
                         interpretability_needed, speed_needed):
    return {
        "task_type":               task_type,
        "n_rows":                  n_rows,
        "n_features":              n_features,
        "interpretability_needed": interpretability_needed,
        "speed_needed":            speed_needed
    }
```

Always ask the user:
"How important is it that you can explain exactly why the model makes each
prediction? For example, if this model will be used to make decisions about
people, explainability may be required."

---

### Step 2 — Recommend models

```python
def recommend_models(task_type, n_rows, n_features, interpretability_needed):
    recommendations = []

    if task_type == "binary_classification":
        # Always start with logistic regression as baseline
        recommendations.append({
            "id":       "logistic_regression",
            "name":     "Logistic Regression",
            "role":     "baseline",
            "reason":   "Simple, fast, and highly interpretable. A great starting point — if a simple model works well, there is no need for complexity.",
            "tradeoff": "May not capture complex patterns in the data. Works best when the relationship between features and outcome is roughly linear.",
            "interpretable": True
        })
        if n_rows > 500:
            recommendations.append({
                "id":       "random_forest",
                "name":     "Random Forest",
                "role":     "strong_candidate",
                "reason":   "Handles complex patterns well, robust to outliers, and works on most datasets without much tuning.",
                "tradeoff": "Less interpretable than Logistic Regression — harder to explain individual predictions.",
                "interpretable": False
            })
        if n_rows > 1000:
            recommendations.append({
                "id":       "xgboost",
                "name":     "XGBoost",
                "role":     "strong_candidate",
                "reason":   "Often the best performing model on tabular data. Handles missing values and class imbalance natively.",
                "tradeoff": "Requires more tuning than Random Forest. Less interpretable without SHAP analysis.",
                "interpretable": False
            })

    elif task_type == "regression":
        recommendations.append({
            "id":       "ridge",
            "name":     "Ridge Regression",
            "role":     "baseline",
            "reason":   "Simple, fast, and interpretable. Built-in regularisation prevents overfitting. Good baseline for any regression task.",
            "tradeoff": "Assumes a linear relationship between features and outcome.",
            "interpretable": True
        })
        if n_rows > 500:
            recommendations.append({
                "id":       "random_forest_regressor",
                "name":     "Random Forest",
                "role":     "strong_candidate",
                "reason":   "Handles non-linear relationships and interactions between features automatically.",
                "tradeoff": "Less interpretable than Ridge Regression.",
                "interpretable": False
            })
        if n_rows > 1000:
            recommendations.append({
                "id":       "xgboost_regressor",
                "name":     "XGBoost",
                "role":     "strong_candidate",
                "reason":   "Strong performer on tabular regression tasks. Handles complex patterns well.",
                "tradeoff": "Requires more tuning. Less interpretable without SHAP.",
                "interpretable": False
            })

    elif task_type == "multiclass_classification":
        recommendations.append({
            "id":       "logistic_regression_multi",
            "name":     "Logistic Regression (multiclass)",
            "role":     "baseline",
            "reason":   "Extends naturally to multiple classes. Simple and interpretable.",
            "tradeoff": "May struggle with complex non-linear boundaries between classes.",
            "interpretable": True
        })
        recommendations.append({
            "id":       "random_forest",
            "name":     "Random Forest",
            "role":     "strong_candidate",
            "reason":   "Handles multiple classes natively and well. Good default choice.",
            "tradeoff": "Less interpretable than logistic regression.",
            "interpretable": False
        })

    # If interpretability is required, flag non-interpretable models
    if interpretability_needed:
        for r in recommendations:
            if not r["interpretable"]:
                r["warning"] = (
                    "This model is harder to explain. "
                    "We can still use SHAP to explain individual predictions, "
                    "but it requires an extra step."
                )

    return recommendations
```

---

## Regularisation

Regularisation prevents the model from memorising the training data instead of
learning general patterns. Always apply it.

```python
REGULARISATION_CONFIGS = {
    "logistic_regression": {
        "parameter":    "C",
        "default":      1.0,
        "description":  "Controls how strongly we prevent the model from overfitting. Lower values = stronger regularisation.",
        "plain_english": "We are telling the model to keep its rules simple rather than memorising every detail of the training data."
    },
    "ridge": {
        "parameter":    "alpha",
        "default":      1.0,
        "description":  "Controls the strength of L2 regularisation. Higher values = stronger regularisation.",
        "plain_english": "We are adding a small penalty for complexity to keep the model generalising well."
    },
    "random_forest": {
        "parameter":    "max_depth",
        "default":      10,
        "description":  "Limits how deep each decision tree can grow. Shallower trees generalise better.",
        "plain_english": "We are stopping each decision tree from growing too deep and memorising the training data."
    },
    "xgboost": {
        "parameter":    ["reg_alpha", "reg_lambda", "subsample"],
        "defaults":     {"reg_alpha": 0.1, "reg_lambda": 1.0, "subsample": 0.8},
        "description":  "L1 and L2 regularisation plus row subsampling to prevent overfitting.",
        "plain_english": "We are applying several techniques to stop the model from over-specialising on the training data."
    }
}
```

---

## Class Imbalance Handling During Training

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def get_class_weights(y_train, task_type):
    if "classification" not in task_type:
        return None, "Class weights are not applicable to regression tasks."

    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    minority_pct = y_train.value_counts(normalize=True).min()

    if minority_pct < 0.2:
        plain_english = (
            f"Your data has an imbalance — the rarer outcome makes up only "
            f"{minority_pct:.1%} of rows. We will tell the model to pay extra "
            f"attention to the rarer outcome so it does not simply learn to "
            f"predict the common one all the time."
        )
        return class_weight_dict, plain_english
    else:
        return None, "Your classes are reasonably balanced — no adjustment needed."
```

---

## Overfitting Detection

Always compare training and validation performance after fitting.

```python
def detect_overfitting(train_score, val_score, threshold=0.1):
    gap = train_score - val_score

    if gap > threshold:
        return {
            "overfitting_detected": True,
            "gap": round(gap, 4),
            "plain_english": (
                f"The model performs significantly better on the training data "
                f"({train_score:.3f}) than on the validation data ({val_score:.3f}). "
                f"This suggests it is memorising the training data rather than "
                f"learning general patterns. We recommend increasing regularisation "
                f"or simplifying the model."
            ),
            "action": "increase_regularisation"
        }
    else:
        return {
            "overfitting_detected": False,
            "gap": round(gap, 4),
            "plain_english": (
                f"The model performs similarly on both training ({train_score:.3f}) "
                f"and validation data ({val_score:.3f}). "
                f"This is a good sign — the model is generalising well."
            ),
            "action": None
        }
```

---

## Building and Training Models

```python
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import pickle
from pathlib import Path

MODEL_REGISTRY = {
    "logistic_regression": lambda cw: LogisticRegression(
        C=1.0, penalty="l2", max_iter=1000,
        class_weight=cw, random_state=42
    ),
    "logistic_regression_multi": lambda cw: LogisticRegression(
        C=1.0, multi_class="multinomial", max_iter=1000,
        class_weight=cw, random_state=42
    ),
    "ridge": lambda _: Ridge(alpha=1.0),
    "random_forest": lambda cw: RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight=cw, n_jobs=-1, random_state=42
    ),
    "random_forest_regressor": lambda _: RandomForestRegressor(
        n_estimators=200, max_depth=10,
        n_jobs=-1, random_state=42
    ),
    "xgboost": lambda cw: XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
        scale_pos_weight=_get_scale_pos_weight(cw),
        use_label_encoder=False, eval_metric="logloss",
        random_state=42
    ),
    "xgboost_regressor": lambda _: XGBRegressor(
        n_estimators=300, learning_rate=0.05,
        reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
        random_state=42
    )
}

def _get_scale_pos_weight(class_weight_dict):
    if class_weight_dict and len(class_weight_dict) == 2:
        classes = sorted(class_weight_dict.keys())
        return class_weight_dict[classes[1]] / class_weight_dict[classes[0]]
    return 1.0

def train_model(model_id, X_train, y_train, X_val, y_val,
                task_type, class_weights, session_id):

    model = MODEL_REGISTRY[model_id](class_weights)

    # For XGBoost — use early stopping on validation set
    if "xgboost" in model_id:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    # Score on both sets
    train_score = model.score(X_train, y_train)
    val_score   = model.score(X_val, y_val)

    # Save model
    model_path = f"sessions/{session_id}/models/{model_id}.pkl"
    Path(f"sessions/{session_id}/models").mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model, train_score, val_score, model_path
```

---

## Running the Full Training Pipeline

```python
def run_training(X_train, X_val, y_train, y_val,
                 task_type, user_decisions, session_id):

    actions_log = []

    # 1. Get class weights if needed
    class_weights, cw_msg = get_class_weights(y_train, task_type)
    if class_weights:
        actions_log.append(cw_msg)

    # 2. Train baseline first
    baseline_id = user_decisions.get("baseline_model")
    baseline, train_s, val_s, baseline_path = train_model(
        baseline_id, X_train, y_train, X_val, y_val,
        task_type, class_weights, session_id
    )
    actions_log.append(
        f"Trained baseline {baseline_id} — "
        f"Training score: {train_s:.3f}, Validation score: {val_s:.3f}"
    )

    overfit = detect_overfitting(train_s, val_s)
    actions_log.append(overfit["plain_english"])

    # 3. Train main model if different from baseline
    main_model_id = user_decisions.get("main_model", baseline_id)
    if main_model_id != baseline_id:
        main_model, train_s2, val_s2, main_path = train_model(
            main_model_id, X_train, y_train, X_val, y_val,
            task_type, class_weights, session_id
        )
        overfit2 = detect_overfitting(train_s2, val_s2)
        actions_log.append(overfit2["plain_english"])
        best_model    = main_model
        best_path     = main_path
        best_val_score = val_s2
    else:
        best_model    = baseline
        best_path     = baseline_path
        best_val_score = val_s

    # Save best model reference
    import json
    with open(f"sessions/{session_id}/models/best_model.json", "w") as f:
        json.dump({
            "model_id":  main_model_id,
            "model_path": best_path,
            "val_score":  best_val_score
        }, f)

    return best_model, actions_log, best_path
```

---

## Output Written to Session

**Trained model:**
`sessions/{session_id}/models/{model_id}.pkl`

**Best model reference:**
`sessions/{session_id}/models/best_model.json`

**Result JSON:**
`sessions/{session_id}/outputs/training/result.json`

```json
{
  "stage": "training",
  "status": "success",
  "model_id": "xgboost",
  "model_path": "sessions/{session_id}/models/xgboost.pkl",
  "train_score": 0.934,
  "val_score": 0.891,
  "overfitting_detected": false,
  "class_weights_applied": true,
  "decisions_required": [],
  "plain_english_summary": "We trained an XGBoost model on your data. It correctly predicted 89.1% of outcomes on the validation data it had not seen during training. The model is learning genuine patterns rather than memorising the training data.",
  "report_section": {
    "stage": "training",
    "title": "Training Your Model",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": "...",
    "why_this_matters": "Training is where the model learns the patterns in your data. The choices made here — which model, how much regularisation, how to handle imbalance — all directly affect how well it performs on new data."
  },
  "config_updates": {
    "model_id": "xgboost",
    "model_path": "...",
    "class_weights": { ... },
    "regularisation": { ... }
  }
}
```

---

## What to Tell the User

Before model selection:
"Now we are ready to train a model. A model is a mathematical pattern learned
from your data that can predict '{target_col}' for new rows it has never seen.
Here are the options we recommend, from simplest to most powerful:"

After training:
"Training is complete. The model studied {train_size} rows of your data.
Here is how it performed:
- On the data it trained on: {train_score:.1%} accuracy
- On data it had not seen: {val_score:.1%} accuracy

{overfitting_message}

We will now move to a thorough evaluation to understand performance in more detail."

---

## Reference Files

- `references/model-guide.md` — full model catalogue with when to use each
- `references/regularisation-guide.md` — how regularisation works and when to adjust it
