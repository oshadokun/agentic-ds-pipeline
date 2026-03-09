# Tuning Guide

## How Optuna Works

Optuna uses a method called TPE (Tree-structured Parzen Estimator) — a smart
search strategy that learns from each trial it runs.

Plain English: "Instead of randomly guessing settings or trying every combination,
Optuna pays attention to which settings worked well so far and focuses its next
attempts in those areas. It gets smarter as it goes."

This is significantly more efficient than Grid Search (tries every combination)
or Random Search (tries random combinations without learning).

---

## When Tuning Helps Most

- The model is already performing reasonably (verdict: fair or good)
- The dataset is large enough that there is genuine signal to optimise
- The model has multiple meaningful hyperparameters (XGBoost, Random Forest)

## When Tuning Helps Least

- The model is already performing strongly (little room to improve)
- The dataset is very small — tuning on small data tends to overfit the validation set
- The model is simple (Ridge Regression has only one meaningful parameter)

---

## The Risk of Over-Tuning

If you run too many trials on a small dataset, the tuned model may perform well
on the validation set but poorly on the test set. This is called "overfitting
the hyperparameters" — the settings are too specific to the validation data.

Signs of this:
- Tuned validation score is much higher than test score
- The best parameters are extreme values at the edge of the search space

Mitigations built into this skill:
- Number of trials is scaled to dataset size (fewer trials for smaller datasets)
- Final model is retrained on train + validation combined
- Test set evaluation is always the definitive measure

---

## Scoring Metrics Used for Tuning

| Task | Scoring metric | Why |
|---|---|---|
| Binary classification (balanced) | roc_auc | Threshold-independent ranking quality |
| Binary classification (imbalanced) | average_precision | Focuses on minority class performance |
| Multiclass classification | f1_weighted | Balances performance across all classes |
| Regression | neg_root_mean_squared_error | Penalises large errors |

---

## After Tuning — Retraining on Train + Validation

After finding the best parameters, the final model is retrained on the
combined training and validation sets. This gives the model more data to
learn from before it faces the test set.

This is safe because:
- The hyperparameters were already decided — we are not making new decisions
- We are just giving the final model more examples to learn from
- The test set has never been touched at any point

The scaler parameters (from Normalisation) do NOT need to be re-fitted —
the same scaler used for the original splits is used for the combined set.
