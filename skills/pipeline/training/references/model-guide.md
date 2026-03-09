# Model Guide

## Choosing a Model — Decision Guide

```
Is interpretability essential?
  Yes → Logistic Regression (classification) or Ridge (regression)

Is the dataset small (< 500 rows)?
  Yes → Logistic Regression or Ridge with cross-validation

Is the data tabular with complex patterns?
  Yes → Random Forest first, then XGBoost if more accuracy is needed

Is training speed critical?
  Yes → Logistic Regression or LightGBM

Is the data text?
  Yes → TF-IDF + Logistic Regression baseline, then transformers

Is the data images?
  Yes → Outside the scope of this pipeline — specialist skill needed
```

---

## Model Profiles

### Logistic Regression
- **Task:** Binary and multiclass classification
- **Plain English:** "Finds a straight-line boundary that separates your outcomes"
- **Strengths:** Fast, interpretable, works well with scaled data
- **Weaknesses:** Assumes a roughly linear relationship — misses complex patterns
- **Regularisation:** C parameter (default 1.0) — lower = stronger regularisation
- **Best when:** Interpretability is required, dataset is small, quick baseline needed

### Ridge Regression
- **Task:** Regression
- **Plain English:** "Fits a straight line through your data with a built-in penalty for complexity"
- **Strengths:** Fast, interpretable, built-in L2 regularisation
- **Weaknesses:** Assumes linear relationships
- **Regularisation:** alpha parameter (default 1.0) — higher = stronger regularisation
- **Best when:** Regression baseline, interpretability needed, many correlated features

### Random Forest
- **Task:** Classification and regression
- **Plain English:** "Builds hundreds of decision trees and combines their votes"
- **Strengths:** Handles non-linear patterns, robust to outliers, minimal tuning needed
- **Weaknesses:** Slower to train on large datasets, less interpretable
- **Regularisation:** max_depth, min_samples_leaf, n_estimators
- **Best when:** General purpose strong performer, don't need high interpretability

### XGBoost
- **Task:** Classification and regression
- **Plain English:** "Builds decision trees in sequence, each one correcting the mistakes of the previous"
- **Strengths:** Often best performing on tabular data, handles missing values, built-in regularisation
- **Weaknesses:** More hyperparameters to tune, less interpretable
- **Regularisation:** reg_alpha (L1), reg_lambda (L2), subsample, colsample_bytree
- **Best when:** Maximum performance on tabular data, dataset > 1000 rows

---

## Baseline vs Main Model

Always train a baseline first. The baseline is typically:
- Logistic Regression for classification
- Ridge for regression

The baseline tells you:
- Whether a simple model is sufficient (if so, use it)
- A performance floor that the more complex model must beat
- A sanity check — if XGBoost cannot beat Logistic Regression, something may be wrong
