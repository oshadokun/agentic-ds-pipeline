# Regularisation Guide

## What is regularisation?
Regularisation is a technique that prevents the model from memorising the
training data. Without it, the model will learn the training data perfectly
but fail on new data — this is called overfitting.

Plain English: "We add a small penalty for complexity. The model learns to
find simple patterns that work broadly, rather than complex rules that only
work for the training data."

---

## Signs of overfitting
- Training score is much higher than validation score (gap > 0.10)
- Model performs well in development but poorly when deployed
- Validation score gets worse as training continues

## Signs of underfitting
- Both training and validation scores are low
- The model is too simple to capture the patterns in the data
- Solution: try a more complex model or reduce regularisation

---

## Regularisation by Model

### Logistic Regression — C parameter
- C = 1/regularisation_strength
- **Lower C = stronger regularisation** (more constrained)
- **Higher C = weaker regularisation** (more flexible)
- Default: C=1.0
- If overfitting: try C=0.1 or C=0.01
- If underfitting: try C=10.0

### Ridge Regression — alpha parameter
- **Higher alpha = stronger regularisation**
- Default: alpha=1.0
- If overfitting: try alpha=10.0 or alpha=100.0
- If underfitting: try alpha=0.1

### Random Forest — max_depth, min_samples_leaf
- **Lower max_depth = stronger regularisation** (shallower trees)
- **Higher min_samples_leaf = stronger regularisation** (less specific splits)
- Default: max_depth=10, min_samples_leaf=1
- If overfitting: reduce max_depth to 5 or 6, increase min_samples_leaf to 5

### XGBoost — reg_alpha, reg_lambda, subsample
- **reg_alpha** (L1): promotes sparsity — some features get zero weight
- **reg_lambda** (L2): shrinks all weights — default 1.0
- **subsample**: fraction of rows used per tree — reduces overfitting
- If overfitting: increase reg_alpha and reg_lambda, reduce subsample to 0.6–0.7

---

## L1 vs L2 Regularisation
- **L1 (Lasso):** Pushes some feature weights to exactly zero — acts as feature selection
- **L2 (Ridge):** Shrinks all weights toward zero but rarely to exactly zero — keeps all features
- **ElasticNet:** Combination of L1 and L2
- Plain English: "L1 eliminates some features entirely. L2 makes all features smaller but keeps them all."
