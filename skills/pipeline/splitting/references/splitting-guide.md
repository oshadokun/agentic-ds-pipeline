# Splitting Strategy Guide

## The Three Splits Explained

### Training Set
- What the model learns from
- Typically 70–80% of the data
- The more the better — but diminishing returns above a certain point
- Plain English: "The textbook the model studies from"

### Validation Set
- Used to tune the model and make decisions during development
- Typically 10% of the data
- The model does not train on this — it is used to check how well training is going
- Plain English: "Practice exams taken while still studying"

### Test Set
- Used once only — at the very end for final evaluation
- Typically 10–20% of the data
- Never used during training or tuning
- Plain English: "The final exam — only taken once, at the end"

---

## Common Mistakes

### Mistake 1: Using the test set more than once
If you evaluate the model on the test set, see the score, make adjustments,
and evaluate again — the test set is no longer independent. Your adjustments
were influenced by it. The reported performance is now optimistic.
**Rule:** Touch the test set exactly once.

### Mistake 2: Random splits for time series
If your data has a time dimension, a random split will put future data into
the training set and past data into the test set. The model learns from the
future to predict the past — which is not possible in production.
**Rule:** Always split time series by time, earliest to latest.

### Mistake 3: Fitting the scaler before splitting
If you scale the full dataset before splitting, the scaler has seen the test
data. This is data leakage.
**Rule:** Split first, then fit the scaler on training data only.

### Mistake 4: No stratification for imbalanced classification
Without stratification, a random split might by chance put most of the
minority class into the training set — leaving the test set with very few
examples to evaluate on.
**Rule:** Always use stratified splits for classification tasks.

---

## Cross-Validation vs Single Split

### Use a single train/val/test split when:
- Dataset has more than 500 rows
- Training is computationally expensive
- Speed matters

### Use cross-validation when:
- Dataset has fewer than 500 rows
- You want the most reliable performance estimate
- You can afford the extra computation (trains k times instead of once)

### Cross-validation does not replace the test set
Even with cross-validation, keep a held-out test set for the final evaluation.
Cross-validation is used for training and tuning — the test set is for the
final honest score.
