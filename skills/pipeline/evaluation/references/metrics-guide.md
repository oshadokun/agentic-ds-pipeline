# Metrics Guide

## Classification Metrics

### Accuracy
- **What it measures:** The percentage of all predictions that were correct
- **Plain English:** "The model got X out of every 100 predictions right"
- **When to use:** Only when classes are balanced
- **When NOT to use:** Imbalanced classes — a model that always predicts the majority class will have high accuracy but be useless

### Precision
- **What it measures:** Of all the times the model predicted positive, how often was it right
- **Plain English:** "When the model raises an alarm, how often is it a real alarm?"
- **When to prioritise:** When false positives are costly — e.g. incorrectly flagging a healthy patient

### Recall (Sensitivity)
- **What it measures:** Of all the actual positives, how many did the model catch
- **Plain English:** "Out of all the real cases, how many did the model find?"
- **When to prioritise:** When false negatives are costly — e.g. missing a cancer diagnosis

### F1 Score
- **What it measures:** The balance between precision and recall
- **Plain English:** "A single score that balances how precise and how thorough the model is"
- **When to use:** When both precision and recall matter equally

### ROC-AUC
- **What it measures:** How well the model ranks positive cases above negative cases
- **Scale:** 0.5 (random) to 1.0 (perfect)
- **Plain English:** "If we picked one positive case and one negative case at random, this is the probability the model ranks the positive case higher"
- **When to use:** Balanced or mildly imbalanced classes, when ranking matters

### PR-AUC (Precision-Recall AUC)
- **What it measures:** The tradeoff between precision and recall across all thresholds
- **Plain English:** "How well the model performs specifically on the minority class"
- **When to use:** Heavily imbalanced classes — more informative than ROC-AUC in this case

---

## Regression Metrics

### RMSE (Root Mean Squared Error)
- **What it measures:** The typical size of prediction errors, penalising large errors more
- **Units:** Same as the target column
- **Plain English:** "On average, the model's predictions are off by about X"
- **Sensitive to:** Large errors — one very wrong prediction increases RMSE significantly

### MAE (Mean Absolute Error)
- **What it measures:** The average absolute prediction error
- **Units:** Same as the target column
- **Plain English:** "The model's predictions are typically within X of the actual value"
- **Sensitive to:** Not sensitive to outliers — each error counts equally

### R² (R-squared)
- **What it measures:** The proportion of variation in the target explained by the model
- **Scale:** 0 (explains nothing) to 1.0 (perfect)
- **Plain English:** "The model explains X% of why the values differ"
- **Watch out:** Can be misleading with many features — prefer adjusted R² then

---

## Interpreting Results in Context

Always relate metrics to the user's original goal:

| Goal | Key metric | Plain English frame |
|---|---|---|
| Catch as many fraud cases as possible | Recall | "Out of all fraud cases, the model caught X%" |
| Only alert when very confident | Precision | "When the model flags fraud, it is right X% of the time" |
| Balance catching and precision | F1 | "The model balances thoroughness and accuracy at X" |
| Predict a price | MAE | "Predictions are typically within £X of the actual price" |
| Rank customers by risk | ROC-AUC | "The model correctly ranks higher-risk customers X% of the time" |
