# Outlier Handling Guide

## What is an outlier?
A value that is unusually far from the rest of the data. Detected using the IQR method:
- Lower boundary = Q1 - 1.5 × IQR
- Upper boundary = Q3 + 1.5 × IQR
Anything outside these boundaries is flagged as an outlier.

---

## When to cap outliers (recommended default)
- The outlier is likely a genuine but extreme value (e.g. a very high salary)
- You want to keep all rows
- The model you plan to use is sensitive to extreme values (linear models, neural nets)
- **Plain English:** "We kept all your rows but limited the extreme values to a sensible boundary"

## When to remove outlier rows
- The outlier is clearly an error (e.g. age = 999, temperature = -500)
- Removing it does not lose much data (< 2% of rows affected)
- **Plain English:** "We removed rows where values were clearly incorrect"

## When to keep outliers as-is
- The outlier is meaningful and removing it would lose important signal
- The model you plan to use is robust to outliers (tree-based models like Random Forest, XGBoost)
- **Plain English:** "We kept the extreme values because the type of model we are using handles them well"

---

## Models and outlier sensitivity

| Model | Outlier sensitivity | Recommendation |
|---|---|---|
| Linear / Logistic Regression | High | Cap or remove |
| SVM | High | Cap or remove |
| Neural Networks | High | Cap or remove |
| Random Forest | Low | Keep or cap |
| XGBoost / LightGBM | Low | Keep or cap |
| KNN | Medium | Cap |

---

## IQR vs Z-score
This skill uses IQR (interquartile range) by default because it is robust to
non-normal distributions. Z-score assumes normality and is not used here.
