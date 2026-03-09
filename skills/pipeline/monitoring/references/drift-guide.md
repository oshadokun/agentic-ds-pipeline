# Drift Guide

## Types of Drift

### Data Drift (Covariate Drift)
- **What it is:** The distribution of input features changes
- **Example:** Average customer age shifts upward as the product attracts older users
- **Detected by:** KS test (numeric), PSI (categorical)
- **Plain English:** "The data coming in now looks different from what the model was trained on"

### Prediction Drift
- **What it is:** The distribution of the model's outputs changes
- **Example:** The model starts predicting churn for 40% of customers when it used to predict 15%
- **Detected by:** Comparing mean and std of predictions against baseline
- **Plain English:** "The model is giving different kinds of answers than it used to"

### Concept Drift
- **What it is:** The relationship between inputs and the target changes
- **Example:** Features that predicted churn last year no longer predict it today
- **Detected by:** Performance decay against ground truth labels
- **Plain English:** "The rules the model learned are no longer as accurate"
- **Hardest to detect:** Requires labelled recent data

---

## Drift Severity Thresholds

### Numeric features (KS test)
| Severity | p-value | Mean shift |
|---|---|---|
| None | ≥ 0.05 | Any |
| Low | < 0.05 | < 0.5 std |
| Medium | < 0.05 | 0.5–1.0 std |
| High | < 0.01 | > 1.0 std |

### Categorical features (PSI)
| Severity | PSI value |
|---|---|
| None / Negligible | < 0.1 |
| Low | 0.1 – 0.2 |
| Medium | 0.2 – 0.25 |
| High | > 0.25 |

---

## When to Retrain

| Situation | Action |
|---|---|
| High severity drift in 1+ features | Investigate → likely retrain |
| Medium drift in 3+ features | Monitor closely → retrain soon |
| Prediction mean shifted > 10% | Investigate → consider retraining |
| Performance dropped > 10% | Retrain immediately |
| Performance dropped 5–10% | Monitor closely |
| New categories appearing | Update training data → retrain |

---

## Retraining Strategy

When retraining is recommended:
1. Collect recent labelled data — ideally from the same time period showing drift
2. Combine with original training data (or replace if concept drift is confirmed)
3. Run the full pipeline again from the Cleaning stage onward
4. Compare new model against old model on a recent holdout set
5. Deploy new model only if it outperforms the old one
6. Update the monitoring baseline with new training data statistics

---

## Monitoring Frequency Recommendations

| Use case | Recommended frequency |
|---|---|
| High-stakes decisions (credit, medical) | Daily |
| Customer-facing predictions | Weekly |
| Internal reporting tools | Monthly |
| Low-frequency batch predictions | Per batch run |
