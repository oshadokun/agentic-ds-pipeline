# SHAP Guide

## What is SHAP?

SHAP stands for SHapley Additive exPlanations. It is a method for explaining
machine learning model predictions based on game theory.

Plain English: "SHAP measures how much each feature contributed to pushing a
prediction up or down from the average. It answers the question: if I know the
value of this feature, how much does that change my prediction?"

---

## Why SHAP over other methods?

- **Consistent:** SHAP values always add up to the difference between the
  prediction and the average prediction
- **Fair:** Each feature gets credit proportional to its actual contribution
- **Universal:** Works for any model type
- **Both global and local:** Explains the overall model and individual predictions

---

## Chart Types

### Feature Importance Bar Chart
- Shows the average absolute SHAP value for each feature
- Longer bar = this feature has more influence on predictions overall
- Does not show direction — just magnitude
- Good for: understanding which features matter most at a glance

### Beeswarm / Summary Plot
- Each dot is one row from the dataset
- Horizontal position = SHAP value (right = pushed prediction up, left = pushed down)
- Colour = feature value (red = high, blue = low)
- Good for: understanding both importance AND direction — does a high value
  of this feature increase or decrease the predicted outcome?

### Waterfall Chart (individual prediction)
- Shows how each feature contributed to one specific prediction
- Starts from the baseline (average prediction) and builds up to the final prediction
- Red bars push the prediction up, blue bars push it down
- Good for: explaining why the model made a specific prediction

### Dependence Plot
- Shows how SHAP values change as a feature's value changes
- Good for: understanding non-linear relationships
- Not included by default — available on request

---

## How to Read a Waterfall Chart

Example: "Why did the model predict this customer will churn?"

```
Base value: 0.32 (the average churn probability across all customers)
+ contract_type = month-to-month  → +0.18 (strong push toward churn)
+ tenure = 3 months               → +0.12 (short tenure increases churn risk)
- total_charges = £450            → -0.06 (moderate charges, slight reduction)
+ support_calls = 5               → +0.09 (many support calls, churn risk up)
= Final prediction: 0.65 (65% probability of churning)
```

Plain English narration:
"The model predicts a 65% chance this customer will churn. The main reasons are:
they are on a monthly contract (which customers can cancel easily), they have
only been a customer for 3 months, and they have made 5 support calls which
suggests dissatisfaction."

---

## Bias Detection

A feature should only drive predictions if it is genuinely predictive of the
outcome — not because it is a proxy for a protected characteristic.

Examples of proxy variables to watch for:
- Postcode as a proxy for ethnicity or socioeconomic status
- First name as a proxy for gender or ethnicity
- Age when the task does not legitimately require it

What to do if bias is detected:
1. Investigate whether the relationship is genuine and appropriate
2. Consider whether the feature should be removed
3. Consider whether the model should be retrained without it
4. Document the finding and the decision made

This skill flags potential issues — the decision of what to do is always
presented to the user.
