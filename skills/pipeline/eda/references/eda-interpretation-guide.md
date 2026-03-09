# EDA Interpretation Guide

How to narrate common patterns found during EDA in plain English.

---

## Distributions

| Pattern | Plain English Explanation |
|---|---|
| Normal / bell-shaped | "Most values are clustered around the middle, with fewer at the extremes. This is a healthy distribution." |
| Right skew (tail on right) | "Most values are low, but there are some very high values pulling the average up. For example, income data often looks like this." |
| Left skew (tail on left) | "Most values are high, but there are some very low values. This is less common." |
| Bimodal (two peaks) | "There appear to be two distinct groups in this column. This could be meaningful — for example, it might represent two different customer types." |
| Uniform (flat) | "Values are spread fairly evenly across the range. This column may be a category that has been assigned numbers." |
| Heavily skewed | "This column has extreme values that could affect the model. We will consider transforming it during feature engineering." |

---

## Correlations

| Value | Plain English |
|---|---|
| 0.9 – 1.0 | "These two columns are almost identical. You likely only need one of them." |
| 0.7 – 0.9 | "These columns are strongly related. It is worth considering whether both are needed." |
| 0.4 – 0.7 | "These columns have a moderate relationship." |
| 0.0 – 0.4 | "These columns have a weak or no linear relationship." |
| Negative values | "As one goes up, the other tends to go down." |

---

## Missing Values

| Missing % | Plain English |
|---|---|
| < 1% | "A very small number of values are missing — this is easy to handle." |
| 1–10% | "Some values are missing. We will fill these in during the cleaning stage." |
| 10–30% | "A notable portion of values are missing. We will discuss how to handle this." |
| 30–80% | "A large portion of values are missing. This column may have limited value." |
| > 80% | "This column is mostly empty and may not be useful for modelling." |

---

## Outliers

| Outlier % | Plain English |
|---|---|
| < 1% | "A few unusual values. These are likely genuine but we will review them." |
| 1–5% | "Some unusual values are present. We will decide whether to keep or remove them." |
| > 5% | "A significant number of unusual values. This column may need special treatment." |

---

## Class Imbalance

| Minority % | Plain English |
|---|---|
| 40–50% | "Your outcomes are well balanced — roughly equal numbers of each." |
| 20–40% | "There is some imbalance between outcomes, but this is manageable." |
| 5–20% | "One outcome is much less common than the other. We will account for this during training." |
| < 5% | "One outcome is very rare. This is a significant imbalance and will need special handling." |

---

## High Cardinality

| Unique % | Plain English |
|---|---|
| > 90% unique | "Almost every row has a different value — this looks like an ID column rather than a useful feature." |
| 50–90% unique | "This column has many different values. We will need to decide how to handle it during feature engineering." |
| < 50% unique | "This column has a manageable number of unique values and can be used as a category." |
