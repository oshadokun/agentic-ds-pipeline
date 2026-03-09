# Encoding Strategy Guide

## Decision Tree

```
Is the column an ID or free text with no structure?
  → Drop it

Does it have 2 unique values?
  → Label encode (convert to 0 and 1)

Does it have 3–15 unique values?
  → One-hot encode (create yes/no columns)

Does it have 16–50 unique values?
  → Frequency encode (replace with how often it appears)

Does it have 50+ unique values?
  → Target encode (replace with average outcome per category)
  → Or drop if it looks like an ID
```

---

## One-Hot Encoding — Detail
- Creates one new column per category (minus one to avoid redundancy)
- Column named: `{original_col}_{category_value}`
- Value is 1 if the row belongs to that category, 0 otherwise
- **Watch out:** Can create many columns if cardinality is high
- **Safe for:** All model types

## Label Encoding — Detail
- Assigns a number to each category: cat=0, dog=1, bird=2
- Implies an ordering that does not exist for most categories
- **Only safe for:** Tree-based models (Random Forest, XGBoost) which split on values
- **Not safe for:** Linear models, neural networks — they will treat the numbers as having mathematical meaning

## Frequency Encoding — Detail
- Replaces each category with the proportion of rows that have that value
- Example: if 'London' appears in 30% of rows, all London rows get value 0.30
- Useful when the popularity of a category is itself predictive
- **Safe for:** All model types

## Target Encoding — Detail
- Replaces each category with the average target value for that category
- Example: if customers from 'London' churn at 25%, London gets value 0.25
- Very powerful but risks data leakage if not handled carefully
- This skill applies smoothing to reduce leakage risk
- **Best for:** High cardinality columns with a strong relationship to the target
- **Safe for:** All model types when smoothing is applied
