# Imputation Strategy Guide

## When to use each strategy

### Median (fill with middle value)
- **Best for:** Numeric columns with outliers or skewed distributions
- **Why:** The median is not affected by extreme values the way the average is
- **Example:** Income, house prices, age — all can have extreme values
- **Plain English:** "We used the middle value because your data has some very high or low values that would distort the average"

### Mean (fill with average)
- **Best for:** Numeric columns with a roughly normal distribution and no significant outliers
- **Why:** Simple and accurate when data is well-behaved
- **Plain English:** "We used the average value because your data is evenly distributed"

### KNN (fill using similar rows)
- **Best for:** When accuracy matters more than speed, and the dataset is not too large (< 100k rows)
- **Why:** Looks at rows that are similar in other columns to estimate the missing value
- **Avoid when:** Dataset is very large (slow) or columns are mostly unrelated
- **Plain English:** "We looked at rows that were similar to the ones with missing values and used their values as an estimate"

### Mode (fill with most common value)
- **Best for:** Categorical / text columns
- **Why:** For categories, the most common value is the most reasonable default
- **Plain English:** "We filled in the gaps with the most commonly occurring value in that column"

### Drop rows
- **Best for:** When missing values are very few (< 5%) and rows are genuinely incomplete
- **Avoid when:** You would lose more than 5% of your data
- **Plain English:** "We removed the rows where this value was missing — there were very few of them"

### Drop column
- **Best for:** Columns that are more than 80% empty
- **Avoid when:** The column is likely to be important to the model
- **Plain English:** "We removed this column because it was mostly empty and unlikely to help the model"

---

## Multiple missing columns — order matters

When multiple columns need imputation, apply in this order:
1. Drop columns first (reduces noise for KNN)
2. Drop rows next (smaller dataset for imputation)
3. Apply KNN last (benefits from other columns being clean)
