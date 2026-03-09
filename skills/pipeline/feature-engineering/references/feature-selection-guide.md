# Feature Selection Guide

## Why feature selection matters
More columns is not always better. Irrelevant or redundant columns:
- Add noise that confuses the model
- Slow down training
- Can reduce accuracy on new data (overfitting)
- Make the model harder to explain

## Method used: Mutual Information
Measures how much knowing a column's value reduces uncertainty about the target.
- Score of 0 = knowing this column tells us nothing about the outcome
- Higher score = more informative

Works for both classification and regression.
Does not assume a linear relationship — captures complex patterns.

## How many features to keep
| Dataset size | Recommended max features |
|---|---|
| < 500 rows | 10 |
| 500–5,000 rows | 20 |
| 5,000–50,000 rows | 30 |
| > 50,000 rows | 50 |

These are guidelines. The user always has final say.

## When NOT to drop a feature
- The user says it is important from a business perspective — domain knowledge overrides statistics
- The feature has a very low score but is the only representation of an important concept
- The feature was expensive to collect — worth understanding the tradeoff before discarding

## After feature selection
Always show the user:
1. The top 5 most important features with a brief plain English explanation of why they might matter
2. The bottom 5 being recommended for removal with why they scored low
3. The option to override any decision
