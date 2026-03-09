# Scaling Strategy Guide

## Standard Scaling (Z-score normalisation)
- **What it does:** Subtracts the mean and divides by the standard deviation
- **Result:** Values centred around 0, most between -3 and +3
- **Plain English:** "We shift your numbers so the average is 0 and the spread is consistent"
- **Sensitive to outliers:** Yes — one extreme value shifts the mean and std
- **Use when:** Data is roughly normally distributed, no extreme outliers

## Min-Max Scaling
- **What it does:** Subtracts the minimum and divides by the range
- **Result:** All values between 0 and 1
- **Plain English:** "We compress your numbers into the range 0 to 1"
- **Sensitive to outliers:** Very — one extreme value compresses everything else into a tiny range
- **Use when:** Neural networks, when a hard 0–1 boundary is needed

## Robust Scaling
- **What it does:** Subtracts the median and divides by the interquartile range (middle 50%)
- **Result:** Values centred around 0 based on the typical spread, ignoring extremes
- **Plain English:** "We scale based on the typical range of your data, ignoring the extreme values"
- **Sensitive to outliers:** No — uses median and IQR which are not affected by extremes
- **Use when:** Data has significant outliers that cannot be removed

## Power Transformation (Yeo-Johnson)
- **What it does:** Applies a mathematical transformation to make the distribution more symmetrical
- **Result:** More normally shaped distributions
- **Plain English:** "We reshape your data to remove the long tail and make it more balanced"
- **Sensitive to outliers:** Moderate — helps reduce the effect of extreme values
- **Use when:** Heavily skewed columns, linear models benefit from normality

## No Scaling
- **What it does:** Leaves values unchanged
- **Use when:** Tree-based models (Random Forest, XGBoost, LightGBM) — they split on
  thresholds and are not affected by scale

---

## The Data Leakage Rule — Explained

**What is data leakage?**
Leakage happens when information from outside the training set influences the model —
making it appear more accurate than it really is.

**How scaling causes leakage:**
If you fit the scaler on the full dataset (train + validation + test), the scaler
learns statistics (mean, std, min, max) that include the test data. When the model
is later evaluated on the test set, the test data has already influenced how it
was processed — giving the model an unfair advantage.

**The rule:**
1. Split data into train / val / test FIRST
2. Fit the scaler on train ONLY
3. Use that same fitted scaler to transform val and test
4. Save the fitted scaler — use it for all new data in production

**Why save the scaler?**
When the deployed model receives new data, that data must be scaled using exactly
the same parameters (same mean, same std) that were used during training.
If you re-fit the scaler on new data, the model will receive differently scaled
inputs and its predictions will be unreliable.
