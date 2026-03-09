# Stage Views Reference

Each stage view follows the same three-zone layout:
1. **Stage Header** — title, plain English explanation, what to expect
2. **Stage Body** — stage-specific content (charts, decisions, results)
3. **Stage Navigation** — back / continue buttons

---

## Goal Capture View
**Plain English label:** "What Would You Like to Predict?"

Zone 1 — Header:
- Title: "Let's start with your goal"
- Explanation: "Tell us what you want to predict. We will ask a few questions to make sure we understand, then guide you through the rest."

Zone 2 — Body:
- Large open text input: "Describe what you want the model to predict, in your own words"
- Followed by guided clarification questions (task type, target column, what success looks like)
- Summary card once goal is confirmed

Zone 3 — Navigation:
- Single button: "Confirm Goal and Load Data →"

---

## Ingestion View
**Plain English label:** "Load Your Data"

Zone 1 — Header:
- Title: "Load Your Data"
- Explanation: "Start by connecting the data you want to work with. We support CSV files, databases, and API endpoints."

Zone 2 — Body:
- Three option cards: Upload CSV file / Connect Database / Connect API
- File drop zone (CSV path)
- On success: data preview table (first 10 rows), row/column count, file name

Zone 3 — Navigation:
- "Continue to Quality Check →"

---

## Validation View
**Plain English label:** "Check Data Quality"

Zone 2 — Body:
- Validation results displayed as a checklist
- Each check shows: icon (pass/warn/fail) + plain English result
- Warnings shown as amber alert banners with decision prompts
- Hard stops shown as red alert banners — cannot proceed

---

## EDA View
**Plain English label:** "Explore Your Data"

Zone 2 — Body:
- Charts presented one at a time, not all at once
- Each chart has a plain English caption below it
- Navigation between charts: "← Previous" / "Next →"
- Summary findings shown after all charts viewed

---

## Cleaning View
**Plain English label:** "Clean Your Data"

Zone 2 — Body:
- Issues listed as cards — one per problem found
- Each card: what the problem is / recommended fix / alternative options
- User selects their preferred approach per issue
- Preview of before/after row count after decisions made

---

## Feature Engineering View
**Plain English label:** "Prepare Features"

Zone 2 — Body:
- Proposed transformations listed with plain English explanations
- User can approve all at once or review individually
- Feature importance preview chart after engineering completes

---

## Normalisation View
**Plain English label:** "Scale Your Data"

Zone 2 — Body:
- Recommended scaler shown with plain English explanation and tradeoffs
- Before/after distribution comparison chart
- Simple approve/change interface

---

## Splitting View
**Plain English label:** "Divide Your Data"

Zone 2 — Body:
- Visual diagram of the split (train/val/test proportions as a segmented bar)
- Recommended ratios with plain English explanation
- Slider to adjust ratios (within recommended bounds)

---

## Training View
**Plain English label:** "Train the Model"

Zone 2 — Body:
- Baseline model runs first — shown with explanation of why
- Progress indicator during training
- Model selection card once training completes
- Baseline metric shown with plain English interpretation

---

## Evaluation View
**Plain English label:** "Measure Performance"

Zone 2 — Body:
- Performance verdict first (strong / good / fair / poor) — large and clear
- Plain English metric interpretations
- Charts (confusion matrix, ROC curve, or residual plots) with captions
- Confusion matrix breakdown in plain English

---

## Tuning View
**Plain English label:** "Fine-Tune the Model"

Zone 2 — Body:
- Explanation of what tuning will do
- Time estimate shown before tuning starts
- Progress indicator during Optuna search
- Before/after comparison once complete
- Plain English explanation of what changed

---

## Explainability View
**Plain English label:** "Understand the Model"

Zone 2 — Body:
- Global feature importance chart with narration
- SHAP summary plot with caption
- Individual prediction examples (tabbed — Example 1, 2, 3)
- Bias check results — amber warning if any issues found

---

## Deployment View
**Plain English label:** "Deploy the Model"

Zone 2 — Body:
- Plain English explanation of what deployment means
- Pre-deployment checklist (model trained, scaler saved, test passed)
- Deploy button — triggers API generation and startup
- Success state: API URL, docs link, copy-to-clipboard for curl example
- Test prediction interface — fill in values, see live result

---

## Monitoring View
**Plain English label:** "Monitor Performance"

Zone 2 — Body:
- Health status summary — large and clear (healthy / warning / attention needed)
- Drift summary chart
- Feature-by-feature drift table (feature / status / plain English finding)
- Retraining recommendation if triggered
- "Run Check Again" button
