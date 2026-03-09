# Interaction Flow Maps

Each stage follows: Message → Decision(s) → Confirm → Proceed

---

## Goal Capture
1. OrchestratorMessage: "Tell us what you want to predict"
2. FreeTextInput: user describes goal
3. OrchestratorMessage: clarification questions (task type, target column)
4. SingleChoiceDecision: task type (classification / regression)
5. OrchestratorMessage: goal summary
6. Confirm → proceed to Ingestion

## Ingestion
1. OrchestratorMessage: "Load your data"
2. SingleChoiceDecision: CSV / Database / API
3. [CSV] FileDropZone → upload → success
4. [DB] CredentialForm → test connection → success
5. ChartDisplay: data preview table (first 10 rows)
6. OrchestratorMessage: row count, column count, any immediate issues
7. Confirm → proceed to Validation

## Validation
1. OrchestratorMessage: "Checking your data quality"
2. AgentRunning (brief — validation is fast)
3. ChecklistDisplay: each validation check with pass/warn/fail
4. [If warnings] AlertBanner + SingleChoiceDecision per warning
5. [If hard stop] AlertBanner type="error" + clear instruction to fix
6. Confirm → proceed to EDA

## EDA
1. OrchestratorMessage: "Let's explore your data"
2. Chart carousel — one chart at a time
   - ChartDisplay with caption
   - "Next →" / "← Previous" navigation
3. OrchestratorMessage: summary of key findings
4. Confirm → proceed to Cleaning

## Cleaning
1. OrchestratorMessage: issues found summary
2. One DecisionCard per issue (missing values, outliers, duplicates)
3. Each card: recommended + alternatives with tradeoffs
4. OrchestratorMessage: preview of what will change (row count before/after)
5. ConfirmModal: "Apply these changes — this will modify your data"
6. AgentRunning → Confirm → proceed to Feature Engineering

## Feature Engineering
1. OrchestratorMessage: proposed transformations
2. SingleChoiceDecision or approve-all toggle
3. ChartDisplay: feature importance preview (if available)
4. Confirm → proceed to Normalisation

## Normalisation
1. OrchestratorMessage: recommended scaler + plain English explanation
2. SingleChoiceDecision: approve recommended or change
3. ChartDisplay: before/after distribution comparison
4. Confirm → proceed to Splitting

## Splitting
1. OrchestratorMessage: recommended ratios + explanation
2. SplitDiagram: visual segmented bar
3. Sliders to adjust ratios (within bounds)
4. OrchestratorMessage: confirms stratification if applicable
5. Confirm → proceed to Training

## Training
1. OrchestratorMessage: "We will start with a simple model"
2. AgentRunning: baseline training (with progress)
3. OrchestratorMessage: baseline result in plain English
4. ComparisonTable: model options (if moving beyond baseline)
5. SingleChoiceDecision: which model to use
6. AgentRunning: full training
7. OrchestratorMessage: training complete summary
8. Confirm → proceed to Evaluation

## Evaluation
1. OrchestratorMessage: performance verdict (large, prominent)
2. Plain English metric interpretations (not raw numbers alone)
3. ChartDisplay: confusion matrix (classification) or residuals (regression)
4. ChartDisplay: ROC curve (classification)
5. OrchestratorMessage: recommendation (proceed to tuning / revisit)
6. Confirm → proceed to Tuning

## Tuning
1. OrchestratorMessage: what tuning will do + time estimate
2. SingleChoiceDecision: proceed / adjust number of trials
3. AgentRunning: Optuna search (with trial count progress)
4. OrchestratorMessage: before/after comparison + parameter explanation
5. Confirm → proceed to Explainability

## Explainability
1. OrchestratorMessage: "Let's look inside the model"
2. ChartDisplay: global feature importance chart + narration
3. ChartDisplay: SHAP summary plot + caption
4. Tabbed individual examples (Example 1 / 2 / 3)
   - Each tab: waterfall chart + plain English narration
5. [If bias warnings] AlertBanner type="warning" + explanation
6. Confirm → proceed to Deployment

## Deployment
1. OrchestratorMessage: what deployment means in plain English
2. Pre-deployment checklist (visual — all items must be green)
3. ConfirmModal: "Deploy your model — this will start a live API"
4. AgentRunning: API generation + startup
5. OrchestratorMessage: success — API URL, docs link, how to use
6. LiveTestWidget: fill in values → see live prediction result
7. Confirm → proceed to Monitoring (or mark pipeline complete)

## Monitoring
1. OrchestratorMessage: health check summary
2. DriftSummaryChart
3. FeatureDriftTable: each feature / status / plain English finding
4. [If retraining recommended] AlertBanner + next steps
5. "Run Check Again" button
6. Pipeline complete state if all stages done
