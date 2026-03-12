# Guided Data Science Pipeline

## What This Project Is

A fully guided, web-based data science pipeline application for non-technical users.
The user uploads data, describes their goal in plain English, and is walked through
every step — cleaning, training, evaluating, and deploying a machine learning model
as a live REST API. Every decision is explained. Every recommendation includes
alternatives with honest tradeoffs. No technical knowledge is required to use it.

---

## Current State (as of March 2026)

The full 14-stage pipeline is **working end-to-end**:

```
Goal Capture → Data Load → Validation → EDA → Cleaning →
Feature Engineering → Scaling → Splitting → Training →
Evaluation → Tuning → Explainability (SHAP) → Deployment → Monitoring
```

### Architecture — Compiler-Driven Orchestration (v2)

The backend was refactored in March 2026 to a compiler-driven architecture
that eliminates data leakage and makes every stage derive its decisions from
a single, validated source of truth.

**How it works**

1. **Validation** builds a `DatasetManifest`, routes the task family through
   `task_router`, compiles a `RunSpec`, and saves it to
   `sessions/{id}/artifacts/run_spec.json`. This is the first and only time
   the manifest is built and the task family is resolved.

2. **Cleaning / Feature Engineering / Normalisation** record their decisions
   (imputation strategies, scaling choice, feature columns) into the session
   config. They do not fit any transformers.

3. **Splitting** loads the saved `RunSpec`, merges all accumulated decisions
   into its preprocessing plan, then executes in strict order:
   - `split_service.split()` — pure index manipulation
   - `preprocessing_service.fit_transform()` — fits on `X_train` only,
     transforms all three splits, saves `preprocessor.pkl`

4. **Training / Evaluation** load the `RunSpec` and use it as the single
   source of truth for task family, metrics, and model selection. Both agents
   hard-fail with a clear message if `run_spec.json` is absent.

**New services and runners**

| Module | Responsibility |
|---|---|
| `contracts/schemas.py` | `DatasetManifest`, `RunSpec`, `EvaluationPayload` dataclasses |
| `services/manifest_builder.py` | Inspects raw DataFrame; builds `DatasetManifest` |
| `services/task_router.py` | Validates declared task against data; locks primary metric |
| `services/pipeline_compiler.py` | Builds `RunSpec` + unfitted sklearn `Pipeline` |
| `services/split_service.py` | Pure index split; returns `SplitResult` |
| `services/preprocessing_service.py` | Fits on `X_train` only; saves parquet splits + `preprocessor.pkl` |
| `services/evaluation_service.py` | Single-pass frozen `EvaluationPayload`; no `y.round()` |
| `services/artifact_service.py` | Packages model + preprocessor for deployment inference |
| `runners/classification_runner.py` | Baseline → SMOTE/weights → fit → baseline check |
| `runners/regression_runner.py` | Baseline → fit → baseline check |
| `runners/timeseries_runner.py` | Naive persistence baseline → fit → baseline check |

### What works

**Pipeline flow**
- All 14 stages complete without errors on classification and regression datasets
- Session state persists across browser refreshes; stages can be re-run
- Report download generated client-side after monitoring

**Data integrity**
- Scaler, imputer, and IQR capper all fitted on `X_train` only (no test-set leakage)
- Custom `_IQRCapper` transformer inside the sklearn Pipeline enforces this
- `EvaluationPayload` is frozen — all metrics from one pass, no re-computation
- Classification labels never silently rounded; continuous floats are a hard error
- SMOTE applied to `X_train` only, after preprocessing, inside `classification_runner`

**Task routing**
- Task family validated against data at validation time; `TaskRoutingError` raised
  on mismatches (e.g. regression declared on a categorical target)
- All task-family checks use strict set membership (`task_family in {"binary_classification", ...}`);
  no substring matching anywhere in the codebase

**EDA**
- When 3+ columns share the same issue, they are grouped into a single decision card
  (e.g. "27 columns with extreme values — how would you like to handle them?")
- ID-like columns (3+) grouped into a single exclusion card

**Cleaning**
- EDA outlier decisions are carried forward — cleaning never re-asks about
  columns already decided in EDA
- "What we did" summary groups repeated actions into a single collapsed line
  (e.g. "Capped extreme values in 25 columns (V1, V2, V3… and 22 more)")
  with a "show all / show less" toggle
- Structural cleaning only (dtype fixes, duplicate removal, text normalisation);
  no imputation or outlier fitting before the split

**Scaling (Normalisation)**
- Two-phase model-aware flow: first asks which model the user plans to use,
  then recommends the appropriate scaling method
  - Tree models (Random Forest, XGBoost, Decision Tree) → No scaling
  - Linear / Logistic Regression, KNN, SVM → Standard scaling
  - Neural Network → Min-Max scaling
  - ARIMA / Prophet → No scaling
  - Not decided yet → Standard scaling (safe fallback)
- Decision recorded only; scaler fitted post-split in `preprocessing_service`

**Splitting**
- User-adjustable train / validation / test sliders; always sum to 100%
- Row counts shown in real time based on dataset size
- Recommended ratios pre-filled from backend (based on dataset size)
- Confirmed ratios correctly passed to the backend split call
- Hard-fails if `run_spec.json` not found (validation must run first)
- Cross-validation path disabled until per-fold preprocessing is implemented

**Evaluation**
- Classification metrics: Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR,
  MCC, Specificity, Log Loss, Confusion Matrix
- Regression metrics: MAE, RMSE, R², MAPE
- Time series metrics: MAE and RMSE only (no R²)
- Confusion matrix rendered interactively with plain-English cell labels:
  TN = "Correctly cleared", FP = "False alarm", FN = "Missed", TP = "Correctly identified"
- Amber warning card when R² > 0.98 (possible data leakage)
- Amber warning card when AUC-ROC > 0.99 on imbalanced data

**Explainability**
- Full SHAP waterfall and bar charts for all supported model types

**Monitoring**
- Drift detection on new data; report downloadable from the frontend

### Known open issues

| Issue | Status |
|---|---|
| Tuning before/after scores show 0.0000 for classification | In progress |
| ROC curve occasionally unavailable (model wrapper mismatch) | In progress |
| Lag feature screen appears for non-time-series datasets | Needs `is_time_series` guard in FeatureEngineeringView |
| Cleaning stage grouped outlier card not always rendering (individual cards may show) | In progress |
| Cross-validation not yet supported with new preprocessing pipeline | Disabled — hard-fails with message |

---

## For Claude Code — Read This First

You are building this application. This README is your master brief.

**Before you write a single file, follow these rules:**

1. Read this README in full before doing anything else
2. Before building any component, read the skill file listed for it
3. Build in the exact order defined in this document
4. Never skip a skill file — they contain the patterns, data contracts,
   and design decisions that everything depends on
5. After completing each major section, confirm with the user before moving on
6. When creating the root `.env` file, populate it from the template defined
   in this README — never invent values

---

## Project Structure

This is the complete file and folder layout. What exists now and what you will build:

```
ds-pipeline/
│
├── README.md                          ← You are reading this
├── .env                               ← You create this (template below)
├── .env.example                       ← You create this
├── .gitignore                         ← You create this
│
├── skills/                            ← Already exists — read only, never modify
│   ├── pipeline/
│   │   ├── orchestrator/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── agent-contracts.md
│   │   │       ├── failure-policies.md
│   │   │       └── plain-english-glossary.md
│   │   ├── ingestion/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── supported-formats.md
│   │   │       └── database-connectors.md
│   │   ├── validation/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── validation-thresholds.md
│   │   ├── eda/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── eda-interpretation-guide.md
│   │   ├── cleaning/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── imputation-guide.md
│   │   │       └── outlier-guide.md
│   │   ├── feature-engineering/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── encoding-guide.md
│   │   │       └── feature-selection-guide.md
│   │   ├── normalisation/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── scaling-guide.md
│   │   ├── splitting/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── splitting-guide.md
│   │   ├── training/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       ├── model-guide.md
│   │   │       └── regularisation-guide.md
│   │   ├── evaluation/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── metrics-guide.md
│   │   ├── tuning/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── tuning-guide.md
│   │   ├── explainability/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── shap-guide.md
│   │   ├── deployment/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── deployment-guide.md
│   │   └── monitoring/
│   │       ├── SKILL.md
│   │       └── references/
│   │           └── drift-guide.md
│   ├── system/
│   │   ├── session-state/
│   │   │   ├── SKILL.md
│   │   │   └── references/
│   │   │       └── session-schema.md
│   │   └── privacy/
│   │       ├── SKILL.md
│   │       └── references/
│   │           └── privacy-regulations.md
│   └── ui/
│       ├── ui-shell/
│       │   ├── SKILL.md
│       │   └── references/
│       │       ├── stage-views.md
│       │       └── responsive-design.md
│       ├── ui-interaction/
│       │   ├── SKILL.md
│       │   └── references/
│       │       └── copy-guide.md
│       └── ui-charts/
│           ├── SKILL.md
│           └── references/
│               └── chart-data-contracts.md
│
├── backend/                           ← You build this
│   ├── main.py
│   ├── requirements.txt
│   ├── contracts/                     ← Core data contracts (v2)
│   │   └── schemas.py                 ← DatasetManifest, RunSpec, EvaluationPayload
│   ├── services/                      ← Stateless, reusable services (v2)
│   │   ├── manifest_builder.py
│   │   ├── task_router.py
│   │   ├── pipeline_compiler.py
│   │   ├── split_service.py
│   │   ├── preprocessing_service.py
│   │   ├── evaluation_service.py
│   │   └── artifact_service.py
│   ├── runners/                       ← Task-family branch runners (v2)
│   │   ├── classification_runner.py
│   │   ├── regression_runner.py
│   │   └── timeseries_runner.py
│   └── agents/
│       ├── ingestion.py
│       ├── validation.py              ← Compiles RunSpec (v2)
│       ├── eda.py
│       ├── cleaning.py                ← Structural only, no fitting (v2)
│       ├── feature_engineering.py
│       ├── normalisation.py           ← Decision recording only, no fitting (v2)
│       ├── splitting.py               ← Consumes RunSpec, orchestrates fit (v2)
│       ├── training.py                ← Delegates to runners (v2)
│       ├── evaluation.py              ← Uses evaluation_service exclusively (v2)
│       ├── tuning.py
│       ├── explainability.py
│       ├── deployment.py
│       └── monitoring.py
│
├── frontend/                          ← You build this
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       ├── components/
│       │   ├── shell/
│       │   │   ├── AppShell.jsx
│       │   │   ├── Header.jsx
│       │   │   ├── ProgressSidebar.jsx
│       │   │   └── StageNavigation.jsx
│       │   ├── shared/
│       │   │   ├── AlertBanner.jsx
│       │   │   ├── AgentRunning.jsx
│       │   │   ├── DecisionCard.jsx
│       │   │   ├── ExplanationPanel.jsx
│       │   │   ├── DataPreviewTable.jsx
│       │   │   ├── ConfirmModal.jsx
│       │   │   └── StatusBadge.jsx
│       │   ├── charts/
│       │   │   ├── FeatureDistributionChart.jsx
│       │   │   ├── TargetDistributionChart.jsx
│       │   │   ├── SplitRatioDiagram.jsx
│       │   │   ├── InteractiveConfusionMatrix.jsx
│       │   │   ├── TuningTrialChart.jsx
│       │   │   ├── FeatureImportanceChart.jsx
│       │   │   ├── PerformanceTrendChart.jsx
│       │   │   ├── DriftSummaryDonut.jsx
│       │   │   ├── StaticChart.jsx
│       │   │   ├── TimeSeriesChart.jsx        ← Upgrade 1: actual vs predicted over time
│       │   │   └── chartTheme.js
│       │   └── stages/
│       │       ├── GoalCaptureView.jsx
│       │       ├── IngestionView.jsx
│       │       ├── ValidationView.jsx
│       │       ├── EDAView.jsx
│       │       ├── CleaningView.jsx
│       │       ├── FeatureEngineeringView.jsx
│       │       ├── NormalisationView.jsx
│       │       ├── SplittingView.jsx
│       │       ├── TrainingView.jsx
│       │       ├── EvaluationView.jsx
│       │       ├── TuningView.jsx
│       │       ├── ExplainabilityView.jsx
│       │       ├── DeploymentView.jsx
│       │       ├── MonitoringView.jsx
│       │       └── InsightsDashboard.jsx      ← Upgrade 2: cross-stage insights tab
│       ├── contexts/
│       │   ├── SessionContext.jsx
│       │   └── PipelineContext.jsx
│       ├── stageConfig.js                     ← STAGE_ORDER and STAGE_LABELS constants
│       └── styles/
│           └── globals.css
│
└── sessions/                          ← Created at runtime, never by you
    └── {session_id}/
        ├── session.json
        ├── .env
        ├── data/
        │   ├── raw/
        │   ├── interim/
        │   └── processed/
        ├── models/
        ├── outputs/
        ├── reports/
        ├── monitoring/
        └── api/                       ← Created by deployment agent at end of pipeline
            ├── app.py
            ├── Dockerfile
            ├── requirements.txt
            └── models/
```

---

## Two Backends — Important Distinction

There are two separate backend services in this project. Do not confuse them.

**Pipeline management backend** (`backend/main.py`)
- Runs on port **8001**
- You build this
- Manages sessions, runs pipeline agents, serves chart files to the frontend
- Always running while the user is working in the app

**Deployed model API** (`sessions/{id}/api/app.py`)
- Runs on port **8000**
- Created by the deployment agent at the end of a pipeline run
- This is the user's deliverable — the live prediction endpoint
- Not built by you — generated by the pipeline at runtime

---

## Environment Files

### `.env.example` — create this file

```
# Pipeline Management Backend
BACKEND_PORT=8001
ENVIRONMENT=development
SESSIONS_DIR=sessions
CORS_ORIGINS=http://localhost:5173
```

### `.env` — create this file from the template above

Copy `.env.example` to `.env` and use the same default values.
Do not invent other values. The user will edit this file if they need to.

### `.gitignore` — create this file

```
.env
sessions/
node_modules/
__pycache__/
*.pyc
.DS_Store
dist/
*.pkl
*.tmp
```

---

## Build Order

Build in this exact sequence. Read the listed skill files before starting
each section. Do not begin a section until the previous one is complete.

---

### Phase 1 — Project Scaffold

Create the root files:
- `.env.example`
- `.env`
- `.gitignore`

No skill file required for this phase.

---

### Phase 2 — Backend

**Read before building:**
- `skills/system/session-state/SKILL.md`
- `skills/system/session-state/references/session-schema.md`
- `skills/pipeline/orchestrator/SKILL.md`
- `skills/pipeline/orchestrator/references/agent-contracts.md`

**Build `backend/main.py`**

This is the FastAPI pipeline management server. It must expose these endpoints:

```
GET  /sessions                              — list all sessions
POST /sessions                              — create new session
GET  /sessions/{id}                         — load session
DELETE /sessions/{id}                       — delete session (requires confirm=true param)

POST /sessions/{id}/stages/{stage}/run      — run a pipeline stage
GET  /sessions/{id}/stages/{stage}/result   — get stage result JSON

POST /sessions/{id}/data                    — upload CSV file
GET  /sessions/{id}/charts                  — serve a chart PNG by path param
GET  /sessions/{id}/report                  — get assembled report

GET  /health                                — health check
```

CORS must allow requests from `http://localhost:5173` (the Vite dev server).

**Build `backend/requirements.txt`**

```
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
optuna>=3.4.0
shap>=0.44.0
matplotlib>=3.7.0
seaborn>=0.13.0
scipy>=1.11.0
sqlalchemy>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
```

**Build `backend/agents/` — one file per stage**

For each agent file, read the corresponding skill before writing the code.
Each agent is a Python module with a single `run(input_path, session_id, decisions)` 
function that reads its input, does its work, and writes its result JSON to
`sessions/{session_id}/outputs/{stage}/result.json`.

| File | Read before building |
|---|---|
| `agents/ingestion.py` | `skills/pipeline/ingestion/SKILL.md` + all references |
| `agents/validation.py` | `skills/pipeline/validation/SKILL.md` + all references |
| `agents/eda.py` | `skills/pipeline/eda/SKILL.md` + all references |
| `agents/cleaning.py` | `skills/pipeline/cleaning/SKILL.md` + all references |
| `agents/feature_engineering.py` | `skills/pipeline/feature-engineering/SKILL.md` + all references |
| `agents/normalisation.py` | `skills/pipeline/normalisation/SKILL.md` + all references |
| `agents/splitting.py` | `skills/pipeline/splitting/SKILL.md` + all references |
| `agents/training.py` | `skills/pipeline/training/SKILL.md` + all references |
| `agents/evaluation.py` | `skills/pipeline/evaluation/SKILL.md` + all references |
| `agents/tuning.py` | `skills/pipeline/tuning/SKILL.md` + all references |
| `agents/explainability.py` | `skills/pipeline/explainability/SKILL.md` + all references |
| `agents/deployment.py` | `skills/pipeline/deployment/SKILL.md` + all references |
| `agents/monitoring.py` | `skills/pipeline/monitoring/SKILL.md` + all references |

Privacy checks are not a separate agent file — they are called from within
`main.py` at the checkpoints defined in `skills/system/privacy/SKILL.md`.
Read that skill before completing `main.py`.

---

### Phase 3 — Frontend Shell and Shared Components

**Read before building:**
- `skills/ui/ui-shell/SKILL.md`
- `skills/ui/ui-shell/references/stage-views.md`
- `skills/ui/ui-shell/references/responsive-design.md`
- `skills/ui/ui-interaction/SKILL.md`
- `skills/ui/ui-interaction/references/copy-guide.md`

**Build `frontend/package.json`**

```json
{
  "name": "ds-pipeline-frontend",
  "version": "1.0.0",
  "scripts": {
    "dev":   "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react":         "^18.2.0",
    "react-dom":     "^18.2.0",
    "recharts":      "^2.10.0",
    "lucide-react":  "^0.263.1"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "autoprefixer":         "^10.4.16",
    "postcss":              "^8.4.32",
    "tailwindcss":          "^3.4.0",
    "vite":                 "^5.0.0"
  }
}
```

**Build in this order:**
1. `frontend/index.html`
2. `frontend/vite.config.js`
3. `frontend/src/styles/globals.css` — design tokens and base styles
4. `frontend/src/contexts/SessionContext.jsx`
5. `frontend/src/contexts/PipelineContext.jsx`
6. `frontend/src/api.js`
7. `frontend/src/components/shell/` — all 4 shell components
8. `frontend/src/components/shared/` — all 7 shared components
9. `frontend/src/main.jsx`
10. `frontend/src/App.jsx`

---

### Phase 4 — Chart Components

**Read before building:**
- `skills/ui/ui-charts/SKILL.md`
- `skills/ui/ui-charts/references/chart-data-contracts.md`

Build all files in `frontend/src/components/charts/` — 9 components + theme file.

The chart assignment table in the skill defines which charts are interactive
(Recharts) and which are static PNG. Follow it exactly.

---

### Phase 5 — Stage Views

**Read before building each view:**

| View file | Read before building |
|---|---|
| `GoalCaptureView.jsx` | `skills/ui/ui-interaction/SKILL.md` |
| `IngestionView.jsx` | `skills/pipeline/ingestion/SKILL.md` + `skills/ui/ui-interaction/SKILL.md` |
| `ValidationView.jsx` | `skills/pipeline/validation/SKILL.md` |
| `EDAView.jsx` | `skills/pipeline/eda/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |
| `CleaningView.jsx` | `skills/pipeline/cleaning/SKILL.md` |
| `FeatureEngineeringView.jsx` | `skills/pipeline/feature-engineering/SKILL.md` |
| `NormalisationView.jsx` | `skills/pipeline/normalisation/SKILL.md` |
| `SplittingView.jsx` | `skills/pipeline/splitting/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |
| `TrainingView.jsx` | `skills/pipeline/training/SKILL.md` |
| `EvaluationView.jsx` | `skills/pipeline/evaluation/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |
| `TuningView.jsx` | `skills/pipeline/tuning/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |
| `ExplainabilityView.jsx` | `skills/pipeline/explainability/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |
| `DeploymentView.jsx` | `skills/pipeline/deployment/SKILL.md` |
| `MonitoringView.jsx` | `skills/pipeline/monitoring/SKILL.md` + `skills/ui/ui-charts/SKILL.md` |

Each stage view connects to the backend via `api.js`, passes decisions collected
from the user to the relevant stage endpoint, and displays the result returned
in the stage result JSON.

---

## How to Run the App

### Install backend dependencies
```bash
cd backend
pip install -r requirements.txt
```

### Install frontend dependencies
```bash
cd frontend
npm install
```

### Start the backend
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Start the frontend
```bash
cd frontend
npm run dev
```

The app is then available at `http://localhost:5175`
(Vite will auto-select the next available port if 5173/5174 are in use.)

---

## Core Architectural Rules

These rules apply everywhere in the codebase. Never violate them.

**Session isolation**
All session data lives inside `sessions/{session_id}/`. Nothing is written
outside this boundary. No agent reads from another session's directory.

**Single source of truth**
`session.json` is the only record of session state. If it is not in
`session.json`, it did not happen. Every agent writes its result to its
own output file AND updates session.json via the session-state module.

**Test set rule**
The test set is never used during training, tuning, or intermediate evaluation.
It is used exactly once — for the final evaluation after tuning is complete.
This rule is enforced in the evaluation agent.

**Atomic session writes**
`session.json` is always written via a temp file then renamed. Never write
directly to `session.json` — this prevents corruption on interrupted writes.

**Credentials never in session.json**
Database and API credentials go to `sessions/{id}/.env` only. They must
never appear in `session.json`, result files, logs, or any frontend response.

**Plain English in the UI**
Technical terms are never shown to the user. Every metric, error, warning,
and decision must be expressed in plain English. The copy guide at
`skills/ui/ui-interaction/references/copy-guide.md` defines the rules.

**Privacy blocks progress**
If sensitive columns are detected after ingestion, the privacy decision flow
must be completed before any pipeline stage runs. This is enforced in
`main.py` — the stage run endpoint checks `session.privacy.user_acknowledged`
before executing.

**Preprocessing fitted on training data only**
All learned transformers (imputer, IQR capper, scaler, encoder) are fitted
on `X_train` only inside `preprocessing_service.fit_transform()`, called by
the splitting agent after the split. No agent before splitting fits any
transformer. `scaled.csv` is a pass-through copy of `features.csv` and is
not treated as authoritative preprocessed input.

**RunSpec is the single source of truth**
`run_spec.json` is compiled by the validation agent and lives at
`sessions/{id}/artifacts/run_spec.json`. Every downstream agent loads it
to determine task family, primary metric, preprocessing plan, and model
selection. No agent re-infers these values from session goal text. Training,
evaluation, and splitting all hard-fail if `run_spec.json` is absent.

**Task family by strict equality only**
Task family checks always use set membership:
`task_family in {"binary_classification", "multiclass_classification"}`.
Substring matching (`"classification" in task_type`) is prohibited across
the entire codebase.

**EvaluationPayload is frozen**
All metrics are computed exactly once in `evaluation_service.evaluate()` and
stored in a frozen `EvaluationPayload`. No widget, stage, or chart may
recompute metrics independently from that payload.

---

## Resuming This Build in a Future Session

If you are resuming a build that was interrupted, do the following before
continuing:

1. Read this README in full
2. Check which files already exist
3. Read the skill file for the next file to be built
4. Continue from where the build stopped — do not rebuild completed files
   unless the user explicitly asks you to

---

### Phase 6 — Advanced Features (Upgrades applied post-build)

These upgrades extend the app after the initial build is complete.
Read the relevant skill files listed above before making any changes.

**Upgrade 1 — Time Series Forecasting**

| File | Change |
|---|---|
| `agents/feature_engineering.py` | Detect date columns; extract month/year/day-of-week/quarter/week; offer lag feature decision |
| `agents/training.py` | Add Prophet and ARIMA options; auto-recommend when date column detected |
| `agents/evaluation.py` | Add actual vs predicted time series chart output |
| `charts/TimeSeriesChart.jsx` | CREATE — interactive Recharts line chart, actual vs predicted |
| `stages/EvaluationView.jsx` | Show TimeSeriesChart when model is time series type |
| `stages/TrainingView.jsx` | Show Prophet / ARIMA options in model selection |
| `stages/FeatureEngineeringView.jsx` | Add LagFeatureCard for lag configuration decisions |

**Upgrade 2 — Insights Dashboard Tab**

| File | Change |
|---|---|
| `stages/InsightsDashboard.jsx` | CREATE — permanent tab aggregating cross-stage insights |
| `shell/AppShell.jsx` | Add Insights as always-visible tab |
| `shell/ProgressSidebar.jsx` | Reflect Insights tab in navigation |
| `contexts/SessionContext.jsx` | Expose stage result data needed by the dashboard |
| `backend/main.py` | Add `GET /sessions/{id}/insights` endpoint |

---

## App Map

This table shows how each part of the app connects end-to-end.

```
User action in browser
  → Frontend tab / stage view (frontend/src/components/stages/)
    → api.js call
      → FastAPI endpoint in backend/main.py
        → Agent module in backend/agents/{stage}.py
          → Skill file reference in skills/pipeline/{stage}/SKILL.md
            → Session state written to sessions/{id}/outputs/{stage}/result.json
              → session.json updated
                → Result returned to frontend for display
```

| Frontend Tab | Component | Endpoint | Agent | Skill File |
|---|---|---|---|---|
| Goal | GoalCaptureView | POST /sessions | — | ui-interaction/SKILL.md |
| Upload Data | IngestionView | POST /sessions/{id}/data | ingestion.py | ingestion/SKILL.md |
| Check Data | ValidationView | POST /stages/validation/run | validation.py | validation/SKILL.md |
| Explore Data | EDAView | POST /stages/eda/run | eda.py | eda/SKILL.md |
| Clean Data | CleaningView | POST /stages/cleaning/run | cleaning.py | cleaning/SKILL.md |
| Prepare Features | FeatureEngineeringView | POST /stages/feature_engineering/run | feature_engineering.py | feature-engineering/SKILL.md |
| Scale Data | NormalisationView | POST /stages/normalisation/run | normalisation.py | normalisation/SKILL.md |
| Split Data | SplittingView | POST /stages/splitting/run | splitting.py | splitting/SKILL.md |
| Train Model | TrainingView | POST /stages/training/run | training.py | training/SKILL.md |
| Evaluate | EvaluationView | POST /stages/evaluation/run | evaluation.py | evaluation/SKILL.md |
| Tune Model | TuningView | POST /stages/tuning/run | tuning.py | tuning/SKILL.md |
| Explain | ExplainabilityView | POST /stages/explainability/run | explainability.py | explainability/SKILL.md |
| Deploy | DeploymentView | POST /stages/deployment/run | deployment.py | deployment/SKILL.md |
| Monitor | MonitoringView | POST /stages/monitoring/run | monitoring.py | monitoring/SKILL.md |
| Insights | InsightsDashboard | GET /sessions/{id}/insights | reads existing outputs | session-state/SKILL.md |

**Session state** for each stage lives at:
`sessions/{id}/outputs/{stage}/result.json`

**Shared chart images** (matplotlib/seaborn, served by backend):
`sessions/{id}/reports/{stage}/*.png`

---

## Advanced Capabilities

### Time Series Forecasting

When the uploaded dataset contains a date or datetime column, the pipeline
automatically switches to time series mode:

- **Feature Engineering**: extracts month, year, day of week, quarter, week
  number, and optionally creates lag features (user selects columns and
  look-back periods: 1, 3, 7, 14, 30)
- **Training**: offers Prophet (trend + seasonality decomposition) and ARIMA
  (statistical autoregression) as model options alongside the standard
  regression models; the recommended default is auto-selected based on data
  size and date column presence
- **Evaluation**: shows an actual vs predicted chart over time
  (`TimeSeriesChart.jsx`) in addition to standard regression metrics
- **Splitting**: uses temporal split (chronological order) instead of random
  split when a date column is present

Activate automatically when a date column is detected — no user configuration
required.

---

### Insights Dashboard

A permanent tab visible throughout the pipeline that aggregates results from
every completed stage into a single plain-English view:

- **Before any data**: friendly prompt to get started
- **After ingestion**: data shape, column types, sample rows
- **After EDA**: distribution charts, correlation highlights
- **After training/evaluation**: model performance, metric cards
- **After explainability**: feature importance chart
- **If date column present**: time series trends

All charts are interactive (Recharts). All data is read from existing
`result.json` files — no new analysis is performed. Consistent with the
design system: #1B3A5C primary, #D97706 accent, DM Serif Display headings.

---

## Questions Before Building

Before you start Phase 1, confirm the following with the user:

1. Is the skills directory structure exactly as shown above?
2. Are you starting a fresh build or resuming an interrupted one?
3. Shall I build all phases in one session or pause between phases for review?
