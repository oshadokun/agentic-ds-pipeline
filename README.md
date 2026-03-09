# Guided Data Science Pipeline

## What This Project Is

A fully guided, web-based data science pipeline application for non-technical users.
The user uploads data, describes their goal in plain English, and is walked through
every step — cleaning, training, evaluating, and deploying a machine learning model
as a live REST API. Every decision is explained. Every recommendation includes
alternatives with honest tradeoffs. No technical knowledge is required to use it.

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
│   └── agents/
│       ├── ingestion.py
│       ├── validation.py
│       ├── eda.py
│       ├── cleaning.py
│       ├── feature_engineering.py
│       ├── normalisation.py
│       ├── splitting.py
│       ├── training.py
│       ├── evaluation.py
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
│       │       └── MonitoringView.jsx
│       ├── contexts/
│       │   ├── SessionContext.jsx
│       │   └── PipelineContext.jsx
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

The app is then available at `http://localhost:5173`

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

**Scaler fitted on training data only**
The scaler is always fitted on `X_train` only and then applied to `X_val`
and `X_test`. It is never fitted on the full dataset. This is enforced in
the normalisation agent.

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

## Questions Before Building

Before you start Phase 1, confirm the following with the user:

1. Is the skills directory structure exactly as shown above?
2. Are you starting a fresh build or resuming an interrupted one?
3. Shall I build all phases in one session or pause between phases for review?
