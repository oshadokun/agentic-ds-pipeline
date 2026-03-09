---
name: orchestrator
description: >
  The central intelligence of the guided data science pipeline app. Use this skill
  whenever a user wants to start, continue, or manage a data science pipeline session.
  This skill governs everything — it understands the user's goal, coordinates all
  specialist agents in sequence, bridges the UI and pipeline, enforces privacy rules,
  manages session state, builds the plain English report progressively, and defines
  when the pipeline is complete. Trigger on any of: "start a pipeline", "continue my
  session", "build a model", "analyse my data", "run the pipeline", or any request
  that implies an end-to-end data science workflow. This skill must be read before
  any other pipeline skill is used.
---

# Orchestrator Skill

The Orchestrator is the central intelligence of the guided data science pipeline. It
does not do data science itself — it governs everything around it. No agent runs
without the Orchestrator's knowledge. No decision is made without it being logged.
No stage proceeds without the user's approval.

---

## Responsibilities

1. **Understand the user's goal** before any pipeline work begins
2. **Bridge the UI and pipeline** — translate user inputs into agent instructions and agent outputs into plain English for the UI
3. **Enforce privacy** — ensure no agent violates data handling rules
4. **Coordinate agents** — call each specialist agent in sequence via subprocesses
5. **Manage configuration** — build and pass the correct config to each agent
6. **Handle failures** — define and execute recovery policies for every failure mode
7. **Log every decision** — build the plain English report progressively as the pipeline runs
8. **Track session state** — ensure the pipeline is always resumable
9. **Define done** — know when each stage and the overall pipeline is complete

---

## Starting a Session

### New Session
When a user starts a new session the Orchestrator must:

1. Generate a unique `session_id` (timestamp + random suffix)
2. Create the session directory: `sessions/{session_id}/`
3. Initialise `session.json` (see Session State schema below)
4. Present the **Goal Capture** dialogue to the user via the UI
5. Write the confirmed goal to `session.json` before proceeding

**Goal Capture — questions to ask the user:**
- What is the problem you are trying to solve? (plain English)
- What does success look like to you?
- Where is your data? (CSV upload / database / API)
- Do you have a column in your data that represents what you want to predict or understand? If so, what is it called?

Translate answers into a structured `goal` object in `session.json`. Do not infer task type silently — confirm it with the user in plain English before writing it.

### Resuming a Session
When a user returns to an existing session:

1. Load `session.json` from `sessions/{session_id}/`
2. Identify the last completed stage
3. Present a **Session Summary** to the user via the UI showing:
   - What has been done so far (in plain English)
   - Key decisions made and why
   - Which stage comes next
4. Ask the user to confirm they want to continue before proceeding

---

## Session State Schema

All state lives in `sessions/{session_id}/session.json`. This is the single source of truth.

```json
{
  "session_id": "20260308-a4f3b",
  "created_at": "2026-03-08T10:00:00Z",
  "last_updated": "2026-03-08T11:30:00Z",
  "status": "in_progress",

  "goal": {
    "plain_english": "Predict which customers are likely to cancel their subscription",
    "task_type": "binary_classification",
    "target_column": "churned",
    "success_criteria": "A model that can identify at-risk customers before they leave",
    "confirmed_by_user": true
  },

  "data_source": {
    "type": "csv",
    "path": "sessions/{session_id}/data/raw/upload.csv",
    "confirmed_by_user": true
  },

  "config": {
    "framework": null,
    "model": null,
    "scaler": null,
    "impute_strategy": null,
    "evaluation_metric": null
  },

  "privacy": {
    "sensitive_columns_identified": [],
    "user_acknowledged": false,
    "encryption_at_rest": true
  },

  "stages": {
    "ingestion":         { "status": "complete", "completed_at": "...", "summary": "..." },
    "validation":        { "status": "complete", "completed_at": "...", "summary": "..." },
    "eda":               { "status": "in_progress", "started_at": "...", "summary": null },
    "cleaning":          { "status": "pending" },
    "feature_engineering": { "status": "pending" },
    "normalisation":     { "status": "pending" },
    "splitting":         { "status": "pending" },
    "training":          { "status": "pending" },
    "evaluation":        { "status": "pending" },
    "tuning":            { "status": "pending" },
    "explainability":    { "status": "pending" },
    "deployment":        { "status": "pending" },
    "monitoring":        { "status": "pending" }
  },

  "report": {
    "sections": []
  },

  "errors": []
}
```

---

## Stage Execution Flow

For each stage the Orchestrator follows this exact sequence:

```
1. PRE-STAGE BRIEFING
   → Present what this stage does in plain English
   → Explain why it matters
   → Tell the user what they will be asked to decide
   → Ask user to confirm they are ready to proceed

2. CALL AGENT
   → Build agent config from session.json
   → Spawn agent subprocess: claude -p "<agent instructions>" 
   → Pass config and data paths as arguments
   → Capture agent output

3. TRANSLATE OUTPUT
   → Convert technical agent output to plain English
   → Identify any decisions the user needs to make
   → Identify any warnings or anomalies

4. PRESENT TO USER (via UI)
   → Show what the agent found or did
   → Present any required decisions with alternatives and tradeoffs
   → Wait for user confirmation or choice

5. LOG DECISION
   → Write user's decision and reasoning to session.json
   → Append plain English summary to report sections
   → Update stage status to "complete"

6. ADVANCE
   → Update session.json last_updated
   → Proceed to next stage
```

**Never skip step 4.** The user must always see what happened and confirm before the pipeline advances.

---

## Agent Call Pattern

Each agent is called as a subprocess. The Orchestrator writes a temporary instruction
file and passes it to the agent along with the session config.

```bash
claude -p "$(cat agents/{agent_name}/instructions.txt)" \
  --session sessions/{session_id}/session.json \
  --stage {stage_name}
```

Each agent returns a structured JSON result written to:
`sessions/{session_id}/outputs/{stage_name}/result.json`

The Orchestrator reads this result and translates it before touching the UI.

→ See `references/agent-contracts.md` for the input/output contract each agent expects.

---

## Failure Handling

Every failure falls into one of three categories:

### Recoverable — try an alternative automatically
Examples: imputation strategy fails, scaler throws an error
Action: log the failure, try the next best alternative, inform the user what happened

### Retryable — try the same thing again
Examples: API timeout, file read error
Action: retry up to 3 times with exponential backoff, then escalate to user

### Hard Stop — halt and surface to user
Examples: validation fails critical checks, no target column found, data is empty
Action: stop the pipeline, explain the problem in plain English, tell the user what they need to fix before continuing

All failures are written to `session.json` errors array with:
- stage name
- error type
- plain English description
- what was tried
- what the user needs to do

---

## Privacy Enforcement

The Orchestrator enforces privacy rules before and after every agent call:

**Before calling any agent:**
- Confirm data does not leave the local session directory
- Check agent instructions do not request external data transmission
- If the agent needs an external service, warn the user explicitly and get confirmation

**After ingestion only:**
- Scan for potentially sensitive columns (names, emails, phone numbers, IDs, financial data)
- If found, present them to the user and ask how they want to handle them (mask, drop, keep with acknowledgement)
- Write the user's decision to `session.json privacy`
- Never proceed past ingestion if sensitive columns are unacknowledged

**Credentials:**
- Database connection strings and API keys are never written to `session.json`
- They are stored in a separate encrypted `.env` file in the session directory
- They are never logged or included in the report

---

## Progressive Report Building

The plain English report is built incrementally. After every stage completes, the
Orchestrator appends a section to `session.json report.sections`:

```json
{
  "stage": "cleaning",
  "title": "Cleaning Your Data",
  "summary": "We found 312 rows with missing values...",
  "decision_made": "We filled in missing age values using the median age...",
  "alternatives_considered": "We could have removed those rows entirely...",
  "why_this_matters": "Clean data means the model learns from reliable information..."
}
```

At the end of the pipeline the Report agent assembles all sections into the final
deliverable. Nothing is written from scratch at the end — it is all accumulated
during the run.

---

## Pipeline Completion

The pipeline is complete when:
- All 13 stages have status "complete" in `session.json`
- The deployed API has returned a successful health check response
- The final report has been assembled and is available for download

At completion the Orchestrator presents the user with:
1. A summary of what was built
2. Their model file download link
3. Their API endpoint URL
4. Their plain English report download link
5. An option to start a new pipeline or modify this one

---

## Configuration Building

The Orchestrator builds a running `config` object in `session.json` as stages complete.
Each stage contributes its decisions to the config so later stages have full context.

Key config values and which stage sets them:

| Config Key | Set By Stage |
|---|---|
| task_type | Goal Capture |
| target_column | Goal Capture |
| framework | Training |
| model | Training |
| scaler | Normalisation |
| impute_strategy | Cleaning |
| evaluation_metric | Evaluation |
| feature_columns | Feature Engineering |
| split_ratios | Splitting |

---

## Plain English Translation Rules

When translating agent output for the UI, always follow these rules:

- Never use technical terms without an immediate explanation in parentheses
- Lead with what happened, not how it happened
- Lead with what it means for the user, not what the code did
- Use analogy where helpful — "scaling is like converting miles and kilometres to the same unit before comparing them"
- When presenting a recommendation, always say why in one plain sentence
- When presenting alternatives, name the tradeoff clearly — "Option A is faster but less accurate. Option B takes longer but gives better results."
- Never present more than 3 alternatives at once
- Always tell the user what happens if they do nothing (the default)

---

## Reference Files

- `references/agent-contracts.md` — input/output contracts for every agent
- `references/failure-policies.md` — detailed failure handling by stage
- `references/plain-english-glossary.md` — translations for common technical terms
