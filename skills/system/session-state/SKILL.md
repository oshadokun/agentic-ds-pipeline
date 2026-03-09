---
name: session-state
description: >
  Responsible for saving, loading, and managing all session state across the
  pipeline. Ensures the pipeline is fully resumable at any stage — if the user
  closes the browser, their progress, decisions, and data are preserved and can
  be continued exactly where they left off. Manages the session directory
  structure, reads and writes session.json as the single source of truth, handles
  session listing and deletion, and validates session integrity on load. Called
  by the Orchestrator at the start and end of every stage. Trigger when any of
  the following are mentioned: "resume session", "continue where I left off",
  "save progress", "load session", "my previous work", "session management",
  "start over", "delete session", or any request involving session persistence
  or resumption.
---

# Session State Skill

The Session State agent is the memory of the pipeline. Every decision the user
makes, every stage that completes, every file that is created — all of it is
recorded and preserved so the user can close the browser at any point and return
later to exactly where they left off.

This skill owns the session directory structure, the `session.json` schema, and
all read/write operations against it. No other agent writes to `session.json`
directly — they return updates to the Orchestrator, which passes them here.

---

## Responsibilities

1. Create new sessions with a unique ID and initialised directory structure
2. Save session state after every stage completes
3. Load and validate session state when resuming
4. List all available sessions for the user to choose from
5. Generate a plain English resume summary when a session is loaded
6. Handle session deletion and cleanup
7. Detect and recover from corrupted or incomplete session files
8. Enforce the session directory structure consistently

---

## Session Directory Structure

Every session lives in a self-contained directory. Nothing escapes this boundary.

```
sessions/
└── {session_id}/
    ├── session.json              ← Single source of truth — never delete
    ├── .env                      ← Credentials only — never logged or exported
    ├── data/
    │   ├── raw/
    │   │   ├── ingested.csv      ← Original loaded data — never modified
    │   │   └── validated.csv     ← Post-validation (same data, confirmed clean)
    │   ├── interim/
    │   │   ├── cleaned.csv       ← Post-cleaning
    │   │   └── features.csv      ← Post-feature engineering
    │   └── processed/
    │       ├── X_train.csv
    │       ├── X_val.csv
    │       ├── X_test.csv
    │       ├── y_train.csv
    │       ├── y_val.csv
    │       └── y_test.csv
    ├── models/
    │   ├── scaler.pkl            ← Fitted scaler
    │   ├── {model_id}.pkl        ← Trained model
    │   ├── tuned_model.pkl       ← Tuned model
    │   ├── best_model.json       ← Reference to current best model
    │   └── best_params.json      ← Best hyperparameters
    ├── outputs/
    │   ├── ingestion/result.json
    │   ├── validation/result.json
    │   ├── eda/result.json
    │   ├── cleaning/result.json
    │   ├── feature_engineering/result.json
    │   ├── normalisation/result.json
    │   ├── splitting/result.json
    │   ├── training/result.json
    │   ├── evaluation/result.json
    │   ├── tuning/result.json
    │   ├── explainability/result.json
    │   ├── deployment/result.json
    │   └── monitoring/result.json
    ├── reports/
    │   ├── eda/
    │   ├── evaluation/
    │   ├── explainability/
    │   └── monitoring/
    ├── monitoring/
    │   ├── baseline.json
    │   └── report_{n}.json
    └── api/
        ├── app.py
        ├── Dockerfile
        ├── requirements.txt
        └── models/
```

---

## Creating a New Session

```python
import json
import uuid
from datetime import datetime
from pathlib import Path

STAGE_ORDER = [
    "ingestion", "validation", "eda", "cleaning",
    "feature_engineering", "normalisation", "splitting",
    "training", "evaluation", "tuning", "explainability",
    "deployment", "monitoring"
]

def create_session(goal_text=None):
    """
    Initialise a new session with a unique ID and clean directory structure.
    Returns the session_id and the initialised session dict.
    """
    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    unique_id  = uuid.uuid4().hex[:6]
    session_id = f"{timestamp}-{unique_id}"

    session = {
        "session_id":   session_id,
        "created_at":   datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "status":       "in_progress",

        "goal": {
            "plain_english":     goal_text,
            "task_type":         None,
            "target_column":     None,
            "success_criteria":  None,
            "confirmed_by_user": False
        },

        "data_source": {
            "type":              None,
            "path":              None,
            "confirmed_by_user": False
        },

        "config": {
            "framework":          None,
            "model_id":           None,
            "scaler":             None,
            "impute_strategy":    None,
            "evaluation_metric":  None,
            "feature_columns":    None,
            "split_ratios":       None,
            "class_weights":      None,
            "best_params":        None
        },

        "privacy": {
            "sensitive_columns_identified": [],
            "user_acknowledged":            False,
            "sensitive_columns_action":     {}
        },

        "stages": {
            stage: {
                "status":       "pending",
                "started_at":   None,
                "completed_at": None,
                "summary":      None,
                "decisions":    []
            }
            for stage in STAGE_ORDER
        },

        "report": {
            "sections": [],
            "assembled": False,
            "path":      None
        },

        "errors": [],
        "warnings": []
    }

    # Create directory structure
    session_dir = Path(f"sessions/{session_id}")
    for subdir in [
        "data/raw", "data/interim", "data/processed/splits",
        "models", "outputs", "reports/eda", "reports/evaluation",
        "reports/explainability", "reports/monitoring",
        "monitoring", "api"
    ]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Write initial session.json
    save_session(session, session_id)

    return session_id, session
```

---

## Saving Session State

```python
def save_session(session, session_id):
    """
    Write session.json atomically — write to temp file then rename
    to prevent corruption if interrupted.
    """
    session["last_updated"] = datetime.now().isoformat()

    session_path = Path(f"sessions/{session_id}/session.json")
    temp_path    = session_path.with_suffix(".tmp")

    with open(temp_path, "w") as f:
        json.dump(session, f, indent=2, default=str)

    # Atomic rename — prevents partial writes from corrupting the file
    temp_path.rename(session_path)


def update_stage(session, stage_name, status,
                  summary=None, decisions=None,
                  config_updates=None, report_section=None,
                  error=None):
    """
    Update a single stage's state and optionally update config and report.
    Called by the Orchestrator after each agent returns its result.
    """
    now = datetime.now().isoformat()
    stage = session["stages"][stage_name]

    stage["status"] = status

    if status == "in_progress" and not stage["started_at"]:
        stage["started_at"] = now

    if status == "complete":
        stage["completed_at"] = now
        if summary:
            stage["summary"] = summary
        if decisions:
            stage["decisions"].extend(decisions)

    if config_updates:
        session["config"].update(config_updates)

    if report_section:
        session["report"]["sections"].append(report_section)

    if error:
        session["errors"].append({
            "stage":       stage_name,
            "timestamp":   now,
            "error":       error
        })

    save_session(session, session["session_id"])
    return session
```

---

## Loading and Validating a Session

```python
def load_session(session_id):
    """
    Load a session from disk and validate its integrity.
    Returns the session dict or raises a descriptive error.
    """
    session_path = Path(f"sessions/{session_id}/session.json")

    if not session_path.exists():
        raise FileNotFoundError(
            f"No session found with ID '{session_id}'. "
            f"Use list_sessions() to see available sessions."
        )

    with open(session_path) as f:
        session = json.load(f)

    # Integrity checks
    integrity = validate_session_integrity(session, session_id)
    if not integrity["valid"]:
        session["_integrity_warnings"] = integrity["warnings"]

    return session, integrity


def validate_session_integrity(session, session_id):
    """
    Check that session.json is consistent with the files on disk.
    Catches cases where files were deleted outside the app.
    """
    warnings = []
    session_dir = Path(f"sessions/{session_id}")

    for stage in STAGE_ORDER:
        stage_data = session["stages"].get(stage, {})
        if stage_data.get("status") == "complete":
            result_path = session_dir / "outputs" / stage / "result.json"
            if not result_path.exists():
                warnings.append(
                    f"Stage '{stage}' is marked complete but its result file "
                    f"is missing. This stage may need to be re-run."
                )

    # Check data files exist for completed stages
    data_checks = {
        "ingestion":          "data/raw/ingested.csv",
        "cleaning":           "data/interim/cleaned.csv",
        "feature_engineering":"data/interim/features.csv",
        "splitting":          "data/processed/splits/X_train.csv",
        "training":           "models/best_model.json",
        "tuning":             "models/tuned_model.pkl"
    }

    for stage, rel_path in data_checks.items():
        if session["stages"][stage]["status"] == "complete":
            if not (session_dir / rel_path).exists():
                warnings.append(
                    f"Expected file '{rel_path}' is missing for completed "
                    f"stage '{stage}'."
                )

    return {
        "valid":    len(warnings) == 0,
        "warnings": warnings
    }
```

---

## Listing Available Sessions

```python
def list_sessions():
    """
    Return a summary of all available sessions for the user to choose from.
    """
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return []

    sessions = []
    for session_dir in sorted(sessions_dir.iterdir(), reverse=True):
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue
        try:
            with open(session_file) as f:
                s = json.load(f)
            last_stage = _get_last_completed_stage(s)
            sessions.append({
                "session_id":   s["session_id"],
                "created_at":   s["created_at"],
                "last_updated": s["last_updated"],
                "status":       s["status"],
                "goal":         s["goal"].get("plain_english", "No goal recorded"),
                "last_stage":   last_stage,
                "progress":     _get_progress_summary(s)
            })
        except Exception:
            continue

    return sessions


def _get_last_completed_stage(session):
    for stage in reversed(STAGE_ORDER):
        if session["stages"][stage]["status"] == "complete":
            return stage
    return "not_started"


def _get_progress_summary(session):
    completed = sum(1 for s in session["stages"].values()
                    if s["status"] == "complete")
    total = len(STAGE_ORDER)
    return f"{completed}/{total} stages complete"
```

---

## Resume Summary — Plain English

```python
def generate_resume_summary(session):
    """
    Generate a plain English summary of what has been done
    and what comes next. Presented to the user when resuming.
    """
    completed_stages = [
        stage for stage in STAGE_ORDER
        if session["stages"][stage]["status"] == "complete"
    ]
    next_stage = next(
        (stage for stage in STAGE_ORDER
         if session["stages"][stage]["status"] != "complete"),
        None
    )

    lines = [
        f"Welcome back. Here is where we left off:",
        f"",
        f"Goal: {session['goal'].get('plain_english', 'Not recorded')}",
        f"Progress: {len(completed_stages)} of {len(STAGE_ORDER)} stages complete",
        f""
    ]

    if completed_stages:
        lines.append("What we have done so far:")
        for stage in completed_stages:
            summary = session["stages"][stage].get("summary")
            if summary:
                lines.append(f"  ✓ {stage.replace('_', ' ').title()}: {summary}")
            else:
                lines.append(f"  ✓ {stage.replace('_', ' ').title()}: Complete")

    if next_stage:
        lines.append(f"")
        lines.append(
            f"Next step: {next_stage.replace('_', ' ').title()}"
        )
        lines.append(
            f"Shall we continue from here?"
        )
    else:
        lines.append(f"")
        lines.append(f"The pipeline is complete.")

    return "\n".join(lines)
```

---

## Deleting a Session

```python
import shutil

def delete_session(session_id, confirm=False):
    """
    Delete a session and all its associated files.
    Requires explicit confirmation — this cannot be undone.
    """
    if not confirm:
        return {
            "status": "requires_confirmation",
            "message": (
                f"This will permanently delete all data, models, and reports "
                f"for session '{session_id}'. This cannot be undone. "
                f"Please confirm to proceed."
            )
        }

    session_dir = Path(f"sessions/{session_id}")
    if session_dir.exists():
        shutil.rmtree(session_dir)
        return {
            "status":  "deleted",
            "message": (
                f"Session '{session_id}' and all associated files have been "
                f"permanently deleted."
            )
        }
    else:
        return {
            "status":  "not_found",
            "message": f"No session found with ID '{session_id}'."
        }
```

---

## Output and Interactions

This skill does not produce pipeline outputs. It is called by the Orchestrator:

- **On session start:** `create_session()` or `load_session()`
- **After every stage:** `update_stage()` → `save_session()`
- **On resume:** `load_session()` → `generate_resume_summary()`
- **From UI session list:** `list_sessions()`
- **On delete request:** `delete_session(session_id, confirm=True)`

---

## What to Tell the User

**When listing sessions:**
"Here are your previous sessions. Select one to continue or start a new one."
[Show each session with: goal, progress, last updated date]

**When resuming:**
[Show the resume summary generated by generate_resume_summary()]

**When integrity warnings exist:**
"We noticed some files are missing from this session. This sometimes happens
if files were moved or deleted outside the app. The affected stages may need
to be re-run. Here is what we found: [list warnings]"

**When deleting:**
Always ask for explicit confirmation. Make clear it cannot be undone.
Never delete without confirmed user intent.

---

## Reference Files

- `references/session-schema.md` — full schema reference for session.json
