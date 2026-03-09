"""
Pipeline Management Backend — runs on port 8001.
Manages sessions, runs pipeline agents, serves chart files to the frontend.
"""

import base64
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, Response
from pydantic import BaseModel

load_dotenv()

SESSIONS_DIR = os.getenv("SESSIONS_DIR", "sessions")
CORS_ORIGINS  = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app = FastAPI(title="DS Pipeline Management Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Stage order — single source of truth for pipeline sequence
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "ingestion", "validation", "eda", "cleaning",
    "feature_engineering", "normalisation", "splitting",
    "training", "evaluation", "tuning", "explainability",
    "deployment", "monitoring"
]

# ---------------------------------------------------------------------------
# Session State helpers  (from session-state SKILL)
# ---------------------------------------------------------------------------

def _sessions_root() -> Path:
    p = Path(SESSIONS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_session(goal_text: Optional[str] = None) -> tuple[str, dict]:
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

        "errors":   [],
        "warnings": []
    }

    session_dir = _sessions_root() / session_id
    for subdir in [
        "data/raw", "data/interim", "data/processed/splits",
        "models", "outputs", "reports/eda", "reports/evaluation",
        "reports/explainability", "reports/monitoring",
        "monitoring", "api"
    ]:
        (session_dir / subdir).mkdir(parents=True, exist_ok=True)

    save_session(session, session_id)
    return session_id, session


def save_session(session: dict, session_id: str) -> None:
    """Write session state to disk."""
    session["last_updated"] = datetime.now().isoformat()
    session_path = _sessions_root() / session_id / "session.json"

    with open(session_path, "w") as f:
        json.dump(session, f, indent=2, default=str)


def load_session(session_id: str) -> tuple[dict, dict]:
    session_path = _sessions_root() / session_id / "session.json"

    if not session_path.exists():
        raise FileNotFoundError(
            f"No session found with ID '{session_id}'."
        )

    with open(session_path) as f:
        session = json.load(f)

    integrity = _validate_session_integrity(session, session_id)
    if not integrity["valid"]:
        session["_integrity_warnings"] = integrity["warnings"]

    return session, integrity


def _validate_session_integrity(session: dict, session_id: str) -> dict:
    warnings = []
    session_dir = _sessions_root() / session_id

    for stage in STAGE_ORDER:
        stage_data = session["stages"].get(stage, {})
        if stage_data.get("status") == "complete":
            result_path = session_dir / "outputs" / stage / "result.json"
            if not result_path.exists():
                warnings.append(
                    f"Stage '{stage}' is marked complete but its result file is missing."
                )

    data_checks = {
        "ingestion":           "data/raw/ingested.csv",
        "cleaning":            "data/interim/cleaned.csv",
        "feature_engineering": "data/interim/features.csv",
        "splitting":           "data/processed/splits/X_train.csv",
        "training":            "models/best_model.json",
        "tuning":              "models/tuned_model.pkl"
    }
    for stage, rel_path in data_checks.items():
        if session["stages"][stage]["status"] == "complete":
            if not (session_dir / rel_path).exists():
                warnings.append(
                    f"Expected file '{rel_path}' is missing for completed stage '{stage}'."
                )

    return {"valid": len(warnings) == 0, "warnings": warnings}


def update_stage(session: dict, stage_name: str, status: str,
                  summary: Optional[str] = None,
                  decisions: Optional[list] = None,
                  config_updates: Optional[dict] = None,
                  report_section: Optional[dict] = None,
                  error: Optional[str] = None) -> dict:
    now   = datetime.now().isoformat()
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
            "stage":     stage_name,
            "timestamp": now,
            "error":     error
        })

    save_session(session, session["session_id"])
    return session


def list_sessions() -> list:
    root = _sessions_root()
    sessions = []
    for session_dir in sorted(root.iterdir(), reverse=True):
        session_file = session_dir / "session.json"
        if not session_file.exists():
            continue
        try:
            with open(session_file) as f:
                s = json.load(f)
            last_stage = next(
                (st for st in reversed(STAGE_ORDER)
                 if s["stages"][st]["status"] == "complete"),
                "not_started"
            )
            completed = sum(
                1 for st in STAGE_ORDER
                if s["stages"][st]["status"] == "complete"
            )
            sessions.append({
                "session_id":   s["session_id"],
                "created_at":   s["created_at"],
                "last_updated": s["last_updated"],
                "status":       s["status"],
                "goal":         s["goal"].get("plain_english", "No goal recorded"),
                "last_stage":   last_stage,
                "progress":     f"{completed}/{len(STAGE_ORDER)} stages complete"
            })
        except Exception:
            continue
    return sessions


def generate_resume_summary(session: dict) -> str:
    completed = [
        st for st in STAGE_ORDER
        if session["stages"][st]["status"] == "complete"
    ]
    next_stage = next(
        (st for st in STAGE_ORDER
         if session["stages"][st]["status"] != "complete"),
        None
    )
    lines = [
        "Welcome back. Here is where we left off:",
        "",
        f"Goal: {session['goal'].get('plain_english', 'Not recorded')}",
        f"Progress: {len(completed)} of {len(STAGE_ORDER)} stages complete",
        ""
    ]
    if completed:
        lines.append("What we have done so far:")
        for st in completed:
            summary = session["stages"][st].get("summary")
            label   = st.replace("_", " ").title()
            lines.append(f"  \u2713 {label}: {summary or 'Complete'}")
    if next_stage:
        lines += ["", f"Next step: {next_stage.replace('_', ' ').title()}", "Shall we continue from here?"]
    else:
        lines += ["", "The pipeline is complete."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Privacy helpers  (from privacy SKILL)
# ---------------------------------------------------------------------------

SENSITIVE_NAME_PATTERNS = {
    "direct_identifiers": [
        "name", "full_name", "first_name", "last_name", "surname",
        "email", "email_address", "phone", "mobile", "telephone",
        "address", "street", "postcode", "zip", "zipcode",
        "passport", "national_id", "ssn", "social_security",
        "driving_licence", "license_number"
    ],
    "quasi_identifiers": [
        "dob", "date_of_birth", "birth_date", "birthdate",
        "age", "gender", "sex", "ethnicity", "race", "nationality",
        "religion", "marital_status", "occupation"
    ],
    "financial": [
        "salary", "income", "wage", "bank_account", "account_number",
        "credit_card", "card_number", "iban", "sort_code",
        "tax_id", "nino", "national_insurance"
    ],
    "technical_identifiers": [
        "ip_address", "ip", "device_id", "mac_address",
        "user_id", "customer_id", "patient_id", "employee_id",
        "cookie", "session_token", "auth_token"
    ],
    "location": [
        "gps", "latitude", "longitude", "lat", "lon", "lng",
        "geolocation", "location", "coordinates"
    ]
}

HANDLING_OPTIONS = {
    "drop": {
        "label": "Remove this column entirely",
        "tradeoff": "Eliminates all privacy risk. The model will not have access to this information.",
        "reversible": False
    },
    "pseudonymise": {
        "label": "Replace with a consistent anonymous code",
        "tradeoff": "Replaces names/IDs with a code. Same person always gets the same code, original value cannot be recovered.",
        "reversible": False
    },
    "mask": {
        "label": "Replace with a generic placeholder",
        "tradeoff": "Replaces all values with [REDACTED]. Simplest option.",
        "reversible": False
    },
    "keep_with_acknowledgement": {
        "label": "Keep as-is — I understand this column contains sensitive data",
        "tradeoff": "The column is kept and used by the model. You take responsibility for appropriateness.",
        "reversible": True
    }
}


def scan_for_sensitive_columns(df) -> list:
    import re
    import pandas as pd
    findings = []

    SENSITIVE_VALUE_PATTERNS = {
        "email":   r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
        "phone":   r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$",
        "ip_address": r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    }

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        matched_category = None
        matched_pattern  = None

        for category, patterns in SENSITIVE_NAME_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower:
                    matched_category = category
                    matched_pattern  = pattern
                    break
            if matched_category:
                break

        if not matched_category and df[col].dtype == object:
            sample = df[col].dropna().head(50).astype(str)
            for value_type, regex in SENSITIVE_VALUE_PATTERNS.items():
                match_rate = sample.str.match(regex, na=False).mean()
                if match_rate > 0.5:
                    matched_category = "value_pattern"
                    matched_pattern  = value_type
                    break

        if matched_category:
            sample_val = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "empty"
            # Partially redact for display
            if len(sample_val) > 4:
                display_val = sample_val[:2] + "*" * (len(sample_val) - 4) + sample_val[-2:]
            else:
                display_val = "****"

            recommended = {
                "direct_identifiers":    "drop",
                "financial":             "drop",
                "technical_identifiers": "drop",
                "quasi_identifiers":     "keep_with_acknowledgement",
                "location":              "keep_with_acknowledgement",
                "value_pattern":         "drop"
            }.get(matched_category, "keep_with_acknowledgement")

            findings.append({
                "column":             col,
                "category":           matched_category,
                "matched_pattern":    matched_pattern,
                "sample_value":       display_val,
                "recommended_action": recommended,
                "options": [
                    {"id": k, **{kk: vv for kk, vv in v.items()}}
                    for k, v in HANDLING_OPTIONS.items()
                ],
                "plain_english": (
                    f"'{col}' appears to contain sensitive information "
                    f"({matched_category.replace('_', ' ')})."
                )
            })

    return findings


def _privacy_check_required(session: dict, stage: str) -> bool:
    """Returns True if stage is blocked by unacknowledged privacy findings."""
    if stage == "ingestion":
        return False
    sensitive = session["privacy"].get("sensitive_columns_identified", [])
    if not sensitive:
        return False
    return not session["privacy"].get("user_acknowledged", False)


# ---------------------------------------------------------------------------
# Agent dispatcher
# ---------------------------------------------------------------------------

def _run_stage_agent(session: dict, stage: str, decisions: dict) -> dict:
    """Import and call the agent module for the given stage."""
    import importlib
    import sys

    agents_dir = Path(__file__).parent / "agents"
    if str(agents_dir) not in sys.path:
        sys.path.insert(0, str(agents_dir))

    module = importlib.import_module(stage)
    importlib.reload(module)
    result = module.run(session, decisions)
    return result


def _write_result(session_id: str, stage: str, result: dict) -> None:
    output_dir = _sessions_root() / session_id / "outputs" / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    goal_text:        Optional[str] = None
    data_source_type: Optional[str] = "csv"

class UpdateGoalRequest(BaseModel):
    plain_english:    Optional[str] = None
    task_type:        Optional[str] = None
    target_column:    Optional[str] = None
    success_criteria: Optional[str] = None
    confirmed_by_user: bool = False

class PrivacyDecisionsRequest(BaseModel):
    decisions: dict  # {column: action}

class StageRunRequest(BaseModel):
    decisions: dict = {}

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "service": "ds-pipeline-backend", "version": "1.0.0"}


@app.get("/sessions")
def get_sessions():
    return {"sessions": list_sessions()}


@app.post("/sessions", status_code=201)
def post_sessions(request: CreateSessionRequest):
    session_id, session = create_session(request.goal_text)
    if request.data_source_type:
        session["data_source"]["type"] = request.data_source_type
        save_session(session, session_id)
    return {"session_id": session_id, "session": session}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    try:
        session, integrity = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    resume_summary = generate_resume_summary(session)
    return {
        "session":        session,
        "integrity":      integrity,
        "resume_summary": resume_summary
    }


@app.delete("/sessions/{session_id}")
def delete_session_endpoint(session_id: str, confirm: bool = Query(False)):
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"No session found with ID '{session_id}'.")

    if not confirm:
        return {
            "status": "requires_confirmation",
            "message": (
                f"This will permanently delete all data, models, and reports "
                f"for session '{session_id}'. This cannot be undone. "
                f"Confirm by adding ?confirm=true to the request."
            )
        }

    shutil.rmtree(session_dir)
    return {
        "status":  "deleted",
        "message": f"Session '{session_id}' and all associated files have been permanently deleted."
    }


@app.patch("/sessions/{session_id}/goal")
def update_goal(session_id: str, request: UpdateGoalRequest):
    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    for field in ["plain_english", "task_type", "target_column",
                  "success_criteria", "confirmed_by_user"]:
        val = getattr(request, field, None)
        if val is not None:
            session["goal"][field] = val

    # Resolve generic "classification" → binary or multiclass based on target cardinality
    if session["goal"].get("task_type") == "classification":
        target_col = session["goal"].get("target_column")
        data_dir   = Path("sessions") / session_id / "data" / "raw"
        csv_files  = list(data_dir.glob("*.csv")) if data_dir.exists() else []
        if target_col and csv_files:
            try:
                import pandas as _pd
                df_sample = _pd.read_csv(csv_files[0], usecols=[target_col], nrows=5000)
                n_unique  = int(df_sample[target_col].nunique())
                session["goal"]["task_type"] = (
                    "binary_classification" if n_unique <= 2 else "multiclass_classification"
                )
            except Exception:
                session["goal"]["task_type"] = "binary_classification"

    save_session(session, session_id)
    return {"session": session}


@app.post("/sessions/{session_id}/data")
async def upload_data(session_id: str, file: UploadFile = File(...)):
    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported at this endpoint.")

    raw_dir  = _sessions_root() / session_id / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest     = raw_dir / "upload.csv"

    contents = await file.read()
    dest.write_bytes(contents)

    session["data_source"]["path"] = str(dest)
    session["data_source"]["type"] = "csv"
    save_session(session, session_id)

    return {
        "status":   "uploaded",
        "path":     str(dest),
        "filename": file.filename,
        "size_bytes": len(contents)
    }


@app.post("/sessions/{session_id}/privacy")
def submit_privacy_decisions(session_id: str, request: PrivacyDecisionsRequest):
    """
    Accept user privacy decisions ({column: action}) and mark privacy as acknowledged.
    Must be called after ingestion surfaces sensitive columns and before any further stage runs.
    """
    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    import pandas as pd
    import hashlib

    session["privacy"]["sensitive_columns_action"].update(request.decisions)

    # Check all identified columns are now acknowledged
    identified = session["privacy"].get("sensitive_columns_identified", [])
    all_ack = all(col in session["privacy"]["sensitive_columns_action"] for col in identified)
    session["privacy"]["user_acknowledged"] = all_ack

    # Apply privacy transformations to the ingested data
    raw_path = _sessions_root() / session_id / "data" / "raw" / "ingested.csv"
    if raw_path.exists():
        df = pd.read_csv(raw_path, low_memory=False)
        for col, action in request.decisions.items():
            if col not in df.columns:
                continue
            if action == "drop":
                df = df.drop(columns=[col])
            elif action == "pseudonymise":
                df[col] = df[col].astype(str).apply(
                    lambda x: "ID_" + hashlib.sha256(x.encode()).hexdigest()[:8]
                )
            elif action == "mask":
                df[col] = "[REDACTED]"
            elif action == "keep_with_acknowledgement":
                pass  # no transformation

        # Save as validated.csv
        validated_path = _sessions_root() / session_id / "data" / "raw" / "validated.csv"
        df.to_csv(validated_path, index=False)

    if not hasattr(session, "privacy_audit"):
        session["privacy_audit"] = []

    save_session(session, session_id)
    return {
        "status":            "acknowledged",
        "user_acknowledged": session["privacy"]["user_acknowledged"],
        "actions_applied":   request.decisions
    }


@app.post("/sessions/{session_id}/stages/{stage}/run")
def run_stage(session_id: str, stage: str, request: StageRunRequest):
    if stage not in STAGE_ORDER:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{stage}'.")

    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Privacy gate — all stages after ingestion require privacy acknowledgment
    if _privacy_check_required(session, stage):
        raise HTTPException(
            status_code=403,
            detail={
                "code":    "privacy_unacknowledged",
                "message": (
                    "Before we can continue, you need to make decisions about the sensitive "
                    "columns we found in your data. Please visit the privacy section first."
                ),
                "sensitive_columns": session["privacy"]["sensitive_columns_identified"]
            }
        )

    # Mark stage as in-progress
    update_stage(session, stage, "in_progress")
    session, _ = load_session(session_id)  # reload after save

    try:
        result = _run_stage_agent(session, stage, request.decisions)
    except Exception as exc:
        update_stage(session, stage, "failed", error=str(exc))
        raise HTTPException(status_code=500, detail={
            "stage": stage,
            "error": str(exc),
            "plain_english": (
                f"Something went wrong during the {stage.replace('_', ' ')} stage. "
                f"Here is the technical detail: {str(exc)}"
            )
        })

    # Write result file
    _write_result(session_id, stage, result)

    # Update session state
    session, _ = load_session(session_id)
    if result.get("status") in ("failed", "hard_stop"):
        stage_status = "failed"
    elif result.get("status") == "decisions_required":
        stage_status = "in_progress"
    else:
        stage_status = "complete"
    update_stage(
        session,
        stage,
        status=stage_status,
        summary=result.get("plain_english_summary"),
        decisions=result.get("decisions_made", []),
        config_updates=result.get("config_updates"),
        report_section=result.get("report_section"),
        error=result.get("error") if result.get("status") == "failed" else None
    )

    return result


@app.get("/sessions/{session_id}/stages/{stage}/result")
def get_stage_result(session_id: str, stage: str):
    if stage not in STAGE_ORDER:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{stage}'.")

    result_path = _sessions_root() / session_id / "outputs" / stage / "result.json"
    if not result_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No result found for stage '{stage}'. Has it been run yet?"
        )

    with open(result_path) as f:
        return json.load(f)


@app.get("/sessions/{session_id}/charts")
def get_chart(session_id: str, path: str = Query(..., description="Relative path to chart PNG within session directory")):
    """Serve a chart PNG file. The `path` param is relative to the session directory."""
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    # Sanitise path — prevent directory traversal
    # Paths may be relative to project root (e.g. sessions/id/reports/...)
    # or relative to session dir (e.g. reports/...) — handle both.
    candidate = Path(path.replace("\\", "/"))
    if not candidate.is_absolute():
        chart_path = (Path(".") / candidate).resolve()
    else:
        chart_path = candidate.resolve()
    if not str(chart_path).startswith(str(session_dir.resolve())):
        raise HTTPException(status_code=403, detail="Access denied.")

    if not chart_path.exists():
        raise HTTPException(status_code=404, detail=f"Chart not found: {path}")

    return FileResponse(str(chart_path), media_type="image/png")


@app.get("/sessions/{session_id}/report")
def get_report(session_id: str):
    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    sections = session.get("report", {}).get("sections", [])
    return {
        "session_id": session_id,
        "goal":       session["goal"],
        "sections":   sections,
        "assembled":  session["report"].get("assembled", False),
        "stage_summaries": {
            stage: session["stages"][stage].get("summary")
            for stage in STAGE_ORDER
            if session["stages"][stage]["status"] == "complete"
        }
    }


@app.get("/sessions/{session_id}/report/html")
def download_report_html(session_id: str):
    """Generate and return an HTML report of the completed pipeline run."""
    try:
        session, _ = load_session(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    goal        = session["goal"]
    sections    = session.get("report", {}).get("sections", [])
    created_at  = session.get("created_at", "")[:10]
    config      = session.get("config", {})

    STAGE_TITLES = {
        "ingestion":           "1. Load Your Data",
        "validation":          "2. Validate Data",
        "eda":                 "3. Explore Your Data",
        "cleaning":            "4. Clean Data",
        "feature_engineering": "5. Prepare Features",
        "normalisation":       "6. Scale Data",
        "splitting":           "7. Split Data",
        "training":            "8. Train Model",
        "evaluation":          "9. Evaluate Model",
        "tuning":              "10. Fine-Tune Model",
        "explainability":      "11. Understand Predictions",
        "deployment":          "12. Deploy Model",
        "monitoring":          "13. Monitor Performance",
    }

    # Map each stage to the image files that belong to it
    session_dir = _sessions_root() / session_id
    STAGE_IMAGES = {
        "eda":                 sorted((session_dir / "reports" / "eda").glob("*.png")) if (session_dir / "reports" / "eda").exists() else [],
        "normalisation":       [session_dir / "reports" / "scaling_comparison.png"] if (session_dir / "reports" / "scaling_comparison.png").exists() else [],
        "evaluation":          sorted((session_dir / "reports" / "evaluation").glob("*.png")) if (session_dir / "reports" / "evaluation").exists() else [],
        "explainability":      [p for p in sorted((session_dir / "reports" / "explainability").glob("*.png")) if "explanation_row" not in p.name] if (session_dir / "reports" / "explainability").exists() else [],
        "monitoring":          sorted((session_dir / "reports" / "monitoring").glob("*.png")) if (session_dir / "reports" / "monitoring").exists() else [],
    }

    def img_tag(path: Path) -> str:
        data = base64.b64encode(path.read_bytes()).decode()
        label = path.stem.replace("_", " ").title()
        return (
            f'<figure class="chart">'
            f'<img src="data:image/png;base64,{data}" alt="{label}" />'
            f'<figcaption>{label}</figcaption>'
            f'</figure>'
        )

    # Build section HTML
    sections_html = ""
    for stage in STAGE_ORDER:
        stage_data = session["stages"].get(stage, {})
        if stage_data.get("status") not in ("complete",):
            continue
        summary = stage_data.get("summary") or ""
        title   = STAGE_TITLES.get(stage, stage.replace("_", " ").title())

        # Find matching report section for extra detail
        extra = next((s for s in sections if s.get("stage") == stage), {})
        why   = extra.get("why_this_matters", "")
        dec   = extra.get("decision_made", "")

        # Embed any charts for this stage
        imgs_html = ""
        for img_path in STAGE_IMAGES.get(stage, []):
            if img_path.exists():
                imgs_html += img_tag(img_path)

        sections_html += f"""
        <div class="section">
          <h2>{title}</h2>
          <p class="summary">{summary}</p>
          {"<p class='decision'><strong>Decision made:</strong> " + dec + "</p>" if dec else ""}
          {('<div class="charts">' + imgs_html + '</div>') if imgs_html else ""}
          {"<p class='why'><em>" + why + "</em></p>" if why else ""}
        </div>
        """

    # Key config stats
    model_id = config.get("model_id", "—").replace("_", " ").title()
    metric   = config.get("primary_metric", "—").upper().replace("_", " ")
    score    = config.get("primary_metric_value")
    score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
    verdict   = config.get("verdict", "—").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Pipeline Report — {created_at}</title>
  <style>
    body {{ font-family: Georgia, serif; max-width: 820px; margin: 40px auto; padding: 0 24px;
            color: #1a1a1a; line-height: 1.7; }}
    h1   {{ font-size: 2rem; color: #1B3A5C; border-bottom: 3px solid #1B3A5C;
            padding-bottom: 12px; margin-bottom: 8px; }}
    .meta {{ color: #666; font-size: 0.9rem; margin-bottom: 32px; }}
    .stats {{ display: flex; gap: 24px; flex-wrap: wrap; margin-bottom: 40px; }}
    .stat  {{ background: #f5f8ff; border: 1px solid #dde6f5; border-radius: 12px;
              padding: 16px 24px; min-width: 140px; text-align: center; }}
    .stat-value {{ font-size: 1.6rem; font-weight: bold; color: #1B3A5C; }}
    .stat-label {{ font-size: 0.75rem; color: #888; margin-top: 4px; }}
    .section {{ margin-bottom: 36px; padding-bottom: 24px;
                border-bottom: 1px solid #eee; }}
    .section h2 {{ font-size: 1.15rem; color: #1B3A5C; margin-bottom: 8px; }}
    .summary {{ margin: 0 0 8px; }}
    .decision {{ font-size: 0.9rem; color: #444; margin: 4px 0; }}
    .why {{ font-size: 0.85rem; color: #777; margin: 6px 0 0; }}
    .footer {{ margin-top: 48px; font-size: 0.8rem; color: #aaa; text-align: center; }}
    .charts {{ display: flex; flex-wrap: wrap; gap: 16px; margin: 16px 0; }}
    .chart  {{ flex: 1 1 340px; }}
    .chart img {{ width: 100%; border-radius: 8px; border: 1px solid #e5e7eb; }}
    .chart figcaption {{ font-size: 0.75rem; color: #999; text-align: center; margin-top: 4px; }}
  </style>
</head>
<body>
  <h1>Data Science Pipeline Report</h1>
  <p class="meta">
    Goal: <strong>{goal.get('plain_english', '—')}</strong><br/>
    Task type: {goal.get('task_type', '—').replace('_', ' ').title()} &nbsp;|&nbsp;
    Target: <code>{goal.get('target_column', '—')}</code> &nbsp;|&nbsp;
    Date: {created_at}
  </p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{model_id}</div>
      <div class="stat-label">Model</div>
    </div>
    <div class="stat">
      <div class="stat-value">{score_str}</div>
      <div class="stat-label">{metric}</div>
    </div>
    <div class="stat">
      <div class="stat-value">{verdict}</div>
      <div class="stat-label">Verdict</div>
    </div>
  </div>

  {sections_html}

  <div class="footer">Generated by the Data Science Pipeline &mdash; {created_at}</div>
</body>
</html>"""

    from fastapi.responses import Response
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=\"pipeline_report_{session_id[:8]}.html\""}
    )


@app.get("/sessions/{session_id}/code/api")
def get_api_code(session_id: str):
    """Return the generated prediction API code (app.py)."""
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    app_py = session_dir / "api" / "app.py"
    if not app_py.exists():
        raise HTTPException(
            status_code=404,
            detail="API code not generated yet. Please run the deployment stage first."
        )

    code = app_py.read_text(encoding="utf-8")
    return {"code": code, "filename": "app.py", "path": str(app_py)}


def _build_analysis_script(session: dict) -> str:
    """Generate a reproducible Python analysis script from session config."""
    cfg    = session.get("config", {})
    goal   = session.get("goal", {})
    stages = session.get("stages", {})

    target       = goal.get("target_column", "target")
    task_type    = goal.get("task_type", "regression")
    model_id     = cfg.get("model_id", "ridge")
    features     = cfg.get("feature_columns", [])
    scaled_cols  = cfg.get("scaled_columns", [])
    scaling      = cfg.get("scaling_strategy", "standard")
    split_strat  = cfg.get("split_strategy", "random")
    split_sizes  = cfg.get("split_sizes", {"train": 1132, "val": 152, "test": 227})
    best_params  = cfg.get("best_params", {})
    n_trials     = 75
    for d in stages.get("tuning", {}).get("decisions", []):
        if d.get("decision") == "n_trials":
            n_trials = d.get("chosen", 75)

    train_n = split_sizes.get("train", 0)
    val_n   = split_sizes.get("val", 0)
    test_n  = split_sizes.get("test", 0)
    total   = train_n + val_n + test_n or 1
    train_r = round(train_n / total, 2)
    val_r   = round(val_n / total, 2)

    scaler_import = {
        "standard": "StandardScaler",
        "minmax":   "MinMaxScaler",
        "robust":   "RobustScaler",
    }.get(scaling, "StandardScaler")

    model_block = {
        "ridge":         f"from sklearn.linear_model import Ridge\nmodel = Ridge(**{best_params or {}})",
        "lasso":         f"from sklearn.linear_model import Lasso\nmodel = Lasso(**{best_params or {}})",
        "random_forest": f"from sklearn.ensemble import RandomForestRegressor\nmodel = RandomForestRegressor(**{best_params or {}}, random_state=42)",
        "xgboost":       f"import xgboost as xgb\nmodel = xgb.XGBRegressor(**{best_params or {}}, random_state=42)",
        "logistic":      f"from sklearn.linear_model import LogisticRegression\nmodel = LogisticRegression(**{best_params or {}}, random_state=42)",
        "random_forest_clf": f"from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(**{best_params or {}}, random_state=42)",
    }.get(model_id, f"from sklearn.linear_model import Ridge\nmodel = Ridge()")

    date_cols = [f for f in features if f.startswith("Date_") or "date" in f.lower()]
    has_date_feat = bool(date_cols)

    lines = [
        '"""',
        "Reproducible analysis script — generated by the Data Science Pipeline.",
        f"Task: {task_type}  |  Target: {target}  |  Model: {model_id}",
        '"""',
        "",
        "import pandas as pd",
        "import numpy as np",
        f"from sklearn.preprocessing import {scaler_import}",
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score",
        "import pickle",
        "",
        "# ── 1. Load data ──────────────────────────────────────────────────────────",
        'df = pd.read_csv("your_data.csv")',
        "print(f'Loaded {len(df)} rows, {len(df.columns)} columns')",
        "",
        "# ── 2. Clean ──────────────────────────────────────────────────────────────",
        "df = df.drop_duplicates()",
    ]

    if has_date_feat:
        # find the date column name
        date_col = next((c for c in ["Date", "date", "DATE"] if c in (cfg.get("feature_columns", []) or [])), "Date")
        # look in original columns from ingestion
        lines += [
            f'df["Date"] = pd.to_datetime(df["Date"])',
            "",
            "# ── 3. Feature engineering ────────────────────────────────────────────────",
            '# Expand date column into numeric features',
            'df["Date_year"]       = df["Date"].dt.year',
            'df["Date_month"]      = df["Date"].dt.month',
            'df["Date_day"]        = df["Date"].dt.day',
            'df["Date_dayofweek"]  = df["Date"].dt.dayofweek',
            'df["Date_quarter"]    = df["Date"].dt.quarter',
            'df["Date_is_weekend"] = df["Date"].dt.dayofweek.ge(5).astype(int)',
            'df["Date_month_sin"]  = np.sin(2 * np.pi * df["Date_month"] / 12)',
            'df["Date_month_cos"]  = np.cos(2 * np.pi * df["Date_month"] / 12)',
            'df["Date_dow_sin"]    = np.sin(2 * np.pi * df["Date_dayofweek"] / 7)',
            'df["Date_dow_cos"]    = np.cos(2 * np.pi * df["Date_dayofweek"] / 7)',
        ]
        if split_strat == "temporal":
            lines += [
                "df = df.sort_values('Date').reset_index(drop=True)",
            ]
    else:
        lines += [
            "",
            "# ── 3. Feature engineering ────────────────────────────────────────────────",
        ]

    lines += [
        "",
        f"FEATURES = {features}",
        f'TARGET   = "{target}"',
        "",
        "X = df[FEATURES]",
        "y = df[TARGET]",
        "",
        "# ── 4. Scale ──────────────────────────────────────────────────────────────",
        f"SCALE_COLS = {scaled_cols}",
        f"scaler = {scaler_import}()",
        "X = X.copy()",
        "X[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])",
        "",
        "# ── 5. Split ──────────────────────────────────────────────────────────────",
    ]

    if split_strat == "temporal":
        lines += [
            f"train_end = int(len(X) * {train_r})",
            f"val_end   = int(len(X) * {train_r + val_r})",
            "X_train, y_train = X.iloc[:train_end],  y.iloc[:train_end]",
            "X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]",
            "X_test,  y_test  = X.iloc[val_end:],   y.iloc[val_end:]",
        ]
    else:
        lines += [
            "from sklearn.model_selection import train_test_split",
            f"X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size={1 - train_r - val_r:.2f}, random_state=42)",
            f"X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size={val_r / (train_r + val_r):.2f}, random_state=42)",
        ]

    lines += [
        f"print(f'Train: {{len(X_train)}}, Val: {{len(X_val)}}, Test: {{len(X_test)}}')",
        "",
        "# ── 6. Train ──────────────────────────────────────────────────────────────",
        model_block,
        "model.fit(X_train, y_train)",
        "",
        "# ── 7. Evaluate ───────────────────────────────────────────────────────────",
        "y_pred = model.predict(X_val)",
        "print(f'Val R²:   {r2_score(y_val, y_pred):.4f}')",
        "print(f'Val RMSE: {mean_squared_error(y_val, y_pred, squared=False):.4f}')",
        "print(f'Val MAE:  {mean_absolute_error(y_val, y_pred):.4f}')",
        "",
        "# Final evaluation on held-out test set",
        "y_test_pred = model.predict(X_test)",
        "print(f'Test R²:   {r2_score(y_test, y_test_pred):.4f}')",
        "print(f'Test RMSE: {mean_squared_error(y_test, y_test_pred, squared=False):.4f}')",
        "",
        "# ── 8. Save model ─────────────────────────────────────────────────────────",
        "with open('model.pkl', 'wb') as f:",
        "    pickle.dump(model, f)",
        "with open('scaler.pkl', 'wb') as f:",
        "    pickle.dump(scaler, f)",
        "print('Model and scaler saved.')",
    ]

    return "\n".join(lines)


def _build_notebook(session: dict) -> dict:
    """Wrap the analysis script into a Jupyter notebook with per-section cells."""
    script = _build_analysis_script(session)
    cfg    = session.get("config", {})
    goal   = session.get("goal", {})
    target = goal.get("target_column", "target")
    model_id = cfg.get("model_id", "ridge")

    def md(text): return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }
    def code(src): return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }

    # Split script into sections by comment markers
    sections = {}
    current  = []
    current_key = "setup"
    for line in script.split("\n"):
        if line.startswith("# ──"):
            if current:
                sections[current_key] = "\n".join(current).strip()
            current_key = line.strip("# ─").strip().split(".")[1].strip() if "." in line else line.strip()
            current = []
        else:
            current.append(line)
    if current:
        sections[current_key] = "\n".join(current).strip()

    cells = [
        md(f"# Analysis Notebook\n\nGenerated by the Data Science Pipeline.\n\n**Target:** `{target}` | **Model:** `{model_id}`"),
    ]
    for title, src in sections.items():
        if src.strip():
            cells.append(md(f"## {title.title()}"))
            cells.append(code(src))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


@app.get("/sessions/{session_id}/code/script")
def download_analysis_script(session_id: str):
    """Download a reproducible Python analysis script for the session."""
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    script  = _build_analysis_script(session)
    return Response(
        content=script,
        media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="analysis_{session_id[:8]}.py"'},
    )


@app.get("/sessions/{session_id}/code/notebook")
def download_analysis_notebook(session_id: str):
    """Download a Jupyter notebook reproducing the full analysis."""
    session_dir = _sessions_root() / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    session  = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    notebook = _build_notebook(session)
    return Response(
        content=json.dumps(notebook, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="analysis_{session_id[:8]}.ipynb"'},
    )
