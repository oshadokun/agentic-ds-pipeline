"""
Ingestion Agent
Loads data from CSV, database, or API. Runs basic structural checks.
Detects sensitive columns and flags them for the privacy flow.
"""

import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Supported source loaders
# ---------------------------------------------------------------------------

def _ingest_csv(upload_path: str) -> pd.DataFrame:
    path = Path(upload_path)
    # Try common encodings
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                df = pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)
                if df.shape[1] > 1:
                    return df
            except Exception:
                continue
    # Last resort
    return pd.read_csv(path, low_memory=False)


def _ingest_database(query: str, conn_string: str) -> pd.DataFrame:
    from sqlalchemy import create_engine
    engine = create_engine(conn_string)
    df     = pd.read_sql(query, engine)
    engine.dispose()
    return df


def _ingest_api(url: str, headers: dict = None, params: dict = None) -> pd.DataFrame:
    import requests
    from io import StringIO

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type:
        data = response.json()
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    return pd.DataFrame(v)
    elif "csv" in content_type:
        return pd.read_csv(StringIO(response.text))

    raise ValueError(f"Unsupported content type from API: {content_type}")


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

def _structural_check(df: pd.DataFrame) -> dict:
    return {
        "row_count":       len(df),
        "column_count":    len(df.columns),
        "duplicate_rows":  int(df.duplicated().sum()),
        "empty_columns":   [c for c in df.columns if df[c].isna().all()],
        "all_null_rows":   int(df.isna().all(axis=1).sum()),
        "column_names":    df.columns.tolist(),
        "dtypes":          df.dtypes.astype(str).to_dict(),
        "missing_per_col": df.isna().sum().to_dict(),
        "sample_rows":     df.head(3).to_dict(orient="records")
    }


def _hard_stop_checks(structural: dict) -> list:
    stops = []
    if structural["row_count"] == 0:
        stops.append({
            "check":   "empty_data",
            "message": "Your file or query returned no data. Please check the source and try again.",
            "action":  "Fix the data source and re-upload."
        })
    elif structural["row_count"] < 50:
        stops.append({
            "check":   "too_few_rows",
            "message": f"Your dataset only has {structural['row_count']} rows. A minimum of 50 rows is needed.",
            "action":  "Add more data and try again."
        })
    if structural["column_count"] == 1:
        stops.append({
            "check":   "single_column",
            "message": "The data loaded with only one column. The delimiter may not be correct.",
            "action":  "Check your CSV uses commas (or semicolons/tabs) between columns."
        })
    return stops


# ---------------------------------------------------------------------------
# Sensitive column detection
# ---------------------------------------------------------------------------

SENSITIVE_PATTERNS = [
    "name", "email", "phone", "mobile", "address", "postcode", "zip",
    "dob", "birth", "age", "gender", "sex", "ethnicity", "race",
    "salary", "income", "wage", "account", "card", "ssn", "passport",
    "national", "ip_address", "device_id", "user_id", "customer_id",
    "patient_id", "employee_id"
]


def _detect_sensitive_columns(df: pd.DataFrame) -> list:
    sensitive = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        matched = next((p for p in SENSITIVE_PATTERNS if p in col_lower), None)
        if matched:
            sample = ""
            non_null = df[col].dropna()
            if not non_null.empty:
                val = str(non_null.iloc[0])
                sample = val[:2] + "****" if len(val) > 4 else "****"
            sensitive.append({
                "column": col,
                "reason": matched,
                "sample": sample
            })
    return sensitive


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    """
    Ingestion agent entry point.
    decisions may contain: {source_type, query, conn_string, api_url, api_headers, api_params}
    For CSV: reads from sessions/{id}/data/raw/upload.csv (uploaded via /data endpoint).
    """
    session_id   = session["session_id"]
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    source_type = decisions.get("source_type") or session["data_source"].get("type") or "csv"

    # --- Load data -----------------------------------------------------------
    try:
        if source_type == "csv":
            upload_path = session_dir / "data" / "raw" / "upload.csv"
            if not upload_path.exists():
                raise FileNotFoundError(
                    "No CSV file found. Please upload a file first via the data upload step."
                )
            df = _ingest_csv(str(upload_path))

        elif source_type == "database":
            conn_string = decisions.get("conn_string", "")
            query       = decisions.get("query", "SELECT * FROM data LIMIT 10000")
            if not conn_string:
                raise ValueError("A database connection string is required.")
            # Store credentials in session .env — never in session.json
            env_path = session_dir / ".env"
            env_path.write_text(f"DB_CONN_STRING={conn_string}\n")
            df = _ingest_database(query, conn_string)

        elif source_type == "api":
            api_url  = decisions.get("api_url", "")
            headers  = decisions.get("api_headers", {})
            params   = decisions.get("api_params", {})
            if not api_url:
                raise ValueError("An API URL is required.")
            df = _ingest_api(api_url, headers, params)

        else:
            raise ValueError(f"Unsupported source type: '{source_type}'.")

    except Exception as exc:
        return {
            "stage":                 "ingestion",
            "status":                "failed",
            "error":                 str(exc),
            "plain_english_summary": f"We could not load your data. {str(exc)}"
        }

    # --- Structural checks ---------------------------------------------------
    structural  = _structural_check(df)
    hard_stops  = _hard_stop_checks(structural)

    if hard_stops:
        return {
            "stage":                 "ingestion",
            "status":                "hard_stop",
            "hard_stops":            hard_stops,
            "plain_english_summary": hard_stops[0]["message"]
        }

    # --- Save ingested data --------------------------------------------------
    output_path = session_dir / "data" / "raw" / "ingested.csv"
    df.to_csv(output_path, index=False)

    # --- Sensitive column detection ------------------------------------------
    sensitive_cols = _detect_sensitive_columns(df)

    # Build plain-English summary
    row_count = structural["row_count"]
    col_count = structural["column_count"]
    summary   = f"We loaded your data successfully. It contains {row_count:,} rows and {col_count} columns."
    if sensitive_cols:
        summary += (
            f" We noticed {len(sensitive_cols)} column(s) that may contain personal information "
            f"— we will ask you how to handle these before continuing."
        )

    warnings = []
    if structural["duplicate_rows"] > 0:
        warnings.append(
            f"{structural['duplicate_rows']} duplicate rows found. These will be addressed during cleaning."
        )
    if structural["empty_columns"]:
        warnings.append(
            f"{len(structural['empty_columns'])} column(s) are completely empty: "
            f"{', '.join(structural['empty_columns'])}."
        )

    return {
        "stage":                "ingestion",
        "status":               "success",
        "output_data_path":     str(output_path),
        "structural_check":     structural,
        "sensitive_columns":    sensitive_cols,
        "decisions_required":   [],
        "decisions_made":       [],
        "warnings":             warnings,
        "errors":               [],
        "plain_english_summary": summary,
        "report_section": {
            "stage":                  "ingestion",
            "title":                  "Loading Your Data",
            "summary":                summary,
            "decision_made":          f"Loaded {row_count:,} rows and {col_count} columns from your {source_type} source.",
            "alternatives_considered": None,
            "why_this_matters": (
                "Starting with correctly loaded data is essential — if the data does not load cleanly, "
                "every step after this will be affected."
            )
        },
        "config_updates": {
            "data_source_type": source_type,
            "row_count":        row_count,
            "column_count":     col_count
        }
    }
