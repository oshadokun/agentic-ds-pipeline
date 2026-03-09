---
name: ingestion
description: >
  Responsible for loading data into the pipeline from any supported source — CSV file
  upload, database connection, or API endpoint. This skill is always the first agent
  called by the Orchestrator after goal capture. It handles all source-specific
  connection logic, loads data into a standardised format, logs what it loaded, detects
  sensitive columns, and writes the result to the session data directory. Trigger when
  any of the following are mentioned: "upload data", "connect to database", "load from
  API", "import data", "data source", "CSV", "SQL", "database connection", or any
  request to bring data into the pipeline for the first time.
---

# Ingestion Skill

The Ingestion agent is the entry point for all data. Its job is to connect to the
user's data source, load the data reliably, understand its basic shape, flag any
obvious issues, and write it to the session directory in a standardised format.

It does not clean, transform, or make decisions about the data. It loads and reports.

---

## Responsibilities

1. Connect to the user's data source (CSV, database, or API)
2. Load the data into a pandas DataFrame
3. Log what was loaded — row count, column count, file size
4. Detect potentially sensitive columns and flag them for the Orchestrator
5. Perform a basic structural check — is there anything obviously wrong?
6. Write the loaded data to `sessions/{session_id}/data/raw/ingested.csv`
7. Return a structured result to the Orchestrator

---

## Supported Sources

### CSV Upload
The most common case. The user uploads a file through the web UI.

```python
import pandas as pd
from pathlib import Path

def ingest_csv(upload_path: str, session_id: str) -> pd.DataFrame:
    path = Path(upload_path)

    # Detect encoding safely
    import chardet
    with open(path, "rb") as f:
        encoding = chardet.detect(f.read())["encoding"] or "utf-8"

    # Try comma first, fall back to other delimiters
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, encoding=encoding, sep=sep, low_memory=False)
            if df.shape[1] > 1:
                break
        except Exception:
            continue

    return df
```

**What to tell the user:**
"We've loaded your file. It contains {n} rows and {m} columns."

### Database Connection
The user provides a connection string. Credentials are stored in `.env` only — never
in `session.json` or any log.

```python
from sqlalchemy import create_engine
import pandas as pd

def ingest_database(query: str, conn_string: str) -> pd.DataFrame:
    engine = create_engine(conn_string)
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df
```

**Connection string is:**
- Received from the UI
- Written immediately to `sessions/{session_id}/.env`
- Never passed as plain text beyond this function
- Never written to session.json or any log file

**What to tell the user:**
"We've connected to your database and loaded {n} rows and {m} columns."

### API Endpoint
The user provides a URL and optionally authentication headers.

```python
import requests
import pandas as pd

def ingest_api(url: str, headers: dict = None, params: dict = None) -> pd.DataFrame:
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type:
        data = response.json()
        # Handle nested JSON — take the first list found
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    df = pd.DataFrame(v)
                    break
    elif "csv" in content_type:
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))

    return df
```

**What to tell the user:**
"We've connected to the API and received {n} rows and {m} columns."

---

## Basic Structural Checks

After loading, always run these checks and include results in the output:

```python
def structural_check(df: pd.DataFrame) -> dict:
    return {
        "row_count":        len(df),
        "column_count":     len(df.columns),
        "duplicate_rows":   int(df.duplicated().sum()),
        "empty_columns":    [c for c in df.columns if df[c].isna().all()],
        "all_null_rows":    int(df.isna().all(axis=1).sum()),
        "mixed_type_cols":  [c for c in df.columns if df[c].dtype == object
                             and df[c].str.match(r"^\d+\.?\d*$").any()],
        "column_names":     df.columns.tolist(),
        "dtypes":           df.dtypes.astype(str).to_dict(),
        "missing_per_col":  df.isna().sum().to_dict(),
        "sample_rows":      df.head(3).to_dict(orient="records")
    }
```

**Hard stop conditions — do not proceed if:**
- `row_count` < 50 → tell user the dataset may be too small
- `row_count` == 0 → tell user the file or query returned no data
- `column_count` == 1 → warn user the data may not have loaded correctly, check delimiter

---

## Sensitive Column Detection

After loading, scan column names and sample values for indicators of sensitive data.
Flag any matches and return them to the Orchestrator — do not make decisions about
them here.

```python
SENSITIVE_PATTERNS = [
    "name", "email", "phone", "mobile", "address", "postcode", "zip",
    "dob", "birth", "age", "gender", "sex", "ethnicity", "race",
    "salary", "income", "wage", "account", "card", "ssn", "passport",
    "national", "ip_address", "device_id", "user_id", "customer_id"
]

def detect_sensitive_columns(df: pd.DataFrame) -> list:
    sensitive = []
    for col in df.columns:
        col_lower = col.lower().replace(" ", "_")
        if any(pattern in col_lower for pattern in SENSITIVE_PATTERNS):
            sensitive.append({
                "column": col,
                "reason": next(p for p in SENSITIVE_PATTERNS if p in col_lower),
                "sample": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "empty"
            })
    return sensitive
```

The Orchestrator will present these to the user and ask how they want to handle them
before the pipeline continues. The Ingestion agent does not make that decision.

---

## Output Written to Session

**Data file:**
`sessions/{session_id}/data/raw/ingested.csv`

Written with:
```python
df.to_csv(output_path, index=False)
```

**Result JSON:**
`sessions/{session_id}/outputs/ingestion/result.json`

```json
{
  "stage": "ingestion",
  "status": "success",
  "output_data_path": "sessions/{session_id}/data/raw/ingested.csv",
  "structural_check": { ... },
  "sensitive_columns": [ ... ],
  "decisions_required": [],
  "warnings": [ ... ],
  "errors": [],
  "plain_english_summary": "We loaded your data successfully. It contains 1,204 rows and 18 columns. We noticed 3 columns that may contain personal information — we will ask you how you want to handle these before continuing.",
  "report_section": {
    "stage": "ingestion",
    "title": "Loading Your Data",
    "summary": "...",
    "decision_made": "...",
    "alternatives_considered": null,
    "why_this_matters": "Starting with correctly loaded data is essential — if the data does not load cleanly, every step after this will be affected."
  },
  "config_updates": {
    "data_source_type": "csv",
    "row_count": 1204,
    "column_count": 18
  }
}
```

---

## What to Tell the User

After ingestion completes the Orchestrator presents the user with:

1. Confirmation the data loaded successfully
2. How many rows and columns were found
3. A preview of the first few rows (shown in the UI as a table)
4. Any warnings about structural issues
5. If sensitive columns were found — what they are and the three options:
   - **Mask them** — replace with anonymised values
   - **Drop them** — remove the column entirely
   - **Keep them with acknowledgement** — proceed knowing they are present

The user must make a decision on sensitive columns before the pipeline advances.

---

## Reference Files

- `references/supported-formats.md` — full list of supported file types and delimiters
- `references/database-connectors.md` — supported databases and connection string formats
