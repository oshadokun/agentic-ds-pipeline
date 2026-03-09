---
name: privacy
description: >
  Responsible for enforcing data privacy rules across the entire pipeline. Called
  by the Orchestrator at key checkpoints — after ingestion, before any external
  call, before report generation, and before session export. Identifies sensitive
  columns, applies user-chosen handling strategies (mask, anonymise, drop), ensures
  credentials are never logged, validates that no data leaves the local session
  directory without user awareness, and produces a privacy audit trail. Privacy is
  a first-class requirement — no stage proceeds past ingestion until sensitive
  columns are acknowledged. Trigger when any of the following are mentioned:
  "privacy", "sensitive data", "personal data", "GDPR", "anonymise", "mask data",
  "credentials", "data security", "PII", "personally identifiable information",
  "data handling", or any situation where user data may be at risk.
---

# Privacy Skill

The Privacy agent enforces data handling rules across the entire pipeline. It is
not a one-time check — it is a continuous presence that ensures the user's data
is handled responsibly at every stage.

Privacy is built in from the start. It is not an afterthought.

The core principle: **the user's data stays under their control at all times.**
Nothing leaves the local session directory without explicit user knowledge.
Credentials are never stored in plain text. Sensitive columns are never processed
without the user's awareness and explicit decision.

---

## Responsibilities

1. Identify potentially sensitive columns after ingestion
2. Present sensitive column findings to the user and collect their decisions
3. Apply chosen handling strategies — mask, anonymise, pseudonymise, or drop
4. Validate that credentials are stored securely in `.env` only
5. Intercept and warn before any operation that sends data externally
6. Scrub sensitive columns from reports and logs
7. Produce a privacy audit trail — every decision recorded
8. Enforce data minimisation — only the columns needed are kept
9. Provide clear data deletion capability

---

## Sensitive Column Detection

```python
import pandas as pd
import re
from pathlib import Path

# Column name patterns that suggest sensitive data
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

# Value patterns that suggest sensitive data regardless of column name
SENSITIVE_VALUE_PATTERNS = {
    "email":   r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
    "phone":   r"^[\+]?[(]?[0-9]{3}[)]?[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}$",
    "postcode_uk": r"^[A-Z]{1,2}[0-9][0-9A-Z]?\s?[0-9][ABD-HJLNP-UW-Z]{2}$",
    "ip_address":  r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$",
    "credit_card": r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"
}


def scan_for_sensitive_columns(df):
    """
    Scan all columns for potentially sensitive data.
    Returns a list of findings with category and recommended action.
    """
    findings = []

    for col in df.columns:
        col_lower = col.lower().replace(" ", "_").replace("-", "_")
        matched_category = None
        matched_pattern  = None

        # Check column name against patterns
        for category, patterns in SENSITIVE_NAME_PATTERNS.items():
            for pattern in patterns:
                if pattern in col_lower:
                    matched_category = category
                    matched_pattern  = pattern
                    break
            if matched_category:
                break

        # If not matched by name, check sample values
        if not matched_category and df[col].dtype == object:
            sample = df[col].dropna().head(50).astype(str)
            for value_type, regex in SENSITIVE_VALUE_PATTERNS.items():
                match_rate = sample.str.match(regex, na=False).mean()
                if match_rate > 0.5:
                    matched_category = "value_pattern"
                    matched_pattern  = value_type
                    break

        if matched_category:
            sample_value = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "empty"
            # Partially redact sample value for display
            display_sample = _redact_sample(sample_value)

            findings.append({
                "column":           col,
                "category":         matched_category,
                "matched_pattern":  matched_pattern,
                "sample_value":     display_sample,
                "recommended_action": _recommend_action(matched_category),
                "plain_english": _describe_sensitivity(col, matched_category,
                                                        matched_pattern)
            })

    return findings


def _redact_sample(value):
    """Partially redact a sample value for safe display."""
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def _recommend_action(category):
    if category == "direct_identifiers":
        return "drop_or_pseudonymise"
    elif category == "quasi_identifiers":
        return "generalise_or_keep_with_acknowledgement"
    elif category == "financial":
        return "drop_or_mask"
    elif category == "technical_identifiers":
        return "drop"
    elif category == "location":
        return "generalise_or_drop"
    elif category == "value_pattern":
        return "drop_or_mask"
    return "keep_with_acknowledgement"


def _describe_sensitivity(col, category, pattern):
    descriptions = {
        "direct_identifiers":    f"'{col}' appears to contain direct identifiers (e.g. names, email addresses, phone numbers) that can identify a person directly.",
        "quasi_identifiers":     f"'{col}' appears to contain quasi-identifiers (e.g. age, gender, location) that could identify someone when combined with other data.",
        "financial":             f"'{col}' appears to contain financial information which is sensitive and regulated in most jurisdictions.",
        "technical_identifiers": f"'{col}' appears to contain a technical identifier (e.g. user ID, device ID) that could be used to track individuals.",
        "location":              f"'{col}' appears to contain precise location data which can be used to identify individuals.",
        "value_pattern":         f"'{col}' contains values that look like {pattern} — a type of sensitive data."
    }
    return descriptions.get(category, f"'{col}' may contain sensitive information.")
```

---

## Handling Strategies

```python
import hashlib
import numpy as np

HANDLING_OPTIONS = {
    "drop": {
        "label": "Remove this column entirely",
        "tradeoff": "Eliminates all privacy risk from this column. The model will not have access to this information.",
        "reversible": False
    },
    "pseudonymise": {
        "label": "Replace with a consistent anonymous code",
        "tradeoff": "Replaces names/IDs with a code. The same person always gets the same code, so relationships are preserved — but the original value cannot be recovered.",
        "reversible": False
    },
    "mask": {
        "label": "Replace with a generic placeholder",
        "tradeoff": "Replaces all values with a fixed label like [REDACTED]. Simplest option — eliminates the data entirely.",
        "reversible": False
    },
    "generalise": {
        "label": "Replace with a broader category",
        "tradeoff": "For example, replace exact age with age band (20-30, 30-40). Reduces precision while preserving some analytical value.",
        "reversible": False
    },
    "keep_with_acknowledgement": {
        "label": "Keep as-is — I understand this column contains sensitive data",
        "tradeoff": "The column is kept and used by the model. You take responsibility for ensuring this is appropriate for your use case.",
        "reversible": True
    }
}


def apply_privacy_handling(df, col, strategy):
    """Apply the user's chosen handling strategy to a column."""

    if strategy == "drop":
        df = df.drop(columns=[col])
        return df, f"Removed '{col}' from the dataset."

    elif strategy == "pseudonymise":
        # Consistent hash — same input always gives same output
        df[col] = df[col].astype(str).apply(
            lambda x: "ID_" + hashlib.sha256(x.encode()).hexdigest()[:8]
        )
        return df, f"Replaced '{col}' with anonymous codes. Original values cannot be recovered."

    elif strategy == "mask":
        df[col] = "[REDACTED]"
        return df, f"Masked all values in '{col}' with [REDACTED]."

    elif strategy == "generalise" and df[col].dtype in ["int64", "float64"]:
        # Generalise numeric to bands
        min_val = df[col].min()
        max_val = df[col].max()
        n_bands = min(10, int((max_val - min_val) / 5) + 1)
        df[col] = pd.cut(df[col], bins=n_bands).astype(str)
        return df, f"Replaced precise values in '{col}' with bands (ranges)."

    elif strategy == "keep_with_acknowledgement":
        return df, f"Kept '{col}' as-is. User has acknowledged this column contains sensitive data."

    return df, f"No action taken on '{col}'."
```

---

## Credential Security

```python
def validate_credential_security(session_id):
    """
    Ensure credentials are only in .env — never in session.json or logs.
    Called after ingestion whenever a database or API source was used.
    """
    issues = []
    session_dir = Path(f"sessions/{session_id}")

    # Check session.json for credential-like strings
    session_file = session_dir / "session.json"
    if session_file.exists():
        content = session_file.read_text().lower()
        credential_keywords = [
            "password", "passwd", "secret", "api_key", "apikey",
            "token", "auth", "connection_string", "jdbc"
        ]
        for keyword in credential_keywords:
            if keyword in content:
                issues.append(
                    f"session.json may contain a credential-like value "
                    f"('{keyword}' found). Please review and remove."
                )

    # Check that .env exists if database/API was used
    env_file = session_dir / ".env"
    data_source_type = _get_data_source_type(session_id)
    if data_source_type in ["database", "api"] and not env_file.exists():
        issues.append(
            ".env file is missing. Database or API credentials should "
            "be stored here."
        )

    # Check .env is not accidentally committed (warn if .gitignore missing)
    gitignore = Path(".gitignore")
    if gitignore.exists():
        content = gitignore.read_text()
        if ".env" not in content and "sessions/" not in content:
            issues.append(
                ".gitignore does not exclude .env or sessions/ directory. "
                "If you are using version control, add these to .gitignore "
                "to prevent credentials from being committed."
            )

    return issues


def _get_data_source_type(session_id):
    try:
        with open(f"sessions/{session_id}/session.json") as f:
            session = json.load(f)
        return session.get("data_source", {}).get("type")
    except Exception:
        return None
```

---

## External Data Transmission Warning

```python
EXTERNAL_SERVICES = [
    "openai", "anthropic", "huggingface", "cohere",
    "aws", "azure", "gcp", "google", "s3"
]

def check_before_external_call(service_name, data_description,
                                contains_user_data=True):
    """
    Warn the user before any operation that sends data to an external service.
    Must be called and confirmed before any external API call is made.
    """
    if not contains_user_data:
        return {"approved": True, "warning": None}

    warning = {
        "service":          service_name,
        "data_description": data_description,
        "plain_english": (
            f"The next step would send your data to {service_name}. "
            f"Specifically: {data_description}. "
            f"This data would leave your local machine and be processed "
            f"by {service_name}'s servers. "
            f"Do you want to proceed?"
        ),
        "requires_confirmation": True
    }
    return {"approved": False, "warning": warning}
```

---

## Report Scrubbing

```python
def scrub_sensitive_from_report(report_text, sensitive_columns,
                                  session):
    """
    Remove sensitive column names and values from the final report.
    Replaces column names with [SENSITIVE COLUMN] in text.
    """
    scrubbed = report_text
    actions  = session.get("privacy", {}).get("sensitive_columns_action", {})

    for col, action in actions.items():
        if action in ["drop", "mask", "pseudonymise"]:
            # Replace exact column name in report text
            scrubbed = scrubbed.replace(f"'{col}'", f"[protected column]")
            scrubbed = scrubbed.replace(col, f"[protected column]")

    return scrubbed
```

---

## Privacy Audit Trail

Every privacy decision is recorded in the session.

```python
import json
from datetime import datetime

def record_privacy_decision(session, col, finding,
                              user_decision, reason=None):
    """
    Append a privacy decision to the audit trail in session.json.
    """
    if "privacy_audit" not in session:
        session["privacy_audit"] = []

    session["privacy_audit"].append({
        "timestamp":     datetime.now().isoformat(),
        "column":        col,
        "category":      finding.get("category"),
        "finding":       finding.get("plain_english"),
        "decision":      user_decision,
        "reason":        reason or "User decision",
        "reversible":    HANDLING_OPTIONS[user_decision]["reversible"]
    })

    session["privacy"]["sensitive_columns_action"][col] = user_decision

    all_acknowledged = all(
        col in session["privacy"]["sensitive_columns_action"]
        for col in [f["column"] for f in
                    session["privacy"].get("sensitive_columns_identified", [])]
    )
    session["privacy"]["user_acknowledged"] = all_acknowledged

    return session
```

---

## Running the Privacy Pipeline

```python
def run_privacy_check(df, session, session_id, checkpoint):
    """
    checkpoint: "post_ingestion" | "pre_report" | "pre_export"
    """
    results = {
        "checkpoint":  checkpoint,
        "issues":      [],
        "decisions_required": []
    }

    if checkpoint == "post_ingestion":
        # 1. Scan for sensitive columns
        findings = scan_for_sensitive_columns(df)

        if findings:
            results["decisions_required"] = [
                {
                    "column":    f["column"],
                    "finding":   f["plain_english"],
                    "recommended": f["recommended_action"],
                    "options":   [
                        {"id": k, **v}
                        for k, v in HANDLING_OPTIONS.items()
                    ]
                }
                for f in findings
            ]
            session["privacy"]["sensitive_columns_identified"] = [
                f["column"] for f in findings
            ]

        # 2. Validate credential security
        credential_issues = validate_credential_security(session_id)
        results["issues"].extend(credential_issues)

    elif checkpoint == "pre_report":
        # Ensure sensitive columns are scrubbed from report content
        results["scrub_required"] = len(
            session["privacy"].get("sensitive_columns_action", {})
        ) > 0

    elif checkpoint == "pre_export":
        # Confirm user is aware data is being exported
        results["issues"].append({
            "type": "export_warning",
            "plain_english": (
                "You are about to export session data. Please ensure you "
                "are complying with any applicable data protection regulations "
                "before sharing or moving this data."
            )
        })

    return results, session
```

---

## Output Written to Session

**Privacy decisions recorded in:**
`sessions/{session_id}/session.json` → `privacy` and `privacy_audit` fields

**Result JSON:**
`sessions/{session_id}/outputs/privacy/result.json`

```json
{
  "checkpoint": "post_ingestion",
  "sensitive_columns_found": 3,
  "decisions_made": {
    "email": "drop",
    "customer_name": "pseudonymise",
    "age": "keep_with_acknowledgement"
  },
  "credential_issues": [],
  "plain_english_summary": "We found 3 columns that may contain sensitive information. You chose to remove the email column, anonymise customer names, and keep age. These decisions have been recorded.",
  "report_section": {
    "stage": "privacy",
    "title": "Data Privacy Decisions",
    "summary": "...",
    "decision_made": "...",
    "why_this_matters": "Handling sensitive data responsibly protects the people whose data you are working with, and protects you from legal and reputational risk."
  }
}
```

---

## What to Tell the User

When sensitive columns are found:
"Before we continue, we need to discuss some of the columns in your data.
We found {n} columns that may contain sensitive or personal information.
For each one, we have a recommendation — but you make the final decision.

Nothing will proceed until you have decided how to handle each of these."

After decisions are made:
"Thank you. Here is a summary of what we will do:
{list each column and its chosen action}

These decisions have been recorded. If you need to change them, you can
do so before the pipeline runs."

Credential warning:
"We noticed your connection credentials. These are stored securely in a
separate file (.env) and will never appear in your session data, logs,
or reports."

---

## Reference Files

- `references/privacy-regulations.md` — overview of GDPR, UK DPA, CCPA and what they mean in practice
