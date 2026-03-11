"""
Cleaning Agent
Fixes data quality issues found during Validation and EDA.
Every significant decision is surfaced to the user before being applied.
"""

from pathlib import Path

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Cleaning modules  (from cleaning SKILL)
# ---------------------------------------------------------------------------

def _remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before  = len(df)
    df      = df.drop_duplicates()
    removed = before - len(df)
    return df, {
        "action":           "remove_duplicates",
        "rows_removed":     removed,
        "rows_remaining":   len(df),
        "plain_english":    f"We removed {removed} duplicate rows. Your dataset now has {len(df):,} rows."
    }


_DATE_NAME_PATTERNS = {"date", "time", "timestamp", "datetime", "dt", "period", "week", "quarter"}
_DATE_VALUE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}",          # YYYY-MM-DD (also catches datetimes)
    r"^\d{1,2}/\d{1,2}/\d{4}",      # M/D/YYYY
    r"^\d{1,2}-\d{1,2}-\d{4}",      # D-M-YYYY
    r"^\d{4}/\d{2}/\d{2}",          # YYYY/MM/DD
    r"^\d{1,2}\s+\w+\s+\d{4}",      # 1 Jan 2022
]


def _is_date_col(col: str, sample) -> bool:
    col_norm = col.lower().replace(" ", "_").replace("-", "_")
    if any(pat in col_norm for pat in _DATE_NAME_PATTERNS):
        return True
    return any(sample.str.match(p).mean() > 0.7 for p in _DATE_VALUE_PATTERNS)


def _fix_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    actions = []

    for col in df.columns:
        if df[col].dtype != object:
            continue

        sample = df[col].dropna().astype(str).str.strip()
        if sample.empty:
            continue

        # 1. Currency: optional leading $, commas as thousand-separators
        is_currency = sample.str.startswith("$").mean() > 0.5
        stripped = sample.str.lstrip("$").str.replace(",", "", regex=False).str.strip()
        converted = pd.to_numeric(stripped, errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.strip().str.lstrip("$")
                       .str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            label = "currency text (removed '$' and commas)" if is_currency else "text with commas"
            actions.append(f"Converted '{col}' from {label} to number.")
            continue

        # 2. Percentage: trailing %
        if sample.str.endswith("%").mean() > 0.8:
            stripped_pct = sample.str.rstrip("%").str.strip()
            converted_pct = pd.to_numeric(stripped_pct, errors="coerce")
            if converted_pct.notna().mean() > 0.9:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.strip().str.rstrip("%").str.strip(),
                    errors="coerce"
                ) / 100
                actions.append(f"Converted '{col}' from percentage text to decimal number.")
                continue

        # 3. Dates: check by name and value patterns then parse
        if _is_date_col(col, sample):
            try:
                parsed = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
                    actions.append(f"Converted '{col}' to a date column.")
                    continue
            except Exception:
                pass

        # 4. Plain numeric fallback
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() > 0.9:
            df[col] = converted
            actions.append(f"Converted '{col}' from text to number.")

    return df, actions


def _standardise_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    actions = []
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()
        before  = df[col].nunique()
        df[col] = df[col].str.lower()
        after   = df[col].nunique()
        if after < before:
            actions.append(
                f"Standardised '{col}' — merged {before - after} variants caused by capitalisation differences."
            )
    return df, actions


def _apply_imputation(df: pd.DataFrame, col: str, strategy: str) -> tuple[pd.DataFrame, str]:
    if col not in df.columns:
        return df, f"Column '{col}' not found — skipped."

    if strategy == "drop_rows":
        before = len(df)
        df     = df.dropna(subset=[col])
        return df, f"Removed {before - len(df)} rows where '{col}' was missing."

    elif strategy == "drop_col":
        df = df.drop(columns=[col])
        return df, f"Removed column '{col}' — it was too empty to be useful."

    elif strategy == "median":
        fill = df[col].median()
        df[col] = df[col].fillna(fill)
        return df, f"Filled missing values in '{col}' with the middle value ({fill:.2f})."

    elif strategy == "mean":
        fill = df[col].mean()
        df[col] = df[col].fillna(fill)
        return df, f"Filled missing values in '{col}' with the average ({fill:.2f})."

    elif strategy == "mode":
        fill = df[col].mode()
        if fill.empty:
            return df, f"No mode found for '{col}' — skipped."
        df[col] = df[col].fillna(fill.iloc[0])
        return df, f"Filled missing values in '{col}' with the most common value ('{fill.iloc[0]}')."

    elif strategy == "knn":
        from sklearn.impute import KNNImputer
        numeric_cols = df.select_dtypes("number").columns.tolist()
        if col in numeric_cols:
            imputer = KNNImputer(n_neighbors=5)
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        return df, f"Filled missing values in '{col}' using patterns from similar rows (KNN)."

    return df, f"No imputation applied to '{col}'."


def _handle_outliers(df: pd.DataFrame, col: str, strategy: str) -> tuple[pd.DataFrame, str]:
    if col not in df.columns or df[col].dtype not in ["int64", "float64"]:
        return df, f"Column '{col}' not found or not numeric — skipped."

    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    lower  = Q1 - 1.5 * IQR
    upper  = Q3 + 1.5 * IQR
    mask   = (df[col] < lower) | (df[col] > upper)
    count  = int(mask.sum())

    if strategy == "cap":
        df[col] = df[col].clip(lower=lower, upper=upper)
        return df, f"Capped extreme values in '{col}'"

    elif strategy == "remove":
        before = len(df)
        df     = df[~mask]
        return df, f"Removed rows with extreme values in '{col}'"

    elif strategy == "keep":
        return df, f"Kept extreme values in '{col}' as-is"

    return df, f"No outlier handling applied to '{col}'."


def _recommend_impute(col: str, df: pd.DataFrame, missing_pct: float) -> tuple[str, list]:
    dtype = df[col].dtype

    if missing_pct > 0.8:
        recommended = "drop_col"
    elif missing_pct < 0.05:
        recommended = "drop_rows"
    elif dtype in ["int64", "float64"]:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        has_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).any()
        recommended  = "median" if has_outliers else "mean"
    else:
        recommended = "mode"

    STRATEGIES = {
        "median":    {"label": "Fill with the middle value (recommended for numbers with outliers)",           "tradeoff": "Safe and reliable. Not thrown off by extreme values."},
        "mean":      {"label": "Fill with the average value",                                                  "tradeoff": "Simple but can be distorted by very high or very low values."},
        "knn":       {"label": "Fill using similar rows (most accurate, slower)",                              "tradeoff": "Best quality but takes longer on large datasets."},
        "mode":      {"label": "Fill with the most common value (for categories)",                             "tradeoff": "Standard approach for text columns."},
        "drop_rows": {"label": "Remove rows with missing values",                                              "tradeoff": "Cleanest option but you lose data. Only recommended if fewer than 5% of rows are affected."},
        "drop_col":  {"label": "Remove this column entirely",                                                  "tradeoff": "Appropriate when a column is more than 80% empty."}
    }

    if dtype in ["int64", "float64"]:
        options = ["median", "mean", "knn", "drop_rows", "drop_col"]
    else:
        options = ["mode", "drop_rows", "drop_col"]

    alternatives = [{"id": s, **STRATEGIES[s]} for s in options]
    return recommended, alternatives


# ---------------------------------------------------------------------------
# Determine what decisions are needed before running
# ---------------------------------------------------------------------------

def _build_decisions_required(df: pd.DataFrame, target_col: str,
                                eda_findings: dict, task_type: str = "") -> list:
    decisions = []
    missing   = df.isna().mean()

    for col, pct in missing[missing > 0].items():
        if col == target_col:
            continue
        if pct >= 0.05:   # significant — ask user
            recommended, alternatives = _recommend_impute(col, df, float(pct))
            decisions.append({
                "id":                   f"missing_{col}",
                "type":                 "missing_value",
                "column":               col,
                "missing_pct":          round(float(pct), 4),
                "question":             (
                    f"'{col}' has {pct:.1%} missing values. "
                    f"How would you like to handle them?"
                ),
                "recommendation":       recommended,
                "recommendation_reason": (
                    f"Based on the data type and distribution, '{recommended}' is the most appropriate strategy."
                ),
                "alternatives":         alternatives
            })

    # Outlier decisions — skip columns already decided in the EDA stage
    outlier_cols            = eda_findings.get("high_outlier_cols", [])
    eda_outlier_strategy    = eda_findings.get("outlier_strategy", {})
    undecided_outlier_cols  = [c for c in outlier_cols if c not in eda_outlier_strategy]

    for col in undecided_outlier_cols:
        if col not in df.columns or df[col].dtype not in ["int64", "float64"]:
            continue
        Q1, Q3  = df[col].quantile([0.25, 0.75])
        IQR     = Q3 - Q1
        out_pct = float(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).mean())
        if out_pct > 0.01:
            decisions.append({
                "id":                   f"outlier_{col}",
                "type":                 "outlier",
                "column":               col,
                "outlier_pct":          round(out_pct, 4),
                "question":             (
                    f"'{col}' has {out_pct:.1%} extreme values. How would you like to handle them?"
                ),
                "recommendation":       "cap",
                "recommendation_reason": "Capping keeps all rows while limiting extreme values.",
                "alternatives": [
                    {"id": "cap",    "label": "Cap at the boundary values (recommended)", "tradeoff": "Keeps all rows but limits extreme values."},
                    {"id": "remove", "label": "Remove rows with extreme values",          "tradeoff": "Cleaner data but you lose rows."},
                    {"id": "keep",   "label": "Keep as-is",                               "tradeoff": "No data lost but extreme values may affect the model."}
                ]
            })

    # Class imbalance — classification only
    if "classification" in task_type and target_col and target_col in df.columns:
        vc          = df[target_col].value_counts(normalize=True)
        minority_pct = float(vc.min())
        majority_pct = float(vc.max())
        if minority_pct < 0.30:
            decisions.append({
                "id":    "balance_classes",
                "type":  "class_imbalance",
                "column": target_col,
                "minority_pct": round(minority_pct, 4),
                "majority_pct": round(majority_pct, 4),
                "question": (
                    f"Your data has imbalanced classes — {minority_pct:.0%} vs {majority_pct:.0%}. "
                    f"Would you like us to balance them?"
                ),
                "recommendation": "smote",
                "recommendation_reason": (
                    "When one outcome is much rarer than another, the model tends to ignore it. "
                    "Balancing creates synthetic examples of the minority class so the model learns both equally."
                ),
                "alternatives": [
                    {
                        "id":       "smote",
                        "label":    "Yes — balance my data (Recommended)",
                        "tradeoff": "Adds synthetic rows to the minority class. May slightly reduce overall accuracy but improves detection of the rarer outcome."
                    },
                    {
                        "id":       "none",
                        "label":    "No — leave as-is",
                        "tradeoff": "The model may struggle to predict the rarer outcome correctly."
                    }
                ]
            })

    # Duplicate confirmation
    dup_pct = float(df.duplicated().mean())
    if dup_pct > 0.1:
        decisions.append({
            "id":                   "handle_duplicates",
            "type":                 "duplicates",
            "column":               None,
            "dup_pct":              round(dup_pct, 4),
            "question":             f"{dup_pct:.1%} of your rows are duplicates. Shall we remove them?",
            "recommendation":       "remove",
            "recommendation_reason": "Duplicate rows add no information and can bias the model.",
            "alternatives": [
                {"id": "remove", "label": "Remove duplicates (recommended)", "tradeoff": "Fewer rows, cleaner data."},
                {"id": "keep",   "label": "Keep duplicates",                 "tradeoff": "No data lost but results may be biased."}
            ]
        })

    return decisions


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id   = session["session_id"]
    target_col   = session["goal"].get("target_column")
    task_type    = session["goal"].get("task_type", "")
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    data_path = session_dir / "data" / "raw" / "ingested.csv"
    if not data_path.exists():
        return {
            "stage":                 "cleaning",
            "status":                "failed",
            "plain_english_summary": "No ingested data found. Please run ingestion first."
        }

    df = pd.read_csv(data_path, low_memory=False)

    # Get EDA findings for outlier / correlation context
    eda_result_path = session_dir / "outputs" / "eda" / "result.json"
    eda_findings    = {}
    if eda_result_path.exists():
        import json
        with open(eda_result_path) as f:
            eda_data = json.load(f)
        eda_findings = eda_data.get("config_updates", {})

    # If the frontend is only asking for decisions_required (no user decisions yet)
    if not decisions or decisions.get("phase") == "request_decisions":
        dr = _build_decisions_required(df, target_col, eda_findings, task_type)
        if dr:
            return {
                "stage":              "cleaning",
                "status":             "decisions_required",
                "decisions_required": dr,
                "plain_english_summary": (
                    "Here is what we found that needs attention. For each item, we have a recommendation "
                    "— but you can choose a different approach if you prefer."
                )
            }
        # No decisions needed — fall through to apply automatic fixes

    # Apply cleaning based on user decisions
    rows_before = len(df)
    cols_before = len(df.columns)
    actions_log = []

    # 1. Duplicates
    dup_decision = decisions.get("handle_duplicates", "remove")
    if dup_decision == "remove":
        df, dup_result = _remove_duplicates(df)
        actions_log.append(dup_result["plain_english"])

    # 2. Data type fixes (automatic)
    df, dtype_actions = _fix_dtypes(df)
    actions_log.extend(dtype_actions)

    # 3. Text standardisation (automatic)
    df, text_actions = _standardise_categoricals(df)
    actions_log.extend(text_actions)

    # 4. Missing values — user decisions
    missing_decisions = {
        k.replace("missing_", ""): v
        for k, v in decisions.items()
        if k.startswith("missing_")
    }
    # Also handle columns with < 5% missing automatically
    for col in df.columns:
        if col == target_col:
            continue
        pct = float(df[col].isna().mean())
        if 0 < pct < 0.05 and col not in missing_decisions:
            rec, _ = _recommend_impute(col, df, pct)
            missing_decisions[col] = rec

    for col, strategy in missing_decisions.items():
        if col in df.columns:
            df, msg = _apply_imputation(df, col, strategy)
            actions_log.append(msg)

    # 5. Outliers — start from EDA decisions, overlay any cleaning-stage decisions
    outlier_decisions = dict(eda_findings.get("outlier_strategy", {}))
    # Expand grouped EDA decision: outliers__grouped → per-column strategy
    if "outliers__grouped" in decisions:
        grouped_strategy = decisions["outliers__grouped"]
        for col in eda_findings.get("high_outlier_cols", []):
            outlier_decisions.setdefault(col, grouped_strategy)
    # Overlay per-column decisions confirmed during cleaning
    for k, v in decisions.items():
        if k.startswith("outlier_"):
            outlier_decisions[k.replace("outlier_", "", 1)] = v
    # Auto-cap minor outliers (< 1%)
    for col in df.select_dtypes("number").columns:
        if col == target_col or col in outlier_decisions:
            continue
        if df[col].isnull().all():
            continue
        Q1, Q3  = df[col].quantile([0.25, 0.75])
        IQR     = Q3 - Q1
        out_pct = float(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).mean())
        if 0 < out_pct <= 0.01:
            df, msg = _handle_outliers(df, col, "cap")
            actions_log.append(msg)

    for col, strategy in outlier_decisions.items():
        if col in df.columns:
            df, msg = _handle_outliers(df, col, strategy)
            actions_log.append(msg)

    # 6. Class imbalance — record user decision; SMOTE is applied in training.py
    #    after the data is split, so it only affects the training split.
    balance_choice        = decisions.get("balance_classes", "none")
    class_imbalance_found = False
    imbalance_note        = ""
    if "classification" in task_type and target_col and target_col in df.columns:
        vc           = df[target_col].value_counts(normalize=True)
        minority_pct = float(vc.min())
        majority_pct = float(vc.max())
        if minority_pct < 0.30:
            class_imbalance_found = True
            if balance_choice == "smote":
                imbalance_note = (
                    f"Your data had imbalanced classes ({minority_pct:.0%} vs {majority_pct:.0%}). "
                    f"We will create synthetic examples of the minority class to help the model "
                    f"learn both outcomes equally."
                )
                actions_log.append(imbalance_note)

    # Save cleaned data
    output_path = session_dir / "data" / "interim" / "cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    rows_after = len(df)
    cols_after = len(df.columns)

    summary = (
        f"We cleaned your data. "
        + (f"We removed {rows_before - rows_after} rows and {cols_before - cols_after} columns. " if rows_before != rows_after or cols_before != cols_after else "")
        + f"Your dataset now has {rows_after:,} rows and {cols_after} columns ready for the next stage."
    )

    return {
        "stage":                "cleaning",
        "status":               "success",
        "output_data_path":     str(output_path),
        "actions_taken":        actions_log,
        "rows_before":          rows_before,
        "rows_after":           rows_after,
        "cols_before":          cols_before,
        "cols_after":           cols_after,
        "decisions_required":   [],
        "decisions_made":       [{"decision": k, "chosen": v} for k, v in decisions.items()],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "cleaning",
            "title":   "Cleaning Your Data",
            "summary": summary,
            "decision_made": "; ".join(actions_log[:5]) + ("..." if len(actions_log) > 5 else ""),
            "alternatives_considered": "Multiple strategies were available for handling missing values and outliers.",
            "why_this_matters": (
                "Clean data means the model learns from reliable information rather than gaps and errors."
            )
        },
        "config_updates": {
            "impute_strategies":       missing_decisions,
            "outlier_strategies":      outlier_decisions,
            "rows_removed":            rows_before - rows_after,
            "class_imbalance_detected": class_imbalance_found,
            "imbalance_strategy":      balance_choice if class_imbalance_found else "none"
        }
    }
