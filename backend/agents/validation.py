"""
Validation Agent
Runs data quality checks and categorises findings by severity.
Never transforms data — inspect and report only.
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# Date / time-series column detection
# ---------------------------------------------------------------------------

# Tokens that strongly indicate a date column by name
_DATE_NAME_TOKENS = {"date", "time", "timestamp", "datetime", "day", "month", "year"}


def _detect_time_series_columns(df: pd.DataFrame, target_col: str) -> list:
    """
    Return a list of column names that appear to contain date/time values.
    Uses two independent signals:
      1. Name-based: the column name contains a known date-related token.
      2. Value-based: pandas can parse >70% of sampled values as dates.
    A column matching either signal is treated as a date column.
    """
    date_cols = []
    for col in df.columns:
        if col == target_col:
            continue
        # --- Signal 1: name ---
        col_norm = col.lower().replace(" ", "_").replace("-", "_")
        if any(tok in col_norm for tok in _DATE_NAME_TOKENS):
            date_cols.append(col)
            continue
        # --- Signal 2: pandas date parsing on object columns ---
        if df[col].dtype == object:
            sample = df[col].dropna().head(50)
            if len(sample) == 0:
                continue
            try:
                parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    date_cols.append(col)
            except Exception:
                pass
    return date_cols


# ---------------------------------------------------------------------------
# Individual check functions  (from validation SKILL)
# ---------------------------------------------------------------------------

def _validate_target(df: pd.DataFrame, target_col: str, task_type: str) -> list:
    results = []

    if target_col not in df.columns:
        results.append({
            "check":    "target_exists",
            "severity": "hard_stop",
            "message":  f"The column '{target_col}' was not found in your data.",
            "action":   "Please check the column name and try again."
        })
        return results

    target = df[target_col]

    if target.isna().all():
        results.append({
            "check":    "target_not_empty",
            "severity": "hard_stop",
            "message":  f"The column '{target_col}' appears to be completely empty.",
            "action":   "Please check your data source and try again."
        })

    missing_pct = target.isna().mean()
    if missing_pct > 0.05:
        results.append({
            "check":    "target_missing_values",
            "severity": "hard_stop",
            "message":  f"{missing_pct:.1%} of your target column values are missing.",
            "action":   "Rows with a missing target value cannot be used for training. We recommend removing them."
        })
    elif missing_pct > 0:
        results.append({
            "check":    "target_missing_values",
            "severity": "warning",
            "message":  f"{target.isna().sum()} rows have a missing target value ({missing_pct:.1%}).",
            "action":   "These rows will be dropped before training."
        })

    if task_type in ("binary_classification", "multiclass_classification") and not target.isna().all():
        counts       = target.value_counts()
        pcts         = target.value_counts(normalize=True)
        minority_pct = float(pcts.min())
        majority_pct = float(pcts.max())
        minority_cls = pcts.idxmin()
        majority_cls = pcts.idxmax()
        if minority_pct < 0.05:
            results.append({
                "check":       "class_imbalance",
                "severity":    "warning",
                "minority_pct": minority_pct,
                "majority_pct": majority_pct,
                "minority_cls": str(minority_cls),
                "majority_cls": str(majority_cls),
                "counts":       counts.to_dict(),
                "message": (
                    f"Your data is heavily imbalanced. "
                    f"{majority_pct:.0%} of rows are '{majority_cls}', "
                    f"but only {minority_pct:.1%} are '{minority_cls}'. "
                    f"A model that just always predicts '{majority_cls}' would score {majority_pct:.0%} accuracy — "
                    f"but it would never actually identify '{minority_cls}' at all."
                ),
                "action": "You will be asked to choose how to handle this."
            })
        elif minority_pct < 0.30:
            results.append({
                "check":       "class_imbalance",
                "severity":    "advisory",
                "minority_pct": minority_pct,
                "majority_pct": majority_pct,
                "minority_cls": str(minority_cls),
                "majority_cls": str(majority_cls),
                "counts":       counts.to_dict(),
                "message": (
                    f"Your data has a moderate imbalance — {minority_pct:.0%} '{minority_cls}' vs "
                    f"{majority_pct:.0%} '{majority_cls}'. "
                    f"The model may learn to favour the majority class unless we correct for this."
                ),
                "action": "You will be asked to choose how to handle this."
            })

    return results


def _validate_size(df: pd.DataFrame) -> list:
    results = []
    n_rows  = len(df)

    if n_rows < 50:
        results.append({
            "check":    "minimum_rows",
            "severity": "hard_stop",
            "message":  f"Your dataset only has {n_rows} rows. This is too small to train a reliable model.",
            "action":   "A minimum of 50 rows is needed. Ideally 500 or more."
        })
    elif n_rows < 200:
        results.append({
            "check":    "minimum_rows",
            "severity": "warning",
            "message":  f"Your dataset has {n_rows} rows. This is quite small — the model may not learn reliably.",
            "action":   "We will use cross-validation to get the most out of the available data."
        })
    elif n_rows < 1000:
        results.append({
            "check":    "minimum_rows",
            "severity": "advisory",
            "message":  f"Your dataset has {n_rows} rows. This is workable but more data generally produces better results."
        })

    return results


def _validate_missing(df: pd.DataFrame) -> list:
    results = []
    missing = df.isna().mean()

    for col, pct in missing[missing > 0].items():
        if pct > 0.8:
            results.append({
                "check":       "missing_values",
                "severity":    "warning",
                "column":      col,
                "missing_pct": round(float(pct), 4),
                "message":     f"'{col}' is {pct:.1%} empty. This column may not be useful.",
                "action":      "We recommend dropping this column. You will be asked to confirm."
            })
        elif pct > 0.3:
            results.append({
                "check":       "missing_values",
                "severity":    "advisory",
                "column":      col,
                "missing_pct": round(float(pct), 4),
                "message":     f"'{col}' has {pct:.1%} missing values. We will handle this during cleaning."
            })

    return results


def _validate_duplicates(df: pd.DataFrame) -> list:
    results   = []
    dup_count = int(df.duplicated().sum())
    dup_pct   = dup_count / len(df)

    if dup_pct > 0.1:
        results.append({
            "check":    "duplicates",
            "severity": "warning",
            "message":  f"{dup_count} duplicate rows found ({dup_pct:.1%} of your data).",
            "action":   "We recommend removing duplicates. You will be asked to confirm."
        })
    elif dup_count > 0:
        results.append({
            "check":    "duplicates",
            "severity": "advisory",
            "message":  f"{dup_count} duplicate rows found. These will be removed during cleaning."
        })

    return results


def _validate_variance(df: pd.DataFrame, date_cols: list) -> list:
    results = []

    for col in df.columns:
        if col in date_cols:
            continue  # Date columns are handled separately — skip all cardinality/ID checks
        n_unique = df[col].nunique()
        if n_unique == 1:
            results.append({
                "check":    "constant_column",
                "severity": "warning",
                "column":   col,
                "message":  f"'{col}' has only one unique value — it contains no useful information.",
                "action":   "We recommend dropping this column. You will be asked to confirm."
            })
        elif n_unique / len(df) > 0.95 and df[col].dtype == object:
            results.append({
                "check":    "high_cardinality",
                "severity": "advisory",
                "column":   col,
                "message":  f"'{col}' has {n_unique} unique text values — it may be an ID column rather than a useful feature.",
                "action":   "We will flag this during feature engineering."
            })

    return results


def _validate_dtypes(df: pd.DataFrame, target_col: str, date_cols: list) -> list:
    results = []

    for col in df.columns:
        if col == target_col:
            continue
        if col in date_cols:
            continue  # Skip date columns — they will be feature-engineered, not converted
        if df[col].dtype == object:
            sample      = df[col].dropna().head(100)
            if len(sample) == 0:
                continue
            numeric_pct = sample.str.match(r"^-?\d+\.?\d*$", na=False).mean()
            if numeric_pct > 0.9:
                results.append({
                    "check":    "dtype_mismatch",
                    "severity": "advisory",
                    "column":   col,
                    "message":  f"'{col}' looks like it contains numbers but is stored as text.",
                    "action":   "We will convert this to a number during cleaning."
                })

    return results


def _run_validation(df: pd.DataFrame, target_col: str, task_type: str) -> dict:
    # Detect date/time columns first — these are excluded from cardinality and dtype checks
    date_cols    = _detect_time_series_columns(df, target_col)

    all_results  = []
    all_results += _validate_target(df, target_col, task_type)
    all_results += _validate_size(df)
    all_results += _validate_missing(df)
    all_results += _validate_duplicates(df)
    all_results += _validate_variance(df, date_cols)
    all_results += _validate_dtypes(df, target_col, date_cols)

    # Add advisory entries for detected date columns
    for col in date_cols:
        all_results.append({
            "check":    "date_column",
            "severity": "advisory",
            "column":   col,
            "message":  (
                f"We found a date column '{col}'. Rather than treating it as a raw value, "
                f"we will extract useful time features from it — such as month, year, "
                f"day of week, and quarter."
            ),
            "action":   "No action needed. Date features will be created automatically during the feature preparation stage."
        })

    hard_stops = [r for r in all_results if r["severity"] == "hard_stop"]
    warnings   = [r for r in all_results if r["severity"] == "warning"]
    advisories = [r for r in all_results if r["severity"] == "advisory"]

    overall = "hard_stop" if hard_stops else "warnings" if warnings else "passed"

    return {
        "overall_status":     overall,
        "hard_stops":         hard_stops,
        "warnings":           warnings,
        "advisories":         advisories,
        "total_checks_run":   len(all_results),
        "time_series_columns": date_cols,
    }


# ---------------------------------------------------------------------------
# Data summary helper
# ---------------------------------------------------------------------------

def _build_data_summary(df: pd.DataFrame) -> dict:
    """Build column list, numeric stats, and shape for the UI data summary panel."""
    columns = []
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        columns.append({
            "name":          col,
            "dtype":         str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count":    null_count,
            "null_pct":      round(float(df[col].isna().mean()), 4),
        })

    numeric_stats = []
    for col in df.select_dtypes("number").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        try:
            numeric_stats.append({
                "name":   col,
                "min":    round(float(s.min()),    4),
                "max":    round(float(s.max()),    4),
                "mean":   round(float(s.mean()),   4),
                "median": round(float(s.median()), 4),
                "std":    round(float(s.std()),    4),
            })
        except Exception:
            pass

    return {
        "shape":         {"rows": int(len(df)), "columns": int(len(df.columns))},
        "columns":       columns,
        "numeric_stats": numeric_stats,
    }


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id   = session["session_id"]
    target_col   = session["goal"].get("target_column")
    task_type    = session["goal"].get("task_type", "binary_classification")
    sessions_dir = Path("sessions")

    if not target_col:
        return {
            "stage":                 "validation",
            "status":                "hard_stop",
            "plain_english_summary": (
                "We need to know which column you want to predict before we can validate the data. "
                "Please confirm your goal first."
            )
        }

    # Load ingested data
    data_path = sessions_dir / session_id / "data" / "raw" / "ingested.csv"
    if not data_path.exists():
        return {
            "stage":                 "validation",
            "status":                "failed",
            "plain_english_summary": "No ingested data found. Please run the ingestion stage first."
        }

    df = pd.read_csv(data_path, low_memory=False)

    # Build data summary from the full (unfiltered) dataframe
    data_summary = _build_data_summary(df)

    # Drop rows where target is missing (for validation purposes)
    if target_col in df.columns:
        df_for_validation = df.dropna(subset=[target_col])
    else:
        df_for_validation = df

    report             = _run_validation(df_for_validation, target_col, task_type)
    hard_stops         = report["hard_stops"]
    warnings           = report["warnings"]
    advisories         = report["advisories"]
    overall            = report["overall_status"]
    time_series_cols   = report.get("time_series_columns", [])

    # Build decisions_required from warnings that need user input
    decisions_required = []
    for w in warnings + advisories:
        if w["check"] == "class_imbalance":
            minority_pct = w.get("minority_pct", 0)
            minority_cls = w.get("minority_cls", "minority")
            majority_cls = w.get("majority_cls", "majority")
            # Recommend class weights for mild imbalance, SMOTE for severe
            rec = "class_weights" if minority_pct >= 0.05 else "smote"
            decisions_required.append({
                "id":       "imbalance_strategy",
                "question": w["message"],
                "recommendation": rec,
                "recommendation_reason": (
                    "Class weights tell the model to treat each '{minority_cls}' example as more important, "
                    "without changing the data itself. It is the safest option."
                    if rec == "class_weights" else
                    f"With only {minority_pct:.1%} '{minority_cls}' rows, class weights alone may not be enough. "
                    f"Oversampling duplicates minority rows so the model sees a more balanced picture during training."
                ).replace("'{minority_cls}'", f"'{minority_cls}'"),
                "alternatives": [
                    {
                        "id":       "class_weights",
                        "label":    "Adjust class weights (recommended for mild imbalance)",
                        "tradeoff": f"Tells the model to penalise mistakes on '{minority_cls}' more heavily. No data is added or removed.",
                    },
                    {
                        "id":       "smote",
                        "label":    f"SMOTE — create synthetic '{minority_cls}' examples",
                        "tradeoff": (
                            f"Generates new realistic-looking '{minority_cls}' rows by interpolating between existing ones. "
                            f"More effective than simple duplication, but requires enough minority rows to work from."
                        ),
                    },
                    {
                        "id":       "undersample",
                        "label":    f"Undersample — remove '{majority_cls}' rows",
                        "tradeoff": f"Randomly removes majority rows to match the minority count. Balanced, but you lose data.",
                    },
                    {
                        "id":       "none",
                        "label":    "Do nothing",
                        "tradeoff": "The model may learn to always predict the majority class. Only choose this if you have a specific reason.",
                    },
                ]
            })
    for w in warnings:
        if w["check"] == "duplicates":
            decisions_required.append({
                "id":                   "handle_duplicates",
                "question":             w["message"],
                "recommendation":       "remove",
                "recommendation_reason": "Duplicate rows add no information and can bias the model.",
                "alternatives": [
                    {"id": "remove", "label": "Remove duplicate rows (recommended)", "tradeoff": "Cleaner data, slightly fewer rows."},
                    {"id": "keep",   "label": "Keep all rows",                        "tradeoff": "No data lost but duplicates may bias results."}
                ]
            })
        elif w["check"] == "missing_values" and w.get("missing_pct", 0) > 0.8:
            decisions_required.append({
                "id":                   f"drop_col_{w['column']}",
                "question":             w["message"],
                "recommendation":       "drop",
                "recommendation_reason": "Columns that are mostly empty are unlikely to be useful.",
                "alternatives": [
                    {"id": "drop", "label": "Remove this column (recommended)", "tradeoff": "Removes an uninformative column."},
                    {"id": "keep", "label": "Keep it for now",                  "tradeoff": "Imputation will be attempted during cleaning."}
                ]
            })

    # Plain English summary
    if overall == "hard_stop":
        summary = (
            "Before we can continue, there is something that needs your attention: "
            + hard_stops[0]["message"]
        )
    elif overall == "warnings":
        summary = (
            f"Your data passed the critical checks. "
            f"We found {len(warnings)} thing(s) worth your attention before we continue."
        )
    else:
        summary = (
            f"Great news — your data passed all {report['total_checks_run']} quality checks. "
            f"Everything looks good to proceed."
        )

    class_imbalance = any(
        r["check"] == "class_imbalance" and r["severity"] in ("warning", "advisory")
        for r in hard_stops + warnings + advisories
    )

    # ── Compile and save initial RunSpec (best-effort; non-fatal) ─────────
    # This establishes the task family, primary metric, and split strategy
    # that all downstream stages will use.  Later stages update the RunSpec
    # with their tactical decisions (scaling, imputation, features, etc.).
    run_spec_compiled = False
    if overall != "hard_stop":
        try:
            import json as _json
            from services.manifest_builder import build as _build_manifest
            from services.task_router import resolve as _route_task
            from services.pipeline_compiler import compile_run_spec as _compile_run_spec

            artifacts_dir = sessions_dir / session_id / "artifacts"
            artifacts_dir.mkdir(parents=True, exist_ok=True)

            _manifest = _build_manifest(df, target_column=target_col)
            _time_col = time_series_cols[0] if time_series_cols else None
            _routing  = _route_task(task_type, _manifest, target_col, _time_col)

            # Only decisions available at validation time
            _initial_decisions = {
                "target_column":   target_col,
                "balance_classes": decisions.get("imbalance_strategy", "none"),
            }
            _run_spec = _compile_run_spec(session_id, _manifest, _routing, _initial_decisions)
            _run_spec.save(artifacts_dir / "run_spec.json")

            # Save manifest for downstream reference
            _manifest_dict = {
                "dataset_id":               _manifest.dataset_id,
                "row_count":                _manifest.row_count,
                "column_count":             _manifest.column_count,
                "numeric_columns":          _manifest.numeric_columns,
                "categorical_columns":      _manifest.categorical_columns,
                "datetime_columns":         _manifest.datetime_columns,
                "binary_columns":           _manifest.binary_columns,
                "text_columns":             _manifest.text_columns,
                "candidate_target_columns": _manifest.candidate_target_columns,
                "candidate_time_columns":   _manifest.candidate_time_columns,
            }
            (artifacts_dir / "manifest.json").write_text(
                _json.dumps(_manifest_dict, indent=2)
            )
            run_spec_compiled = True
        except Exception as _e:
            print(f"[validation] RunSpec compilation failed (non-fatal): {_e}")

    return {
        "stage":                "validation",
        "status":               overall,
        "output_data_path":     str(data_path),
        "validation_report":    report,
        "data_summary":         data_summary,
        "decisions_required":   decisions_required,
        "decisions_made":       [],
        "plain_english_summary": summary,
        "time_series_columns":  time_series_cols,
        "report_section": {
            "stage":   "validation",
            "title":   "Checking Your Data Quality",
            "summary": f"We ran {report['total_checks_run']} checks on your data. {summary}",
            "decision_made": (
                f"{len(warnings)} items flagged for your review."
                if warnings else "No decisions required — data passed all checks."
            ),
            "alternatives_considered": None,
            "why_this_matters": (
                "Catching data quality issues early means the model learns from reliable "
                "information rather than noise."
            )
        },
        "run_spec_compiled": run_spec_compiled,
        "config_updates": {
            "validation_passed":        overall != "hard_stop",
            "class_imbalance_detected": class_imbalance,
            "imbalance_strategy":       decisions.get("imbalance_strategy", None),
            "is_time_series":           len(time_series_cols) > 0,
            "time_series_columns":      time_series_cols,
        }
    }
