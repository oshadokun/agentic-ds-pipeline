"""
Feature Engineering Agent
Encodes categoricals, expands datetimes, removes redundant features,
and selects the most informative features.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _apply_encoding(df: pd.DataFrame, col: str, strategy: str,
                     target_col: str = None) -> tuple[pd.DataFrame, str]:
    if col not in df.columns:
        return df, f"Column '{col}' not found — skipped."

    if strategy == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df, f"Created {len(dummies.columns)} yes/no column(s) from '{col}'."

    elif strategy == "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("missing"))
        return df, f"Converted '{col}' categories to numbers."

    elif strategy == "frequency":
        freq_map = df[col].value_counts(normalize=True)
        df[col + "_freq"] = df[col].map(freq_map)
        df = df.drop(columns=[col])
        return df, f"Replaced '{col}' with how often each category appears."

    elif strategy == "target" and target_col and target_col in df.columns:
        global_mean = float(df[target_col].mean())
        agg = df.groupby(col)[target_col].agg(["mean", "count"])
        smoothing = 10
        agg["smoothed"] = (
            (agg["mean"] * agg["count"] + global_mean * smoothing)
            / (agg["count"] + smoothing)
        )
        df[col + "_target_enc"] = df[col].map(agg["smoothed"])
        df = df.drop(columns=[col])
        return df, f"Replaced '{col}' with the average outcome for each category."

    elif strategy == "ordinal":
        # Detect the order from the known ordinal sets
        order = None
        for known_set, ord_order in _ORDINAL_SETS:
            vals = set(df[col].dropna().str.lower().str.strip().unique())
            if vals <= set(known_set):
                order = [v for v in ord_order if v in vals]
                break
        if order:
            rank_map = {v: i for i, v in enumerate(order)}
            df[col] = df[col].str.lower().str.strip().map(rank_map)
            return df, f"Encoded '{col}' as ordered numbers ({' < '.join(order)})."
        # Fallback to label if order not found
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).fillna("missing"))
        return df, f"Converted '{col}' categories to numbers."

    elif strategy == "drop":
        df = df.drop(columns=[col])
        return df, f"Removed '{col}' — it did not appear useful for prediction."

    return df, f"No encoding applied to '{col}'."


_GEO_NAMES = {
    "country", "nation", "continent",
    "region", "province", "territory", "county", "district", "zone", "area",
    "state", "prefecture",
    "city", "town", "village", "municipality", "suburb", "borough",
    "postcode", "postal", "zip", "zipcode",
    "latitude", "longitude", "lat", "lon", "lng",
    "location", "address", "street",
}

_ORDINAL_SETS = [
    # size
    (["xs", "s", "m", "l", "xl", "xxl"],         ["xs", "s", "m", "l", "xl", "xxl"]),
    (["xsmall", "small", "medium", "large", "xlarge"], ["xsmall", "small", "medium", "large", "xlarge"]),
    (["extra small", "small", "medium", "large", "extra large"], ["extra small", "small", "medium", "large", "extra large"]),
    # level
    (["low", "medium", "high"],                   ["low", "medium", "high"]),
    (["none", "low", "medium", "high"],            ["none", "low", "medium", "high"]),
    (["very low", "low", "medium", "high", "very high"], ["very low", "low", "medium", "high", "very high"]),
    # rating
    (["poor", "fair", "good", "very good", "excellent"], ["poor", "fair", "good", "very good", "excellent"]),
    (["bad", "neutral", "good"],                  ["bad", "neutral", "good"]),
    # frequency
    (["never", "rarely", "sometimes", "often", "always"], ["never", "rarely", "sometimes", "often", "always"]),
    # agreement
    (["strongly disagree", "disagree", "neutral", "agree", "strongly agree"],
     ["strongly disagree", "disagree", "neutral", "agree", "strongly agree"]),
    # education
    (["high school", "bachelor", "master", "phd"], ["high school", "bachelor", "master", "phd"]),
    # priority
    (["low", "normal", "high", "critical"],        ["low", "normal", "high", "critical"]),
]

_ID_NAME_PATTERNS = {"id", "uuid", "guid", "key", "code", "number", "num", "no", "ref", "index", "idx", "hash", "token"}

_DATE_NAME_PATTERNS = {"date", "time", "timestamp", "datetime", "dt", "period", "week", "quarter"}

_DATE_VALUE_PATTERNS = [
    r"^\d{4}-\d{2}-\d{2}",          # YYYY-MM-DD
    r"^\d{1,2}/\d{1,2}/\d{4}",      # M/D/YYYY or MM/DD/YYYY
    r"^\d{1,2}-\d{1,2}-\d{4}",      # D-M-YYYY
    r"^\d{4}/\d{2}/\d{2}",          # YYYY/MM/DD
    r"^\d{4}-\d{2}-\d{2}[T ]",      # ISO datetime
]


def _is_date_column(col: str, df: pd.DataFrame) -> bool:
    """Return True if column looks like a date by name or by sampled values."""
    col_norm = col.lower().replace(" ", "_").replace("-", "_")
    if any(pat in col_norm for pat in _DATE_NAME_PATTERNS):
        return True
    sample = df[col].dropna().astype(str).head(50)
    if len(sample) == 0:
        return False
    return any(sample.str.match(p).mean() > 0.7 for p in _DATE_VALUE_PATTERNS)


def _is_geo_column(col: str) -> bool:
    col_norm = col.lower().replace(" ", "_").replace("-", "_")
    return any(geo in col_norm for geo in _GEO_NAMES)


def _detect_ordinal(col: str, df: pd.DataFrame):
    """Return (order_list, reason) if column values match a known ordinal set, else None."""
    values = set(df[col].dropna().str.lower().str.strip().unique())
    for known_set, order in _ORDINAL_SETS:
        if values <= set(known_set) and len(values) >= 2:
            matched_order = [v for v in order if v in values]
            return matched_order, f"values follow a natural order ({' < '.join(matched_order)})"
    return None


def _looks_like_id(col: str, n_unique: int, n_rows: int, df: pd.DataFrame = None) -> bool:
    col_norm = col.lower().replace(" ", "_").replace("-", "_")
    # Never flag date-like columns as IDs
    if df is not None and _is_date_column(col, df):
        return False
    if n_unique / max(n_rows, 1) > 0.9:
        return True
    return any(col_norm == pat or col_norm.endswith("_" + pat) for pat in _ID_NAME_PATTERNS)


def _recommend_encoding(col: str, df: pd.DataFrame) -> tuple[str, str]:
    n_unique   = df[col].nunique()
    n_rows     = len(df)

    # ID / free-text columns — drop (but never drop date-like columns)
    if _looks_like_id(col, n_unique, n_rows, df):
        return "drop", f"'{col}' looks like an identifier — it won't help the model learn patterns."

    # Binary
    if n_unique <= 2:
        return "label", f"'{col}' has only 2 values — we will convert them to 0 and 1."

    # Ordinal — check before cardinality rules
    ordinal = _detect_ordinal(col, df)
    if ordinal:
        order, reason = ordinal
        return "ordinal", f"'{col}' has {reason} — we will encode them in that order."

    # Geographic — target encoding captures regional patterns best
    if _is_geo_column(col):
        if n_unique <= 6:
            return "onehot", (
                f"'{col}' is a geographic column with only {n_unique} values "
                f"— yes/no columns work well at this scale."
            )
        return "target", (
            f"'{col}' is a geographic column — we will replace each location with "
            f"the average outcome for that area, which captures regional patterns better than counting."
        )

    # Nominal by cardinality
    if n_unique <= 10:
        return "onehot", f"'{col}' has {n_unique} categories — we will create a yes/no column for each."
    if n_unique <= 50:
        return "target", (
            f"'{col}' has {n_unique} categories — we will replace each with the average outcome "
            f"for that category, which is more informative than counting how often it appears."
        )
    return "target", f"'{col}' has {n_unique} categories — we will use target encoding."


# ---------------------------------------------------------------------------
# Datetime expansion
# ---------------------------------------------------------------------------

def _expand_datetime(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, str]:
    dt = pd.to_datetime(df[col], errors="coerce")
    df[f"{col}_year"]       = dt.dt.year
    df[f"{col}_month"]      = dt.dt.month
    df[f"{col}_day"]        = dt.dt.day
    df[f"{col}_dayofweek"]  = dt.dt.dayofweek
    df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df[f"{col}_quarter"]    = dt.dt.quarter
    df[f"{col}_month_sin"]  = np.sin(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_month_cos"]  = np.cos(2 * np.pi * dt.dt.month / 12)
    df[f"{col}_dow_sin"]    = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df[f"{col}_dow_cos"]    = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    df = df.drop(columns=[col])
    return df, (
        f"Expanded '{col}' into 10 date-based features: "
        f"year, month, day, day of week, weekend flag, quarter, "
        f"and cyclical encodings for month and day of week."
    )


# ---------------------------------------------------------------------------
# Redundant feature removal
# ---------------------------------------------------------------------------

def _remove_redundant(df: pd.DataFrame, target_col: str,
                       high_corr_pairs: list, threshold: float = 0.95) -> tuple[pd.DataFrame, list]:
    to_drop = set()
    actions = []
    for pair in high_corr_pairs:
        col_a = pair.get("col_a", "")
        col_b = pair.get("col_b", "")
        corr  = pair.get("correlation", 0)
        if corr < threshold:
            continue
        if col_a not in df.columns or col_b not in df.columns:
            continue
        if target_col in df.columns:
            ca = abs(df[col_a].corr(df[target_col])) if df[col_a].dtype in ["int64","float64"] else 0
            cb = abs(df[col_b].corr(df[target_col])) if df[col_b].dtype in ["int64","float64"] else 0
            drop = col_b if ca >= cb else col_a
        else:
            drop = col_b
        to_drop.add(drop)
        keep = col_b if drop == col_a else col_a
        actions.append(
            f"Removed '{drop}' because it is almost identical to '{keep}' (similarity: {corr:.2f}). "
            f"Keeping the one more related to the outcome."
        )
    df = df.drop(columns=list(to_drop), errors="ignore")
    return df, actions


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def _select_features(df: pd.DataFrame, target_col: str,
                      task_type: str, max_features: int = 30) -> tuple[list, pd.DataFrame, str]:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fill nulls for scoring only
    X_filled = X.fillna(X.median(numeric_only=True))
    # Fill object cols
    for col in X_filled.select_dtypes("object").columns:
        X_filled[col] = LabelEncoder().fit_transform(X_filled[col].astype(str))

    try:
        if "classification" in task_type:
            scores = mutual_info_classif(X_filled, y, random_state=42)
        else:
            scores = mutual_info_regression(X_filled, y, random_state=42)
    except Exception:
        # Fallback: keep all
        return X.columns.tolist() + [target_col], pd.DataFrame(), "All features kept."

    importance_df = pd.DataFrame({
        "feature":    X.columns,
        "importance": scores
    }).sort_values("importance", ascending=False)

    n_keep   = min(max_features, max(5, len(importance_df)))
    selected = importance_df.head(n_keep)["feature"].tolist()
    dropped  = importance_df.tail(len(importance_df) - n_keep)["feature"].tolist()

    msg = (
        f"We measured how informative each column is for predicting '{target_col}'. "
        f"We kept the top {n_keep} most informative columns"
        + (f" and removed {len(dropped)} columns that added little value." if dropped else ".")
    )

    return selected + [target_col], importance_df, msg


# ---------------------------------------------------------------------------
# Decisions required (pre-run)
# ---------------------------------------------------------------------------

def _build_decisions_required(df: pd.DataFrame, target_col: str) -> list:
    decisions = []
    # Exclude date-like object columns — they will be expanded as datetimes, not encoded
    cat_cols  = [
        c for c in df.select_dtypes("object").columns
        if c != target_col and not _is_date_column(c, df)
    ]

    # Detect geographic hierarchy — flag if multiple geo levels exist together
    geo_cols = [c for c in cat_cols if _is_geo_column(c)]

    OPTS = {
        "onehot":    {"label": "Create a separate yes/no column for each category",                    "tradeoff": "Simple and interpretable, but creates many extra columns for high-cardinality data."},
        "target":    {"label": "Replace each category with the average outcome for that category",     "tradeoff": "Powerful and compact. Works especially well for geographic and high-cardinality columns."},
        "frequency": {"label": "Replace each category with how often it appears in the data",          "tradeoff": "Useful when how common a category is matters, but loses outcome information."},
        "ordinal":   {"label": "Encode as ordered numbers (e.g. low=0, medium=1, high=2)",            "tradeoff": "Only appropriate when the categories have a genuine order."},
        "label":     {"label": "Replace each category with an arbitrary number",                       "tradeoff": "Compact but implies a false order. Usually only correct for binary columns."},
        "drop":      {"label": "Remove this column entirely",                                          "tradeoff": "Loses all information. Appropriate for ID columns or columns unlikely to help prediction."},
    }

    for col in cat_cols:
        rec, reason = _recommend_encoding(col, df)
        n_unique = int(df[col].nunique())

        # Build context note
        notes = []
        if _is_geo_column(col) and len(geo_cols) > 1:
            other_geo = [c for c in geo_cols if c != col]
            notes.append(
                f"Note: '{col}' and {other_geo} all represent geographic location. "
                f"They overlap — consider whether you need all of them."
            )
        ordinal_match = _detect_ordinal(col, df)
        if ordinal_match and rec != "ordinal":
            notes.append(f"Note: values appear to follow a natural order.")

        question = (
            f"How should we convert '{col}' "
            f"({'geographic column, ' if _is_geo_column(col) else ''}"
            f"{n_unique} unique values) to numbers?"
        )

        decisions.append({
            "id":                    f"encoding_{col}",
            "type":                  "encoding",
            "column":                col,
            "n_unique":              n_unique,
            "is_geographic":         _is_geo_column(col),
            "question":              question,
            "recommendation":        rec,
            "recommendation_reason": reason,
            "notes":                 notes,
            "alternatives":          [{"id": k, **v} for k, v in OPTS.items()],
        })

    return decisions


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id = session["session_id"]
    target_col = session["goal"].get("target_column")
    task_type  = session["goal"].get("task_type", "binary_classification")
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    data_path = session_dir / "data" / "interim" / "cleaned.csv"
    if not data_path.exists():
        return {
            "stage":                 "feature_engineering",
            "status":                "failed",
            "plain_english_summary": "No cleaned data found. Please run the cleaning stage first."
        }

    df = pd.read_csv(data_path, low_memory=False)

    # Re-parse date columns — CSV doesn't preserve datetime dtype, so ISO date
    # strings written by cleaning come back as object and need re-detection.
    for col in list(df.select_dtypes("object").columns):
        if col == target_col:
            continue
        if _is_date_column(col, df):
            try:
                parsed = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
            except Exception:
                pass

    # Get EDA correlation findings
    eda_result_path = session_dir / "outputs" / "eda" / "result.json"
    high_corr_pairs = []
    if eda_result_path.exists():
        with open(eda_result_path) as f:
            eda_data = json.load(f)
        high_corr_pairs = (
            eda_data.get("findings", {})
                    .get("correlations", {})
                    .get("high_corr_pairs", [])
        )

    # Phase 1: return decisions_required if no user decisions yet
    if not decisions or decisions.get("phase") == "request_decisions":
        dr = _build_decisions_required(df, target_col)
        if dr:
            return {
                "stage":              "feature_engineering",
                "status":             "decisions_required",
                "decisions_required": dr,
                "plain_english_summary": (
                    f"Your data contains {len(dr)} text or category column(s). "
                    f"The model cannot use text directly — we need to convert them to numbers. "
                    f"Here is what we recommend for each:"
                )
            }
        # No categorical columns — fall through to apply transformations directly

    # Phase 2: apply transformations
    actions_log = []
    is_time_series = "time_series" in task_type or "forecast" in task_type

    # 1. Remove redundant correlated features
    df, redundant_actions = _remove_redundant(df, target_col, high_corr_pairs)
    actions_log.extend(redundant_actions)

    # 2. Expand datetime columns (also catch any date-like object columns not yet parsed)
    for col in list(df.select_dtypes("object").columns):
        if col == target_col:
            continue
        if _is_date_column(col, df):
            try:
                parsed = pd.to_datetime(df[col], dayfirst=False, errors="coerce")
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
            except Exception:
                pass

    date_cols_expanded = []
    for col in list(df.select_dtypes("datetime").columns):
        if is_time_series:
            # Sort by the earliest date column before expanding, to preserve temporal order
            df = df.sort_values(col).reset_index(drop=True)
        date_cols_expanded.append(col)
        df, msg = _expand_datetime(df, col)
        actions_log.append(msg)

    if is_time_series and date_cols_expanded:
        actions_log.append(
            f"Data sorted chronologically by '{date_cols_expanded[0]}' to preserve time order."
        )

    # 3. Encode categoricals using user decisions
    cat_cols = [c for c in df.select_dtypes("object").columns if c != target_col]
    encoding_strategies = {}
    for col in cat_cols:
        strategy = decisions.get(f"encoding_{col}")
        if not strategy:
            strategy, _ = _recommend_encoding(col, df)
        df, msg = _apply_encoding(df, col, strategy, target_col)
        encoding_strategies[col] = strategy
        actions_log.append(msg)

    # 4. Feature selection — present results
    feature_selection_approved = decisions.get("feature_selection_approved", True)
    selected_features, importance_df, selection_msg = _select_features(df, target_col, task_type)
    actions_log.append(selection_msg)

    if feature_selection_approved:
        available = [c for c in selected_features if c in df.columns]
        # For time series, always keep date-derived features (year, month, day, etc.)
        if is_time_series:
            date_derived = [
                c for c in df.columns
                if any(c.endswith(suf) for suf in
                       ("_year", "_month", "_day", "_dayofweek", "_is_weekend",
                        "_quarter", "_month_sin", "_month_cos", "_dow_sin", "_dow_cos"))
            ]
            for dc in date_derived:
                if dc not in available:
                    available.append(dc)
        df = df[available]

    # Save output
    output_path = session_dir / "data" / "interim" / "features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    feature_cols = [c for c in df.columns if c != target_col]

    summary = (
        f"We transformed your data ready for modelling. "
        f"We encoded {len(encoding_strategies)} category column(s)"
        + (f", removed {len(redundant_actions)} redundant column(s)" if redundant_actions else "")
        + f", and selected {len(feature_cols)} informative features."
    )

    return {
        "stage":                "feature_engineering",
        "status":               "success",
        "output_data_path":     str(output_path),
        "actions_taken":        actions_log,
        "feature_importance":   importance_df.to_dict(orient="records") if not importance_df.empty else [],
        "decisions_required":   [],
        "decisions_made":       [{"decision": k, "chosen": v} for k, v in decisions.items()],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "feature_engineering",
            "title":   "Preparing Your Features",
            "summary": summary,
            "decision_made": "; ".join(actions_log[:4]),
            "alternatives_considered": "Multiple encoding strategies were available for each category column.",
            "why_this_matters": (
                "Models can only learn from numbers — this stage turns all your data into a form "
                "the model can understand, while keeping the most useful information."
            )
        },
        "config_updates": {
            "feature_columns":    feature_cols,
            "encoding_strategies": encoding_strategies,
            "features_selected":  len(feature_cols)
        }
    }
