"""
EDA Agent
Explores and summarises the data. Produces charts and plain-English insights.
Never transforms data.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Analysis modules  (from EDA SKILL)
# ---------------------------------------------------------------------------

def _overview(df: pd.DataFrame, target_col: str) -> dict:
    numeric_cols  = df.select_dtypes("number").columns.tolist()
    cat_cols      = df.select_dtypes("object").columns.tolist()
    datetime_cols = df.select_dtypes("datetime").columns.tolist()
    return {
        "row_count":        len(df),
        "column_count":     len(df.columns),
        "numeric_cols":     numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols":    datetime_cols,
        "total_missing":    int(df.isna().sum().sum()),
        "missing_pct":      round(float(df.isna().mean().mean()), 4),
        "memory_mb":        round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "plain_english": (
            f"Your dataset has {len(df):,} rows and {len(df.columns)} columns. "
            f"{len(numeric_cols)} columns contain numbers, {len(cat_cols)} contain text or categories. "
            f"Overall, {df.isna().mean().mean():.1%} of values are missing across the whole dataset."
        )
    }


def _analyse_target(df: pd.DataFrame, target_col: str, task_type: str, output_dir: str) -> dict:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    target = df[target_col].dropna()
    result = {"column": target_col}

    if task_type in ["binary_classification", "multiclass_classification"]:
        counts = target.value_counts()
        pcts   = target.value_counts(normalize=True)
        result["class_counts"] = counts.to_dict()
        result["class_pcts"]   = {str(k): round(float(v), 4) for k, v in pcts.items()}

        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="#2E75B6", edgecolor="white")
        ax.set_title("How often each outcome appears in your data")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Number of rows")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        chart_path = Path(output_dir) / "target_distribution.png"
        plt.savefig(chart_path, bbox_inches="tight", dpi=100)
        plt.close()
        result["chart_path"] = str(chart_path)

        majority = counts.index[0]
        minority = counts.index[-1]
        result["plain_english"] = (
            f"Your data has {len(counts)} possible outcomes. "
            f"'{majority}' is the most common ({pcts.iloc[0]:.1%} of rows). "
            f"'{minority}' is the least common ({pcts.iloc[-1]:.1%} of rows)."
        )

    elif task_type == "regression":
        result.update({
            "mean":   round(float(target.mean()), 4),
            "median": round(float(target.median()), 4),
            "std":    round(float(target.std()), 4),
            "min":    round(float(target.min()), 4),
            "max":    round(float(target.max()), 4),
            "skew":   round(float(target.skew()), 4)
        })

        fig, ax = plt.subplots(figsize=(8, 4))
        target.hist(bins=40, ax=ax, color="#2E75B6", edgecolor="white")
        ax.set_title(f"Distribution of '{target_col}'")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Number of rows")
        plt.tight_layout()
        chart_path = Path(output_dir) / "target_distribution.png"
        plt.savefig(chart_path, bbox_inches="tight", dpi=100)
        plt.close()
        result["chart_path"] = str(chart_path)

        skew_val  = result["skew"]
        skew_desc = "fairly symmetrical"
        if abs(skew_val) > 1:
            skew_desc = "skewed to the right" if skew_val > 0 else "skewed to the left"

        result["plain_english"] = (
            f"The values in '{target_col}' range from {result['min']} to {result['max']}. "
            f"The typical value is around {result['median']} (the middle value). "
            f"The distribution is {skew_desc}."
        )

    return result


def _analyse_features(df: pd.DataFrame, target_col: str, output_dir: str,
                       time_series_columns: list = None) -> list:
    results              = []
    output_dir           = Path(output_dir)
    time_series_columns  = time_series_columns or []

    for col in df.columns:
        if col == target_col:
            continue
        if col in time_series_columns:
            continue  # Date columns are fully handled by validation — skip entirely
        col_result = {
            "column":       col,
            "dtype":        str(df[col].dtype),
            "missing_pct":  round(float(df[col].isna().mean()), 4),
            "unique_count": int(df[col].nunique())
        }

        if df[col].dtype in ["int64", "float64"]:
            col_result.update({
                "mean":   round(float(df[col].mean()), 4),
                "median": round(float(df[col].median()), 4),
                "std":    round(float(df[col].std()), 4),
                "skew":   round(float(df[col].skew()), 4)
            })
            Q1, Q3   = df[col].quantile([0.25, 0.75])
            IQR      = Q3 - Q1
            outlier_count = int(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum())
            col_result["outlier_count"] = outlier_count
            col_result["outlier_pct"]   = round(outlier_count / max(len(df), 1), 4)
            if outlier_count > 0:
                col_result["advisory"] = (
                    f"'{col}' has {outlier_count} unusually high or low values "
                    f"({col_result['outlier_pct']:.1%}). These will be reviewed during cleaning."
                )
        else:
            top = df[col].value_counts().head(5).to_dict()
            col_result["top_values"] = {str(k): int(v) for k, v in top.items()}
            if col_result["unique_count"] / max(len(df), 1) > 0.9:
                col_result["advisory"] = (
                    f"'{col}' has almost as many unique values as rows — "
                    f"it may be an ID column and not useful for modelling."
                )

        results.append(col_result)

    # Distribution grid for numeric columns
    numeric_cols = [c for c in df.columns if c != target_col and df[c].dtype in ["int64", "float64"]]
    if numeric_cols:
        n           = min(len(numeric_cols), 12)
        cols_to_plt = numeric_cols[:n]
        ncols       = 3
        nrows       = (n + ncols - 1) // ncols
        fig, axes   = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
        axes_flat   = np.array(axes).flatten()
        for i, col in enumerate(cols_to_plt):
            df[col].hist(bins=30, ax=axes_flat[i], color="#2E75B6", edgecolor="white")
            axes_flat[i].set_title(col, fontsize=10)
        for j in range(len(cols_to_plt), len(axes_flat)):
            axes_flat[j].set_visible(False)
        plt.suptitle("Distribution of your numeric columns", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_distributions.png", bbox_inches="tight", dpi=100)
        plt.close()

    return results


def _analyse_correlations(df: pd.DataFrame, target_col: str, output_dir: str) -> dict:
    import seaborn as sns
    output_dir = Path(output_dir)
    numeric_df = df.select_dtypes("number")

    if len(numeric_df.columns) < 2:
        return {"plain_english": "Not enough numeric columns to analyse correlations."}

    corr_matrix = numeric_df.corr()

    n_cols = len(numeric_df.columns)
    fig_size = (min(16, n_cols * 1.2), min(14, n_cols * 1.0))
    fig, ax  = plt.subplots(figsize=fig_size)
    annot    = n_cols <= 15
    sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap="Blues",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title(
        "How strongly your numeric columns relate to each other\n"
        "(1.0 = identical, 0 = no relationship, -1.0 = opposite)"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight", dpi=100)
    plt.close()

    target_corrs = {}
    if target_col in numeric_df.columns:
        tc = corr_matrix[target_col].drop(target_col, errors="ignore").abs().sort_values(ascending=False)
        target_corrs = {k: round(float(v), 4) for k, v in tc.items()}

    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] == target_col or cols[j] == target_col:
                continue
            val = abs(float(corr_matrix.iloc[i, j]))
            if val > 0.85:
                high_corr_pairs.append({
                    "col_a":       cols[i],
                    "col_b":       cols[j],
                    "correlation": round(val, 4),
                    "advisory": (
                        f"'{cols[i]}' and '{cols[j]}' are very similar to each other ({val:.2f}). "
                        f"You may only need one of them."
                    )
                })

    plain = []
    if target_corrs:
        top     = list(target_corrs.items())[:3]
        top_str = ", ".join([f"'{c}' ({v:.2f})" for c, v in top])
        plain.append(f"The columns most related to '{target_col}' are: {top_str}.")
    if high_corr_pairs:
        plain.append(
            f"We found {len(high_corr_pairs)} pairs of columns that are very similar — "
            f"you may not need both. We will flag these during feature engineering."
        )

    return {
        "target_correlations": target_corrs,
        "high_corr_pairs":     high_corr_pairs,
        "chart_path":          str(output_dir / "correlation_heatmap.png"),
        "plain_english":       " ".join(plain) if plain else "No strong relationships found between columns."
    }


def _analyse_feature_vs_target(df: pd.DataFrame, target_col: str,
                                task_type: str, output_dir: str) -> dict:
    output_dir   = Path(output_dir)
    numeric_cols = [c for c in df.select_dtypes("number").columns if c != target_col]
    chart_path   = str(output_dir / "feature_vs_target.png")

    if not numeric_cols or task_type not in ["binary_classification", "multiclass_classification"]:
        return {"chart_path": None, "insights": []}

    n = min(len(numeric_cols), 6)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
    for i, col in enumerate(numeric_cols[:n]):
        for label in df[target_col].dropna().unique():
            subset = df[df[target_col] == label][col].dropna()
            axes[i].hist(subset, alpha=0.6, bins=20, label=str(label))
        axes[i].set_title(col, fontsize=10)
        axes[i].legend(fontsize=8)
    plt.suptitle("How each column varies across your outcomes", fontsize=12)
    plt.tight_layout()
    plt.savefig(chart_path, bbox_inches="tight", dpi=100)
    plt.close()

    return {
        "chart_path": chart_path,
        "insights": [
            "Columns where the distributions look different between outcomes "
            "are likely to be useful for prediction."
        ]
    }


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id          = session["session_id"]
    target_col          = session["goal"].get("target_column")
    task_type           = session["goal"].get("task_type", "binary_classification")
    sessions_dir        = Path("sessions")
    # Columns already identified as date/time by the validation stage — never treat as identifiers
    time_series_columns = session.get("config", {}).get("time_series_columns", [])

    if not target_col:
        return {
            "stage":                 "eda",
            "status":                "failed",
            "plain_english_summary": "Target column is not set. Please confirm your goal first."
        }

    data_path = sessions_dir / session_id / "data" / "raw" / "ingested.csv"
    if not data_path.exists():
        return {
            "stage":                 "eda",
            "status":                "failed",
            "plain_english_summary": "No ingested data found. Please run ingestion first."
        }

    df         = pd.read_csv(data_path, low_memory=False)
    output_dir = str(sessions_dir / session_id / "reports" / "eda")

    # Coerce string-formatted numbers (e.g. "$1,200", "50%") to float for analysis.
    # This is read-only — the raw file is never modified.
    for col in df.select_dtypes("object").columns:
        cleaned = (df[col].astype(str)
                   .str.replace(r'[$%\s]', '', regex=True)
                   .str.replace(',', '', regex=False))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.8:
            df[col] = numeric

    try:
        ov            = _overview(df, target_col)
        target_result = _analyse_target(df, target_col, task_type, output_dir)
        features      = _analyse_features(df, target_col, output_dir, time_series_columns)
        correlations  = _analyse_correlations(df, target_col, output_dir)
        vs_target     = _analyse_feature_vs_target(df, target_col, task_type, output_dir)
    except Exception as exc:
        return {
            "stage":                 "eda",
            "status":                "failed",
            "plain_english_summary": f"EDA encountered an error: {str(exc)}"
        }

    # Collect chart paths
    charts = []
    for obj in [target_result, correlations, vs_target]:
        cp = obj.get("chart_path")
        if cp:
            charts.append(cp)
    feat_dist = str(Path(output_dir) / "feature_distributions.png")
    if Path(feat_dist).exists():
        charts.append(feat_dist)

    # Config updates
    high_outlier_cols = [
        f["column"] for f in features
        if f.get("outlier_pct", 0) > 0.01
    ]
    skewed_cols = [
        f["column"] for f in features
        if abs(f.get("skew", 0)) > 1 and f.get("dtype") in ("int64", "float64")
    ]
    high_card_cols = [
        f["column"] for f in features
        if f.get("advisory", "").startswith(f"'{f['column']}' has almost")
        and f["column"] not in time_series_columns  # date cols are never identifiers
    ]

    # Build decisions list if first run (no decisions provided yet)
    decisions_required = []
    decisions_made     = []

    if not decisions:
        # Ask about ID-like columns — group if 3+ share the same issue
        if len(high_card_cols) >= 3:
            decisions_required.append({
                "id":                    "id_col__grouped",
                "question":              f"We found {len(high_card_cols)} columns that look like identifiers (row IDs or codes). Should we exclude them?",
                "recommendation":        "exclude",
                "recommendation_reason": (
                    "These columns have nearly as many unique values as rows, which usually means "
                    "they are ID or reference numbers — not useful for prediction."
                ),
                "grouped_columns":       high_card_cols,
                "alternatives": [
                    {"id": "exclude", "label": "Exclude all of them", "tradeoff": "Remove them from the model — recommended for ID columns"},
                    {"id": "keep",    "label": "Keep all of them",    "tradeoff": "Include them — only if they genuinely carry meaning"}
                ]
            })
        else:
            for col in high_card_cols:
                decisions_required.append({
                    "id":                    f"id_col__{col}",
                    "question":              f'Is "{col}" just an identifier (like a row number or code)?',
                    "recommendation":        "exclude",
                    "recommendation_reason": (
                        f'"{col}" has nearly as many unique values as rows, which usually means '
                        f"it's an ID or reference number — not useful for prediction."
                    ),
                    "alternatives": [
                        {"id": "exclude", "label": "Exclude it", "tradeoff": "Remove it from the model — recommended for ID columns"},
                        {"id": "keep",    "label": "Keep it",    "tradeoff": "Include it — choose this if it genuinely carries meaning"}
                    ]
                })

        # Ask about outlier handling — group if 3+ columns share the same recommendation
        if len(high_outlier_cols) >= 3:
            decisions_required.append({
                "id":                    "outliers__grouped",
                "question":              f"We found {len(high_outlier_cols)} columns with extreme values. How would you like to handle them?",
                "recommendation":        "cap",
                "recommendation_reason": (
                    f"These {len(high_outlier_cols)} columns each have an unusual number of very high "
                    f"or very low values. Capping pulls them to the edge of the normal range without "
                    f"removing any rows — the safest approach."
                ),
                "grouped_columns":       high_outlier_cols,
                "alternatives": [
                    {"id": "cap",    "label": "Cap at boundary (apply to all)",  "tradeoff": "Pull extreme values in to the edge of the normal range — usually the safest choice"},
                    {"id": "remove", "label": "Remove those rows (apply to all)", "tradeoff": "Delete rows with extreme values — use if they are clearly errors"},
                    {"id": "keep",   "label": "Keep as-is (apply to all)",        "tradeoff": "Leave them unchanged — choose if the extremes are real and meaningful"}
                ]
            })
        else:
            for col in high_outlier_cols:
                pct = next((f["outlier_pct"] for f in features if f["column"] == col), 0)
                decisions_required.append({
                    "id":                    f"outliers__{col}",
                    "question":              f'How should we handle the extreme values in "{col}"?',
                    "recommendation":        "cap",
                    "recommendation_reason": (
                        f'"{col}" has {pct:.1%} of values that are unusually high or low. '
                        f"These can skew the model if left untreated."
                    ),
                    "alternatives": [
                        {"id": "cap",    "label": "Cap at boundary",  "tradeoff": "Pull extreme values in to the edge of the normal range — usually the safest choice"},
                        {"id": "remove", "label": "Remove those rows", "tradeoff": "Delete rows with extreme values — use if they are clearly errors"},
                        {"id": "keep",   "label": "Keep as-is",        "tradeoff": "Leave them unchanged — choose if the extremes are real and meaningful"}
                    ]
                })
    else:
        # Record decisions made — handle grouped and individual decisions
        for key, value in decisions.items():
            if key == "id_col__grouped":
                label = "Excluded" if value == "exclude" else "Kept"
                for col in high_card_cols:
                    decisions_made.append({"column": col, "decision_type": "id_column", "value": value,
                                           "plain_english": f'{label} "{col}" (grouped ID column handling)'})
            elif key == "outliers__grouped":
                label = {"cap": "Cap at boundary", "remove": "Remove rows", "keep": "Keep as-is"}.get(value, value)
                for col in high_outlier_cols:
                    decisions_made.append({"column": col, "decision_type": "outlier_handling", "value": value,
                                           "plain_english": f'"{col}" outliers: {label} (grouped)'})
            else:
                col = key.split("__", 1)[-1]
                if key.startswith("id_col__"):
                    label = "Excluded" if value == "exclude" else "Kept"
                    decisions_made.append({"column": col, "decision_type": "id_column", "value": value,
                                           "plain_english": f'{label} "{col}" (ID column handling)'})
                elif key.startswith("outliers__"):
                    label = {"cap": "Cap at boundary", "remove": "Remove rows", "keep": "Keep as-is"}.get(value, value)
                    decisions_made.append({"column": col, "decision_type": "outlier_handling", "value": value,
                                           "plain_english": f'"{col}" outliers: {label}'})

    if decisions_required:
        summary = (
            f"We explored your data. {ov['plain_english']} "
            + (target_result.get("plain_english", ""))
            + f" We found {len(decisions_required)} thing(s) we need your input on before continuing."
        )
        return {
            "stage":             "eda",
            "status":            "decisions_required",
            "output_data_path":  str(data_path),
            "findings": {
                "overview":          ov,
                "target_analysis":   target_result,
                "feature_analysis":  features,
                "correlations":      correlations,
                "feature_vs_target": vs_target
            },
            "charts":             charts,
            "decisions_required": decisions_required,
            "decisions_made":     [],
            "advisories":         [f["advisory"] for f in features if "advisory" in f],
            "plain_english_summary": summary,
        }

    summary = (
        f"We explored your data. {ov['plain_english']} "
        + (target_result.get("plain_english", ""))
    )

    # Resolve column exclusions and outlier strategies from decisions
    exclude_cols     = [d["column"] for d in decisions_made if d["decision_type"] == "id_column"    and d["value"] == "exclude"]
    outlier_strategy = {d["column"]: d["value"] for d in decisions_made if d["decision_type"] == "outlier_handling"}

    return {
        "stage":             "eda",
        "status":            "success",
        "output_data_path":  str(data_path),
        "findings": {
            "overview":          ov,
            "target_analysis":   target_result,
            "feature_analysis":  features,
            "correlations":      correlations,
            "feature_vs_target": vs_target
        },
        "charts":             charts,
        "decisions_required": [],
        "decisions_made":     decisions_made,
        "advisories": [
            f["advisory"] for f in features if "advisory" in f
        ],
        "plain_english_summary": summary,
        "report_section": {
            "stage":   "eda",
            "title":   "Understanding Your Data",
            "summary": summary,
            "decision_made": (
                "; ".join([d["plain_english"] for d in decisions_made])
                if decisions_made else "Explored data — no column changes requested."
            ),
            "alternatives_considered": None,
            "why_this_matters": (
                "Understanding your data before modelling helps us make better decisions "
                "at every stage that follows."
            )
        },
        "config_updates": {
            "exclude_cols":      exclude_cols,
            "outlier_strategy":  outlier_strategy,
            "highly_correlated_pairs": correlations.get("high_corr_pairs", []),
            "high_outlier_cols":       high_outlier_cols,
            "skewed_cols":             skewed_cols,
            "high_cardinality_cols":   high_card_cols
        }
    }
