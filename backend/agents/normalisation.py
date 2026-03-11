"""
Normalisation Agent
Scales numeric features. Fits the scaler on the FULL features dataset
(consistent with STAGE_ORDER: normalisation before splitting).
The fitted scaler is saved for deployment use.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
)


# Task types that never need scaling
TASK_TYPES_NOT_REQUIRING_SCALING = [
    "time_series", "forecast", "time_series_forecast",
]

# Model options shown to the user — includes recommended scaling for each
MODEL_OPTIONS = [
    {"id": "random_forest",      "label": "Random Forest",                      "scaling": "none",     "group": "tree"},
    {"id": "xgboost",            "label": "XGBoost / Gradient Boosting",         "scaling": "none",     "group": "tree"},
    {"id": "decision_tree",      "label": "Decision Tree",                       "scaling": "none",     "group": "tree"},
    {"id": "linear_regression",  "label": "Linear / Ridge Regression",           "scaling": "standard", "group": "linear"},
    {"id": "logistic_regression","label": "Logistic Regression",                 "scaling": "standard", "group": "linear"},
    {"id": "knn",                "label": "K-Nearest Neighbours (KNN)",          "scaling": "standard", "group": "distance"},
    {"id": "svm",                "label": "Support Vector Machine (SVM)",        "scaling": "standard", "group": "distance"},
    {"id": "neural_network",     "label": "Neural Network",                      "scaling": "minmax",   "group": "neural"},
    {"id": "arima_prophet",      "label": "ARIMA / Prophet (time series model)", "scaling": "none",     "group": "timeseries"},
    {"id": "not_decided",        "label": "Not decided yet",                     "scaling": "standard", "group": "unknown"},
]
MODEL_SCALING_MAP = {m["id"]: m["scaling"] for m in MODEL_OPTIONS}
MODEL_GROUP_MAP   = {m["id"]: m["group"]   for m in MODEL_OPTIONS}

_SCALING_NONE_REASONS = {
    "tree":       "Tree-based models (like Random Forest and XGBoost) split on value thresholds — the scale of the data doesn't affect their decisions, so no scaling is needed.",
    "timeseries": "Time series models work directly on raw values. Scaling would make error metrics like MAE harder to interpret.",
    "unknown":    "No scaling applied.",
}

SCALING_STRATEGIES = {
    "standard": {
        "label":     "Standard scaling — centre around zero with consistent spread (recommended default)",
        "tradeoff":  "Works well for most data. Sensitive to outliers.",
        "best_for":  "Normally distributed data, linear models, neural networks",
    },
    "minmax": {
        "label":     "Min-Max scaling — squeeze all values into the range 0 to 1",
        "tradeoff":  "Bounded range. Very sensitive to outliers.",
        "best_for":  "Neural networks, when a bounded range is required",
    },
    "robust": {
        "label":     "Robust scaling — scales using the middle 50% of values, ignoring extremes",
        "tradeoff":  "Much less affected by outliers. Slightly less intuitive.",
        "best_for":  "Data with significant outliers",
    },
    "power": {
        "label":     "Power transformation — makes skewed distributions more symmetrical",
        "tradeoff":  "Effective for heavily skewed data. More complex transformation.",
        "best_for":  "Heavily skewed numeric columns before linear models",
    },
    "none": {
        "label":     "No scaling — leave values as they are",
        "tradeoff":  "Only appropriate for tree-based models (Random Forest, XGBoost).",
        "best_for":  "Tree-based models only",
    }
}

def _make_scaler(strategy: str):
    return {
        "standard": StandardScaler(),
        "minmax":   MinMaxScaler(),
        "robust":   RobustScaler(),
        "power":    PowerTransformer(method="yeo-johnson"),
        "none":     None
    }.get(strategy)


def _recommend_strategy(df: pd.DataFrame, numeric_cols: list, task_type: str = "") -> tuple:
    """Return (strategy_key, reason_string). Never raises — falls back to standard scaling."""
    task_type = (task_type or "").lower()

    # --- Task-type rules (take priority over data-driven heuristics) ---
    if any(t in task_type for t in TASK_TYPES_NOT_REQUIRING_SCALING):
        return (
            "none",
            "Time series models (ARIMA, Prophet) work directly on raw values. "
            "Scaling the data would make error metrics like MAE uninterpretable — "
            "they would no longer correspond to real units."
        )

    # --- Data-driven heuristics for classification / regression ---
    if not numeric_cols:
        return "standard", "Standard scaling is the most widely used approach."

    outlier_cols = []
    skewed_cols  = []
    for col in numeric_cols:
        if df[col].isnull().all():
            continue
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR    = Q3 - Q1
        if IQR > 0:
            out_pct = float(((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).mean())
            if out_pct > 0.05:
                outlier_cols.append(col)
        try:
            skew = abs(float(df[col].skew()))
            if skew > 1.5:
                skewed_cols.append(col)
        except Exception:
            pass

    if len(outlier_cols) > len(numeric_cols) * 0.3:
        return "robust", f"{len(outlier_cols)} of your columns have significant extreme values. Robust scaling handles these well."
    elif len(skewed_cols) > len(numeric_cols) * 0.3:
        return "power",  f"{len(skewed_cols)} of your columns have skewed distributions. Power transformation will make them more symmetrical."
    else:
        return "standard", "Standard scaling is the most widely used approach and works well for your data."


def _identify_skip_cols(df: pd.DataFrame, target_col: str) -> list:
    skip = []
    for col in df.columns:
        if col == target_col:
            skip.append(col)
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1, 0.0, 1.0}):
            skip.append(col)
    return skip


def _plot_comparison(before: pd.DataFrame, after: pd.DataFrame,
                     cols: list, output_dir: str, n: int = 4) -> str:
    cols_to_plot = [c for c in cols if c in before.columns and c in after.columns][:n]
    if not cols_to_plot:
        return ""

    fig, axes = plt.subplots(2, len(cols_to_plot), figsize=(4 * len(cols_to_plot), 6))
    if len(cols_to_plot) == 1:
        axes = axes.reshape(2, 1)

    for i, col in enumerate(cols_to_plot):
        axes[0, i].hist(before[col].dropna(), bins=30, color="#AECDE8", edgecolor="white")
        axes[0, i].set_title(f"{col}\nBefore scaling", fontsize=9)
        axes[1, i].hist(after[col].dropna(), bins=30, color="#2E75B6", edgecolor="white")
        axes[1, i].set_title("After scaling", fontsize=9)

    plt.suptitle(
        "How scaling changed your numeric columns\n"
        "The shape stays the same — only the numbers on the axis change.",
        fontsize=11
    )
    plt.tight_layout()
    path = Path(output_dir) / "scaling_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", dpi=100)
    plt.close()
    return str(path)


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def run(session: dict, decisions: dict) -> dict:
    session_id   = session["session_id"]
    target_col   = session["goal"].get("target_column")
    task_type    = session["goal"].get("task_type", "")
    sessions_dir = Path("sessions")
    session_dir  = sessions_dir / session_id

    data_path = session_dir / "data" / "interim" / "features.csv"
    if not data_path.exists():
        return {
            "stage":                 "normalisation",
            "status":                "failed",
            "plain_english_summary": "No feature-engineered data found. Please run the feature engineering stage first."
        }

    df = pd.read_csv(data_path, low_memory=False)

    skip_cols    = _identify_skip_cols(df, target_col)
    numeric_cols = [
        c for c in df.select_dtypes("number").columns
        if c not in skip_cols and c != target_col
    ]

    # Check for stored model preference from a previous session choice
    stored_model = session.get("config", {}).get("model_preference")
    model_preference = decisions.get("model_preference") or stored_model

    # ── Phase 1: Ask which model the user is planning to use ──────────────────
    if not model_preference and "scaling_strategy" not in decisions:
        return {
            "stage":  "normalisation",
            "status": "decisions_required",
            "decisions_required": [{
                "id":                    "model_preference",
                "question":              "Which model are you planning to use?",
                "recommendation":        None,
                "recommendation_reason": (
                    "Your choice of model affects which scaling approach is best. "
                    "Some models (like Random Forest) don't need scaling at all."
                ),
                "alternatives": [
                    {"id": m["id"], "label": m["label"],
                     "tradeoff": _SCALING_NONE_REASONS.get(m["group"], "")
                                 or SCALING_STRATEGIES.get(m["scaling"], {}).get("best_for", "")}
                    for m in MODEL_OPTIONS
                ]
            }],
            "plain_english_summary": (
                "Before we scale your data, we need to know which model you're planning to use — "
                "different models have different requirements."
            )
        }

    # ── Phase 2: Recommend scaling based on chosen model ─────────────────────
    if model_preference and "scaling_strategy" not in decisions:
        model_rec = MODEL_SCALING_MAP.get(model_preference, "standard")
        model_grp = MODEL_GROUP_MAP.get(model_preference, "unknown")

        if model_rec == "none":
            rec    = "none"
            reason = _SCALING_NONE_REASONS.get(model_grp, "This model type doesn't require scaling.")
            intro  = (
                "Based on your model choice, scaling is not required for this pipeline. "
                "You can still choose a different approach below if you prefer."
            )
        elif model_rec == "minmax":
            rec    = "minmax"
            reason = "Neural networks work best when inputs are in a bounded 0-to-1 range. Min-Max scaling achieves this."
            intro  = (
                "Neural networks require inputs in a bounded range. We'll scale your columns to 0-to-1."
            )
        else:
            # model suggests standard — let data heuristics refine (robust/power if needed)
            try:
                data_rec, data_reason = _recommend_strategy(df, numeric_cols, task_type)
            except Exception:
                data_rec, data_reason = "standard", "Standard scaling is the most widely used approach."
            rec    = data_rec  # may be robust/power if data warrants it
            reason = data_reason
            intro  = (
                "Some of your columns have very different value ranges. "
                "Here is what we recommend based on your model choice and data:"
            )

        return {
            "stage":  "normalisation",
            "status": "decisions_required",
            "model_preference": model_preference,
            "decisions_required": [{
                "id":                    "scaling_strategy",
                "question":              "How would you like to scale your numeric columns?",
                "recommendation":        rec,
                "recommendation_reason": reason,
                "alternatives": [
                    {"id": k, **{kk: vv for kk, vv in v.items()}}
                    for k, v in SCALING_STRATEGIES.items()
                ]
            }],
            "plain_english_summary": intro
        }

    strategy = decisions.get("scaling_strategy") or "standard"

    if not numeric_cols or strategy == "none":
        # No scaling — just copy features.csv to scaled.csv
        output_path = session_dir / "data" / "processed" / "scaled.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return {
            "stage":            "normalisation",
            "status":           "success",
            "scaling_strategy": "none",
            "columns_scaled":   [],
            "columns_skipped":  numeric_cols,
            "scaler_path":      None,
            "chart_path":       None,
            "output_data_path": str(output_path),
            "decisions_required": [],
            "decisions_made":   [{"decision": "scaling_strategy", "chosen": "none"}],
            "plain_english_summary": "No scaling applied — the model you are using does not require it.",
            "report_section": {
                "stage":   "normalisation",
                "title":   "Scaling Your Data",
                "summary": "No scaling applied.",
                "decision_made": "Scaling skipped — tree-based models do not require it.",
                "alternatives_considered": "Standard, Min-Max, and Robust scaling were available.",
                "why_this_matters": "Tree-based models handle different value scales natively."
            },
            "config_updates": {
                "scaling_strategy": "none",
                "scaled_columns":   [],
                "scaler_path":      None,
                "model_preference": model_preference
            }
        }

    scaler = _make_scaler(strategy)
    df_before = df.copy()

    scaler.fit(df[numeric_cols])
    df_after        = df.copy()
    df_after[numeric_cols] = scaler.transform(df[numeric_cols])

    # Save scaler
    models_dir  = session_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = models_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save scaled full dataset
    output_path = session_dir / "data" / "processed" / "scaled.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_after.to_csv(output_path, index=False)

    # Comparison chart
    reports_dir = str(session_dir / "reports")
    chart_path  = _plot_comparison(df_before, df_after, numeric_cols, reports_dir)

    msg = (
        f"Scaled {len(numeric_cols)} numeric column(s) using {SCALING_STRATEGIES[strategy]['label']}. "
        f"The shape of your data has not changed — only the numbers on the axis. "
        f"The scaler has been saved for use when the model is deployed."
    )

    return {
        "stage":             "normalisation",
        "status":            "success",
        "scaling_strategy":  strategy,
        "columns_scaled":    numeric_cols,
        "columns_skipped":   skip_cols,
        "scaler_path":       str(scaler_path),
        "chart_path":        chart_path,
        "output_data_path":  str(output_path),
        "decisions_required": [],
        "decisions_made":    [{"decision": "scaling_strategy", "chosen": strategy}],
        "plain_english_summary": msg,
        "report_section": {
            "stage":   "normalisation",
            "title":   "Scaling Your Data",
            "summary": msg,
            "decision_made": f"Applied {SCALING_STRATEGIES[strategy]['label']}.",
            "alternatives_considered": "Standard, Min-Max, Robust, and Power scaling were available.",
            "why_this_matters": (
                "Scaling ensures the model treats all columns fairly — a column with values in the "
                "thousands will not overshadow a column with values between 0 and 1 simply because "
                "its numbers are bigger."
            )
        },
        "config_updates": {
            "scaling_strategy": strategy,
            "scaled_columns":   numeric_cols,
            "scaler_path":      str(scaler_path),
            "model_preference": model_preference
        }
    }
