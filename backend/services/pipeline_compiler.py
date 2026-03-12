"""
pipeline_compiler.py

Two responsibilities:

1. compile_run_spec()
   Compiles a RunSpec from the DatasetManifest, validated task_family,
   and user-confirmed decisions collected during cleaning / FE / normalisation.
   This is the single place that assembles the preprocessing_plan.

2. build_pipeline()
   Builds a fitted-or-unfitted sklearn Pipeline from a RunSpec's preprocessing_plan.
   Enforces the deterministic-vs-learned separation:
     - Deterministic transforms (date parts, boolean coercions) are assumed
       already applied to the DataFrame before the Pipeline is built.
     - Learned transforms (imputation, encoding, scaling) are placed INSIDE
       the Pipeline so they are fit on training data only via preprocessing_service.

Usage:
  run_spec = pipeline_compiler.compile_run_spec(manifest, routing_result, decisions)
  pipeline = pipeline_compiler.build_pipeline(run_spec, X_train)
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
)

from contracts.schemas import DatasetManifest, RunSpec
from services.task_router import SUPPORTED_TASK_FAMILIES

# ---------------------------------------------------------------------------
# Encoding strategies that must be post-split (learned)
# ---------------------------------------------------------------------------
_LEARNED_ENCODERS = {"target", "frequency"}
_DETERMINISTIC_ENCODERS = {"onehot", "label", "ordinal", "drop"}

# Scaling strategy → sklearn scaler
_SCALER_MAP = {
    "standard": StandardScaler,
    "minmax":   MinMaxScaler,
    "robust":   RobustScaler,
    "power":    lambda: PowerTransformer(method="yeo-johnson"),
    "none":     None,
}


# ---------------------------------------------------------------------------
# compile_run_spec
# ---------------------------------------------------------------------------

def compile_run_spec(
    session_id: str,
    manifest: DatasetManifest,
    routing_result: dict,
    decisions: dict[str, Any],
) -> RunSpec:
    """
    Build and return a RunSpec.

    Parameters
    ----------
    session_id      : the current session UUID
    manifest        : DatasetManifest from manifest_builder
    routing_result  : dict returned by task_router.resolve()
    decisions       : flattened dict of all user-confirmed decisions
                      collected from cleaning / FE / normalisation stages.
                      Keys used:
                        impute_{col}        → imputation strategy for col
                        outlier_{col}       → outlier strategy for col
                        encode_{col}        → encoding strategy for col
                        ordinal_order_{col} → list of ordered categories
                        scaling_strategy    → str
                        feature_columns     → list[str]
                        drop_columns        → list[str]
                        balance_classes     → "smote" | "undersample" | "class_weights" | "none"
                        model_selection     → model_id string
    """
    task_family = routing_result["task_family"]

    # --- Feature columns ---------------------------------------------------
    feature_columns: list[str] = decisions.get("feature_columns", [])
    drop_columns: list[str] = decisions.get("drop_columns", [])

    # --- Imputation strategies -------------------------------------------
    imputation_strategies: dict[str, str] = {
        k.replace("missing_", ""): v
        for k, v in decisions.items()
        if k.startswith("missing_")
    }

    # --- Outlier strategies -----------------------------------------------
    outlier_strategies: dict[str, str] = {
        k.replace("outlier_", "", 1): v
        for k, v in decisions.items()
        if k.startswith("outlier_")
    }

    # --- Encoding strategies ---------------------------------------------
    encoding_strategies: dict[str, str] = {
        k.replace("encode_", ""): v
        for k, v in decisions.items()
        if k.startswith("encode_")
    }
    ordinal_orders: dict[str, list] = {
        k.replace("ordinal_order_", ""): v
        for k, v in decisions.items()
        if k.startswith("ordinal_order_")
    }

    # --- Scaling ---------------------------------------------------------
    scaling_strategy: str = decisions.get("scaling_strategy", "standard")
    if task_family == "time_series":
        scaling_strategy = "none"  # hard rule

    # --- Resampling ------------------------------------------------------
    _CLASSIFICATION_FAMILIES = {"binary_classification", "multiclass_classification"}
    resampling_plan: dict | None = None
    if task_family in _CLASSIFICATION_FAMILIES:
        balance_choice = decisions.get("balance_classes", "none")
        resampling_plan = {
            "strategy": balance_choice,
            "min_class_count": None,  # filled at training time
        }

    # --- Model candidates ------------------------------------------------
    model_candidates: list[str] = _default_model_candidates(task_family, manifest.row_count)
    selected_model = decisions.get("model_selection")

    # --- Assemble -----------------------------------------------------------
    preprocessing_plan = {
        "imputation_strategies": imputation_strategies,
        "outlier_strategies":    outlier_strategies,
        "encoding_strategies":   encoding_strategies,
        "ordinal_orders":        ordinal_orders,
        "scaling_strategy":      scaling_strategy,
        "feature_columns":       feature_columns,
        "drop_columns":          drop_columns,
    }

    run_spec = RunSpec(
        run_id=str(uuid.uuid4()),
        session_id=session_id,
        task_family=task_family,
        target_column=decisions.get("target_column") or manifest.candidate_target_columns[0]
            if manifest.candidate_target_columns else None,
        time_column=decisions.get("time_column"),
        split_strategy=routing_result["split_strategy"],
        random_seed=42,
        preprocessing_plan=preprocessing_plan,
        resampling_plan=resampling_plan,
        model_candidates=model_candidates,
        selected_model_id=selected_model,
        primary_metric=routing_result["primary_metric"],
        secondary_metrics=routing_result["secondary_metrics"],
        compiled_at=datetime.datetime.utcnow().isoformat(),
        compiler_version="1.0",
    )

    return run_spec


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

def build_pipeline(
    run_spec: RunSpec,
    X_reference: pd.DataFrame,
) -> Pipeline:
    """
    Build an unfitted sklearn Pipeline from RunSpec.preprocessing_plan.

    The returned Pipeline is NOT yet fitted — call preprocessing_service.fit()
    to fit it on training data only.

    Parameters
    ----------
    run_spec      : RunSpec with preprocessing_plan populated
    X_reference   : DataFrame used only to determine column names and dtypes.
                    Must NOT include the target column.
                    This is typically X_train (after split) but can be any
                    representative sample — no fitting happens here.
    """
    plan = run_spec.preprocessing_plan
    feature_cols = plan.get("feature_columns") or list(X_reference.columns)
    # Remove any columns that were dropped
    drop_cols = set(plan.get("drop_columns", []))
    feature_cols = [c for c in feature_cols if c in X_reference.columns and c not in drop_cols]

    if not feature_cols:
        raise ValueError("Pipeline compiler: no feature columns available to build pipeline.")

    X_ref = X_reference[feature_cols]

    # Separate columns by type for ColumnTransformer
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(X_ref[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(X_ref[c])
    ]

    transformers = []

    # ── Numeric branch ──────────────────────────────────────────────────────
    if numeric_cols:
        num_steps = _build_numeric_steps(plan, numeric_cols, run_spec.scaling_strategy)
        transformers.append(("numeric", Pipeline(num_steps), numeric_cols))

    # ── Categorical branch ─────────────────────────────────────────────────
    if categorical_cols:
        cat_steps = _build_categorical_steps(plan, categorical_cols)
        transformers.append(("categorical", Pipeline(cat_steps), categorical_cols))

    if not transformers:
        raise ValueError("Pipeline compiler: no transformers could be built.")

    col_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",           # drop any columns not in feature_cols
        sparse_threshold=0,         # always return dense array
        n_jobs=1,
    )

    # The ColumnTransformer IS the pipeline (no further steps needed here;
    # the scaler is embedded in the numeric branch)
    pipeline = Pipeline([("preprocessor", col_transformer)])

    # Store metadata for downstream use
    pipeline._feature_columns = feature_cols  # type: ignore[attr-defined]
    pipeline._numeric_columns = numeric_cols   # type: ignore[attr-defined]
    pipeline._categorical_columns = categorical_cols  # type: ignore[attr-defined]

    return pipeline


# ---------------------------------------------------------------------------
# Internal step builders
# ---------------------------------------------------------------------------

def _build_numeric_steps(
    plan: dict,
    numeric_cols: list[str],
    scaling_strategy: str,
) -> list[tuple]:
    """Build the sequence of steps for numeric columns."""
    imputation_strategies = plan.get("imputation_strategies", {})
    outlier_strategies    = plan.get("outlier_strategies", {})

    # Check if KNN imputation is requested for any numeric column
    use_knn = any(
        imputation_strategies.get(c) == "knn"
        for c in numeric_cols
    )

    # For simplicity, use KNNImputer if any column requests it,
    # otherwise use SimpleImputer with the most common strategy.
    # Column-specific strategies are approximated: the first dominant strategy wins.
    if use_knn:
        imputer = KNNImputer(n_neighbors=5)
    else:
        # Pick the most frequently requested strategy, or median as default
        strategy_counts: dict[str, int] = {}
        for c in numeric_cols:
            s = imputation_strategies.get(c, "median")
            if s in ("mean", "median"):  # SimpleImputer supports these
                strategy_counts[s] = strategy_counts.get(s, 0) + 1
        strategy = max(strategy_counts, key=lambda k: strategy_counts[k]) if strategy_counts else "median"
        imputer = SimpleImputer(strategy=strategy)

    steps: list[tuple] = [("imputer", imputer)]

    # Outlier capping — implemented as a custom transformer
    # that caps at IQR boundaries computed ONLY on training data
    steps.append(("outlier_capper", _IQRCapper(outlier_strategies, numeric_cols)))

    # Scaler
    if scaling_strategy and scaling_strategy != "none":
        scaler_cls = _SCALER_MAP.get(scaling_strategy)
        if scaler_cls is not None:
            scaler = scaler_cls() if isinstance(scaler_cls, type) else scaler_cls()
            steps.append(("scaler", scaler))

    return steps


def _build_categorical_steps(
    plan: dict,
    categorical_cols: list[str],
) -> list[tuple]:
    """Build the sequence of steps for categorical columns."""
    encoding_strategies = plan.get("encoding_strategies", {})
    ordinal_orders      = plan.get("ordinal_orders", {})

    # Separate by encoding type
    onehot_cols  = []
    ordinal_cols = []
    drop_cols    = []

    for col in categorical_cols:
        strategy = encoding_strategies.get(col, "onehot")
        if strategy in _LEARNED_ENCODERS:
            # Target/frequency encoding should have been collected as decisions
            # and will be handled post-split — fall back to onehot here
            onehot_cols.append(col)
        elif strategy == "ordinal":
            ordinal_cols.append(col)
        elif strategy == "drop":
            drop_cols.append(col)
        else:
            onehot_cols.append(col)  # onehot or label → onehot is safer in Pipeline

    # Impute before encoding (fill missing with constant "__missing__")
    imputer = SimpleImputer(strategy="constant", fill_value="__missing__")

    # Build a nested ColumnTransformer for encoding
    enc_transformers = []
    if onehot_cols:
        enc_transformers.append((
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            # Note: indices relative to the output of the imputer
            # We use column names via make_column_selector approach
            _col_indices(categorical_cols, onehot_cols),
        ))
    if ordinal_cols:
        categories = [
            ordinal_orders.get(c, "auto") for c in ordinal_cols
        ]
        enc_transformers.append((
            "ordinal",
            OrdinalEncoder(
                categories=categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            _col_indices(categorical_cols, ordinal_cols),
        ))
    if drop_cols:
        # remainder="drop" on the outer ColumnTransformer handles this,
        # but we explicitly handle it here to avoid confusion
        pass

    if not enc_transformers:
        # All categorical columns are being dropped — just impute
        return [("imputer", imputer)]

    # When there is only one encoding type and no drops, skip the nested transformer
    if len(enc_transformers) == 1 and not drop_cols:
        name, encoder, indices = enc_transformers[0]
        return [("imputer", imputer), ("encoder", encoder)]

    # Multiple encoding types → nested ColumnTransformer
    enc_ct = ColumnTransformer(enc_transformers, remainder="drop", sparse_threshold=0)
    return [("imputer", imputer), ("encoder", enc_ct)]


def _col_indices(all_cols: list[str], subset: list[str]) -> list[int]:
    """Return integer indices of subset within all_cols."""
    return [all_cols.index(c) for c in subset if c in all_cols]


# ---------------------------------------------------------------------------
# Custom transformer: IQR outlier capper (fit on train only)
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class _IQRCapper(BaseEstimator, TransformerMixin):
    """
    Caps outliers at IQR boundaries computed during .fit() (training data only).

    Only applies capping to columns with strategy == "cap".
    Columns with strategy == "remove" are capped here (row removal
    cannot be done inside a Pipeline transformer since it changes
    the number of rows — row removal must happen before split in
    structural cleaning).
    Columns with strategy == "keep" are passed through unchanged.
    """

    def __init__(self, outlier_strategies: dict[str, str], numeric_cols: list[str]):
        self.outlier_strategies = outlier_strategies
        self.numeric_cols = numeric_cols

    def fit(self, X, y=None):
        # X here is a numpy array (output of the imputer step)
        X_arr = np.asarray(X, dtype=float)
        self.lower_bounds_: dict[int, float] = {}
        self.upper_bounds_: dict[int, float] = {}

        for i, col in enumerate(self.numeric_cols):
            strategy = self.outlier_strategies.get(col, "keep")
            if strategy in ("cap", "remove"):
                q1, q3 = np.nanpercentile(X_arr[:, i], [25, 75])
                iqr = q3 - q1
                self.lower_bounds_[i] = float(q1 - 1.5 * iqr)
                self.upper_bounds_[i] = float(q3 + 1.5 * iqr)
        return self

    def transform(self, X, y=None):
        X_arr = np.asarray(X, dtype=float).copy()
        for i in self.lower_bounds_:
            X_arr[:, i] = np.clip(X_arr[:, i], self.lower_bounds_[i], self.upper_bounds_[i])
        return X_arr

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return np.array(self.numeric_cols, dtype=object)


# ---------------------------------------------------------------------------
# Model candidate defaults
# ---------------------------------------------------------------------------

def _default_model_candidates(task_family: str, n_rows: int) -> list[str]:
    if task_family == "binary_classification":
        candidates = ["logistic_regression"]
        if n_rows > 500:
            candidates.append("random_forest")
        if n_rows > 1000:
            candidates.append("xgboost")
        return candidates

    elif task_family == "multiclass_classification":
        candidates = ["logistic_regression_multi", "random_forest"]
        if n_rows > 1000:
            candidates.append("xgboost")
        return candidates

    elif task_family == "regression":
        candidates = ["ridge"]
        if n_rows > 500:
            candidates.append("random_forest_regressor")
        if n_rows > 1000:
            candidates.append("xgboost_regressor")
        return candidates

    elif task_family == "time_series":
        candidates = ["random_forest_regressor", "ridge", "arima"]
        if n_rows > 1000:
            candidates.insert(1, "xgboost_regressor")
        return candidates

    return []
