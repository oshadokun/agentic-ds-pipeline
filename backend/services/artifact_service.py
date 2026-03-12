"""
artifact_service.py

Packages the complete inference artifact bundle for deployment.

Bundle contents:
  model.pkl                       — fitted sklearn-compatible model
  preprocessor.pkl                — fitted preprocessing pipeline
  feature_names.json              — ordered list of input feature names
  run_spec.json                   — full RunSpec (task_family, metrics, etc.)
  input_schema.json               — per-column dtype and allowed range metadata
  training_column_order.json      — exact column order fed into the preprocessor
  preprocessing_feature_names.json — output feature names after transformation
  target_schema.json              — target column name, dtype, label mapping

Inference validation rule (mandatory):
  Any dataset passed into deployment or monitoring MUST be validated against
  input_schema.json before reaching preprocessor.pkl. If columns are missing,
  extra, or wrong dtype, raise a descriptive error with actionable guidance.

This module does NOT generate the FastAPI app.py — that remains in deployment.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from contracts.schemas import RunSpec, EvaluationPayload
from services.preprocessing_service import transform_inference_batch


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ArtifactBundle:
    """Paths to all packaged artifacts."""

    def __init__(self, artifact_dir: Path):
        self.artifact_dir = artifact_dir

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / "model.pkl"

    @property
    def preprocessor_path(self) -> Path:
        return self.artifact_dir / "preprocessor.pkl"

    @property
    def input_schema_path(self) -> Path:
        return self.artifact_dir / "input_schema.json"

    @property
    def run_spec_path(self) -> Path:
        return self.artifact_dir / "run_spec.json"

    @property
    def feature_names_path(self) -> Path:
        return self.artifact_dir / "feature_names.json"

    @property
    def target_schema_path(self) -> Path:
        return self.artifact_dir / "target_schema.json"

    def to_dict(self) -> dict:
        return {
            "artifact_dir":      str(self.artifact_dir),
            "model_path":        str(self.model_path),
            "preprocessor_path": str(self.preprocessor_path),
            "input_schema_path": str(self.input_schema_path),
            "run_spec_path":     str(self.run_spec_path),
            "feature_names_path":str(self.feature_names_path),
            "target_schema_path":str(self.target_schema_path),
        }


# ---------------------------------------------------------------------------
# Public: package artifacts
# ---------------------------------------------------------------------------

def package(
    run_spec: RunSpec,
    eval_payload: EvaluationPayload,
    model_source_path: str | Path,
    preprocessor_source_path: str | Path,
    X_train_sample: pd.DataFrame,
    artifact_dir: str | Path,
) -> ArtifactBundle:
    """
    Copy model and preprocessor into artifact_dir and generate metadata files.

    Parameters
    ----------
    run_spec                  : RunSpec for this run
    eval_payload              : frozen EvaluationPayload (for metrics/label mapping)
    model_source_path         : path to the trained model .pkl
    preprocessor_source_path  : path to the fitted preprocessor .pkl
    X_train_sample            : small sample of preprocessed training features
                                used to infer input_schema dtype ranges
    artifact_dir              : destination directory for the bundle

    Returns
    -------
    ArtifactBundle with all paths populated
    """
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Copy core binaries ────────────────────────────────────────────────
    shutil.copy2(model_source_path,        out / "model.pkl")
    shutil.copy2(preprocessor_source_path, out / "preprocessor.pkl")

    # ── Save run_spec ─────────────────────────────────────────────────────
    run_spec.save(out / "run_spec.json")

    # ── Build input_schema.json ───────────────────────────────────────────
    feature_cols = run_spec.feature_columns or list(X_train_sample.columns)
    input_schema = _build_input_schema(X_train_sample, feature_cols)
    with open(out / "input_schema.json", "w") as f:
        json.dump(input_schema, f, indent=2)

    # ── feature_names.json ────────────────────────────────────────────────
    with open(out / "feature_names.json", "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    # ── training_column_order.json ────────────────────────────────────────
    with open(out / "training_column_order.json", "w") as f:
        json.dump({"column_order": feature_cols}, f, indent=2)

    # ── preprocessing_feature_names.json (output side) ───────────────────
    preprocessor = joblib.load(out / "preprocessor.pkl")
    try:
        from services.preprocessing_service import _get_feature_names
        output_features = _get_feature_names(preprocessor, feature_cols)
    except Exception:
        output_features = feature_cols
    with open(out / "preprocessing_feature_names.json", "w") as f:
        json.dump({"output_features": output_features}, f, indent=2)

    # ── target_schema.json ────────────────────────────────────────────────
    target_schema = _build_target_schema(run_spec, eval_payload)
    with open(out / "target_schema.json", "w") as f:
        json.dump(target_schema, f, indent=2)

    # ── Summary manifest ─────────────────────────────────────────────────
    with open(out / "artifact_manifest.json", "w") as f:
        bundle_meta = {
            "run_id":      run_spec.run_id,
            "session_id":  run_spec.session_id,
            "task_family": run_spec.task_family,
            "model_id":    eval_payload.model_id,
            "primary_metric": eval_payload.primary_metric,
            "primary_metric_value": eval_payload.primary_metric_value,
            "verdict":     eval_payload.verdict,
            "n_input_features": len(feature_cols),
            "n_output_features": len(output_features),
        }
        json.dump(bundle_meta, f, indent=2)

    return ArtifactBundle(out)


# ---------------------------------------------------------------------------
# Public: validate inference input
# ---------------------------------------------------------------------------

def validate_inference_input(
    df: pd.DataFrame,
    input_schema: dict,
    strict_dtypes: bool = False,
) -> pd.DataFrame:
    """
    Validate an inference batch against input_schema.json before preprocessing.

    Parameters
    ----------
    df            : incoming DataFrame (one or more rows)
    input_schema  : loaded from input_schema.json
    strict_dtypes : if True, raise on dtype mismatches; else coerce

    Returns
    -------
    Validated (and optionally coerced) DataFrame, column-aligned to schema order

    Raises
    ------
    ValueError with a descriptive, actionable message
    """
    required_columns: list[str] = [f["name"] for f in input_schema["features"]]
    schema_map: dict[str, dict] = {f["name"]: f for f in input_schema["features"]}

    # ── Missing columns ───────────────────────────────────────────────────
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required column(s): {missing}. "
            f"Provide all {len(required_columns)} features listed in input_schema.json."
        )

    # ── Extra columns ─────────────────────────────────────────────────────
    extra = [c for c in df.columns if c not in schema_map]
    if extra:
        # Silently drop extra columns
        df = df.drop(columns=extra)

    # ── Reorder to match training column order ────────────────────────────
    df = df[required_columns]

    # ── Dtype coercion / validation ───────────────────────────────────────
    for feat in input_schema["features"]:
        col  = feat["name"]
        kind = feat.get("dtype_kind")  # "numeric" or "categorical"
        if kind == "numeric":
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                if strict_dtypes:
                    raise ValueError(
                        f"Column '{col}' could not be converted to numeric. "
                        f"Expected dtype: numeric."
                    )

    return df


def load_inference_bundle(artifact_dir: str | Path) -> dict:
    """
    Load model, preprocessor, and schema metadata from an artifact bundle.

    Returns dict with keys: model, preprocessor, input_schema, run_spec,
                            feature_names, target_schema
    """
    p = Path(artifact_dir)
    model        = joblib.load(p / "model.pkl")
    preprocessor = joblib.load(p / "preprocessor.pkl")

    with open(p / "input_schema.json")  as f: input_schema = json.load(f)
    with open(p / "feature_names.json") as f: feature_names = json.load(f)
    with open(p / "target_schema.json") as f: target_schema = json.load(f)
    with open(p / "run_spec.json")      as f: run_spec_dict = json.load(f)

    return {
        "model":         model,
        "preprocessor":  preprocessor,
        "input_schema":  input_schema,
        "feature_names": feature_names["features"],
        "target_schema": target_schema,
        "run_spec":      run_spec_dict,
    }


def predict(
    df: pd.DataFrame,
    bundle: dict,
) -> dict:
    """
    Make a prediction using a loaded bundle.

    Validates input against input_schema before preprocessing.
    """
    input_schema  = bundle["input_schema"]
    preprocessor  = bundle["preprocessor"]
    model         = bundle["model"]
    feature_names = bundle["feature_names"]
    task_family   = bundle["run_spec"].get("task_family", "")

    # Validate input
    df_valid = validate_inference_input(df, input_schema)

    # Preprocess
    X_transformed = transform_inference_batch(preprocessor, df_valid, feature_names)

    # Predict
    predictions = model.predict(X_transformed)
    result: dict[str, Any] = {"predictions": predictions.tolist()}

    if "classification" in task_family and hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_transformed)
            result["probabilities"] = probas.tolist()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_input_schema(X_sample: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Build per-column dtype and range metadata from a training data sample."""
    features = []
    for col in feature_cols:
        if col not in X_sample.columns:
            features.append({"name": col, "dtype_kind": "unknown", "nullable": True})
            continue

        series = X_sample[col].dropna()
        if pd.api.types.is_numeric_dtype(X_sample[col]):
            feat: dict[str, Any] = {
                "name":       col,
                "dtype_kind": "numeric",
                "dtype":      str(X_sample[col].dtype),
                "nullable":   bool(X_sample[col].isna().any()),
                "min":        float(series.min()) if len(series) > 0 else None,
                "max":        float(series.max()) if len(series) > 0 else None,
            }
        else:
            uniq = series.unique().tolist()
            feat = {
                "name":       col,
                "dtype_kind": "categorical",
                "dtype":      str(X_sample[col].dtype),
                "nullable":   bool(X_sample[col].isna().any()),
                "categories": [str(u) for u in uniq[:50]],  # cap at 50
            }
        features.append(feat)

    return {
        "schema_version": "1.0",
        "n_features":     len(feature_cols),
        "features":       features,
    }


def _build_target_schema(run_spec: RunSpec, eval_payload: EvaluationPayload) -> dict:
    return {
        "target_column":  run_spec.target_column,
        "task_family":    run_spec.task_family,
        "primary_metric": eval_payload.primary_metric,
        "label_order":    eval_payload.label_order,
        "class_mapping":  eval_payload.class_mapping,
        "threshold_used": eval_payload.threshold_used,
    }
