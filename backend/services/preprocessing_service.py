"""
preprocessing_service.py

Single entry point for all fit/transform logic.

Rules (non-negotiable):
  - .fit() is called ONLY on X_train.
  - .transform() on X_val and X_test applies the already-fitted pipeline.
  - The fitted pipeline is saved as preprocessor.pkl.
  - Transformed splits are saved as parquet.
  - This module never reads y — labels are passed through untouched.
  - This module is the ONLY place a preprocessing pipeline is fitted.
    No agent or runner may call .fit() on a transformer directly.
"""

from __future__ import annotations

import joblib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.schemas import RunSpec
from services.pipeline_compiler import build_pipeline


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class PreprocessingResult:
    """Paths and metadata produced after fit/transform."""

    def __init__(
        self,
        X_train_path: Path,
        X_val_path:   Path,
        X_test_path:  Path,
        preprocessor_path: Path,
        feature_names_out: list[str],
        n_features_in: int,
        n_features_out: int,
        warnings: list[str],
    ):
        self.X_train_path      = X_train_path
        self.X_val_path        = X_val_path
        self.X_test_path       = X_test_path
        self.preprocessor_path = preprocessor_path
        self.feature_names_out = feature_names_out
        self.n_features_in     = n_features_in
        self.n_features_out    = n_features_out
        self.warnings          = warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_transform(
    run_spec: RunSpec,
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    splits_dir: str | Path,
    models_dir: str | Path,
) -> PreprocessingResult:
    """
    Fit the preprocessing pipeline on X_train ONLY.
    Transform X_train, X_val, X_test.
    Save transformed splits and fitted pipeline.

    Parameters
    ----------
    run_spec    : RunSpec — provides preprocessing_plan
    X_train     : raw training features (unprocessed)
    X_val       : raw validation features
    X_test      : raw test features
    splits_dir  : directory to save transformed parquet files
    models_dir  : directory to save preprocessor.pkl

    Returns
    -------
    PreprocessingResult with paths and metadata
    """
    splits_dir = Path(splits_dir)
    models_dir = Path(models_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []

    # Align feature columns across all splits
    feature_cols = run_spec.feature_columns
    if feature_cols:
        missing_train = [c for c in feature_cols if c not in X_train.columns]
        if missing_train:
            warnings.append(f"Feature columns missing from training data: {missing_train}")
        X_train = _select_columns(X_train, feature_cols, "train")
        X_val   = _select_columns(X_val,   feature_cols, "val")
        X_test  = _select_columns(X_test,  feature_cols, "test")

    n_features_in = X_train.shape[1]

    # Build unfitted pipeline
    pipeline = build_pipeline(run_spec, X_train)

    # ── FIT on training data ONLY ──────────────────────────────────────────
    pipeline.fit(X_train)

    # ── TRANSFORM all splits ───────────────────────────────────────────────
    X_train_t = _safe_transform(pipeline, X_train, "train", warnings)
    X_val_t   = _safe_transform(pipeline, X_val,   "val",   warnings)
    X_test_t  = _safe_transform(pipeline, X_test,  "test",  warnings)

    # Recover output feature names
    feature_names_out = _get_feature_names(pipeline, X_train.columns.tolist())
    n_features_out = X_train_t.shape[1]

    # ── Save transformed splits as parquet ───────────────────────────────
    X_train_df = pd.DataFrame(X_train_t, columns=feature_names_out)
    X_val_df   = pd.DataFrame(X_val_t,   columns=feature_names_out)
    X_test_df  = pd.DataFrame(X_test_t,  columns=feature_names_out)

    X_train_path = splits_dir / "X_train.parquet"
    X_val_path   = splits_dir / "X_val.parquet"
    X_test_path  = splits_dir / "X_test.parquet"

    X_train_df.to_parquet(X_train_path, index=False)
    X_val_df.to_parquet(X_val_path,     index=False)
    X_test_df.to_parquet(X_test_path,   index=False)

    # ── Save CSV mirrors for backwards compatibility ──────────────────────
    # (existing agents and frontend may still load CSVs)
    X_train_df.to_csv(splits_dir / "X_train.csv", index=False)
    X_val_df.to_csv(splits_dir   / "X_val.csv",   index=False)
    X_test_df.to_csv(splits_dir  / "X_test.csv",  index=False)

    # ── Save fitted pipeline ──────────────────────────────────────────────
    preprocessor_path = models_dir / "preprocessor.pkl"
    joblib.dump(pipeline, preprocessor_path)

    # ── Save feature name metadata ────────────────────────────────────────
    feature_names_path = models_dir / "preprocessing_feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump({
            "input_features":  list(X_train.columns),
            "output_features": feature_names_out,
        }, f, indent=2)

    return PreprocessingResult(
        X_train_path=X_train_path,
        X_val_path=X_val_path,
        X_test_path=X_test_path,
        preprocessor_path=preprocessor_path,
        feature_names_out=feature_names_out,
        n_features_in=n_features_in,
        n_features_out=n_features_out,
        warnings=warnings,
    )


def load_preprocessor(models_dir: str | Path):
    """Load a previously fitted preprocessing pipeline."""
    path = Path(models_dir) / "preprocessor.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {path}. "
            "Run the splitting stage first to fit and save the preprocessor."
        )
    return joblib.load(path)


def transform_inference_batch(
    preprocessor,
    df: pd.DataFrame,
    feature_names_in: list[str],
) -> np.ndarray:
    """
    Apply a fitted preprocessor to an inference batch.

    Validates that all required columns are present and in the correct order
    before passing into the pipeline. Called by artifact_service.
    """
    missing = [c for c in feature_names_in if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required feature columns: {missing}. "
            f"Provide all {len(feature_names_in)} features listed in input_schema.json."
        )
    extra = [c for c in df.columns if c not in feature_names_in]
    if extra:
        # Extra columns are harmless — silently drop them
        df = df[feature_names_in]
    else:
        df = df[feature_names_in]

    return preprocessor.transform(df)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_columns(df: pd.DataFrame, feature_cols: list[str], split_name: str) -> pd.DataFrame:
    """Select only feature_cols from df; fill missing with NaN."""
    available = [c for c in feature_cols if c in df.columns]
    result = df[available].copy()
    # Add any missing columns as NaN (will be imputed by the pipeline)
    for col in feature_cols:
        if col not in result.columns:
            result[col] = np.nan
    return result[feature_cols]


def _safe_transform(pipeline, X: pd.DataFrame, split_name: str, warnings: list[str]) -> np.ndarray:
    """Transform X, catching and re-raising with a clear message."""
    try:
        return pipeline.transform(X)
    except Exception as e:
        raise RuntimeError(
            f"Preprocessing pipeline failed to transform {split_name} split: {e}"
        ) from e


def _get_feature_names(pipeline, input_columns: list[str]) -> list[str]:
    """
    Attempt to recover output feature names from the pipeline.
    Falls back to generic names if the pipeline does not support get_feature_names_out().
    """
    try:
        names = pipeline.get_feature_names_out()
        # sklearn ColumnTransformer prefixes transformer names — clean them up
        cleaned = []
        for n in names:
            n_str = str(n)
            # Remove "preprocessor__numeric__" or "preprocessor__categorical__onehot__" prefixes
            for prefix in ("numeric__", "categorical__", "preprocessor__"):
                n_str = n_str.replace(prefix, "")
            cleaned.append(n_str)
        return cleaned
    except Exception:
        # Fallback: generate generic feature_0, feature_1, ...
        col_transformer = pipeline.named_steps.get("preprocessor")
        if col_transformer is not None:
            try:
                names = col_transformer.get_feature_names_out()
                return [str(n) for n in names]
            except Exception:
                pass
        return [f"feature_{i}" for i in range(len(input_columns))]
