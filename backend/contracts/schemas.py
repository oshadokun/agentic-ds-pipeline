"""
Core data contracts for the compiler-driven ML pipeline.

These dataclasses are the single source of truth for:
  - DatasetManifest  : what was observed about the raw dataset
  - RunSpec          : compiled execution plan (set once during validation, read everywhere else)
  - EvaluationPayload: frozen metrics from one evaluation pass

Rules:
  - RunSpec is produced by pipeline_compiler during the validation stage.
  - No downstream stage may re-infer task_family, models, metrics, split logic,
    or resampling policy. They must read from RunSpec.
  - EvaluationPayload is produced by evaluation_service and is the only source
    of metrics. No widget or stage may recompute predictions independently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------

@dataclass
class DatasetManifest:
    """
    Describes the raw uploaded dataset.
    Produced by manifest_builder, consumed by task_router and pipeline_compiler.
    """
    dataset_id: str
    row_count: int
    column_count: int

    # Column type lists
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)

    # Candidate columns for special roles
    candidate_target_columns: list[str] = field(default_factory=list)
    candidate_time_columns: list[str] = field(default_factory=list)

    # Missingness
    missingness_summary: dict[str, float] = field(default_factory=dict)  # col → fraction missing
    duplicate_row_count: int = 0

    # Quality flags
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    possible_leakage_columns: list[str] = field(default_factory=list)

    # Target analysis (populated when target_column is known)
    target_distribution: dict[str, Any] | None = None

    # Task routing hints
    task_hypotheses: list[str] = field(default_factory=list)  # ordered by confidence

    # Warnings for the user
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "DatasetManifest":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# ---------------------------------------------------------------------------
# RunSpec
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """
    Compiled execution plan.

    Produced ONCE by pipeline_compiler during the validation stage.
    Saved to run_spec.json.

    Every downstream stage reads from this. No stage may re-infer task_family,
    model choices, metrics, split logic, or resampling policy after this point.
    """
    run_id: str
    session_id: str

    # Task identity — set once, never changed downstream
    task_family: str          # "binary_classification" | "multiclass_classification"
                              # | "regression" | "time_series"
    target_column: str | None
    time_column: str | None   # None for non-time-series tasks

    # Split configuration
    split_strategy: str       # "stratified_holdout" | "standard_holdout" | "time_ordered_holdout"
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

    # Preprocessing decisions (populated as user confirms decisions through cleaning/FE/norm stages)
    preprocessing_plan: dict[str, Any] = field(default_factory=dict)
    # Schema:
    #   imputation_strategies: dict[str, str]   col → "median" | "mean" | "mode" | "drop_rows" | "drop_col"
    #   outlier_strategies:    dict[str, str]   col → "cap" | "remove" | "keep"
    #   encoding_strategies:   dict[str, str]   col → "onehot" | "label" | "frequency" | "target" | "ordinal" | "drop"
    #   ordinal_orders:        dict[str, list]  col → ordered category list
    #   scaling_strategy:      str              "standard" | "minmax" | "robust" | "power" | "none"
    #   feature_columns:       list[str]        final ordered feature list after selection
    #   drop_columns:          list[str]        columns to drop before preprocessing

    # Resampling — only allowed for classification, only on training data
    resampling_plan: dict[str, Any] | None = None
    # Schema:
    #   strategy: "smote" | "undersample" | "class_weights" | "none"
    #   min_class_count: int  (for SMOTE k-neighbor guard)

    # Model selection
    model_candidates: list[str] = field(default_factory=list)
    selected_model_id: str | None = None

    # Metrics — determined by task_family, not by user choice
    primary_metric: str = ""
    secondary_metrics: list[str] = field(default_factory=list)

    # Compilation metadata
    compiled_at: str = ""
    compiler_version: str = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "RunSpec":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_classification(self) -> bool:
        return "classification" in self.task_family

    @property
    def is_regression(self) -> bool:
        return self.task_family == "regression"

    @property
    def is_time_series(self) -> bool:
        return self.task_family == "time_series"

    @property
    def scaling_strategy(self) -> str:
        return self.preprocessing_plan.get("scaling_strategy", "none")

    @property
    def feature_columns(self) -> list[str]:
        return self.preprocessing_plan.get("feature_columns", [])

    @property
    def imputation_strategies(self) -> dict[str, str]:
        return self.preprocessing_plan.get("imputation_strategies", {})

    @property
    def encoding_strategies(self) -> dict[str, str]:
        return self.preprocessing_plan.get("encoding_strategies", {})

    @property
    def outlier_strategies(self) -> dict[str, str]:
        return self.preprocessing_plan.get("outlier_strategies", {})

    @property
    def resampling_strategy(self) -> str:
        if self.resampling_plan is None:
            return "none"
        return self.resampling_plan.get("strategy", "none")


# ---------------------------------------------------------------------------
# EvaluationPayload
# ---------------------------------------------------------------------------

@dataclass
class EvaluationPayload:
    """
    Frozen, single source of truth for all metrics from one evaluation pass.

    Produced by evaluation_service.evaluate().
    No stage, chart, or widget may recompute predictions independently.

    y_true_path, y_pred_path, y_score_path point to parquet files containing
    the raw arrays — available for downstream chart rendering without
    recomputing predictions.
    """
    task_family: str
    model_id: str
    split_name: str            # "val" | "test"

    # Paths to raw prediction arrays (parquet, one column each)
    y_true_path: str
    y_pred_path: str
    y_score_path: str | None   # predicted probabilities / scores (classification only)

    # Classification metadata
    threshold_used: float | None       # decision threshold (default 0.5)
    label_order: list | None           # class label order matching confusion matrix axes
    class_mapping: dict | None         # int → original label (if labels were encoded)

    # Metrics — all values are Python float, never numpy types
    confusion_matrix: list[list[int]] | None
    metrics: dict[str, float]

    # Summary
    primary_metric: str
    verdict: str               # "strong" | "good" | "fair" | "poor"
    verdict_message: str = ""

    # Provenance
    timestamp: str = ""
    run_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationPayload":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @property
    def primary_metric_value(self) -> float:
        return self.metrics.get(self.primary_metric, 0.0)
