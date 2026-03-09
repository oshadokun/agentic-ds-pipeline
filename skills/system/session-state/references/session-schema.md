# Session Schema Reference

## Top-Level Fields

| Field | Type | Description |
|---|---|---|
| session_id | string | Unique identifier — timestamp + random hex |
| created_at | ISO datetime | When the session was first created |
| last_updated | ISO datetime | When session.json was last written |
| status | string | "in_progress", "complete", "failed", "abandoned" |

## goal Object
| Field | Type | Description |
|---|---|---|
| plain_english | string | User's goal in their own words |
| task_type | string | "binary_classification", "multiclass_classification", "regression" |
| target_column | string | Column name the model predicts |
| success_criteria | string | What success looks like to the user |
| confirmed_by_user | boolean | True once user has confirmed the goal capture |

## data_source Object
| Field | Type | Description |
|---|---|---|
| type | string | "csv", "database", "api" |
| path | string | Path to ingested data file |
| confirmed_by_user | boolean | True once user confirmed data loaded correctly |

## config Object
All values are null until set by the relevant stage.
| Field | Set by stage |
|---|---|
| framework | Training |
| model_id | Training |
| scaler | Normalisation |
| impute_strategy | Cleaning |
| evaluation_metric | Evaluation |
| feature_columns | Feature Engineering |
| split_ratios | Splitting |
| class_weights | Training |
| best_params | Tuning |

## privacy Object
| Field | Type | Description |
|---|---|---|
| sensitive_columns_identified | array | List of column names flagged as sensitive |
| user_acknowledged | boolean | True once user has made decisions on sensitive columns |
| sensitive_columns_action | object | {column: action} — "mask", "drop", or "keep" |

## stages Object
Each stage has:
| Field | Type | Description |
|---|---|---|
| status | string | "pending", "in_progress", "complete", "failed", "skipped" |
| started_at | ISO datetime | When the stage started |
| completed_at | ISO datetime | When the stage completed |
| summary | string | Plain English one-line summary of what happened |
| decisions | array | List of {decision, chosen, alternatives, reason} |

## report Object
| Field | Type | Description |
|---|---|---|
| sections | array | Ordered list of report sections, one per stage |
| assembled | boolean | True once final report has been generated |
| path | string | Path to the assembled report file |

## errors Array
Each error entry:
| Field | Description |
|---|---|
| stage | Which stage the error occurred in |
| timestamp | When the error occurred |
| error | Plain English description of the error |

---

## Stage Status Flow

```
pending → in_progress → complete
                      ↘ failed → (retry) → in_progress
                               → (skip)  → skipped
```

A stage can only move forward — never back to pending once started.
If a stage fails and is retried, its status returns to in_progress.

---

## Atomic Write Pattern

session.json is always written atomically:
1. Write to session.json.tmp
2. Rename .tmp to session.json

This prevents corruption if the process is interrupted mid-write.
The .tmp file can be safely deleted if found on startup.
