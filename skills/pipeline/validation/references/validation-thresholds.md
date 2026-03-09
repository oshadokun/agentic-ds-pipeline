# Validation Thresholds

These are the default thresholds used by the Validation agent.
They can be overridden in the session config if needed.

## Dataset Size
| Threshold | Value | Severity |
|---|---|---|
| Minimum rows (hard stop) | 50 | hard_stop |
| Small dataset warning | 200 | warning |
| Small dataset advisory | 1000 | advisory |

## Missing Values (per column)
| Threshold | Value | Severity |
|---|---|---|
| Mostly empty — recommend drop | 80% | warning |
| High missing — flag for cleaning | 30% | advisory |

## Target Column
| Threshold | Value | Severity |
|---|---|---|
| Missing target values (hard stop) | > 5% | hard_stop |
| Missing target values (warning) | > 0% | warning |

## Class Imbalance (classification)
| Threshold | Value | Severity |
|---|---|---|
| Severe imbalance | < 5% minority | warning |
| Moderate imbalance | < 20% minority | advisory |

## Duplicates
| Threshold | Value | Severity |
|---|---|---|
| High duplicate rate | > 10% | warning |
| Low duplicate count | > 0 | advisory |

## Column Variance
| Threshold | Value | Severity |
|---|---|---|
| Constant column | 1 unique value | warning |
| High cardinality text | > 95% unique | advisory |
