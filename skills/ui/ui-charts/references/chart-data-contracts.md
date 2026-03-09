# Chart Data Contracts

Every interactive chart expects a specific JSON shape from the backend.
These are included in the stage result.json under a `chart_data` key.

---

## Feature Distribution (EDA)

Endpoint: included in `outputs/eda/result.json`

```json
{
  "feature_distributions": {
    "age": {
      "type": "numeric",
      "bins": [
        { "range": "18–25", "count": 312, "pct": 8.2, "isOutlier": false },
        { "range": "25–35", "count": 891, "pct": 23.4, "isOutlier": false },
        { "range": "65–75", "count": 24,  "pct": 0.6,  "isOutlier": true  }
      ]
    },
    "contract_type": {
      "type": "categorical",
      "values": [
        { "label": "Month-to-month", "count": 2153, "pct": 56.6 },
        { "label": "One year",       "count": 1023, "pct": 26.9 },
        { "label": "Two year",       "count": 629,  "pct": 16.5 }
      ]
    }
  }
}
```

---

## Target Distribution (EDA)

```json
{
  "target_distribution": {
    "task_type": "binary_classification",
    "data": [
      { "label": "No churn", "count": 5174, "pct": 73.5 },
      { "label": "Churned",  "count": 1869, "pct": 26.5 }
    ]
  }
}
```

---

## Split Ratios (Splitting)

The frontend computes this live from the slider — no backend call needed.
The backend receives the confirmed ratios when the user clicks continue:

```json
{
  "split_ratios": { "train": 0.75, "val": 0.10, "test": 0.15 }
}
```

---

## Confusion Matrix (Evaluation)

```json
{
  "confusion_matrix": {
    "matrix":      [[3421, 215], [312, 1095]],
    "class_names": ["No churn", "Churned"],
    "total_rows":  5043
  }
}
```

---

## Tuning Trial History (Tuning)

```json
{
  "trial_history": [
    { "trial": 1,  "score": 0.821 },
    { "trial": 2,  "score": 0.834 },
    { "trial": 3,  "score": 0.829 },
    { "trial": 50, "score": 0.912 }
  ],
  "baseline_score": 0.891,
  "metric_name":    "ROC-AUC"
}
```

---

## Feature Importance (Explainability)

```json
{
  "feature_importance": [
    { "feature": "tenure",        "importance": 0.2341 },
    { "feature": "contract_type", "importance": 0.1893 },
    { "feature": "total_charges", "importance": 0.1204 }
  ]
}
```

---

## Performance Trend (Monitoring)

```json
{
  "performance_trend": {
    "metric_name":         "ROC-AUC",
    "baseline_score":      0.912,
    "warning_threshold":   0.870,
    "critical_threshold":  0.820,
    "reports": [
      { "reportNumber": 1, "date": "2025-03-01", "score": 0.908 },
      { "reportNumber": 2, "date": "2025-03-08", "score": 0.901 },
      { "reportNumber": 3, "date": "2025-03-15", "score": 0.884 }
    ]
  }
}
```

---

## Drift Summary (Monitoring)

```json
{
  "drift_summary": {
    "total_features":   14,
    "drifted":          3,
    "high_severity":    1,
    "medium_severity":  1,
    "low_severity":     1
  }
}
```

---

## Static Chart Paths

These are PNG files served by the backend. The frontend fetches them via:
`GET /sessions/{session_id}/charts?path={encoded_path}`

| Chart | Path in session |
|---|---|
| Correlation heatmap | `reports/eda/correlation_heatmap.png` |
| Feature vs target | `reports/eda/feature_vs_target.png` |
| ROC curve | `reports/evaluation/roc_curve.png` |
| Residual plot | `reports/evaluation/residual_plot.png` |
| SHAP summary | `reports/explainability/shap_summary.png` |
| Waterfall row 0 | `reports/explainability/explanation_row_0.png` |
| Waterfall row 1 | `reports/explainability/explanation_row_1.png` |
| Waterfall row 2 | `reports/explainability/explanation_row_2.png` |
| Before/after scaling | `reports/normalisation/scaling_comparison.png` |
