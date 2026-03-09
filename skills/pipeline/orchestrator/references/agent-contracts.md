# Agent Input/Output Contracts

Every agent receives a standardised input and returns a standardised output.
The Orchestrator is responsible for building inputs and reading outputs.

---

## Standard Input (all agents)

Written to: `sessions/{session_id}/inputs/{stage_name}/input.json`

```json
{
  "session_id": "20260308-a4f3b",
  "stage": "cleaning",
  "data_path": "sessions/{session_id}/data/interim/validated.csv",
  "config": { ... },
  "goal": { ... },
  "privacy": { ... }
}
```

---

## Standard Output (all agents)

Written to: `sessions/{session_id}/outputs/{stage_name}/result.json`

```json
{
  "stage": "cleaning",
  "status": "success",
  "output_data_path": "sessions/{session_id}/data/interim/cleaned.csv",
  "decisions_made": [ ... ],
  "decisions_required": [ ... ],
  "warnings": [ ... ],
  "errors": [ ... ],
  "plain_english_summary": "...",
  "report_section": { ... },
  "config_updates": { ... }
}
```

### decisions_required format
```json
{
  "id": "impute_strategy",
  "question": "Some of your data has missing values. How would you like to handle them?",
  "recommendation": "fill_median",
  "recommendation_reason": "Your data has some extreme values so the median is more reliable than the average.",
  "alternatives": [
    {
      "id": "fill_median",
      "label": "Fill with the middle value (recommended)",
      "tradeoff": "Safe and reliable. Works well when data has outliers."
    },
    {
      "id": "fill_mean",
      "label": "Fill with the average value",
      "tradeoff": "Simple but can be thrown off by very high or very low values."
    },
    {
      "id": "drop_rows",
      "label": "Remove rows with missing values",
      "tradeoff": "Cleanest option but you lose data. Only suitable if less than 5% of rows are affected."
    }
  ]
}
```

---

## Per-Agent Data Path Chain

Each agent reads from the previous agent's output path:

| Agent | Reads From | Writes To |
|---|---|---|
| Ingestion | user upload / source | data/raw/ingested.csv |
| Validation | data/raw/ingested.csv | data/raw/validated.csv |
| EDA | data/raw/validated.csv | reports/eda/ (no data transform) |
| Cleaning | data/raw/validated.csv | data/interim/cleaned.csv |
| Feature Engineering | data/interim/cleaned.csv | data/interim/features.csv |
| Normalisation | data/interim/features.csv | data/processed/scaled.csv + scaler.pkl |
| Splitting | data/processed/scaled.csv | data/processed/splits/ |
| Training | data/processed/splits/ | models/trained_model.pkl |
| Evaluation | models/trained_model.pkl + splits | reports/evaluation/ |
| Tuning | models/trained_model.pkl | models/tuned_model.pkl |
| Explainability | models/tuned_model.pkl + splits | reports/explainability/ |
| Deployment | models/tuned_model.pkl | api/ |
| Monitoring | api/ + data/raw/ | reports/monitoring/ |
