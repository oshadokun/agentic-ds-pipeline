# Failure Policies by Stage

## Failure Categories

| Category | Definition | Action |
|---|---|---|
| Recoverable | A different approach can be tried automatically | Log, try alternative, inform user |
| Retryable | Transient error, same approach may succeed | Retry up to 3x with backoff |
| Hard Stop | Cannot proceed without user intervention | Halt, explain, await user action |

---

## Per-Stage Failure Policies

### Ingestion
| Failure | Category | Action |
|---|---|---|
| File not found | Hard Stop | Ask user to re-upload |
| File is empty | Hard Stop | Tell user file has no data |
| Database connection timeout | Retryable | Retry 3x, then ask user to check credentials |
| API returns 4xx | Hard Stop | Explain the API rejected the request, show error in plain English |
| API returns 5xx | Retryable | Retry 3x with backoff |
| Unrecognised file format | Hard Stop | Tell user which formats are supported |

### Validation
| Failure | Category | Action |
|---|---|---|
| Target column not found | Hard Stop | Ask user to confirm the column name |
| All rows have missing target | Hard Stop | Explain the target column appears empty |
| Dataset has fewer than 50 rows | Hard Stop | Warn user the dataset may be too small to train reliably |
| More than 80% of a column is missing | Recoverable | Recommend dropping the column, present to user |

### EDA
| Failure | Category | Action |
|---|---|---|
| Plot generation fails | Recoverable | Skip that plot, continue with others, note in summary |
| Profiling library unavailable | Recoverable | Fall back to basic stats, inform user |

### Cleaning
| Failure | Category | Action |
|---|---|---|
| Imputation produces all-null column | Recoverable | Try next strategy, inform user |
| Outlier removal leaves fewer than 30 rows | Hard Stop | Warn user, ask them to approve before proceeding |

### Training
| Failure | Category | Action |
|---|---|---|
| Model runs out of memory | Recoverable | Try a lighter model automatically, inform user |
| Training takes more than 10 minutes | Recoverable | Suggest a faster model, present to user |
| Model fails to converge | Recoverable | Adjust learning rate or increase iterations, inform user |

### Deployment
| Failure | Category | Action |
|---|---|---|
| Port already in use | Recoverable | Try next available port |
| Docker not available | Recoverable | Fall back to local uvicorn, inform user |
| Health check fails after 3 attempts | Hard Stop | Show user the error log in plain English |

---

## Error Message Template

All errors surfaced to the user follow this structure:

```
What happened: [plain English description of the error]
Why it happened: [plain English cause]
What this means for you: [impact on the pipeline]
What you need to do: [clear action for the user, or "nothing — we handled it automatically"]
```
