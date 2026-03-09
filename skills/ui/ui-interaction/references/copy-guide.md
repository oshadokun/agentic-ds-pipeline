# Plain English Copy Guide

The audience is non-technical. Every word in the UI should pass this test:
would a confident, knowledgeable friend explain it this way?

---

## Voice and Tone

**Warm but not casual.** This is professional work. The user is making real
decisions about real data. Treat them as an intelligent adult who simply
does not know data science terminology — not as someone who needs protection
from complexity.

**Confident but not arrogant.** "We recommend" — not "You should" or "Obviously".
"This approach works well when..." — not "This is the best option".

**Honest about tradeoffs.** Never hide the downside of a recommendation.
If something has a cost, say so. The user will trust the tool more for it.

---

## Terminology Replacements

| Technical term | Plain English alternative |
|---|---|
| Ingestion | Loading your data |
| Validation | Checking data quality |
| EDA | Exploring your data |
| Imputation | Filling in missing values |
| Outlier | Unusual value |
| Feature engineering | Preparing features |
| Normalisation / scaling | Scaling your data |
| Train/val/test split | Dividing your data |
| Hyperparameter | Model setting |
| Overfitting | Memorising instead of learning |
| ROC-AUC | How well the model separates outcomes (0–1 scale) |
| Precision | How often the model is right when it raises an alert |
| Recall | How many real cases the model catches |
| RMSE | Typical prediction error (in the same units as your target) |
| R² | How much of the variation the model explains |
| SHAP value | How much this factor influenced this prediction |
| Drift | The data has changed compared to what the model was trained on |
| Endpoint | Web address for the model API |
| Pickle file | The saved model file |
| DataFrame | Your data table |

---

## Decision Copy Patterns

### Presenting a recommendation:
"Based on [reason], we recommend [option]. [One sentence plain English rationale]."

### Presenting a tradeoff:
"[Option] works well for [situation], but [honest downside] if [condition]."

### Asking for confirmation:
"Does this match what you want to do?" / "Shall we proceed?"

### Irreversible actions:
"This will permanently [action]. It cannot be undone."
Never soften irreversible actions — be direct.

---

## Error Copy Patterns

### Recoverable error:
"Something went wrong with [stage]. [Plain English explanation of what happened].
Here is what you can do: [action]."

### Hard stop:
"We cannot continue until [issue] is resolved. [Plain English explanation].
[Specific next step]."

### Warning (user decision needed):
"We noticed [issue]. This may affect [outcome]. How would you like to handle it?"

---

## Progress Copy Patterns

### Stage complete:
"[Stage name] complete. [One sentence summary of what was found or done]."

### Stage in progress:
"[Verb]ing your data…" / "Training the model…" / "Running the check…"
Use the -ing form. Tell the user what is happening right now.

### Estimated time:
"This will take approximately [time]."
Be honest — do not under-promise. A user who expects 2 minutes and waits 5
is more frustrated than one who expected 5 and waited 5.

---

## Things Never to Say

- "Error 422" or any HTTP status code — translate it
- "null" or "undefined" — handle it before display
- "Please wait..." — say what is happening instead
- "Success!" alone — follow with what happened
- "Warning!" alone — always explain what the warning means
- "Are you sure?" alone — say what they are confirming and why it matters
- "Invalid input" — say specifically what is wrong and how to fix it
