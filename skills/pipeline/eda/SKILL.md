---
name: eda
description: >
  Responsible for exploring and summarising the loaded data before any transformation
  occurs. Always called by the Orchestrator after Validation passes. Analyses
  distributions, relationships, patterns, and anomalies across all columns. Produces
  charts, statistics, and plain English insights tailored to non-technical users.
  Does not transform data — only explores and reports. Surfaces findings that will
  inform decisions in later stages. Trigger when any of the following are mentioned:
  "explore data", "understand my data", "data summary", "EDA", "exploratory analysis",
  "what does my data look like", "distributions", "correlations", or any request to
  gain insight into a dataset before modelling.
---

# EDA Skill

The EDA agent explores the data and translates what it finds into plain English
insights that a non-technical user can understand and act on. It does not make
decisions — it surfaces information. Every chart, statistic, and observation it
produces feeds into the decisions the user will make in Cleaning, Feature Engineering,
and Training.

Its output is both a visual report for the user and a structured findings object
that later agents use to make better decisions.

---

## Responsibilities

1. Summarise the overall shape and character of the data
2. Analyse each column individually — distribution, missing values, unique values
3. Analyse the target column in depth
4. Find relationships between features and the target
5. Find relationships between features (correlations)
6. Identify patterns that may affect modelling decisions
7. Produce charts saved to the session reports directory
8. Write plain English insights for every finding
9. Return structured findings to the Orchestrator

---

## Analysis Modules

### 1. Dataset Overview
```python
def overview(df, target_col):
    numeric_cols  = df.select_dtypes("number").columns.tolist()
    cat_cols      = df.select_dtypes("object").columns.tolist()
    datetime_cols = df.select_dtypes("datetime").columns.tolist()

    return {
        "row_count":       len(df),
        "column_count":    len(df.columns),
        "numeric_cols":    numeric_cols,
        "categorical_cols": cat_cols,
        "datetime_cols":   datetime_cols,
        "total_missing":   int(df.isna().sum().sum()),
        "missing_pct":     round(df.isna().mean().mean(), 4),
        "memory_mb":       round(df.memory_usage(deep=True).sum() / 1e6, 2)
    }
```

**Plain English summary template:**
"Your dataset has {row_count} rows and {column_count} columns.
{numeric_count} columns contain numbers, {cat_count} contain text or categories.
Overall, {missing_pct:.1%} of values are missing across the whole dataset."

---

### 2. Target Column Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyse_target(df, target_col, task_type, output_dir):
    target = df[target_col].dropna()
    result = {"column": target_col}

    if task_type in ["binary_classification", "multiclass_classification"]:
        counts = target.value_counts()
        pcts   = target.value_counts(normalize=True)
        result["class_counts"]  = counts.to_dict()
        result["class_pcts"]    = pcts.round(4).to_dict()

        # Chart
        fig, ax = plt.subplots(figsize=(8, 4))
        counts.plot(kind="bar", ax=ax, color="#2E75B6")
        ax.set_title("How often each outcome appears in your data")
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Number of rows")
        plt.tight_layout()
        path = Path(output_dir) / "target_distribution.png"
        plt.savefig(path)
        plt.close()
        result["chart_path"] = str(path)

        # Plain English
        majority_class = counts.index[0]
        minority_class = counts.index[-1]
        result["plain_english"] = (
            f"Your data has {len(counts)} possible outcomes. "
            f"'{majority_class}' is the most common ({pcts.iloc[0]:.1%} of rows). "
            f"'{minority_class}' is the least common ({pcts.iloc[-1]:.1%} of rows)."
        )

    elif task_type == "regression":
        result["mean"]   = round(float(target.mean()), 4)
        result["median"] = round(float(target.median()), 4)
        result["std"]    = round(float(target.std()), 4)
        result["min"]    = round(float(target.min()), 4)
        result["max"]    = round(float(target.max()), 4)
        result["skew"]   = round(float(target.skew()), 4)

        fig, ax = plt.subplots(figsize=(8, 4))
        target.hist(bins=40, ax=ax, color="#2E75B6", edgecolor="white")
        ax.set_title(f"Distribution of '{target_col}'")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Number of rows")
        plt.tight_layout()
        path = Path(output_dir) / "target_distribution.png"
        plt.savefig(path)
        plt.close()
        result["chart_path"] = str(path)

        skew_desc = "fairly symmetrical"
        if abs(result["skew"]) > 1:
            skew_desc = "skewed to the right" if result["skew"] > 0 else "skewed to the left"

        result["plain_english"] = (
            f"The values in '{target_col}' range from {result['min']} to {result['max']}. "
            f"The typical value is around {result['median']} (the middle value). "
            f"The distribution is {skew_desc}."
        )

    return result
```

---

### 3. Feature Distributions
```python
def analyse_features(df, target_col, output_dir):
    results = []
    output_dir = Path(output_dir)

    for col in df.columns:
        if col == target_col:
            continue

        col_result = {"column": col, "dtype": str(df[col].dtype)}
        col_result["missing_pct"] = round(df[col].isna().mean(), 4)
        col_result["unique_count"] = int(df[col].nunique())

        if df[col].dtype in ["int64", "float64"]:
            col_result["mean"]   = round(float(df[col].mean()), 4)
            col_result["median"] = round(float(df[col].median()), 4)
            col_result["std"]    = round(float(df[col].std()), 4)
            col_result["skew"]   = round(float(df[col].skew()), 4)

            # Flag outliers using IQR
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outlier_count = int(((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum())
            col_result["outlier_count"] = outlier_count
            col_result["outlier_pct"]   = round(outlier_count / len(df), 4)

            if outlier_count > 0:
                col_result["advisory"] = f"'{col}' has {outlier_count} unusually high or low values ({col_result['outlier_pct']:.1%}). These will be reviewed during cleaning."

        else:
            top_values = df[col].value_counts().head(5).to_dict()
            col_result["top_values"] = {str(k): v for k, v in top_values.items()}

            if col_result["unique_count"] / len(df) > 0.9:
                col_result["advisory"] = f"'{col}' has almost as many unique values as rows — it may be an ID column and not useful for modelling."

        results.append(col_result)

    # Save a grid of distribution plots for numeric columns
    numeric_cols = [c for c in df.columns if c != target_col and df[c].dtype in ["int64", "float64"]]
    if numeric_cols:
        n = min(len(numeric_cols), 12)
        cols_to_plot = numeric_cols[:n]
        fig, axes = plt.subplots(nrows=(n+2)//3, ncols=3, figsize=(15, 4*((n+2)//3)))
        axes = axes.flatten()
        for i, col in enumerate(cols_to_plot):
            df[col].hist(bins=30, ax=axes[i], color="#2E75B6", edgecolor="white")
            axes[i].set_title(col, fontsize=10)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle("Distribution of your numeric columns", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_distributions.png", bbox_inches="tight")
        plt.close()

    return results
```

---

### 4. Correlation Analysis
```python
def analyse_correlations(df, target_col, output_dir):
    output_dir = Path(output_dir)
    numeric_df = df.select_dtypes("number")

    if len(numeric_df.columns) < 2:
        return {"plain_english": "Not enough numeric columns to analyse correlations."}

    corr_matrix = numeric_df.corr()

    # Heatmap
    fig, ax = plt.subplots(figsize=(min(16, len(numeric_df.columns) * 1.2),
                                    min(14, len(numeric_df.columns) * 1.0)))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("How strongly your numeric columns relate to each other\n(1.0 = identical, 0 = no relationship, -1.0 = opposite)")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # Target correlations
    target_corrs = {}
    if target_col in numeric_df.columns:
        target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        target_corrs = target_corrs.round(4).to_dict()

    # Highly correlated pairs (potential redundancy)
    high_corr_pairs = []
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if cols[i] == target_col or cols[j] == target_col:
                continue
            val = abs(corr_matrix.iloc[i, j])
            if val > 0.85:
                high_corr_pairs.append({
                    "col_a": cols[i],
                    "col_b": cols[j],
                    "correlation": round(float(val), 4),
                    "advisory": f"'{cols[i]}' and '{cols[j]}' are very similar to each other ({val:.2f}). You may only need one of them."
                })

    return {
        "target_correlations": target_corrs,
        "high_corr_pairs": high_corr_pairs,
        "chart_path": str(output_dir / "correlation_heatmap.png"),
        "plain_english": _correlation_plain_english(target_corrs, high_corr_pairs, target_col)
    }

def _correlation_plain_english(target_corrs, high_corr_pairs, target_col):
    lines = []
    if target_corrs:
        top = list(target_corrs.items())[:3]
        top_str = ", ".join([f"'{c}' ({v:.2f})" for c, v in top])
        lines.append(f"The columns most related to '{target_col}' are: {top_str}.")
    if high_corr_pairs:
        lines.append(f"We found {len(high_corr_pairs)} pairs of columns that are very similar to each other — you may not need both. We will flag these during feature engineering.")
    return " ".join(lines) if lines else "No strong relationships found between columns."
```

---

### 5. Feature vs Target Relationships
```python
def analyse_feature_vs_target(df, target_col, task_type, output_dir):
    output_dir = Path(output_dir)
    numeric_cols = [c for c in df.select_dtypes("number").columns if c != target_col]
    insights = []

    if task_type in ["binary_classification", "multiclass_classification"] and numeric_cols:
        n = min(len(numeric_cols), 6)
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        if n == 1:
            axes = [axes]
        for i, col in enumerate(numeric_cols[:n]):
            for label in df[target_col].dropna().unique():
                subset = df[df[target_col] == label][col].dropna()
                axes[i].hist(subset, alpha=0.6, bins=20, label=str(label))
            axes[i].set_title(col, fontsize=10)
            axes[i].legend(fontsize=8)
        plt.suptitle(f"How each column varies across your outcomes", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / "feature_vs_target.png", bbox_inches="tight")
        plt.close()
        insights.append("We have plotted how each numeric column is distributed across your different outcomes. Columns where the distributions look different between outcomes are likely to be useful for prediction.")

    return {"chart_path": str(output_dir / "feature_vs_target.png"), "insights": insights}
```

---

## Assembling the Full EDA Report

```python
def run_eda(df, target_col, task_type, session_id):
    output_dir = f"sessions/{session_id}/reports/eda"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    findings = {
        "overview":             overview(df, target_col),
        "target_analysis":      analyse_target(df, target_col, task_type, output_dir),
        "feature_analysis":     analyse_features(df, target_col, output_dir),
        "correlations":         analyse_correlations(df, target_col, output_dir),
        "feature_vs_target":    analyse_feature_vs_target(df, target_col, task_type, output_dir)
    }

    return findings
```

---

## Output Written to Session

**No data transformation** — EDA reads but never writes data files.

**Charts saved to:**
`sessions/{session_id}/reports/eda/`
- `target_distribution.png`
- `feature_distributions.png`
- `correlation_heatmap.png`
- `feature_vs_target.png`

**Result JSON:**
`sessions/{session_id}/outputs/eda/result.json`

```json
{
  "stage": "eda",
  "status": "success",
  "output_data_path": "sessions/{session_id}/data/raw/ingested.csv",
  "findings": { ... },
  "charts": [ ... ],
  "decisions_required": [],
  "advisories": [ ... ],
  "plain_english_summary": "...",
  "report_section": {
    "stage": "eda",
    "title": "Understanding Your Data",
    "summary": "...",
    "decision_made": "No decisions required at this stage — this is purely exploratory.",
    "alternatives_considered": null,
    "why_this_matters": "Understanding your data before modelling helps us make better decisions at every stage that follows."
  },
  "config_updates": {
    "highly_correlated_pairs": [ ... ],
    "high_outlier_cols": [ ... ],
    "skewed_cols": [ ... ],
    "high_cardinality_cols": [ ... ]
  }
}
```

---

## What to Tell the User

Present EDA findings as a guided tour, not a data dump:

1. Start with the big picture — "Here's what your data looks like overall"
2. Show the target column — "This is what we're trying to predict"
3. Walk through the most interesting feature findings — lead with the ones most relevant to the target
4. Show the correlation heatmap with a plain explanation
5. Flag anything that will require a decision in a later stage — outliers, high cardinality, skewed distributions, redundant columns
6. End with a summary of what we found and what comes next

**Never show all charts at once.** Present them in the guided sequence above.
**Always caption every chart** in plain English directly beneath it.

---

## Reference Files

- `references/eda-interpretation-guide.md` — how to interpret and narrate common EDA patterns
