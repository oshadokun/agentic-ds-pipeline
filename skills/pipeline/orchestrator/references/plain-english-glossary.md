# Plain English Glossary

When the Orchestrator or any agent needs to explain a technical concept to the user,
use these translations. Never use the technical term alone.

---

| Technical Term | Plain English Explanation |
|---|---|
| DataFrame | A table of data, like a spreadsheet, that the computer can work with |
| Feature | A column in your data that the model uses to make predictions |
| Target column | The column you want the model to predict or explain |
| Model | A mathematical pattern the computer learns from your data to make predictions |
| Training | The process of the computer learning patterns from your data |
| Overfitting | When the model memorises your data instead of learning from it — it performs well on old data but poorly on new data |
| Underfitting | When the model is too simple to pick up the patterns in your data |
| Regularisation | A technique that stops the model from memorising your data by keeping it simpler |
| Normalisation / Scaling | Adjusting your numbers so they are all on a comparable scale — like converting miles and kilometres to the same unit before comparing them |
| Imputation | Filling in missing values in your data using a calculated estimate |
| Outlier | A value that is unusually high or low compared to the rest of the data |
| Train/test split | Dividing your data into two groups — one for the model to learn from, one for us to test how well it learned |
| Cross-validation | Testing the model multiple times on different slices of your data to get a reliable measure of performance |
| ROC-AUC | A score between 0 and 1 that measures how well the model distinguishes between the two outcomes — 1.0 is perfect, 0.5 is no better than a coin flip |
| Precision | Of all the times the model predicted yes, how often was it right |
| Recall | Of all the actual yes cases, how many did the model catch |
| F1 Score | A single number that balances precision and recall — useful when both matter equally |
| Hyperparameter | A setting you choose before training that controls how the model learns — like the temperature dial on an oven |
| Hyperparameter tuning | Systematically trying different settings to find the combination that gives the best results |
| SHAP values | A way of measuring how much each column in your data contributed to a specific prediction |
| REST API | A way of making your model available so other applications can send it new data and receive predictions back |
| Drift | When the patterns in new data start to differ from the data the model was trained on — a sign the model may need retraining |
| Feature engineering | Creating new, more informative columns from the ones you already have |
| Categorical variable | A column that contains labels or categories rather than numbers — for example, country or product type |
| One-hot encoding | Turning a category column into multiple yes/no columns so the model can understand it |
| Class imbalance | When one outcome appears much more often than the other in your data — for example, 95% of customers stayed and only 5% churned |
| Ensemble | Combining the predictions of multiple models to get a more reliable result |
| Gradient boosting | A technique that builds many small models in sequence, each one correcting the mistakes of the previous one |
| Dimensionality reduction | Reducing the number of columns in your data while keeping the most important information |
| Null / NaN | A missing value — a cell in your data that has no entry |
