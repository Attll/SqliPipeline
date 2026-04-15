# SQL Injection Detection Pipeline

A recall-oriented end-to-end machine learning pipeline for detecting SQL injection attacks using character TF-IDF features and handcrafted SQL-pattern signals.

---

## Project Structure

```
sqli-fresh-pipeline/
  data/
    raw/
      sql_injection_dataset.csv
  artifacts/
  reports/
  scripts/
    train.py
    evaluate.py
  src/
    sqli_pipeline/
      __init__.py
      config.py
      data.py
      features.py
      preprocessor.py
      trainer.py
      evaluator.py
      predictor.py
  tests/
    test_features.py
  streamlit_app.py
  requirements.txt
  README.md
  .gitignore
```

### Dataset Format

The dataset must live at `data/raw/sql_injection_dataset.csv` with exactly these columns:

| Column | Description |
|--------|-------------|
| `Query` | Raw SQL query or request parameter string |
| `Label` | `1` = SQL injection / malicious, `0` = benign |

---

## 1. Install Python

Install Python **3.10** or **3.11** on the target system.

```bash
python --version
# Python 3.10.x  or  Python 3.11.x
```

> Avoid very old Python versions. Python 3.10 or 3.11 is the recommended choice for this project.

---

## 2. Open Terminal In The Project

**Windows PowerShell:**
```powershell
cd "C:\Projects\sqli-fresh-pipeline"
```

**macOS / Linux:**
```bash
cd /path/to/sqli-fresh-pipeline
```

---

## 3. Install Dependencies

**Simple install into current environment:**
```bash
python -m pip install -r requirements.txt
```

**Recommended: use a virtual environment:**

*Windows:*
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

*macOS / Linux:*
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies included: `pandas`, `numpy`, `scikit-learn`, `joblib`, `streamlit`, `matplotlib`, `seaborn`

---

## 4. Verify Dataset

Before training, confirm the dataset is present:

```bash
dir data\raw          # Windows
ls data/raw           # macOS/Linux
```

You should see `sql_injection_dataset.csv`.

**Quick validation:**
```bash
python -c "import pandas as pd; df=pd.read_csv('data/raw/sql_injection_dataset.csv'); print(df.shape); print(df.columns); print(df['Label'].value_counts())"
```

Expected output:
- Columns include: `Query`, `Label`
- Both labels `0` and `1` are present

> **Important:** The model cannot train properly if only attack samples or only benign samples are present.

---

## 5. Train The Models

```bash
python scripts/train.py
```

This trains the three models:
- **Random Forest**
- **Logistic Regression**
- **Multinomial Naive Bayes**

The trainer does the following:
1. Loads the dataset
2. Performs basic EDA
3. Splits data into train/test
4. Builds character TF-IDF features
5. Adds simple SQLi-focused handcrafted features
6. Trains RF, LR, and NB
7. Chooses the best model using **F1 score**
8. Saves models, metrics, and reports

> **Why F1 score?** F1 gives equal weight to precision and recall, making it a balanced metric for evaluating the overall quality of the detector.

**After training, these files will be created:**

```
artifacts/
  best_model.joblib
  model_metadata.json
  holdout_test.csv
  models/
    random_forest.joblib
    logistic_regression.joblib
    naive_bayes.joblib

reports/
  eda_summary.json
  eda_rows.csv
  model_metrics.csv
  model_metrics.json
  feature_importance.csv
```

---

## 6. Evaluate The Best Model

```bash
python scripts/evaluate.py
```

This evaluates the saved best model on the holdout test split.

**It creates:**
```
reports/
  best_model_evaluation.json
  confusion_matrix.csv
  holdout_predictions.csv
```

**Open this file to compare model performance:**
```
reports/model_metrics.csv
```

| Metric | Description |
|--------|-------------|
| `recall` | How many attacks were caught |
| `precision` | How many flagged attacks were actually attacks |
| `f1_score` | Balance between precision and recall |
| `f1_score` | Balance between precision and recall |
| `roc_auc` | Ranking quality of probabilities |
| `average_precision` | Useful for imbalanced data |

> **Since this is a security detector, prioritize:** `recall`, `f1_score`, and `fn` (false negatives). False negatives are dangerous — they mean attacks were missed.

---

## 7. Run The Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Streamlit will print a local URL, usually: `http://localhost:8501`

**The dashboard includes:**

| Tab | Description |
|-----|-------------|
| **Single Query** | Paste one query or payload and predict Attack/Benign |
| **Batch CSV** | Upload a CSV with a `Query` column and download predictions |
| **EDA** | View dataset size, duplicates, label distribution, and query length patterns |
| **Metrics** | Compare Random Forest, Logistic Regression, and Naive Bayes |
| **Feature Importance** | See the strongest model signals when supported by the selected model |

---

## 8. Batch Prediction CSV Format

Upload a CSV like:

```csv
Query
' OR 1=1 --
select * from users where id = 5
admin' #
```

The app will return columns: `malicious_probability`, `prediction`, `risk_label`, `threshold`, `model`

---

## 9. Recommended Workflow

Use this order every time on a fresh system:

```bash
cd "path\to\sqli-fresh-pipeline"
python -m pip install -r requirements.txt
python scripts/train.py
python scripts/evaluate.py
streamlit run streamlit_app.py
```

---

## 10. Best Practices

- Keep raw data unchanged: `data/raw/sql_injection_dataset.csv`
- Do not manually edit trained model artifacts: `artifacts/*.joblib`
- If the dataset changes, retrain: `python scripts/train.py && python scripts/evaluate.py`
- Use the same train/evaluate scripts instead of training manually in notebooks — this keeps results reproducible
- Check `reports/model_metrics.csv` after every training run, focusing especially on `recall`, `f1_score`, and `fn`
- **Do not judge the model only by accuracy.** A model can have high accuracy and still miss dangerous SQL injection attempts.

---

## 11. Troubleshooting

| Problem | Solution |
|---------|----------|
| `pandas` or `sklearn` missing | `python -m pip install -r requirements.txt` |
| `streamlit: command not found` | `python -m streamlit run streamlit_app.py` |
| Dashboard says model is missing | Run `python scripts/train.py` then `python scripts/evaluate.py` |
| Training fails because of column names | Ensure the CSV has `Query,Label` (case-sensitive) |
