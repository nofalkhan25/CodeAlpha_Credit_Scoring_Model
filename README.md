# 💳 Credit Scoring Model

A machine learning model that predicts an individual's **creditworthiness** using past financial data — comparing Logistic Regression, Decision Tree, and Random Forest side-by-side, with a built-in credit score (300–850 FICO-scale) output.

Built as **Task 1** for the **CodeAlpha Machine Learning Internship Program**.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Flask](https://img.shields.io/badge/flask-optional%20API-black)
![Status](https://img.shields.io/badge/status-complete-success)

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Quickstart](#quickstart)
- [Example Output](#example-output)
- [Optional Flask API](#optional-flask-api)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Roadmap](#roadmap)

---

## Overview

**Credit scoring** is a classification problem: given an applicant's financial history, predict whether they will **repay** (0) or **default** (1) on a loan.

**Highlights:**

- Three models trained and compared: **Logistic Regression**, **Decision Tree**, **Random Forest**
- Full evaluation suite: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- **Feature engineering**: debt-to-income ratio, loan-to-income ratio, repayment burden
- **Credit score output**: probability → 300–850 score (same scale as FICO)
- `class_weight='balanced'` to handle class imbalance without oversampling
- Optional **Flask REST API** for real-time predictions

---

## How It Works

```
Applicant Data (CSV)
        │
        ▼
  Load & Clean (EDA)         ← nulls, duplicates, dtype coercion
        │
        ▼
  Feature Engineering        ← debt_to_income, loan_to_income,
        │                       payment_to_income, repayment_burden
        ▼
  StandardScaler + Split     ← 80/20 stratified train/test split
        │
        ▼
  Train 3 Models             ← Logistic Regression | Decision Tree | Random Forest
        │
        ▼
  Evaluate All               ← Accuracy, Precision, Recall, F1, ROC-AUC
        │
        ▼
  Best Model → Predict       ← probability → credit score (300–850)
```

**Engineered Features:**

| Feature | Formula |
|---|---|
| `debt_to_income` | existing_debts / (income / 12) |
| `loan_to_income` | loan_amount / income |
| `payment_to_income` | (loan_amount / term) / (income / 12) |
| `repayment_burden` | (debts + est. payment) / monthly_income |

---

## Quickstart

```bash
# Clone and enter the repo
git clone https://github.com/YOUR_USERNAME/CodeAlpha_CreditScoringModel
cd CodeAlpha_CreditScoringModel

# Set up a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Regenerate the dataset
python3 make_dataset.py

# Run the full pipeline
python3 credit_scorer.py
```

### Run your own query

```python
from credit_scorer import (
    load_data, run_eda, engineer_features,
    preprocess, train_all, predict_single
)

df = engineer_features(run_eda(load_data("data/applicants.csv")))
X_tr, X_te, y_tr, y_te, scaler = preprocess(df)
model = train_all(X_tr, y_tr)["rf"]

result = predict_single({
    "age": 35, "income": 55000, "employment_years": 5,
    "loan_amount": 20000, "loan_term_months": 36,
    "credit_history": 1, "existing_debts": 600,
    "num_credit_lines": 3, "missed_payments": 1,
}, model, scaler)

print(result)
```

### Use your own dataset

```bash
python3 credit_scorer.py --data path/to/your_data.csv
```

Any CSV with the required columns works:
`age`, `income`, `employment_years`, `loan_amount`, `loan_term_months`,
`credit_history`, `existing_debts`, `num_credit_lines`, `missed_payments`, `default`

---

## Example Output

```
============================================================
MODEL EVALUATION
============================================================
Model                  Accuracy  Precision   Recall       F1   ROC-AUC
----------------------------------------------------------------------
Logistic Regression      0.7750     0.7273   0.7273   0.7273    0.8423
Decision Tree            0.7500     0.7241   0.7273   0.7257    0.7606
Random Forest            0.8250     0.8000   0.7273   0.7619    0.9012

✓ Best model: Random Forest (ROC-AUC = 0.9012)

SAMPLE PREDICTIONS
============================================================
Applicant : Ahmed (Low Risk)
  Verdict          : Creditworthy
  Default Prob     : 4.50%
  Credit Score     : 824  (Exceptional)

Applicant : Sara (Medium Risk)
  Verdict          : Creditworthy
  Default Prob     : 38.20%
  Credit Score     : 573  (Fair)

Applicant : Bilal (High Risk)
  Verdict          : Default Risk
  Default Prob     : 82.10%
  Credit Score     : 397  (Poor)
```

---

## Optional Flask API

```bash
python3 app.py
```

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Predict creditworthiness + credit score |
| `/health` | GET | Uptime check |
| `/features` | GET | List required input fields |

**Example request:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "income": 55000, "employment_years": 5,
    "loan_amount": 20000, "loan_term_months": 36,
    "credit_history": 1, "existing_debts": 600,
    "num_credit_lines": 3, "missed_payments": 1
  }'
```

**Example response:**

```json
{
  "credit_score": 573,
  "default_probability": 0.3820,
  "predicted_class": 0,
  "score_band": "Fair",
  "verdict": "Creditworthy"
}
```

---

## Project Structure

```
CodeAlpha_CreditScoringModel/
├── data/
│   └── applicants.csv          # dataset (200 applicants)
├── credit_scorer.py             # EDA, feature engineering, models, CLI
├── make_dataset.py              # generates data/applicants.csv
├── app.py                       # optional Flask REST API
├── requirements.txt
└── README.md
```

---

## Dataset

`data/applicants.csv` ships with **200 synthetic applicants** spanning low, medium, and high risk profiles. Each record includes financial features and a `default` label.

**Schema:**

| Column | Description |
|---|---|
| `applicant_id` | Unique identifier (APP0001 … APP0200) |
| `age` | Applicant age in years |
| `income` | Annual income (USD) |
| `employment_years` | Years at current employer |
| `loan_amount` | Requested loan amount (USD) |
| `loan_term_months` | Requested loan term |
| `credit_history` | 0 = bad/none, 1 = good |
| `existing_debts` | Monthly existing debt obligations (USD) |
| `num_credit_lines` | Number of open credit accounts |
| `missed_payments` | Missed payments in last 2 years |
| `default` | **Target**: 1 = defaulted, 0 = repaid |

Compatible with UCI German Credit and Kaggle Give Me Some Credit schemas — swap in real data with no code changes.

---

## Evaluation Metrics

| Metric | Why it matters for credit scoring |
|---|---|
| **Precision** | Of predicted defaults, how many truly defaulted? (cost of false alarms) |
| **Recall** | Of actual defaults, how many did we catch? (cost of missed defaults) |
| **F1-Score** | Balance of precision and recall |
| **ROC-AUC** | Overall discrimination ability across all thresholds |
| **Confusion Matrix** | Exact TP/FP/TN/FN counts |

`class_weight='balanced'` is applied to all three models so the minority class (defaulters) is not drowned out by the majority class.

---

## Credit Score Band

| Score Range | Band |
|---|---|
| 800–850 | Exceptional |
| 740–799 | Very Good |
| 670–739 | Good |
| 580–669 | Fair |
| 300–579 | Poor |

---

## Roadmap

- [ ] Integrate real dataset (UCI German Credit / Kaggle Give Me Some Credit)
- [ ] Add XGBoost / LightGBM for comparison
- [ ] Add SHAP values for model explainability
- [ ] Hyperparameter tuning via GridSearchCV / RandomizedSearchCV
- [ ] Cross-validation (k-fold) instead of single train/test split
- [ ] Streamlit web UI for interactive predictions

---

## Acknowledgments

- [CodeAlpha](https://www.codealpha.tech) — Machine Learning Internship Program
- [UCI German Credit Dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) — schema reference
- Built with [pandas](https://pandas.pydata.org/), [scikit-learn](https://scikit-learn.org/), and [Flask](https://flask.palletsprojects.com/)

---

*This project was built for educational purposes as part of an internship program. Feel free to fork and adapt.*
