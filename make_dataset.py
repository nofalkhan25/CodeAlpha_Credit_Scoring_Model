"""
Generates data/applicants.csv — the 200-applicant curated sample dataset
used to validate the full Credit Scoring pipeline end-to-end.

Run:
    python3 make_dataset.py

The CSV schema matches common credit scoring datasets (e.g., UCI German
Credit, Kaggle Give Me Some Credit), so the same credit_scorer.py code
works against real data with no changes — just matching column names.

Columns:
    applicant_id      : unique identifier
    age               : applicant age (years)
    income            : annual income (USD)
    employment_years  : years at current employer
    loan_amount       : requested loan amount (USD)
    loan_term_months  : requested loan term
    credit_history    : 0 = bad/none, 1 = good
    existing_debts    : total existing monthly debt obligations (USD)
    num_credit_lines  : number of open credit accounts
    missed_payments   : number of missed payments in last 2 years
    default           : target — 1 = defaulted, 0 = repaid
"""

import os
import random
import numpy as np
import pandas as pd


def create_sample_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    records = []
    for i in range(1, n + 1):
        # Base risk profile
        risk = random.choice(["low", "medium", "high"])

        if risk == "low":
            age              = int(np.clip(np.random.normal(42, 8), 25, 65))
            income           = int(np.clip(np.random.normal(80000, 15000), 40000, 150000))
            employment_years = int(np.clip(np.random.normal(10, 4), 2, 30))
            loan_amount      = int(np.clip(np.random.normal(15000, 5000), 2000, 40000))
            loan_term_months = random.choice([12, 24, 36])
            credit_history   = 1
            existing_debts   = int(np.clip(np.random.normal(300, 100), 0, 800))
            num_credit_lines = int(np.clip(np.random.normal(5, 2), 1, 12))
            missed_payments  = int(np.clip(np.random.poisson(0.2), 0, 2))
            default          = 0 if random.random() < 0.90 else 1

        elif risk == "medium":
            age              = int(np.clip(np.random.normal(35, 10), 20, 60))
            income           = int(np.clip(np.random.normal(50000, 12000), 25000, 100000))
            employment_years = int(np.clip(np.random.normal(5, 3), 0, 20))
            loan_amount      = int(np.clip(np.random.normal(22000, 8000), 5000, 60000))
            loan_term_months = random.choice([24, 36, 48])
            credit_history   = random.choice([0, 1])
            existing_debts   = int(np.clip(np.random.normal(700, 200), 100, 2000))
            num_credit_lines = int(np.clip(np.random.normal(3, 2), 0, 10))
            missed_payments  = int(np.clip(np.random.poisson(1.0), 0, 5))
            default          = 0 if random.random() < 0.60 else 1

        else:  # high
            age              = int(np.clip(np.random.normal(28, 7), 18, 55))
            income           = int(np.clip(np.random.normal(28000, 8000), 12000, 55000))
            employment_years = int(np.clip(np.random.normal(2, 2), 0, 10))
            loan_amount      = int(np.clip(np.random.normal(35000, 12000), 8000, 80000))
            loan_term_months = random.choice([36, 48, 60])
            credit_history   = 0
            existing_debts   = int(np.clip(np.random.normal(1400, 400), 400, 4000))
            num_credit_lines = int(np.clip(np.random.normal(2, 1), 0, 6))
            missed_payments  = int(np.clip(np.random.poisson(3.0), 0, 10))
            default          = 0 if random.random() < 0.25 else 1

        records.append({
            "applicant_id"      : f"APP{i:04d}",
            "age"               : age,
            "income"            : income,
            "employment_years"  : employment_years,
            "loan_amount"       : loan_amount,
            "loan_term_months"  : loan_term_months,
            "credit_history"    : credit_history,
            "existing_debts"    : existing_debts,
            "num_credit_lines"  : num_credit_lines,
            "missed_payments"   : missed_payments,
            "default"           : default,
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = create_sample_dataset(n=200, seed=42)
    df.to_csv("data/applicants.csv", index=False)
    default_rate = df["default"].mean() * 100
    print(
        f"Sample dataset created at data/applicants.csv\n"
        f"  Rows      : {len(df)}\n"
        f"  Columns   : {list(df.columns)}\n"
        f"  Default % : {default_rate:.1f}%\n"
    )
