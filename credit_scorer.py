"""
Credit Scoring Model
CodeAlpha Machine Learning Internship - Task 1

Pipeline:
    1. Load dataset            -> data/applicants.csv
    2. EDA + cleaning          -> nulls, duplicates, dtype coercion,
                                   class-imbalance report
    3. Feature engineering     -> debt-to-income ratio, loan-to-income ratio,
                                   payment-to-income ratio, repayment burden
    4. Preprocessing           -> StandardScaler on numeric features,
                                   train/test split (80/20, stratified)
    5. Model training          -> Logistic Regression, Decision Tree,
                                   Random Forest — all three trained and
                                   compared side-by-side
    6. Evaluation              -> Accuracy, Precision, Recall, F1-Score,
                                   ROC-AUC for each model; confusion matrix
                                   printed for the best model
    7. Credit score mapping    -> probability output -> 300–850 score band
                                   (same scale as FICO)
    8. Prediction API          -> predict_single() accepts a dict of raw
                                   applicant features and returns class,
                                   probability, and credit score

Usage:
    python3 credit_scorer.py                    # full pipeline + demo
    python3 credit_scorer.py --data path.csv    # use your own dataset
    python3 credit_scorer.py --model rf         # lr | dt | rf (default: rf)

Revision note: DEFAULT_DATA_PATH is resolved relative to THIS FILE so the
script works regardless of the caller's working directory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

DEFAULT_DATA_PATH = str(Path(__file__).resolve().parent / "data" / "applicants.csv")

REQUIRED_COLUMNS = {
    "age", "income", "employment_years", "loan_amount",
    "loan_term_months", "credit_history", "existing_debts",
    "num_credit_lines", "missed_payments", "default",
}

# 300–850 mirrors the FICO score band
SCORE_MIN, SCORE_MAX = 300, 850


# ──────────────────────────────────────────────────────────────────────
# 1. LOAD
# ──────────────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Dataset not found at '{path}'. Pass --data /path/to/applicants.csv.")
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        sys.exit(f"Dataset is missing required column(s): {missing}")
    return df


# ──────────────────────────────────────────────────────────────────────
# 2. EDA + CLEANING
# ──────────────────────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 60)
    print("EDA & DATA CLEANING")
    print("=" * 60)
    print(f"Rows: {len(df)}   Columns: {list(df.columns)}\n")

    # Null report
    nulls = df[list(REQUIRED_COLUMNS)].isna().sum()
    print("Null counts:\n", nulls[nulls > 0].to_string() or "  none")

    # Duplicate applicant IDs (if column present)
    if "applicant_id" in df.columns:
        dupes = df.duplicated(subset="applicant_id").sum()
        print(f"\nDuplicate applicant IDs: {dupes}")

    # Drop rows with nulls in required columns and any duplicate targets
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    print(f"\nDropped {before - len(df)} row(s) with nulls -> {len(df)} remain")

    # Coerce numeric columns
    numeric_cols = [
        "age", "income", "employment_years", "loan_amount",
        "loan_term_months", "credit_history", "existing_debts",
        "num_credit_lines", "missed_payments", "default",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    print(f"Rows after dtype coercion: {len(df)}")

    # Class balance
    counts = df["default"].value_counts()
    print(f"\nClass distribution (default=1 → defaulted):")
    print(f"  Repaid (0): {counts.get(0, 0)}  |  Defaulted (1): {counts.get(1, 0)}")
    imbalance_ratio = counts.get(1, 0) / max(counts.get(0, 1), 1)
    if imbalance_ratio < 0.4:
        print("  ⚠  Minority class < 40 % of majority — consider class_weight='balanced'")

    # Basic statistics
    print("\nDescriptive statistics (numeric features):")
    print(df[numeric_cols].describe().to_string())
    print()

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates four derived ratio features:
      - debt_to_income      : existing monthly debts / (monthly income)
      - loan_to_income      : loan amount / annual income
      - payment_to_income   : estimated monthly repayment / monthly income
                              (simple amortisation approximation)
      - repayment_burden    : combined monthly debt + repayment / income
    These ratios are industry-standard credit-risk signals (DTI is
    explicitly used in US mortgage underwriting regulations).
    """
    df = df.copy()
    monthly_income = df["income"] / 12

    # Avoid division by zero
    safe_income   = monthly_income.replace(0, np.nan)
    safe_annual   = df["income"].replace(0, np.nan)
    safe_term     = df["loan_term_months"].replace(0, np.nan)

    df["debt_to_income"]    = (df["existing_debts"] / safe_income).fillna(0)
    df["loan_to_income"]    = (df["loan_amount"]    / safe_annual).fillna(0)
    estimated_payment       = df["loan_amount"] / safe_term
    df["payment_to_income"] = (estimated_payment    / safe_income).fillna(0)
    df["repayment_burden"]  = (
        (df["existing_debts"] + estimated_payment.fillna(0)) / safe_income
    ).fillna(0)

    return df


# ──────────────────────────────────────────────────────────────────────
# 4. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "age", "income", "employment_years", "loan_amount",
    "loan_term_months", "credit_history", "existing_debts",
    "num_credit_lines", "missed_payments",
    "debt_to_income", "loan_to_income", "payment_to_income", "repayment_burden",
]

TARGET_COLUMN = "default"


def preprocess(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Returns X_train, X_test, y_train, y_test, scaler.
    Scaler is fit on train only (no data leakage).
    """
    X = df[FEATURE_COLUMNS].values
    y = df[TARGET_COLUMN].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# ──────────────────────────────────────────────────────────────────────
# 5. MODEL TRAINING
# ──────────────────────────────────────────────────────────────────────
def build_models(seed: int = 42) -> dict:
    return {
        "lr": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        ),
        "dt": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }


def train_all(X_train, y_train, seed: int = 42) -> dict:
    models = build_models(seed)
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


# ──────────────────────────────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────────────────────────────
MODEL_LABELS = {"lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest"}


def evaluate_all(trained: dict, X_test, y_test) -> dict:
    """
    Evaluates every trained model and prints a comparison table.
    Returns a dict of {name: metrics_dict}.
    """
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    results = {}
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4),
        }
        results[name] = metrics

    # Print table
    header = f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'ROC-AUC':>9}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        label = MODEL_LABELS[name]
        print(
            f"{label:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
            f"{m['recall']:>8.4f} {m['f1']:>8.4f} {m['roc_auc']:>9.4f}"
        )

    # Best model by ROC-AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n✓ Best model: {MODEL_LABELS[best_name]} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")

    # Confusion matrix for best model
    best_model = trained[best_name]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nConfusion Matrix — {MODEL_LABELS[best_name]}:")
    print(f"                   Predicted")
    print(f"                Repaid  Default")
    print(f"  Actual Repaid   {cm[0,0]:>4d}    {cm[0,1]:>4d}")
    print(f"  Actual Default  {cm[1,0]:>4d}    {cm[1,1]:>4d}")

    # Full classification report for best model
    print(f"\nClassification Report — {MODEL_LABELS[best_name]}:")
    print(classification_report(y_test, y_pred_best, target_names=["Repaid", "Default"]))

    # Feature importances (Random Forest / Decision Tree only)
    if best_name in ("rf", "dt") and hasattr(best_model, "feature_importances_"):
        fi = sorted(
            zip(FEATURE_COLUMNS, best_model.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )
        print(f"Top feature importances — {MODEL_LABELS[best_name]}:")
        for feat, imp in fi[:8]:
            bar = "█" * int(imp * 40)
            print(f"  {feat:<22} {imp:.4f}  {bar}")
        print()

    return results, best_name


# ──────────────────────────────────────────────────────────────────────
# 7. CREDIT SCORE MAPPING
# ──────────────────────────────────────────────────────────────────────
def prob_to_credit_score(default_prob: float) -> int:
    """
    Maps default probability [0, 1] to a credit score [300, 850].
    A lower default probability → higher score (better creditworthiness).
    """
    repay_prob = 1.0 - float(default_prob)
    score = SCORE_MIN + repay_prob * (SCORE_MAX - SCORE_MIN)
    return int(round(np.clip(score, SCORE_MIN, SCORE_MAX)))


def score_band(score: int) -> str:
    if score >= 800:
        return "Exceptional"
    if score >= 740:
        return "Very Good"
    if score >= 670:
        return "Good"
    if score >= 580:
        return "Fair"
    return "Poor"


# ──────────────────────────────────────────────────────────────────────
# 8. SINGLE-APPLICANT PREDICTION
# ──────────────────────────────────────────────────────────────────────
def predict_single(
    applicant: dict,
    model,
    scaler: StandardScaler,
) -> dict:
    """
    Accepts a raw applicant feature dict (same keys as the CSV, minus
    applicant_id and default), and returns:
        {
          "default_probability": float,
          "predicted_class"    : int,   # 0 = Repaid, 1 = Default
          "verdict"            : str,
          "credit_score"       : int,
          "score_band"         : str,
        }
    """
    # Build a single-row DataFrame for feature engineering
    row = pd.DataFrame([applicant])
    # Add placeholder for columns engineer_features expects
    for col in REQUIRED_COLUMNS - {"default"}:
        if col not in row.columns:
            row[col] = 0
    row = engineer_features(row)

    X = row[FEATURE_COLUMNS].values
    X_scaled = scaler.transform(X)

    prob      = model.predict_proba(X_scaled)[0, 1]
    cls       = int(model.predict(X_scaled)[0])
    verdict   = "Default Risk" if cls == 1 else "Creditworthy"
    c_score   = prob_to_credit_score(prob)

    return {
        "default_probability": round(float(prob), 4),
        "predicted_class"    : cls,
        "verdict"            : verdict,
        "credit_score"       : c_score,
        "score_band"         : score_band(c_score),
    }


# ──────────────────────────────────────────────────────────────────────
# DEMO
# ──────────────────────────────────────────────────────────────────────
def demo(model, scaler):
    sample_applicants = [
        {
            "name"            : "Ahmed (Low Risk)",
            "age"             : 42, "income": 90000, "employment_years": 12,
            "loan_amount"     : 15000, "loan_term_months": 36,
            "credit_history"  : 1, "existing_debts": 250,
            "num_credit_lines": 6, "missed_payments": 0,
        },
        {
            "name"            : "Sara (Medium Risk)",
            "age"             : 31, "income": 45000, "employment_years": 3,
            "loan_amount"     : 25000, "loan_term_months": 48,
            "credit_history"  : 1, "existing_debts": 750,
            "num_credit_lines": 3, "missed_payments": 2,
        },
        {
            "name"            : "Bilal (High Risk)",
            "age"             : 23, "income": 22000, "employment_years": 1,
            "loan_amount"     : 40000, "loan_term_months": 60,
            "credit_history"  : 0, "existing_debts": 1800,
            "num_credit_lines": 1, "missed_payments": 5,
        },
    ]

    print("=" * 60)
    print("SAMPLE PREDICTIONS (qualitative evaluation)")
    print("=" * 60)
    for app in sample_applicants:
        name = app.pop("name")
        result = predict_single(app, model, scaler)
        print(f"\nApplicant : {name}")
        print(f"  Verdict          : {result['verdict']}")
        print(f"  Default Prob     : {result['default_probability']:.2%}")
        print(f"  Credit Score     : {result['credit_score']}  ({result['score_band']})")


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DEFAULT_DATA_PATH, help="path to applicants CSV")
    parser.add_argument("--model", default="rf", choices=["lr", "dt", "rf"],
                        help="model to expose via predict_single (default: rf)")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    # Step 1: Load
    df = load_data(args.data)

    # Step 2: EDA
    df = run_eda(df)

    # Step 3: Feature engineering
    df = engineer_features(df)

    # Step 4: Preprocess
    X_train, X_test, y_train, y_test, scaler = preprocess(df, seed=args.seed)

    # Step 5: Train
    print("=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    trained = train_all(X_train, y_train, seed=args.seed)
    print("  Logistic Regression  ✓")
    print("  Decision Tree        ✓")
    print("  Random Forest        ✓\n")

    # Step 6: Evaluate
    results, best_name = evaluate_all(trained, X_test, y_test)

    # Step 7 + 8: Demo
    best_model = trained[best_name]
    demo(best_model, scaler)

    print("\n" + "=" * 60)
    print("Try your own:")
    print("  from credit_scorer import load_data, run_eda, engineer_features,")
    print("        preprocess, train_all, predict_single")
    print("  df = engineer_features(run_eda(load_data('data/applicants.csv')))")
    print("  X_tr,X_te,y_tr,y_te,scaler = preprocess(df)")
    print("  model = train_all(X_tr,y_tr)['rf']")
    print("  print(predict_single({...}, model, scaler))")
    print("=" * 60)


if __name__ == "__main__":
    main()
