"""
Optional Flask API for the Credit Scoring Model.

Run:
    python3 app.py

Endpoints:
    POST /predict
        Body (JSON):
        {
            "age": 35,
            "income": 55000,
            "employment_years": 5,
            "loan_amount": 20000,
            "loan_term_months": 36,
            "credit_history": 1,
            "existing_debts": 600,
            "num_credit_lines": 3,
            "missed_payments": 1
        }
        Returns:
        {
            "default_probability": 0.3412,
            "predicted_class": 0,
            "verdict": "Creditworthy",
            "credit_score": 553,
            "score_band": "Fair"
        }

    GET /health
        Returns {"status": "ok"} — for uptime checks.

    GET /features
        Returns the list of required input features.

Revision notes:
    - Model and scaler are built once at startup (not per-request).
    - Missing fields in POST body return a descriptive 400, not a 500.
    - Non-numeric field values return a clean 400 with the offending field.
    - DEFAULT_DATA_PATH from credit_scorer.py is used so the path resolves
      regardless of the caller's working directory.
"""

from flask import Flask, jsonify, request

from credit_scorer import (
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    engineer_features,
    load_data,
    predict_single,
    preprocess,
    run_eda,
    train_all,
)

app = Flask(__name__)

# ── Build the model once at startup ─────────────────────────────────
print("Loading dataset and training model, please wait…")
_df = engineer_features(run_eda(load_data(DEFAULT_DATA_PATH)))
X_train, _, y_train, _, _scaler = preprocess(_df)
_model = train_all(X_train, y_train)["rf"]   # Random Forest by default
print("Model ready.\n")

# Required raw input fields (before feature engineering)
RAW_FIELDS = [
    "age", "income", "employment_years", "loan_amount",
    "loan_term_months", "credit_history", "existing_debts",
    "num_credit_lines", "missed_payments",
]


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/features")
def features():
    return jsonify({
        "required_input_fields"   : RAW_FIELDS,
        "engineered_features_used": FEATURE_COLUMNS,
    })


@app.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Check required fields
    missing = [f for f in RAW_FIELDS if f not in body]
    if missing:
        return jsonify({"error": f"Missing field(s): {missing}"}), 400

    # Coerce to numeric
    applicant = {}
    for field in RAW_FIELDS:
        try:
            applicant[field] = float(body[field])
        except (TypeError, ValueError):
            return jsonify({
                "error": f"Field '{field}' must be numeric, got {body[field]!r}"
            }), 400

    result = predict_single(applicant, _model, _scaler)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
