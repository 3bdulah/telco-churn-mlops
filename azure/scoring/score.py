"""
Azure ML Online Endpoint scoring script for ChurnPredictionModel.
"""
import json
import os
import numpy as np
import mlflow.sklearn

model = None

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0,
              "No phone service": 0, "No internet service": 0}
BINARY_COLS = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
               "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies"]
INTERNET_CATS = ["DSL", "Fiber optic", "No"]
CONTRACT_CATS = ["Month-to-month", "One year", "Two year"]
PAYMENT_CATS  = ["Bank transfer (automatic)", "Credit card (automatic)",
                 "Electronic check", "Mailed check"]
SCALER_MEAN = {"tenure": 32.37, "MonthlyCharges": 64.76, "TotalCharges": 2279.73}
SCALER_STD  = {"tenure": 24.56, "MonthlyCharges": 30.09, "TotalCharges": 2266.77}
FEATURE_ORDER = (
    ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
     "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
     "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
     "tenure", "MonthlyCharges", "TotalCharges"]
    + [f"InternetService_{c}" for c in INTERNET_CATS]
    + [f"Contract_{c}" for c in CONTRACT_CATS]
    + [f"PaymentMethod_{c}" for c in PAYMENT_CATS]
)


def init():
    global model
    model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", "."), "best_tuned_model")
    if not os.path.exists(model_path):
        model_path = os.path.join(os.environ.get("AZUREML_MODEL_DIR", "."), "model")
    model = mlflow.sklearn.load_model(model_path)


def build_features(d: dict) -> np.ndarray:
    row = {"SeniorCitizen": int(d.get("SeniorCitizen", 0))}
    for col in BINARY_COLS:
        row[col] = BINARY_MAP.get(str(d.get(col, "No")), 0)
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        row[col] = (float(d[col]) - SCALER_MEAN[col]) / SCALER_STD[col]
    for cat in INTERNET_CATS:
        row[f"InternetService_{cat}"] = int(d.get("InternetService", "") == cat)
    for cat in CONTRACT_CATS:
        row[f"Contract_{cat}"] = int(d.get("Contract", "") == cat)
    for cat in PAYMENT_CATS:
        row[f"PaymentMethod_{cat}"] = int(d.get("PaymentMethod", "") == cat)
    return np.array([[row.get(f, 0) for f in FEATURE_ORDER]], dtype=float)


def get_strategy(prob: float, contract: str, monthly: float) -> tuple:
    if prob >= 0.70:
        risk = "High"
        if contract == "Month-to-month":
            strategy = "HIGH RISK: Offer 20% discount to upgrade to 1-year contract."
        elif monthly > 70:
            strategy = "HIGH RISK: Offer 15% loyalty discount + dedicated CSM."
        else:
            strategy = "HIGH RISK: Call customer within 24 hours."
    elif prob >= 0.40:
        risk = "Medium"
        strategy = "MEDIUM RISK: Send proactive engagement email with referral bonus."
    else:
        risk = "Low"
        strategy = "LOW RISK: No action required. Schedule quarterly NPS survey."
    return risk, strategy


def run(raw_data):
    try:
        data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        results = []
        customers = data if isinstance(data, list) else [data]
        for customer in customers:
            X = build_features(customer)
            prob = float(model.predict_proba(X)[0, 1])
            pred = prob >= 0.5
            risk, strategy = get_strategy(
                prob, customer.get("Contract", ""), float(customer.get("MonthlyCharges", 0))
            )
            annual_risk = round(float(customer.get("MonthlyCharges", 0)) * 12 * prob, 2)
            results.append({
                "churn_probability": round(prob, 4),
                "churn_prediction":  pred,
                "risk_level":        risk,
                "retention_strategy": strategy,
                "annual_revenue_at_risk": annual_risk,
            })
        return json.dumps(results if len(results) > 1 else results[0])
    except Exception as e:
        return json.dumps({"error": str(e)})
