"""
FastAPI deployment for the ChurnPredictionModel.

Endpoints:
  GET  /health   — liveness check
  POST /predict  — churn probability + proactive retention strategy
"""

import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
MODEL_NAME  = "ChurnPredictionModel"
MODEL_STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Real-time customer churn prediction with proactive retention strategies. "
        "Model: GradientBoosting trained on Telco Customer Churn dataset. "
        "Part of AIN-3009 MLOps Term Project."
    ),
    version="1.0.0",
)

# Load model at startup — cached for the lifetime of the process
_model = None


def get_model():
    global _model
    if _model is None:
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        _model = mlflow.sklearn.load_model(model_uri)
    return _model


# ─── Request / Response schemas ───────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    """Raw customer attributes — mirrors the original Telco dataset columns."""
    tenure: float = Field(..., ge=0, description="Months as a customer")
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)
    gender: str = Field("Male", description="'Male' or 'Female'")
    SeniorCitizen: int = Field(0, ge=0, le=1)
    Partner: str = Field("No")
    Dependents: str = Field("No")
    PhoneService: str = Field("Yes")
    MultipleLines: str = Field("No")
    InternetService: str = Field("Fiber optic", description="'DSL', 'Fiber optic', or 'No'")
    OnlineSecurity: str = Field("No")
    OnlineBackup: str = Field("No")
    DeviceProtection: str = Field("No")
    TechSupport: str = Field("No")
    StreamingTV: str = Field("No")
    StreamingMovies: str = Field("No")
    Contract: str = Field("Month-to-month", description="'Month-to-month', 'One year', or 'Two year'")
    PaperlessBilling: str = Field("Yes")
    PaymentMethod: str = Field("Electronic check")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    retention_strategy: str
    model_name: str
    model_stage: str


# ─── Feature engineering (mirrors src/preprocess.py logic) ────────────────────

BINARY_MAP = {
    "Yes": 1, "No": 0, "Male": 1, "Female": 0,
    "No phone service": 0, "No internet service": 0,
}

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Categorical dummies expected by the model (created by pd.get_dummies during training)
INTERNET_CATS   = ["DSL", "Fiber optic", "No"]
CONTRACT_CATS   = ["Month-to-month", "One year", "Two year"]
PAYMENT_CATS    = ["Bank transfer (automatic)", "Credit card (automatic)",
                   "Electronic check", "Mailed check"]

# StandardScaler params learned from training data (approximate population stats)
# In production these would be serialised alongside the model; here we use the
# training-set approximations that are stable for the Telco dataset.
SCALER_MEAN = {"tenure": 32.37, "MonthlyCharges": 64.76, "TotalCharges": 2279.73}
SCALER_STD  = {"tenure": 24.56, "MonthlyCharges": 30.09, "TotalCharges": 2266.77}


def build_feature_vector(customer: CustomerFeatures) -> np.ndarray:
    """Convert raw customer data into the same feature vector the model expects."""
    d = customer.dict()

    # Binary encode
    row = {}
    row["SeniorCitizen"] = int(d["SeniorCitizen"])
    for col in BINARY_COLS:
        row[col] = BINARY_MAP.get(str(d.get(col, "No")), 0)

    # Scale numeric
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        row[col] = (float(d[col]) - SCALER_MEAN[col]) / SCALER_STD[col]

    # One-hot InternetService
    for cat in INTERNET_CATS:
        row[f"InternetService_{cat}"] = int(d["InternetService"] == cat)

    # One-hot Contract
    for cat in CONTRACT_CATS:
        row[f"Contract_{cat}"] = int(d["Contract"] == cat)

    # One-hot PaymentMethod
    for cat in PAYMENT_CATS:
        row[f"PaymentMethod_{cat}"] = int(d["PaymentMethod"] == cat)

    # Build array in the exact column order the model was trained on
    feature_order = (
        ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
         "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
         "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling",
         "tenure", "MonthlyCharges", "TotalCharges"]
        + [f"InternetService_{c}" for c in INTERNET_CATS]
        + [f"Contract_{c}" for c in CONTRACT_CATS]
        + [f"PaymentMethod_{c}" for c in PAYMENT_CATS]
    )
    return np.array([[row.get(f, 0) for f in feature_order]], dtype=float)


# ─── Retention strategy logic ─────────────────────────────────────────────────

def get_retention_strategy(prob: float, customer: CustomerFeatures) -> tuple[str, str]:
    """Return (risk_level, retention_strategy) based on probability and profile."""
    if prob >= 0.70:
        risk = "High"
        if customer.Contract == "Month-to-month":
            strategy = (
                "HIGH RISK: Customer is on a month-to-month contract. "
                "Offer a 20% discount to upgrade to a 1-year contract and "
                "provide a free tech support add-on for the first 3 months."
            )
        elif customer.MonthlyCharges > 70:
            strategy = (
                "HIGH RISK: Customer has high monthly charges. "
                "Offer a loyalty discount of 15% on their current plan and "
                "assign a dedicated customer success manager."
            )
        else:
            strategy = (
                "HIGH RISK: Immediate outreach required. "
                "Call customer within 24 hours with a personalised retention offer."
            )
    elif prob >= 0.40:
        risk = "Medium"
        strategy = (
            "MEDIUM RISK: Send a proactive engagement email highlighting "
            "unused features (e.g., streaming, tech support) and offer a "
            "referral bonus or loyalty points reward."
        )
    else:
        risk = "Low"
        strategy = (
            "LOW RISK: Customer is satisfied. "
            "No immediate action required. Consider a routine NPS survey "
            "in the next quarterly cycle."
        )
    return risk, strategy


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "stage": MODEL_STAGE,
        "tracking_uri": MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict churn probability")
def predict(customer: CustomerFeatures):
    try:
        model = get_model()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model not available: {exc}. Ensure MLflow server is running and model is in Production stage.",
        )

    try:
        X = build_feature_vector(customer)
        prob = float(model.predict_proba(X)[0, 1])
        pred = prob >= 0.5
        risk, strategy = get_retention_strategy(prob, customer)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction error: {exc}")

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_level=risk,
        retention_strategy=strategy,
        model_name=MODEL_NAME,
        model_stage=MODEL_STAGE,
    )


# ─── Local dev runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
