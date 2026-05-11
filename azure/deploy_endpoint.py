"""
Azure ML Online Endpoint Deployment — deploys ChurnPredictionModel to the cloud.

This script:
  1. Finds the Production model in the MLflow/Azure ML Model Registry
  2. Creates a Managed Online Endpoint (HTTPS REST API in Azure)
  3. Deploys the model with auto-scaling
  4. Prints the live Azure endpoint URL + API key
  5. Tests the endpoint with a sample prediction

Usage:
    python azure/deploy_endpoint.py

Prerequisites:
    - azure/setup_workspace.py completed
    - azure/train_azure.py completed (model in Production)
"""

import json
import os
import time

import requests
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    CodeConfiguration,
    Environment,
    IdentityConfiguration,
)
from azure.ai.ml.constants import AssetTypes
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "azure_config.json")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
SCORE_DIR    = os.path.join(os.path.dirname(__file__), "scoring")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def get_credential():
    try:
        cred = DefaultAzureCredential()
        cred.get_token("https://management.azure.com/.default")
        return cred
    except Exception:
        return InteractiveBrowserCredential()


def get_client(cfg: dict) -> MLClient:
    return MLClient(
        credential=get_credential(),
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )


def get_latest_production_model(client: MLClient) -> Model:
    """Find the latest Production-stage model in the registry."""
    model_name = "ChurnPredictionModel"
    print(f"  Looking for model '{model_name}' in registry ...")
    try:
        models = list(client.models.list(name=model_name))
        if not models:
            raise RuntimeError(
                f"No model '{model_name}' found. Run train_azure.py first."
            )
        # Sort by version descending and return latest
        models.sort(key=lambda m: int(m.version), reverse=True)
        model = models[0]
        print(f"  ✓ Found model: {model.name} v{model.version}")
        return model
    except ResourceNotFoundError:
        raise RuntimeError(
            "Model not found in Azure ML registry. "
            "Run azure/train_azure.py first to train and register the model."
        )


def create_scoring_script():
    """Create the scoring script used by Azure ML endpoint."""
    os.makedirs(SCORE_DIR, exist_ok=True)

    score_py = os.path.join(SCORE_DIR, "score.py")
    with open(score_py, "w") as f:
        f.write('''"""
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
''')

    # conda env for scoring
    conda_yaml = os.path.join(SCORE_DIR, "conda.yaml")
    with open(conda_yaml, "w") as f:
        f.write("""name: churn-score-env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
    - mlflow==2.13.0
    - scikit-learn==1.4.2
    - numpy==1.26.4
    - xgboost==2.0.3
    - azureml-defaults
""")


def create_endpoint(client: MLClient, cfg: dict) -> ManagedOnlineEndpoint:
    """Create or get the online endpoint."""
    print(f"\n[2/4] Creating online endpoint '{cfg['endpoint_name']}' ...")
    try:
        endpoint = client.online_endpoints.get(cfg["endpoint_name"])
        print(f"  ✓ Endpoint already exists: {endpoint.scoring_uri}")
        return endpoint
    except ResourceNotFoundError:
        pass

    endpoint = ManagedOnlineEndpoint(
        name=cfg["endpoint_name"],
        description="AIN-3009 MLOps — Telco Churn Prediction API",
        auth_mode="key",
        identity=IdentityConfiguration(type="system_assigned"),
        tags={
            "project": "AIN3009-MLOps",
            "student": "Abdullah-Al-Shobaki-2284612",
        },
    )
    print("  Creating endpoint (takes ~1-2 minutes) ...")
    client.online_endpoints.begin_create_or_update(endpoint).result()
    endpoint = client.online_endpoints.get(cfg["endpoint_name"])
    print(f"  ✓ Endpoint created: {endpoint.scoring_uri}")
    return endpoint


def deploy_model(client: MLClient, cfg: dict, model: Model) -> ManagedOnlineDeployment:
    """Deploy the model to the endpoint."""
    print(f"\n[3/4] Deploying model to endpoint (takes 3-5 minutes) ...")
    create_scoring_script()

    deployment = ManagedOnlineDeployment(
        name=cfg["deployment_name"],
        endpoint_name=cfg["endpoint_name"],
        model=model.id,
        code_configuration=CodeConfiguration(
            code=SCORE_DIR,
            scoring_script="score.py",
        ),
        environment=Environment(
            conda_file=os.path.join(SCORE_DIR, "conda.yaml"),
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        ),
        instance_type="Standard_DS2_v2",
        instance_count=1,
    )

    client.online_deployments.begin_create_or_update(deployment).result()

    # Route 100% of traffic to this deployment
    endpoint = client.online_endpoints.get(cfg["endpoint_name"])
    endpoint.traffic = {cfg["deployment_name"]: 100}
    client.online_endpoints.begin_create_or_update(endpoint).result()

    print(f"  ✓ Model deployed successfully!")
    return deployment


def test_endpoint(client: MLClient, cfg: dict):
    """Send a test prediction to the live Azure endpoint."""
    print(f"\n[4/4] Testing live Azure endpoint ...")
    endpoint = client.online_endpoints.get(cfg["endpoint_name"])
    keys = client.online_endpoints.get_keys(cfg["endpoint_name"])

    sample = {
        "tenure": 5, "MonthlyCharges": 95.5, "TotalCharges": 477.5,
        "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
        "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {keys.primary_key}",
    }
    response = requests.post(
        endpoint.scoring_uri,
        json=sample,
        headers=headers,
        timeout=30,
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"  ✓ Live prediction from Azure:")
        print(f"    Churn probability : {result.get('churn_probability')}")
        print(f"    Risk level        : {result.get('risk_level')}")
        print(f"    Strategy          : {result.get('retention_strategy')}")
    else:
        print(f"  Response: {response.text[:300]}")

    return endpoint.scoring_uri, keys.primary_key


def save_endpoint_config(scoring_uri: str, api_key: str):
    """Save endpoint details for use by Streamlit dashboard."""
    out = {
        "azure_endpoint_url": scoring_uri,
        "azure_api_key": api_key,
    }
    path = os.path.join(os.path.dirname(__file__), "endpoint_config.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Endpoint config saved to: azure/endpoint_config.json")
    print("  (The Streamlit dashboard will use this automatically)")


def main():
    print("=" * 60)
    print("  Azure ML Online Endpoint Deployment")
    print("  AIN-3009 MLOps Term Project")
    print("=" * 60)

    cfg = load_config()
    print(f"\nConnecting to workspace '{cfg['workspace_name']}' ...")
    client = get_client(cfg)
    print("  ✓ Connected")

    print("\n[1/4] Finding production model ...")
    model = get_latest_production_model(client)

    endpoint = create_endpoint(client, cfg)
    deploy_model(client, cfg, model)
    scoring_uri, api_key = test_endpoint(client, cfg)
    save_endpoint_config(scoring_uri, api_key)

    print("\n" + "=" * 60)
    print("  ✓ Deployment complete!")
    print(f"\n  🌐 Azure Endpoint URL:")
    print(f"     {scoring_uri}")
    print(f"\n  🔑 API Key: {api_key[:8]}...")
    print(f"\n  Use this in the Streamlit dashboard (Tab 1 → Azure toggle)")
    print(f"  Or call directly:")
    print(f"""
    curl -X POST "{scoring_uri}" \\
      -H "Content-Type: application/json" \\
      -H "Authorization: Bearer {api_key[:8]}..." \\
      -d '{{"tenure": 5, "MonthlyCharges": 95.5, ...}}'
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()
