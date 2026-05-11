"""
Azure ML Training Job — submits the full training pipeline to cloud compute.

This script:
  1. Connects to your Azure ML workspace
  2. Uploads the project code + dataset as a job input
  3. Submits train.py → tune.py → register.py as sequential Azure ML jobs
  4. All MLflow experiments are logged to Azure ML (visible in Azure ML Studio)
  5. Streams job logs to your terminal

Usage:
    python azure/train_azure.py

Prerequisites:
    - azure/azure_config.json filled in with real values
    - python azure/setup_workspace.py  (run once first)
"""

import json
import os
import time

from azure.ai.ml import MLClient, command, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "azure_config.json")
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")


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


def get_or_create_environment(client: MLClient) -> str:
    """Register a curated environment with all project dependencies."""
    env_name = "churn-mlops-env"
    print(f"  Setting up Azure ML environment '{env_name}' ...")
    try:
        env = client.environments.get(env_name, label="latest")
        print(f"  ✓ Environment already exists (version {env.version})")
        return f"{env_name}@latest"
    except Exception:
        pass

    env = Environment(
        name=env_name,
        version="1",
        description="AIN-3009 MLOps Churn Prediction",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        conda_file={
            "name": "churn-env",
            "channels": ["defaults", "conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {"pip": [
                    "mlflow==2.13.0", "scikit-learn==1.4.2", "pandas==2.2.2",
                    "numpy==1.26.4", "hyperopt==0.2.7", "xgboost==2.0.3",
                    "shap==0.45.0", "evidently==0.4.30", "imbalanced-learn==0.12.2",
                    "matplotlib==3.8.4", "seaborn==0.13.2", "joblib==1.4.2",
                    "azure-ai-ml", "azureml-mlflow",
                ]},
            ],
        },
    )
    client.environments.create_or_update(env)
    print(f"  ✓ Environment '{env_name}' registered")
    return f"{env_name}@latest"


def submit_training_job(client: MLClient, cfg: dict, env_ref: str) -> str:
    """Submit the training script as an Azure ML command job."""
    print(f"\n[2/3] Submitting training job to compute '{cfg['compute_name']}' ...")

    job = command(
        code=PROJECT_ROOT,                     # upload entire project directory
        command=(
            "python src/train.py && "
            "python src/tune.py && "
            "python src/register.py"
        ),
        environment=env_ref,
        compute=cfg["compute_name"],
        display_name="Telco-Churn-Full-Pipeline",
        description=(
            "AIN-3009 MLOps Term Project: train 4 models (LR, RF, GB, XGBoost) "
            "with SMOTE + CV + SHAP, then Hyperopt tuning, then register to Production."
        ),
        tags={
            "project": "AIN3009-MLOps",
            "student": "Abdullah-Al-Shobaki-2284612",
            "dataset": "Telco-Customer-Churn",
        },
        environment_variables={
            "MLFLOW_TRACKING_URI": client.workspaces.get(
                cfg["workspace_name"]
            ).mlflow_tracking_uri,
        },
    )

    submitted = client.jobs.create_or_update(job)
    print(f"  ✓ Job submitted!")
    print(f"  Job name : {submitted.name}")
    print(f"  Studio URL: {submitted.studio_url}")
    return submitted.name


def stream_logs(client: MLClient, job_name: str):
    """Stream job logs to terminal until job completes."""
    print(f"\n[3/3] Streaming logs (Ctrl+C to stop streaming, job continues in cloud) ...")
    print("-" * 60)
    try:
        client.jobs.stream(job_name)
    except KeyboardInterrupt:
        print("\n  (Log streaming stopped — job is still running in Azure)")

    # Final status
    job = client.jobs.get(job_name)
    status = job.status
    icon = "✓" if status == "Completed" else "✗"
    print(f"\n  {icon} Job '{job_name}' status: {status}")
    if status == "Completed":
        print("  ✓ Training complete! Check Azure ML Studio for experiment results.")
        print(f"  ✓ Model 'ChurnPredictionModel' is now in Production stage.")
        print(f"  Next: python azure/deploy_endpoint.py")
    else:
        print(f"  Check Azure ML Studio for error details: {job.studio_url}")


def main():
    print("=" * 60)
    print("  Azure ML Training Job — AIN-3009 MLOps Term Project")
    print("=" * 60)

    cfg = load_config()
    print(f"\nConnecting to workspace '{cfg['workspace_name']}' ...")
    client = get_client(cfg)
    print("  ✓ Connected")

    print("\n[1/3] Setting up compute environment ...")
    env_ref = get_or_create_environment(client)

    job_name = submit_training_job(client, cfg, env_ref)
    stream_logs(client, job_name)


if __name__ == "__main__":
    main()
