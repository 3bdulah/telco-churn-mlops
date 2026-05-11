"""
Azure ML Workspace Setup — Step 1 of Azure integration.

This script:
  1. Connects to (or creates) your Azure ML workspace
  2. Creates a compute cluster for training jobs
  3. Prints your Azure ML MLflow tracking URI
  4. Saves the workspace connection details

Run this ONCE before running train_azure.py or deploy_endpoint.py.

Usage:
    python azure/setup_workspace.py

Prerequisites:
    1. Fill in azure/azure_config.json with your real values
    2. pip install azure-ai-ml azure-identity
    3. Be logged into Azure: az login  (or the script will open a browser)
"""

import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    Workspace,
)
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.resource.resources.models import ResourceGroup

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "azure_config.json")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    missing = [k for k, v in cfg.items() if v.startswith("YOUR_")]
    if missing:
        raise ValueError(
            f"Please fill in these fields in azure/azure_config.json: {missing}\n"
            "You can find these values in the Azure Portal or from your professor."
        )
    return cfg


def get_credential():
    """Try DefaultAzureCredential first (works with az login), fall back to browser."""
    try:
        cred = DefaultAzureCredential()
        # Test it works
        cred.get_token("https://management.azure.com/.default")
        return cred
    except Exception:
        print("  DefaultAzureCredential failed — opening browser login ...")
        return InteractiveBrowserCredential()


def create_resource_group_if_needed(cfg: dict, credential) -> None:
    """Create the resource group if it doesn't exist. Safe to call on existing groups."""
    print(f"\n[1/4] Checking resource group '{cfg['resource_group']}' ...")
    rg_client = ResourceManagementClient(credential, cfg["subscription_id"])
    if rg_client.resource_groups.check_existence(cfg["resource_group"]):
        print(f"  ✓ Resource group already exists")
        return
    print(f"  Creating resource group '{cfg['resource_group']}' in {cfg['location']} ...")
    rg_client.resource_groups.create_or_update(
        cfg["resource_group"],
        ResourceGroup(location=cfg["location"]),
    )
    print(f"  ✓ Resource group created")


def get_or_create_workspace(cfg: dict, credential) -> MLClient:
    """Connect to existing workspace or create a new one."""
    print(f"\n[2/4] Connecting to Azure ML workspace '{cfg['workspace_name']}' ...")
    client = MLClient(
        credential=credential,
        subscription_id=cfg["subscription_id"],
        resource_group_name=cfg["resource_group"],
        workspace_name=cfg["workspace_name"],
    )
    try:
        ws = client.workspaces.get(cfg["workspace_name"])
        print(f"  ✓ Connected to existing workspace: {ws.name}  (region: {ws.location})")
    except Exception:
        print(f"  Workspace not found — creating new workspace '{cfg['workspace_name']}' ...")
        ws = client.workspaces.begin_create(
            Workspace(
                name=cfg["workspace_name"],
                location=cfg["location"],
                resource_group=cfg["resource_group"],
                description="AIN-3009 MLOps Term Project — Telco Churn Prediction",
            )
        ).result()
        print(f"  ✓ Workspace created: {ws.name}")
    return client


def create_compute_cluster(client: MLClient, cfg: dict):
    """Create a CPU compute cluster for training jobs (auto-scales 0→2 nodes)."""
    print(f"\n[3/4] Setting up compute cluster '{cfg['compute_name']}' ...")
    try:
        cluster = client.compute.get(cfg["compute_name"])
        print(f"  ✓ Compute cluster already exists: {cluster.name}  ({cluster.size})")
    except Exception:
        print("  Creating new compute cluster (this may take 1-2 minutes) ...")
        cluster = AmlCompute(
            name=cfg["compute_name"],
            type="amlcompute",
            size="Standard_DS3_v2",   # 4 vCPU, 14 GB RAM — good for sklearn training
            min_instances=0,           # scale to 0 when idle (saves cost)
            max_instances=2,
            idle_time_before_scale_down=120,
            tier="Dedicated",
        )
        client.compute.begin_create_or_update(cluster).result()
        print(f"  ✓ Compute cluster created: {cfg['compute_name']} (Standard_DS3_v2, 0-2 nodes)")
    return cluster


def print_mlflow_uri(client: MLClient):
    """Print the Azure ML MLflow tracking URI."""
    print("\n[4/4] Fetching MLflow tracking URI ...")
    tracking_uri = client.workspaces.get(
        client.workspace_name
    ).mlflow_tracking_uri
    print(f"\n  ✓ Azure ML MLflow Tracking URI:")
    print(f"    {tracking_uri}")
    print(f"\n  To use Azure ML as your MLflow backend, set:")
    print(f"    MLFLOW_TRACKING_URI={tracking_uri}")
    print(f"\n  Or use it in Python:")
    print(f"    import mlflow")
    print(f"    mlflow.set_tracking_uri('{tracking_uri}')")

    # Save tracking URI for use by other scripts
    uri_path = os.path.join(os.path.dirname(__file__), "mlflow_tracking_uri.txt")
    with open(uri_path, "w") as f:
        f.write(tracking_uri)
    print(f"\n  URI saved to: azure/mlflow_tracking_uri.txt")
    return tracking_uri


def main():
    print("=" * 60)
    print("  Azure ML Workspace Setup — AIN-3009 MLOps Term Project")
    print("=" * 60)

    print("\nLoading azure_config.json ...")
    cfg = load_config()
    print(f"  Subscription : {cfg['subscription_id'][:8]}...")
    print(f"  Resource Group: {cfg['resource_group']}")
    print(f"  Workspace    : {cfg['workspace_name']}")
    print(f"  Location     : {cfg['location']}")

    print("\nAuthenticating with Azure ...")
    credential = get_credential()
    print("  ✓ Authentication successful")

    create_resource_group_if_needed(cfg, credential)
    client = get_or_create_workspace(cfg, credential)
    create_compute_cluster(client, cfg)
    tracking_uri = print_mlflow_uri(client)

    print("\n" + "=" * 60)
    print("  ✓ Azure ML setup complete!")
    print("  Next steps:")
    print("    1. Run: python azure/train_azure.py")
    print("    2. Run: python azure/deploy_endpoint.py")
    print("=" * 60)

    return client, tracking_uri


if __name__ == "__main__":
    main()
