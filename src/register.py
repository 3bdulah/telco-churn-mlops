"""
MLflow Model Registry workflow.

Finds the best run from the tuning experiment, registers it as
"ChurnPredictionModel", transitions Staging → Production, and validates it.
"""

import os
import sys
import warnings
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = "http://localhost:5500"
EXPERIMENT_NAME = "Telco_Churn_Experiments"
MODEL_NAME = "ChurnPredictionModel"
F1_THRESHOLD = 0.55    # Minimum F1 to promote to Production


def find_best_run(client: MlflowClient) -> str:
    """Return run_id of the HyperTuning parent run with the highest best_f1."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found. Run src/train.py first."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.tuning_method = 'hyperopt_tpe' AND attributes.status = 'FINISHED'",
        order_by=["metrics.best_f1 DESC"],
        max_results=1,
    )

    if not runs:
        # Fallback: pick best run by roc_auc from baseline training
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.roc_auc DESC"],
            max_results=1,
        )

    if not runs:
        raise RuntimeError("No finished runs found. Run src/train.py first.")

    best_run = runs[0]
    print(f"  Best run: {best_run.info.run_id}")
    print(f"  Run name: {best_run.data.tags.get('mlflow.runName', 'N/A')}")
    f1  = best_run.data.metrics.get("best_f1", best_run.data.metrics.get("f1", 0))
    auc = best_run.data.metrics.get("best_roc_auc", best_run.data.metrics.get("roc_auc", 0))
    print(f"  F1={f1:.4f}  ROC-AUC={auc:.4f}")
    return best_run.info.run_id


def register_model(client: MlflowClient, run_id: str) -> str:
    """Register the model artifact from run_id and return version number."""
    # Try best_tuned_model path first, then generic model path
    for artifact_path in ("best_tuned_model", "model"):
        try:
            model_uri = f"runs:/{run_id}/{artifact_path}"
            result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
            version = result.version
            print(f"  Registered '{MODEL_NAME}' version {version} from artifact '{artifact_path}'")
            return version
        except Exception:
            continue
    raise RuntimeError("Could not find a model artifact in the best run.")


def validate_model(model_uri: str, X_test, y_test, threshold: float) -> bool:
    """Load model, run predictions, return True if F1 meets threshold."""
    model = mlflow.sklearn.load_model(model_uri)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"  Validation → F1={f1:.4f}  ROC-AUC={auc:.4f}  (threshold={threshold})")
    return f1 >= threshold


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    print("\n[1/5] Loading test data for validation ...")
    _, X_test, _, y_test, _, _ = load_and_preprocess()

    print("\n[2/5] Finding best run ...")
    run_id = find_best_run(client)

    print("\n[3/5] Registering model in Model Registry ...")
    version = register_model(client, run_id)

    # Allow MLflow a moment to register
    import time; time.sleep(2)

    print(f"\n[4/5] Transitioning version {version} → Staging ...")
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage="Staging",
        archive_existing_versions=False,
    )
    print(f"  '{MODEL_NAME}' v{version} is now in Staging")

    model_uri_staging = f"models:/{MODEL_NAME}/Staging"
    passed = validate_model(model_uri_staging, X_test, y_test, F1_THRESHOLD)

    if passed:
        print(f"\n[5/5] Validation passed. Transitioning → Production ...")
        client.transition_model_version_stage(
            name=MODEL_NAME, version=version, stage="Production",
            archive_existing_versions=True,
        )
        print(f"  '{MODEL_NAME}' v{version} is now in PRODUCTION ✓")
    else:
        print(f"\n[5/5] Validation FAILED (F1 < {F1_THRESHOLD}). Staying in Staging.")
        return

    # Print registry summary
    print("\n--- Model Registry Summary ---")
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"  Version {mv.version:>2}  |  Stage: {mv.current_stage:<12}  "
              f"|  Run: {mv.run_id[:8]}...")

    print(f"\nTo load production model:")
    print(f"  mlflow.sklearn.load_model('models:/{MODEL_NAME}/Production')")


if __name__ == "__main__":
    main()
