"""
Hyperparameter tuning using Hyperopt with MLflow nested run tracking.

Uses Gradient Boosting as the base model (best performer from baseline training).
50 evaluation trials — all logged as child runs under a parent "HyperTuning" run.
"""

import os
import sys
import warnings
import joblib
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXPERIMENT_NAME = "Telco_Churn_Experiments"
MAX_EVALS = 50


def objective(params, X_train, X_test, y_train, y_test, parent_run_id):
    """Hyperopt objective: train model, log child run, return loss."""
    with mlflow.start_run(run_name="hyperopt_trial", nested=True):
        mlflow.set_tag("parent_run_id", parent_run_id)
        mlflow.set_tag("tuning_method", "hyperopt_tpe")

        n_estimators    = int(params["n_estimators"])
        max_depth       = int(params["max_depth"])
        learning_rate   = params["learning_rate"]
        min_samples_split = int(params["min_samples_split"])
        subsample       = params["subsample"]

        log_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_split": min_samples_split,
            "subsample": subsample,
        }
        mlflow.log_params(log_params)

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_split=min_samples_split,
            subsample=subsample,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        f1  = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        mlflow.log_metrics({"f1": f1, "roc_auc": auc})

        return {"loss": -f1, "status": STATUS_OK, "f1": f1, "auc": auc}


def main():
    print("Loading data ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    search_space = {
        "n_estimators":     scope.int(hp.quniform("n_estimators",     100, 500, 50)),
        "max_depth":        scope.int(hp.quniform("max_depth",         2,   8,   1)),
        "learning_rate":    hp.loguniform("learning_rate",             np.log(0.01), np.log(0.3)),
        "min_samples_split":scope.int(hp.quniform("min_samples_split", 2,   10,  1)),
        "subsample":        hp.uniform("subsample",                    0.6,  1.0),
    }

    print(f"Starting Hyperopt tuning ({MAX_EVALS} trials) ...")

    with mlflow.start_run(run_name="HyperTuning_GradientBoosting") as parent_run:
        mlflow.set_tag("model_type", "GradientBoosting")
        mlflow.set_tag("tuning_method", "hyperopt_tpe")
        mlflow.log_param("max_evals", MAX_EVALS)

        parent_run_id = parent_run.info.run_id

        scaler_path = "/tmp/scaler_tuning.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        trials = Trials()

        best = fmin(
            fn=lambda params: objective(
                params, X_train, X_test, y_train, y_test, parent_run_id
            ),
            space=search_space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
            verbose=False,
        )

        # Convert hyperopt return types to plain Python types
        best_params = {
            "n_estimators":      int(best["n_estimators"]),
            "max_depth":         int(best["max_depth"]),
            "learning_rate":     float(best["learning_rate"]),
            "min_samples_split": int(best["min_samples_split"]),
            "subsample":         float(best["subsample"]),
        }
        print(f"\nBest params: {best_params}")

        # Retrain final model with best params and log to parent run
        final_model = GradientBoostingClassifier(
            **best_params, random_state=42
        )
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        y_prob = final_model.predict_proba(X_test)[:, 1]

        best_f1  = f1_score(y_test, y_pred, zero_division=0)
        best_auc = roc_auc_score(y_test, y_prob)

        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics({"best_f1": best_f1, "best_roc_auc": best_auc})

        mlflow.sklearn.log_model(
            final_model,
            artifact_path="best_tuned_model",
            input_example=X_test[:2],
        )

        print(f"Best tuned model  →  F1={best_f1:.4f}  ROC-AUC={best_auc:.4f}")
        print(f"Parent run ID: {parent_run_id}")
        print(f"Registered artifact: best_tuned_model")

    return best_params, best_f1, parent_run_id


if __name__ == "__main__":
    main()
