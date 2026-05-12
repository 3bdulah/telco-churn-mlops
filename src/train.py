"""
Train four ML models with MLflow tracking, cross-validation, and SHAP explainability.

Experiment: "Telco_Churn_Experiments"
Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
Extras: SMOTE balancing, class_weight, 5-fold CV, SHAP summary plots per model
"""

import os
import sys
import warnings
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)
import xgboost as xgb
import shap

import mlflow
import mlflow.sklearn

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
EXPERIMENT_NAME = "Telco_Churn_Experiments"


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix") -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontweight="bold")
    plt.tight_layout()
    path = f"/tmp/{title.replace(' ', '_')}_cm.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def plot_roc_curve(y_true, y_prob, title="ROC Curve") -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1976D2", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color="#1976D2")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = f"/tmp/{title.replace(' ', '_')}_roc.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def plot_shap_summary(model, X_sample, feature_names, title="SHAP Summary") -> str:
    """Generate and save a SHAP beeswarm summary plot."""
    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_sample)
            # For binary classifiers that return list (RF), take class-1 values
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
        else:
            masker    = shap.maskers.Independent(X_sample)
            explainer = shap.LinearExplainer(model, masker)
            shap_vals = explainer.shap_values(X_sample)
            if hasattr(shap_vals, "values"):
                shap_vals = shap_vals.values

        feat_arr = np.array(feature_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(shap_vals, X_sample, feature_names=feat_arr,
                          show=False, plot_size=None)
        plt.title(f"{title}", fontweight="bold", fontsize=12)
        plt.tight_layout()
        path = f"/tmp/{title.replace(' ', '_')}_shap.png"
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close("all")
        return path
    except Exception as e:
        print(f"    SHAP plot skipped: {e}")
        return None


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def run_experiment(model, params: dict, model_name: str,
                   X_train, X_test, y_train, y_test, feature_names, scaler):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_params(params)

        # Fit model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Test-set metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # 5-fold cross-validation on training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5,
                                    scoring="f1", n_jobs=-1)
        mlflow.log_metric("cv_f1_mean", float(cv_scores.mean()))
        mlflow.log_metric("cv_f1_std", float(cv_scores.std()))

        # Plots
        cm_path  = plot_confusion_matrix(y_test, y_pred, f"{model_name} Confusion Matrix")
        roc_path = plot_roc_curve(y_test, y_prob, f"{model_name} ROC Curve")
        mlflow.log_artifact(cm_path,  artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")

        # SHAP summary plot (on a sample of test data for speed)
        shap_sample = X_test[:200]
        shap_path = plot_shap_summary(model, shap_sample, feature_names,
                                      title=f"{model_name} SHAP")
        if shap_path:
            mlflow.log_artifact(shap_path, artifact_path="plots")

        # Log scaler so the API can load it instead of using hardcoded stats
        scaler_path = f"/tmp/scaler_{model_name.replace(' ', '_')}.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="scaler")

        # Log model artifact
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=None,
            input_example=X_test[:2],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"  [{model_name}]  F1={metrics['f1']:.4f}  "
              f"ROC-AUC={metrics['roc_auc']:.4f}  "
              f"CV-F1={cv_scores.mean():.4f}±{cv_scores.std():.4f}")
        return run_id, metrics


def main():
    print("Loading and preprocessing data (with SMOTE) ...")
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess(use_smote=True)
    print(f"  Train={X_train.shape} (balanced)  Test={X_test.shape}")

    # class_weight ratio for XGBoost: n_negative / n_positive (before SMOTE was 73/27 ≈ 2.7)
    scale_pos_weight = 2.7

    models = [
        (
            LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000,
                               class_weight="balanced", random_state=42),
            {"model": "LogisticRegression", "C": 1.0, "solver": "lbfgs",
             "max_iter": 1000, "class_weight": "balanced", "smote": True},
            "Logistic Regression",
        ),
        (
            RandomForestClassifier(n_estimators=200, max_depth=8,
                                   min_samples_split=5, class_weight="balanced",
                                   random_state=42, n_jobs=-1),
            {"model": "RandomForest", "n_estimators": 200, "max_depth": 8,
             "min_samples_split": 5, "class_weight": "balanced", "smote": True},
            "Random Forest",
        ),
        (
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                       max_depth=4, random_state=42),
            {"model": "GradientBoosting", "n_estimators": 200,
             "learning_rate": 0.1, "max_depth": 4, "smote": True},
            "Gradient Boosting",
        ),
        (
            xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                               scale_pos_weight=scale_pos_weight,
                               eval_metric="logloss", random_state=42,
                               n_jobs=-1, verbosity=0),
            {"model": "XGBoost", "n_estimators": 200, "max_depth": 4,
             "learning_rate": 0.1, "scale_pos_weight": scale_pos_weight, "smote": True},
            "XGBoost",
        ),
    ]

    print(f"\nRunning experiments → MLflow experiment: '{EXPERIMENT_NAME}'")
    results = []
    for model, params, name in models:
        run_id, metrics = run_experiment(
            model, params, name, X_train, X_test, y_train, y_test, feature_names, scaler
        )
        results.append((name, metrics["f1"], metrics["roc_auc"], run_id))

    print("\n--- Summary ---")
    print(f"{'Model':<25} {'F1':>8} {'ROC-AUC':>10}")
    print("-" * 45)
    for name, f1, auc, _ in sorted(results, key=lambda x: x[2], reverse=True):
        print(f"{name:<25} {f1:>8.4f} {auc:>10.4f}")


if __name__ == "__main__":
    main()
