"""
Performance monitoring with simulated data drift + Evidently AI HTML reports.

Logs three time-window monitoring runs to MLflow experiment "Churn_Monitoring"
and generates professional Evidently drift reports as MLflow artifacts.

Windows:
  1 - Baseline    : original test set (no drift)
  2 - Mild drift  : Gaussian noise on numeric features + minor categorical shift
  3 - Heavy drift : heavier noise + 20% label flip (concept drift)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
import mlflow.sklearn

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_and_preprocess, NUMERIC_COLS

warnings.filterwarnings("ignore")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5500")
MONITOR_EXPERIMENT  = "Churn_Monitoring"
MODEL_NAME          = "ChurnPredictionModel"
MODEL_STAGE         = "Production"


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }


def compute_data_stats(X, numeric_indices: list) -> dict:
    stats = {}
    names = ["tenure", "MonthlyCharges", "TotalCharges"]
    for i, name in zip(numeric_indices, names):
        stats[f"data_mean_{name}"] = float(np.mean(X[:, i]))
        stats[f"data_std_{name}"]  = float(np.std(X[:, i]))
    return stats


def apply_mild_drift(X: np.ndarray, numeric_indices: list, rng) -> np.ndarray:
    X_drift = X.copy()
    for idx in numeric_indices:
        X_drift[:, idx] += rng.normal(0, 0.3, size=X_drift.shape[0])
    return X_drift


def apply_heavy_drift(X: np.ndarray, y: np.ndarray,
                      numeric_indices: list, rng) -> tuple:
    X_drift = X.copy()
    y_drift = y.copy()
    for idx in numeric_indices:
        X_drift[:, idx] += rng.normal(0, 0.8, size=X_drift.shape[0])
    flip_mask = rng.random(len(y_drift)) < 0.20
    y_drift[flip_mask] = 1 - y_drift[flip_mask]
    return X_drift, y_drift


def generate_evidently_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame,
                               window_num: int) -> str:
    """Create an Evidently data drift HTML report and return the file path."""
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(),
    ])
    report.run(reference_data=ref_df, current_data=cur_df)
    path = f"/tmp/evidently_drift_window_{window_num}.html"
    report.save_html(path)
    return path


def plot_metric_trend(window_metrics: list, metric: str) -> str:
    windows = [1, 2, 3]
    values  = [m[metric] for m in window_metrics]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(windows, values, "o-", color="#1976D2", linewidth=2.5, markersize=10)
    ax.fill_between(windows, values, alpha=0.12, color="#1976D2")
    for x, y in zip(windows, values):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(windows)
    ax.set_xticklabels(["Window 1\n(Baseline)", "Window 2\n(Mild Drift)",
                         "Window 3\n(Heavy Drift)"])
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.set_title(f"{metric.upper()} Degradation Over Time", fontweight="bold", fontsize=13)
    ax.set_ylim(max(0, min(values) - 0.1), 1.05)
    plt.tight_layout()
    path = f"/tmp/monitor_{metric}_trend.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


def main():
    print("Loading data ...")
    X_train, X_test, _, y_test, feature_names, _ = load_and_preprocess()

    numeric_indices = [feature_names.index(c) for c in
                       ["tenure", "MonthlyCharges", "TotalCharges"]
                       if c in feature_names]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MONITOR_EXPERIMENT)

    print(f"Loading Production model '{MODEL_NAME}' ...")
    try:
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load model: {exc}\n"
            "Run src/register.py first to promote a model to Production."
        )

    rng = np.random.default_rng(seed=99)

    # Reference DataFrame (training set) for Evidently
    ref_df = pd.DataFrame(X_train, columns=feature_names)

    windows = [
        ("Window 1 - Baseline",   X_test,                                    y_test,  "none",  1),
        ("Window 2 - Mild Drift", apply_mild_drift(X_test, numeric_indices, rng), y_test,  "mild",  2),
    ]
    X_heavy, y_heavy = apply_heavy_drift(X_test, y_test, numeric_indices, rng)
    windows.append(("Window 3 - Heavy Drift", X_heavy, y_heavy, "heavy", 3))

    all_metrics = []
    print(f"\nRunning 3 monitoring windows → MLflow experiment: '{MONITOR_EXPERIMENT}'")

    for run_name, X_win, y_win, drift_type, window_num in windows:
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("window", str(window_num))
            mlflow.set_tag("drift_type", drift_type)
            mlflow.log_param("n_samples", len(y_win))
            mlflow.log_param("churn_rate_actual", float(y_win.mean()))

            y_pred = model.predict(X_win)
            metrics = compute_metrics(y_win, y_pred)
            data_stats = compute_data_stats(X_win, numeric_indices)

            mlflow.log_metrics(metrics)
            mlflow.log_metrics(data_stats)
            mlflow.log_metric("churn_rate_predicted", float(y_pred.mean()))

            # Evidently drift report
            cur_df = pd.DataFrame(X_win, columns=feature_names)
            try:
                report_path = generate_evidently_report(ref_df, cur_df, window_num)
                mlflow.log_artifact(report_path, artifact_path="drift_reports")
                print(f"  [{run_name}]  F1={metrics['f1']:.4f}  "
                      f"Accuracy={metrics['accuracy']:.4f}  "
                      f"drift={drift_type}  evidently=✓")
            except Exception as e:
                print(f"  [{run_name}]  F1={metrics['f1']:.4f}  "
                      f"Accuracy={metrics['accuracy']:.4f}  "
                      f"drift={drift_type}  evidently=✗ ({e})")

            all_metrics.append(metrics)

    # Trend charts logged as separate summary runs
    for metric_name in ("f1", "accuracy", "recall"):
        chart_path = plot_metric_trend(all_metrics, metric_name)
        with mlflow.start_run(run_name=f"Trend Chart - {metric_name.upper()}"):
            mlflow.set_tag("chart_type", "drift_trend")
            mlflow.log_artifact(chart_path, artifact_path="monitoring_charts")

    print("\n--- Monitoring Summary ---")
    print(f"{'Window':<30} {'F1':>8} {'Accuracy':>10} {'Recall':>8}")
    print("-" * 60)
    labels = ["Baseline", "Mild Drift", "Heavy Drift"]
    for label, m in zip(labels, all_metrics):
        print(f"  {label:<28} {m['f1']:>8.4f} {m['accuracy']:>10.4f} {m['recall']:>8.4f}")

    f1_drop = all_metrics[0]["f1"] - all_metrics[2]["f1"]
    print(f"\n  F1 degradation (baseline → heavy drift): {f1_drop:.4f}")
    print(f"  Evidently HTML reports saved as MLflow artifacts (drift_reports/)")
    print(f"  View in MLflow UI → experiment '{MONITOR_EXPERIMENT}'")


if __name__ == "__main__":
    main()
