# 🔮 Real-time Customer Churn Prediction & Proactive Retention

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.13-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Azure ML](https://img.shields.io/badge/Azure_ML-Cloud-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)

**AIN-3009 MLOps Term Project — Bahçeşehir University**
**Student:** Abdullah Hani Abdellatif Al-Shobaki · `2284612`

*A production-grade, end-to-end MLOps system that predicts telecom customer churn in real time,
explains every decision with SHAP, and proactively recommends retention strategies —
all tracked, versioned, and monitored with MLflow.*

</div>

---

## 🗺️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│   Telco Customer Churn Dataset  →  preprocess.py (SMOTE)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   EXPERIMENT TRACKING  (MLflow)                 │
│  train.py → 4 models + CV + SHAP  │  tune.py → 50 Hyperopt      │
│  Telco_Churn_Experiments           │  trials (nested runs)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│              MODEL REGISTRY  (MLflow Registry)                  │
│        ChurnPredictionModel  →  Staging  →  Production          │
└──────┬────────────────────────────────────────┬─────────────────┘
       │                                        │
┌──────▼──────────┐                  ┌──────────▼──────────────┐
│  DEPLOYMENT     │                  │  MONITORING             │
│  FastAPI :8000  │                  │  monitor.py + Evidently │
│  /predict       │                  │  3 drift windows        │
│  /health        │                  │  HTML reports in MLflow │
└──────┬──────────┘                  └─────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD  :8501                         │
│  Tab 1: Customer Predictor + SHAP waterfall + ROI calculator    │
│  Tab 2: Model Performance comparison (4 models × 5 metrics)     │
│  Tab 3: Drift monitoring charts + Evidently report links        │
└──────┬──────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                   ☁️  AZURE ML CLOUD                            │
│  Workspace: ain3009-churn-ws  (Sweden Central)                  │
│  Compute:   churn-cluster  (Standard_DS3_v2, auto-scale 0→2)    │
│  Registry:  ChurnPredictionModel v1  (live)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| **4 ML Models** | Logistic Regression · Random Forest · Gradient Boosting · XGBoost |
| **Class Imbalance** | SMOTE oversampling + `class_weight="balanced"` on LR & RF |
| **Cross-Validation** | 5-fold CV F1 logged per model alongside test metrics |
| **Explainability** | SHAP beeswarm plots (training) + per-customer waterfall charts |
| **Hyperparameter Tuning** | Hyperopt TPE · 50 trials · nested MLflow runs |
| **Drift Monitoring** | 3-window simulation · Evidently AI HTML reports as artifacts |
| **REST API** | FastAPI + uvicorn · `/predict` · `/health` · interactive Swagger docs |
| **Live Dashboard** | Streamlit · Plotly gauge · SHAP charts · ROI calculator |
| **Model Registry** | MLflow Registry · Staging → Production lifecycle management |
| **Reproducibility** | `MLproject` file — full pipeline as one-command entrypoints |
| **Cloud** | Azure ML workspace · compute cluster · registered model (live) |

---

## 📊 Key Results

### Baseline Models — with SMOTE + class balancing

| Model | F1 Score | ROC-AUC | Precision | Recall | CV F1 (mean ± std) |
|---|---|---|---|---|---|
| Logistic Regression | 0.6184 | 0.8401 | 0.5042 | 0.7995 | 0.7801 ± 0.0122 |
| **Random Forest** | **0.6303** | **0.8434** | 0.5401 | 0.7567 | 0.8249 ± 0.0303 |
| Gradient Boosting | 0.5953 | 0.8386 | 0.5816 | 0.6096 | 0.8270 ± 0.1019 |
| XGBoost | 0.6186 | 0.8344 | 0.5034 | 0.8021 | 0.8284 ± 0.0378 |

### Hyperopt-Tuned Gradient Boosting (50 trials · TPE)

| Model | F1 Score | ROC-AUC | Best Hyperparameters |
|---|---|---|---|
| **GB (Hyperopt-tuned)** | **0.6015** | **0.8443** | n_estimators=400, max_depth=3, lr=0.0545, subsample=0.712 |

### Drift Monitoring — F1 degradation across 3 windows

| Window | Condition | F1 Score | Accuracy |
|---|---|---|---|
| Window 1 | Baseline (clean data) | **0.6015** | 0.8091 |
| Window 2 | Mild drift (Gaussian noise σ=0.3) | 0.5323 | 0.7793 |
| Window 3 | Heavy drift (σ=0.8 + 20% label flip) | 0.4308 | 0.6437 |

> F1 degrades **0.60 → 0.53 → 0.43** — a clear signal for model retraining alerts.
> Evidently AI HTML reports for each window are logged as MLflow artifacts.

---

## 🗂️ Project Structure

```
PRJ-AbdullahAl-Shobaki-2284612/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # 7,043 customers · 21 features
│
├── notebooks/
│   ├── 01_eda.ipynb                            # EDA — distributions, correlations, churn analysis
│   └── 02_feature_engineering.ipynb            # Feature importance & preprocessing pipeline
│
├── src/
│   ├── preprocess.py    # Cleaning · label encoding · SMOTE · StandardScaler · train/test split
│   ├── train.py         # 4 models · SHAP beeswarm · 5-fold CV · full MLflow tracking
│   ├── tune.py          # Hyperopt TPE · 50 trials · nested MLflow runs · best model saved
│   ├── register.py      # Promote best model: None → Staging → Production
│   └── monitor.py       # 3-window drift simulation · Evidently HTML reports · MLflow logging
│
├── api/
│   └── app.py           # FastAPI service — /predict · /health · Swagger UI
│
├── ui/
│   └── streamlit_app.py # 3-tab dashboard — predictor · model perf · monitoring
│
├── azure/
│   ├── setup_workspace.py   # Create Azure ML resource group + workspace + compute cluster
│   ├── train_azure.py       # Submit full pipeline as Azure ML cloud job
│   ├── deploy_endpoint.py   # Deploy Production model as Azure ML Managed Online Endpoint
│   └── scoring/
│       ├── score.py         # Self-contained scoring script for Azure endpoint
│       └── conda.yaml       # Conda environment for Azure deployment
│
├── MLproject            # MLflow Projects — 6 reproducible entrypoints
├── python_env.yaml      # Python environment spec for MLflow Projects
├── requirements.txt     # All dependencies (pinned versions)
├── start_mlflow.sh      # One-command MLflow server startup (port 5500)
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/AbdullahAl-Shobaki/PRJ-AbdullahAl-Shobaki-2284612.git
cd PRJ-AbdullahAl-Shobaki-2284612
pip install -r requirements.txt
```

### 2. Add the dataset

Download [`WA_Fn-UseC_-Telco-Customer-Churn.csv`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in `data/`.

### 3. Start the MLflow tracking server

```bash
bash start_mlflow.sh
```

Open **http://localhost:5500** — MLflow UI is live.

> **macOS note:** Port 5000 is blocked by AirPlay Receiver. This project uses port **5500**.

---

## 🚀 Running the Full Pipeline

All commands run from the **project root directory**.

### Option A — Manual step by step

```bash
# 1. Train 4 models (SMOTE · class_weight · 5-fold CV · SHAP plots)
python src/train.py

# 2. Hyperparameter tuning (50 Hyperopt trials · nested runs)
python src/tune.py

# 3. Register best model to Production
python src/register.py

# 4. Launch prediction API
cd api && uvicorn app:app --host 0.0.0.0 --port 8000

# 5. Drift monitoring (3 windows · Evidently HTML reports)
python src/monitor.py

# 6. Launch Streamlit dashboard
streamlit run ui/streamlit_app.py
```

### Option B — MLflow Projects (reproducible, one command each)

```bash
mlflow run . -e train      # Train all 4 models
mlflow run . -e tune       # Hyperparameter tuning
mlflow run . -e register   # Register & promote to Production
mlflow run . -e monitor    # Run drift monitoring
mlflow run . -e ui         # Launch Streamlit dashboard
mlflow run . -e all        # Run entire pipeline end-to-end
```

---

## 🌐 Streamlit Dashboard

**URL:** http://localhost:8501

| Tab | Content |
|---|---|
| 🔍 **Customer Predictor** | 19-field input form → probability gauge (0–100%) → risk badge (🔴/🟡/🟢) → SHAP waterfall → retention strategy → ROI calculator |
| 📊 **Model Performance** | Side-by-side bar charts (F1 · AUC · Precision · Recall) + CV scores table for all 4 models |
| 📈 **Monitoring** | F1 & Accuracy trend across 3 drift windows + clickable Evidently HTML report links |

---

## 🔌 Prediction API

### Health check
```bash
curl http://localhost:8000/health
```

### Predict churn (example — high-risk customer)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 5, "MonthlyCharges": 95.5, "TotalCharges": 477.5,
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
    "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Example response:**
```json
{
  "churn_probability": 0.8741,
  "churn_prediction": true,
  "risk_level": "High",
  "retention_strategy": "HIGH RISK: Offer 20% discount to upgrade to 1-year contract.",
  "annual_revenue_at_risk": 1003.21,
  "intervention_cost_usd": 50,
  "expected_roi_pct": 1906.4
}
```

Interactive Swagger docs: **http://localhost:8000/docs**

---

## ☁️ Azure ML Cloud Deployment

Full Azure ML integration — workspace, compute cluster, model registry, and managed endpoint scripts are included.

### Configuration

Edit `azure/azure_config.json` with your Azure Portal values:
```json
{
  "subscription_id":  "your-subscription-id",
  "resource_group":   "rg-ain3009-mlops",
  "workspace_name":   "ain3009-churn-ws",
  "location":         "swedencentral",
  "compute_name":     "churn-cluster",
  "endpoint_name":    "churn-endpoint",
  "deployment_name":  "churn-deployment"
}
```

### Deployment steps

```bash
# Step 1 — Create resource group + workspace + compute cluster
python azure/setup_workspace.py

# Step 2 — Submit full training pipeline to Azure cloud
python azure/train_azure.py

# Step 3 — Deploy model as HTTPS REST endpoint
python azure/deploy_endpoint.py
```

**Live Azure resources (Sweden Central):**

| Resource | Name | Status |
|---|---|---|
| Resource Group | `rg-ain3009-mlops` | ✅ Live |
| ML Workspace | `ain3009-churn-ws` | ✅ Live |
| Compute Cluster | `churn-cluster` (Standard_DS3_v2) | ✅ Live |
| Registered Model | `ChurnPredictionModel v1` | ✅ Live |

---

## 📈 MLflow UI Guide

**URL:** http://localhost:5500

| Section | What to explore |
|---|---|
| **Experiments → Telco_Churn_Experiments** | 4 baseline runs (SHAP plots · CV metrics) + HyperTuning parent with 50 child trials |
| **Experiments → Churn_Monitoring** | 3 windows with Evidently HTML drift reports as artifacts |
| **Models → ChurnPredictionModel** | Version history · Staging/Production stage transitions |
| **Any Run → Artifacts → plots/** | Confusion matrix · ROC curve · SHAP beeswarm |
| **Monitoring Run → Artifacts → drift_reports/** | Evidently AI full HTML drift report |

---

## 🛠️ Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10 |
| ML — Models | scikit-learn | 1.4.2 |
| ML — Boosting | XGBoost | 2.0.3 |
| ML — Imbalance | imbalanced-learn (SMOTE) | 0.12.2 |
| ML — Explainability | SHAP | 0.45.0 |
| ML — Tuning | Hyperopt (TPE) | 0.2.7 |
| Tracking | MLflow | 2.13.0 |
| Monitoring | Evidently AI | 0.4.30 |
| API | FastAPI + uvicorn | 0.111.0 |
| Dashboard | Streamlit + Plotly | 1.33.0 |
| Cloud | Azure ML SDK v2 | ≥1.15.0 |
| Backend Store | SQLite (`mlflow.db`) | — |
| Artifact Store | Local filesystem (`mlartifacts/`) | — |

---

## 📋 Dataset

**[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)**

- **7,043** customer records · **21** raw features · **1** binary target (`Churn`)
- **26.5% churn rate** — significant class imbalance handled via SMOTE
- Features include: tenure, monthly charges, total charges, contract type, internet service, phone service, streaming services, payment method, and demographics

---

## 📦 Submission

| | |
|---|---|
| **Student** | Abdullah Hani Abdellatif Al-Shobaki |
| **Student ID** | 2284612 |
| **Course** | AIN-3009 MLOps |
| **University** | Bahçeşehir University |
| **Semester** | Spring 2026 |
| **Instructor** | Gökşin BAKIR |
| **Due Date** | June 1, 2026 |
