"""
Streamlit Dashboard — Telco Customer Churn Prediction
AIN-3009 MLOps Term Project | Abdullah Hani Abdellatif Al-Shobaki — 2284612

Tabs:
  1. Customer Churn Predictor  — live prediction + SHAP + ROI calculator
  2. Model Performance         — compare all 4 models from MLflow
  3. Monitoring Dashboard      — drift metrics across 3 time windows
"""

import html as _html
import json
import sys
import os
import warnings
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import mlflow
import mlflow.sklearn

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = "http://localhost:5500"
API_URL         = "http://localhost:8000"
MODEL_NAME      = "ChurnPredictionModel"
MODEL_STAGE     = "Production"
EXPERIMENT_NAME = "Telco_Churn_Experiments"
MONITOR_EXP     = "Churn_Monitoring"

# ── Azure endpoint config (loaded from azure/endpoint_config.json if available)
AZURE_ENDPOINT_CONFIG = os.path.join(
    os.path.dirname(__file__), "..", "azure", "endpoint_config.json"
)


def load_azure_config() -> dict:
    """Load Azure endpoint URL and key if deployment has been done."""
    if os.path.exists(AZURE_ENDPOINT_CONFIG):
        with open(AZURE_ENDPOINT_CONFIG) as f:
            return json.load(f)
    return {}


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.preprocess import load_and_preprocess

mlflow.set_tracking_uri(MLFLOW_URI)

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700; color: #1976D2;
        border-bottom: 3px solid #1976D2; padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1rem;
        border-left: 4px solid #1976D2; margin-bottom: 0.5rem;
    }
    .risk-high   { background:#FFEBEE; border-left:4px solid #D32F2F; padding:0.8rem; border-radius:8px; }
    .risk-medium { background:#FFF8E1; border-left:4px solid #F57F17; padding:0.8rem; border-radius:8px; }
    .risk-low    { background:#E8F5E9; border-left:4px solid #2E7D32; padding:0.8rem; border-radius:8px; }
    .strategy-box { background:#E3F2FD; border-radius:8px; padding:1rem; margin-top:0.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model from MLflow...")
def load_model():
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess()
    return X_train, X_test, y_train, y_test, feature_names, scaler


@st.cache_data(show_spinner="Fetching MLflow runs...")
def get_experiment_runs(experiment_name: str) -> pd.DataFrame:
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            return pd.DataFrame()
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
        )
        return runs
    except Exception:
        return pd.DataFrame()


# ── Sidebar — API source toggle ───────────────────────────────────────────────
azure_cfg = load_azure_config()
azure_available = bool(azure_cfg.get("azure_endpoint_url"))

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    if azure_available:
        use_azure = st.toggle("☁️ Use Azure Cloud Endpoint", value=True)
        if use_azure:
            st.success(f"**Azure ML Endpoint active**")
            st.caption(azure_cfg["azure_endpoint_url"][:55] + "...")
        else:
            st.info("Using local FastAPI (localhost:8000)")
    else:
        use_azure = False
        st.info("**Local API** (localhost:8000)")
        st.markdown(
            "To enable Azure cloud endpoint:\n"
            "1. Fill in `azure/azure_config.json`\n"
            "2. Run `python azure/setup_workspace.py`\n"
            "3. Run `python azure/train_azure.py`\n"
            "4. Run `python azure/deploy_endpoint.py`"
        )
    st.markdown("---")
    st.markdown(f"**MLflow UI:** [localhost:5500](http://localhost:5500)")
    st.markdown(f"**API Docs:** [localhost:8000/docs](http://localhost:8000/docs)")
    st.markdown(f"**Streamlit:** localhost:8501")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📡 Telco Customer Churn Prediction Dashboard</div>',
            unsafe_allow_html=True)
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.caption("AIN-3009 MLOps Term Project | Real-time Prediction & Proactive Retention Strategy")
with col_h2:
    if use_azure:
        st.markdown("☁️ **Powered by Azure ML**", unsafe_allow_html=False)

tab1, tab2, tab3 = st.tabs(["🔍 Customer Predictor", "📊 Model Performance", "📈 Monitoring"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Customer Churn Predictor
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1.6], gap="large")

    with col_form:
        st.subheader("Customer Profile")
        with st.form("customer_form"):
            tenure         = st.slider("Tenure (months)", 0, 72, 12)
            monthly        = st.number_input("Monthly Charges ($)", 0.0, 150.0, 65.0, step=0.5)
            total          = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                              float(tenure * monthly), step=10.0)
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior         = st.selectbox("Senior Citizen", [0, 1])
            partner        = st.selectbox("Partner", ["Yes", "No"])
            dependents     = st.selectbox("Dependents", ["Yes", "No"])
            contract       = st.selectbox("Contract",
                                          ["Month-to-month", "One year", "Two year"])
            internet       = st.selectbox("Internet Service",
                                          ["Fiber optic", "DSL", "No"])
            online_sec     = st.selectbox("Online Security", ["No", "Yes"])
            tech_support   = st.selectbox("Tech Support", ["No", "Yes"])
            payment        = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
            phone          = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
            online_backup  = st.selectbox("Online Backup", ["No", "Yes"])
            device_prot    = st.selectbox("Device Protection", ["No", "Yes"])
            streaming_tv   = st.selectbox("Streaming TV", ["No", "Yes"])
            streaming_mov  = st.selectbox("Streaming Movies", ["No", "Yes"])

            submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True,
                                               type="primary")

    with col_result:
        if submitted:
            payload = {
                "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
                "gender": gender, "SeniorCitizen": senior, "Partner": partner,
                "Dependents": dependents, "PhoneService": phone,
                "MultipleLines": multiple_lines, "InternetService": internet,
                "OnlineSecurity": online_sec, "OnlineBackup": online_backup,
                "DeviceProtection": device_prot, "TechSupport": tech_support,
                "StreamingTV": streaming_tv, "StreamingMovies": streaming_mov,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment,
            }

            try:
                if use_azure and azure_available:
                    # ── Call Azure ML Online Endpoint ────────────────────
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {azure_cfg['azure_api_key']}",
                    }
                    resp = requests.post(
                        azure_cfg["azure_endpoint_url"],
                        json=payload,
                        headers=headers,
                        timeout=30,
                    )
                    result = resp.json()
                    # Azure scoring script returns snake_case keys directly
                    st.caption("☁️ Prediction served from **Azure ML Online Endpoint**")
                else:
                    # ── Call local FastAPI ────────────────────────────────
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                    result = resp.json()
                    st.caption("🖥️ Prediction served from **local FastAPI**")

                prob      = result["churn_probability"]
                pred      = result["churn_prediction"]
                risk      = result["risk_level"]
                strategy  = result["retention_strategy"]

                # ── Gauge chart ──────────────────────────────────────────
                gauge_color = "#D32F2F" if risk == "High" else ("#F57F17" if risk == "Medium" else "#2E7D32")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={"text": "Churn Probability", "font": {"size": 20}},
                    number={"suffix": "%", "font": {"size": 36}},
                    delta={"reference": 50, "increasing": {"color": "#D32F2F"},
                           "decreasing": {"color": "#2E7D32"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": gauge_color},
                        "steps": [
                            {"range": [0, 40],  "color": "#E8F5E9"},
                            {"range": [40, 70], "color": "#FFF8E1"},
                            {"range": [70, 100],"color": "#FFEBEE"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 3},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

                # ── Risk badge ───────────────────────────────────────────
                risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
                risk_icon  = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                st.markdown(
                    f'<div class="{risk_class[risk]}">'
                    f'<strong>{risk_icon[risk]} {risk} Risk</strong> — '
                    f'{"Churn predicted" if pred else "Retention likely"}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ── Retention strategy ───────────────────────────────────
                st.markdown(f'<div class="strategy-box">💡 <strong>Retention Strategy</strong><br>{_html.escape(strategy)}</div>',
                            unsafe_allow_html=True)

                # ── ROI calculator ───────────────────────────────────────
                annual_risk      = round(monthly * 12 * prob, 2)
                intervention     = 50.0
                net_savings      = annual_risk - intervention
                roi_pct          = ((net_savings / intervention) * 100) if intervention > 0 else 0

                st.markdown("---")
                st.markdown("**💰 Retention ROI Calculator**")
                r1, r2, r3 = st.columns(3)
                r1.metric("Annual Revenue at Risk", f"${annual_risk:,.0f}")
                r2.metric("Intervention Cost",      f"${intervention:,.0f}")
                r3.metric("Expected ROI",            f"{roi_pct:,.0f}%",
                           delta=f"${net_savings:,.0f} net savings")

            except requests.exceptions.ConnectionError:
                if use_azure and azure_available:
                    st.error("⚠️ Cannot connect to Azure endpoint. Check your internet connection or endpoint status in Azure ML Studio.")
                else:
                    st.error("⚠️ Cannot connect to local API. Make sure it's running:\n"
                             "`cd api && uvicorn app:app --port 8000`")
            except Exception as e:
                st.error(f"Prediction error: {e}")

            # ── SHAP waterfall ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("**🧠 SHAP Feature Explanation**")
            with st.spinner("Computing SHAP values..."):
                try:
                    model       = load_model()
                    _, X_test, _, _, feature_names, scaler = load_data()
                    shap_sample = X_test[:300]

                    if hasattr(model, "feature_importances_"):
                        explainer  = shap.TreeExplainer(model)
                        shap_vals  = explainer.shap_values(shap_sample)
                        if isinstance(shap_vals, list):
                            shap_vals = shap_vals[1]
                        base_val = float(explainer.expected_value
                                         if not isinstance(explainer.expected_value, list)
                                         else explainer.expected_value[1])
                    else:
                        explainer = shap.LinearExplainer(model, shap_sample)
                        shap_vals = explainer.shap_values(shap_sample)
                        base_val  = float(explainer.expected_value)

                    # Show summary beeswarm
                    fig_shap, ax = plt.subplots(figsize=(8, 5))
                    shap.summary_plot(shap_vals, shap_sample,
                                      feature_names=feature_names,
                                      show=False, plot_size=None,
                                      max_display=15)
                    plt.title("SHAP Feature Impact on Churn Prediction",
                              fontweight="bold", fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig_shap, use_container_width=True)
                    plt.close("all")

                except Exception as e:
                    st.info(f"SHAP explanation unavailable: {e}")

        else:
            st.info("👈 Fill in the customer profile and click **Predict Churn** to get started.")
            st.markdown("""
            **This dashboard provides:**
            - 🎯 Real-time churn probability with visual gauge
            - 💡 Proactive retention strategy recommendation
            - 💰 ROI calculator for the retention intervention
            - 🧠 SHAP explanation of which features drove the prediction
            """)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model Performance Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Performance Comparison")
    st.caption(f"Pulling live data from MLflow experiment: `{EXPERIMENT_NAME}`")

    runs_df = get_experiment_runs(EXPERIMENT_NAME)

    if runs_df.empty:
        st.warning("No runs found. Run `python src/train.py` first.")
    else:
        # Filter to baseline model runs (not hyperopt children, not tuning parents)
        metric_cols = ["metrics.f1", "metrics.roc_auc", "metrics.precision",
                        "metrics.recall", "metrics.accuracy",
                        "metrics.cv_f1_mean", "metrics.cv_f1_std"]
        available   = [c for c in metric_cols if c in runs_df.columns]

        baseline = runs_df[
            runs_df["tags.mlflow.runName"].isin(
                ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]
            )
        ].copy()

        if baseline.empty:
            st.warning("No baseline model runs found.")
        else:
            baseline = baseline.sort_values("metrics.roc_auc", ascending=False)
            baseline["Model"] = baseline["tags.mlflow.runName"]

            # Metrics bar chart
            metrics_to_plot = {
                "F1 Score": "metrics.f1",
                "ROC-AUC": "metrics.roc_auc",
                "Precision": "metrics.precision",
                "Recall": "metrics.recall",
                "Accuracy": "metrics.accuracy",
            }
            metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if v in baseline.columns}

            fig_bar = go.Figure()
            colors  = ["#1976D2", "#E53935", "#43A047", "#FB8C00"]
            for i, (_, row) in enumerate(baseline.iterrows()):
                fig_bar.add_trace(go.Bar(
                    name=row["Model"],
                    x=list(metrics_to_plot.keys()),
                    y=[row.get(v, 0) for v in metrics_to_plot.values()],
                    marker_color=colors[i % len(colors)],
                    text=[f"{row.get(v, 0):.3f}" for v in metrics_to_plot.values()],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                barmode="group",
                title="Model Metrics Comparison (Test Set)",
                yaxis=dict(range=[0, 1.1], title="Score"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=430,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # CV scores if available
            if "metrics.cv_f1_mean" in baseline.columns:
                st.markdown("**5-Fold Cross-Validation F1 Scores**")
                cv_data = baseline[["Model", "metrics.cv_f1_mean", "metrics.cv_f1_std"]].copy()
                cv_data.columns = ["Model", "CV F1 Mean", "CV F1 Std"]
                cv_data = cv_data.reset_index(drop=True)

                fig_cv = go.Figure()
                for i, (_, row) in enumerate(cv_data.iterrows()):
                    fig_cv.add_trace(go.Bar(
                        name=row["Model"],
                        x=[row["Model"]],
                        y=[row["CV F1 Mean"]],
                        error_y=dict(type="data", array=[row["CV F1 Std"]], visible=True),
                        marker_color=colors[i % len(colors)],
                        text=[f"{row['CV F1 Mean']:.3f}"],
                        textposition="outside",
                    ))
                fig_cv.update_layout(
                    title="Cross-Validation F1 (mean ± std)",
                    yaxis=dict(range=[0, 1.1], title="CV F1 Score"),
                    showlegend=False,
                    height=330,
                    plot_bgcolor="white",
                )
                st.plotly_chart(fig_cv, use_container_width=True)

            # Full metrics table
            st.markdown("**Full Metrics Table**")
            display_cols = {"Model": "Model"}
            display_cols.update({v: k for k, v in metrics_to_plot.items()})
            if "metrics.cv_f1_mean" in baseline.columns:
                display_cols["metrics.cv_f1_mean"] = "CV F1 Mean"
            available_display = {k: v for k, v in display_cols.items() if k in baseline.columns}
            table_df = baseline[list(available_display.keys())].rename(columns=available_display)
            table_df = table_df.reset_index(drop=True)
            numeric_cols_disp = [c for c in table_df.columns if c != "Model"]
            st.dataframe(
                table_df.style.highlight_max(subset=numeric_cols_disp, color="#C8E6C9")
                               .format({c: "{:.4f}" for c in numeric_cols_disp}),
                use_container_width=True,
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Monitoring Dashboard
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Monitoring")
    st.caption(f"Pulling live data from MLflow experiment: `{MONITOR_EXP}`")

    monitor_runs = get_experiment_runs(MONITOR_EXP)

    if monitor_runs.empty:
        st.warning("No monitoring runs found. Run `python src/monitor.py` first.")
    else:
        # Filter to the 3 window runs (not trend chart runs)
        window_runs = monitor_runs[
            monitor_runs["tags.mlflow.runName"].str.contains("Window", na=False)
        ].copy()

        if window_runs.empty:
            st.warning("No window runs found in monitoring experiment.")
        else:
            window_runs["Window"] = window_runs["tags.window"].astype(int)
            window_runs = window_runs.sort_values("Window")

            drift_labels = {1: "Baseline", 2: "Mild Drift", 3: "Heavy Drift"}
            window_runs["Label"] = window_runs["Window"].map(drift_labels)

            metric_map = {
                "F1 Score":  "metrics.f1",
                "Accuracy":  "metrics.accuracy",
                "Precision": "metrics.precision",
                "Recall":    "metrics.recall",
            }
            available_metrics = {k: v for k, v in metric_map.items()
                                  if v in window_runs.columns}

            # Line chart — all metrics over windows
            fig_drift = go.Figure()
            line_colors = ["#1976D2", "#43A047", "#E53935", "#FB8C00"]
            for i, (label, col) in enumerate(available_metrics.items()):
                fig_drift.add_trace(go.Scatter(
                    x=window_runs["Label"].tolist(),
                    y=window_runs[col].tolist(),
                    mode="lines+markers+text",
                    name=label,
                    line=dict(color=line_colors[i], width=2.5),
                    marker=dict(size=10),
                    text=[f"{v:.3f}" for v in window_runs[col].tolist()],
                    textposition="top center",
                ))
            fig_drift.update_layout(
                title="Model Performance Degradation Over Time (Simulated Drift)",
                yaxis=dict(range=[0, 1.1], title="Score"),
                xaxis_title="Monitoring Window",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=420,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig_drift, use_container_width=True)

            # Drift summary metrics
            if len(window_runs) >= 2:
                st.markdown("**Drift Impact Summary**")
                baseline_row = window_runs[window_runs["Window"] == 1].iloc[0]
                heavy_row    = window_runs[window_runs["Window"] == 3].iloc[0]

                d1, d2, d3, d4 = st.columns(4)
                for col_ui, (label, col) in zip([d1, d2, d3, d4], available_metrics.items()):
                    if col in window_runs.columns:
                        base_val  = baseline_row[col]
                        heavy_val = heavy_row[col]
                        delta     = heavy_val - base_val
                        col_ui.metric(
                            label=f"{label} (Baseline → Heavy)",
                            value=f"{heavy_val:.3f}",
                            delta=f"{delta:+.3f}",
                            delta_color="inverse",
                        )

            # Per-window detailed table
            st.markdown("**Detailed Metrics per Window**")
            table_cols = {"Label": "Window"} | {v: k for k, v in available_metrics.items()}
            available_table = {k: v for k, v in table_cols.items() if k in window_runs.columns}
            mon_table = window_runs[list(available_table.keys())].rename(columns=available_table)
            mon_table = mon_table.reset_index(drop=True)
            num_cols  = [c for c in mon_table.columns if c != "Window"]
            st.dataframe(
                mon_table.style.format({c: "{:.4f}" for c in num_cols}),
                use_container_width=True,
            )

            # Evidently report links
            st.markdown("---")
            st.markdown("**📋 Evidently AI Drift Reports**")
            st.info(
                "Evidently HTML reports are saved as MLflow artifacts under each window run. "
                "To view them:\n"
                "1. Open the MLflow UI at http://localhost:5500\n"
                "2. Go to `Churn_Monitoring` experiment\n"
                "3. Click on a window run → Artifacts → `drift_reports/`"
            )


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "AIN-3009 MLOps Term Project · Bahçeşehir University · "
    "Abdullah Hani Abdellatif Al-Shobaki — 2284612 · "
    "Model: ChurnPredictionModel (Production) · "
    f"MLflow: {MLFLOW_URI}"
)
