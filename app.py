"""Streamlit Application: Bank Customer Churn Intelligence & Relationship Manager Studio."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.churn_engine import ChurnEngine
from src.threshold_optimizer import DEFAULT_OPTIMAL_THRESHOLD

st.set_page_config(
    page_title="RetentionOps AI | Churn & Revenue Risk Platform",
    page_icon="🏦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0.2rem; }
    .sub-header { color: #475569; font-size: 1.05rem; margin-bottom: 1.2rem; }
    .risk-banner {
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        margin-bottom: 15px;
    }
    .banner-critical { background-color: #fee2e2; color: #b91c1c; border: 1px solid #f87171; }
    .banner-high { background-color: #ffedd5; color: #c2410c; border: 1px solid #fb923c; }
    .banner-moderate { background-color: #fef9c3; color: #a16207; border: 1px solid #facc15; }
    .banner-loyal { background-color: #dcfce7; color: #15803d; border: 1px solid #4ade80; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🏦 Bank Customer Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Enterprise churn risk engine with cost-sensitive threshold tuning, CLV revenue loss modeling, TreeSHAP waterfall explainability, and actionable retention playbooks.</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    return ChurnEngine()


engine = load_engine()

# Presets for fast exploration
PRESETS = {
    "Custom Profile": None,
    "High-Balance Inactive Flight Risk": {
        "CreditScore": 620, "Geography": "Germany", "Gender": "Female", "Age": 48,
        "Tenure": 2, "Balance": 95000.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 0, "EstimatedSalary": 85000.0,
    },
    "Single-Product German Account": {
        "CreditScore": 580, "Geography": "Germany", "Gender": "Male", "Age": 42,
        "Tenure": 3, "Balance": 65000.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 70000.0,
    },
    "Loyal Multi-Product Active Account": {
        "CreditScore": 750, "Geography": "France", "Gender": "Male", "Age": 34,
        "Tenure": 6, "Balance": 45000.0, "NumOfProducts": 2, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 65000.0,
    },
}

preset_choice = st.selectbox("Quick-Load Customer Scenario", list(PRESETS.keys()))
preset_data = PRESETS[preset_choice] or {
    "CreditScore": 650, "Geography": "France", "Gender": "Female", "Age": 40,
    "Tenure": 5, "Balance": 50000.0, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 1, "EstimatedSalary": 60000.0,
}

col_in1, col_in2 = st.columns(2)

with col_in1:
    st.subheader("Demographics & Account Profile")
    credit_score = st.slider("Credit Score", 300, 850, int(preset_data["CreditScore"]))
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(preset_data["Geography"]))
    gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(preset_data["Gender"]))
    age = st.slider("Customer Age", 18, 90, int(preset_data["Age"]))
    tenure = st.slider("Tenure with Bank (Years)", 0, 10, int(preset_data["Tenure"]))

with col_in2:
    st.subheader("Financial Assets & Engagement")
    balance = st.number_input("Account Deposit Balance ($)", min_value=0.0, max_value=500000.0, value=float(preset_data["Balance"]), step=2500.0)
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0.0, max_value=500000.0, value=float(preset_data["EstimatedSalary"]), step=2500.0)
    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=int(preset_data["NumOfProducts"]) - 1)
    has_card = st.radio("Has Credit Card?", ["Yes", "No"], index=0 if preset_data["HasCrCard"] else 1, horizontal=True)
    is_active = st.radio("Active Member Status?", ["Yes", "No"], index=0 if preset_data["IsActiveMember"] else 1, horizontal=True)

with st.expander("Decision Threshold & Cost Parameters", expanded=False):
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        threshold = st.slider(
            "Operating Decision Threshold (τ)",
            0.05, 0.95, float(DEFAULT_OPTIMAL_THRESHOLD), 0.01,
            help="Optimal threshold τ* = 0.275 minimizes banking cost matrix ($2,000 lost churner vs $100 campaign).",
        )
    with t_col2:
        st.info("💡 Standard default is 0.50. Banking research shows optimal threshold τ* = 0.275 captures +5.1% more churners while minimizing expected net loss.")

if st.button("Evaluate Churn Risk & Retention Strategy", type="primary", use_container_width=True):
    cust_record = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": salary,
    }

    res = engine.assess_customer(cust_record, threshold=threshold)
    fin = res["financials"]
    exp = res["explanation"]
    ret = res["retention_strategy"]

    st.markdown("---")

    # Risk Banner
    risk_tier = res["risk_tier"]
    banner_cls = {
        "Critical Flight Risk": "banner-critical",
        "High Churn Propensity": "banner-high",
        "Moderate Churn Risk": "banner-moderate",
        "Loyal Account": "banner-loyal",
    }.get(risk_tier, "banner-loyal")

    status_txt = "AT-RISK OF CHURNING" if res["is_churn"] else "ACCOUNT STABLE & LOYAL"
    st.markdown(
        f'<div class="risk-banner {banner_cls}">{risk_tier.upper()} &nbsp;|&nbsp; {status_txt} &nbsp;(P = {res["churn_probability"]*100:.1f}%)</div>',
        unsafe_allow_html=True,
    )

    # 4 KPI Metrics
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Churn Probability", f"{res['churn_probability']*100:.1f}%", f"Threshold: {res['decision_threshold']:.3f}")
    k2.metric("Annual Customer Margin", f"${fin['annual_customer_margin']:,.2f}/yr")
    k3.metric("3-Year Forward CLV", f"${fin['clv_3yr']:,.2f}")
    k4.metric("Customer Value-at-Risk", f"${fin['value_at_risk']:,.2f}", fin["value_tier"].split(":")[0])

    # Visualizations: Gauge & Waterfall
    g_col, w_col = st.columns([1, 2], gap="large")

    with g_col:
        st.subheader("Churn Risk Gauge")
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=res["churn_probability"] * 100,
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#dc2626" if res["is_churn"] else "#16a34a"},
                    "steps": [
                        {"range": [0, 25], "color": "#dcfce7"},
                        {"range": [25, 45], "color": "#fef9c3"},
                        {"range": [45, 70], "color": "#ffedd5"},
                        {"range": [70, 100], "color": "#fee2e2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1e3a8a", "width": 4},
                        "thickness": 0.85,
                        "value": res["decision_threshold"] * 100,
                    },
                },
            )
        )
        gauge_fig.update_layout(height=320, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.caption(f"Blue needle indicates operating threshold τ = {res['decision_threshold']:.3f}")

    with w_col:
        st.subheader("TreeSHAP Local Feature Attribution Waterfall")
        steps = exp["waterfall_steps"]

        x_labels = [s["step"] for s in steps]
        y_values = [s["value"] for s in steps]
        measures = ["absolute" if s["type"] in ["baseline", "total"] else "relative" for s in steps]

        waterfall_fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=measures,
                x=x_labels,
                y=y_values,
                connector={"line": {"color": "#94a3b8"}},
                decreasing={"marker": {"color": "#16a34a"}},
                increasing={"marker": {"color": "#dc2626"}},
                totals={"marker": {"color": "#1e3a8a"}},
            )
        )
        waterfall_fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=30),
            yaxis_title="Log-Odds Contribution",
            showlegend=False,
        )
        st.plotly_chart(waterfall_fig, use_container_width=True)

    # Tactical Retention Playbook
    st.markdown("---")
    st.subheader("📋 Tactical Banking Retention Playbook")

    r_left, r_right = st.columns([2, 1])

    with r_left:
        st.markdown(f"**Primary Strategy**: `{ret['primary_strategy']}`")
        st.write(f"**Clinical Rationale**: *{ret['rationale']}*")
        st.markdown("**Action Checklist for Relationship Manager:**")
        for idx, item in enumerate(ret["action_items"], 1):
            st.write(f"{idx}. {item}")

    with r_right:
        st.metric("Action Urgency SLA", ret["urgency"], f"Window: {ret['sla_hours']} hrs")
        st.metric("Estimated Campaign Cost", f"${ret['estimated_intervention_cost']:,.2f}")
        st.metric("Expected Saved CLV", f"${ret['expected_saved_clv']:,.2f}", f"{ret['expected_retention_success_prob']*100:.0f}% success rate")
        st.metric("Campaign Net Expected ROI", f"${ret['net_expected_profit']:,.2f}", f"{ret['roi_ratio_pct']:+.1f}% ROI")

st.caption("Bank Customer Churn Intelligence • XGBoost Classifier + TreeSHAP + Cost-Sensitive Thresholds • 100% Offline CPU")
