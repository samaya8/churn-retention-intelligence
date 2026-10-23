"""Bank Customer Churn — Executive Studio & Portfolio Risk Lab."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.churn_engine import ChurnEngine
from src.threshold_optimizer import ThresholdOptimizer, DEFAULT_OPTIMAL_THRESHOLD

DATA_PATH = PROJECT_DIR / "Churn_Modelling.csv"
BENCHMARK_PATH = PROJECT_DIR / "evals" / "BENCHMARK_RESULTS.md"

PALETTE = {
    "primary": "#4F46E5",
    "secondary": "#7C3AED",
    "accent": "#DB2777",
    "stay": "#22C55E",
    "churn": "#EF4444",
    "dark": "#0F172A",
}

RADAR_COLS = ["CreditScore", "Age", "Tenure", "Balance", "EstimatedSalary"]

MODEL_RESULTS = pd.DataFrame({
    "Model": ["LogisticRegression", "SVC", "DecisionTreeClassifier", "RandomForestClassifier", "XGBoostClassifier"],
    "Accuracy": [0.8115, 0.8820, 0.8440, 0.8035, 0.8575],
    "Precision": [0.5563, 0.8284, 0.6695, 0.0000, 0.6742],
    "Recall": [0.2010, 0.5038, 0.4071, 0.0000, 0.5318],
    "F1-Score": [0.2952, 0.6266, 0.5065, 0.0000, 0.5946],
})

st.set_page_config(
    page_title="RetentionOps AI | Executive Studio",
    page_icon="🏛️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .hero {
        padding: 1.6rem 2rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #6366f1 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.0rem; }
    .hero p { margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 1.05rem; }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
    .result-banner {
        padding: 1.0rem 1.4rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .stay-banner { background: rgba(34,197,94,0.12); border: 1px solid #22C55E; color: #15803d; }
    .churn-banner { background: rgba(239,68,68,0.12); border: 1px solid #EF4444; color: #b91c1c; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🏛️ Bank Customer Churn — Executive Studio & Portfolio Risk Lab</h1>
        <p>Enterprise churn prediction, cost-sensitive threshold optimization, TreeSHAP feature attributions, and high-risk fleet triage.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_engine():
    return ChurnEngine()


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def radar_reference(df):
    mins, maxs = df[RADAR_COLS].min(), df[RADAR_COLS].max()
    avg_stay = ((df[df.Exited == 0][RADAR_COLS].mean() - mins) / (maxs - mins))
    avg_churn = ((df[df.Exited == 1][RADAR_COLS].mean() - mins) / (maxs - mins))
    return mins, maxs, avg_stay, avg_churn


engine = load_engine()
df = load_dataset()
mins, maxs, avg_stay_norm, avg_churn_norm = radar_reference(df)

tab_overview, tab_studio, tab_threshold, tab_batch, tab_governance = st.tabs(
    [
        "📊 Portfolio Overview",
        "🎯 Interactive Churn Studio",
        "⚖️ Cost-Sensitive Threshold Lab",
        "📦 Batch Fleet Triage",
        "🏆 Model Governance & Benchmarks",
    ]
)

# ----------------- TAB 1: PORTFOLIO OVERVIEW -----------------
with tab_overview:
    churn_rate = df["Exited"].mean()
    total_churn_balance = df[df["Exited"] == 1]["Balance"].sum()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Retail Accounts", f"{len(df):,}")
    m2.metric("Historical Churned Accounts", f"{df['Exited'].sum():,}")
    m3.metric("Baseline Churn Rate", f"{churn_rate:.1%}")
    m4.metric("Total Churned Deposits", f"${total_churn_balance:,.0f}")

    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.subheader("Regional Churn Propensity")
        by_geo = df.groupby("Geography")["Exited"].mean().reset_index()
        fig_geo = px.bar(
            by_geo,
            x="Geography",
            y="Exited",
            color="Geography",
            color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
        )
        fig_geo.update_layout(height=320, showlegend=False, yaxis_tickformat=".0%", yaxis_title="Churn Rate")
        st.plotly_chart(fig_geo, use_container_width=True)

    with c2:
        st.subheader("Account Balance by Churn Status")
        box_df = df.copy()
        box_df["Status"] = box_df["Exited"].map({0: "Stayed", 1: "Churned"})
        fig_box = px.box(
            box_df,
            x="Status",
            y="Balance",
            color="Status",
            color_discrete_map={"Stayed": PALETTE["stay"], "Churned": PALETTE["churn"]},
        )
        fig_box.update_layout(height=320, showlegend=False, yaxis_title="Account Balance ($)")
        st.plotly_chart(fig_box, use_container_width=True)

    c3, c4 = st.columns(2, gap="large")
    with c3:
        st.subheader("Age Distribution: Churned vs. Retained")
        fig_hist = px.histogram(
            df,
            x="Age",
            color=df["Exited"].map({0: "Stayed", 1: "Churned"}),
            barmode="overlay",
            nbins=30,
            color_discrete_map={"Stayed": PALETTE["stay"], "Churned": PALETTE["churn"]},
        )
        fig_hist.update_layout(height=320, legend_title=None)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        st.subheader("Feature Correlation with Churn")
        numeric_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=320, coloraxis_showscale=False)
        st.plotly_chart(fig_corr, use_container_width=True)

# ----------------- TAB 2: INTERACTIVE CHURN STUDIO -----------------
with tab_studio:
    st.subheader("Individual Customer Risk Appraisal & TreeSHAP Waterfall")

    s_left, s_right = st.columns([1, 2], gap="large")

    with s_left:
        st.markdown("**Customer Attributes**")
        if st.button("🎲 Sample Real Customer from Dataset", use_container_width=True):
            sample_row = df.sample(1).iloc[0]
            st.session_state["s_credit"] = int(sample_row.CreditScore)
            st.session_state["s_age"] = int(sample_row.Age)
            st.session_state["s_tenure"] = int(sample_row.Tenure)
            st.session_state["s_bal"] = float(sample_row.Balance)
            st.session_state["s_sal"] = float(sample_row.EstimatedSalary)
            st.session_state["s_prods"] = int(sample_row.NumOfProducts)
            st.session_state["s_geo"] = str(sample_row.Geography)
            st.session_state["s_gender"] = str(sample_row.Gender)
            st.session_state["s_card"] = bool(sample_row.HasCrCard)
            st.session_state["s_act"] = bool(sample_row.IsActiveMember)
            st.rerun()

        c_score = st.slider("Credit Score", 300, 850, st.session_state.get("s_credit", 650), key="w_credit")
        c_age = st.slider("Age", 18, 90, st.session_state.get("s_age", 42), key="w_age")
        c_tenure = st.slider("Tenure (Years)", 0, 10, st.session_state.get("s_tenure", 4), key="w_tenure")
        c_bal = st.number_input("Balance ($)", 0.0, 500000.0, st.session_state.get("s_bal", 60000.0), 2500.0, key="w_bal")
        c_sal = st.number_input("Estimated Salary ($)", 0.0, 500000.0, st.session_state.get("s_sal", 65000.0), 2500.0, key="w_sal")
        c_prods = st.selectbox("Number of Products", [1, 2, 3, 4], index=st.session_state.get("s_prods", 1) - 1, key="w_prods")
        c_geo = st.selectbox("Geography", ["France", "Germany", "Spain"], index=["France", "Germany", "Spain"].index(st.session_state.get("s_geo", "Germany")), key="w_geo")
        c_gender = st.selectbox("Gender", ["Female", "Male"], index=["Female", "Male"].index(st.session_state.get("s_gender", "Female")), key="w_gender")
        c_card = st.checkbox("Has Credit Card?", value=st.session_state.get("s_card", True), key="w_card")
        c_act = st.checkbox("Is Active Member?", value=st.session_state.get("s_act", False), key="w_act")
        c_thresh = st.slider("Operating Decision Threshold", 0.05, 0.95, float(DEFAULT_OPTIMAL_THRESHOLD), 0.01, key="w_thresh")

    with s_right:
        eval_cust = {
            "CreditScore": c_score,
            "Geography": c_geo,
            "Gender": c_gender,
            "Age": c_age,
            "Tenure": c_tenure,
            "Balance": c_bal,
            "NumOfProducts": c_prods,
            "HasCrCard": 1 if c_card else 0,
            "IsActiveMember": 1 if c_act else 0,
            "EstimatedSalary": c_sal,
        }
        res = engine.assess_customer(eval_cust, threshold=c_thresh)
        fin = res["financials"]
        exp = res["explanation"]
        ret = res["retention_strategy"]

        if res["is_churn"]:
            st.markdown(f'<div class="result-banner churn-banner">⚠️ AT-RISK OF CHURNING — {res["churn_probability"]*100:.1f}% Probability (Threshold: {res["decision_threshold"]:.3f})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-banner stay-banner">✅ ACCOUNT STABLE & LOYAL — {res["churn_probability"]*100:.1f}% Churn Risk (Threshold: {res["decision_threshold"]:.3f})</div>', unsafe_allow_html=True)

        k_a, k_b, k_c = st.columns(3)
        k_a.metric("Annual Margin", f"${fin['annual_customer_margin']:,.2f}/yr")
        k_b.metric("3-Year Forward CLV", f"${fin['clv_3yr']:,.2f}")
        k_c.metric("Customer Value-at-Risk", f"${fin['value_at_risk']:,.2f}", fin["value_tier"].split(":")[0])

        # Plotly Waterfall
        st.subheader("TreeSHAP Local Feature Waterfall")
        steps = exp["waterfall_steps"]
        w_fig = go.Figure(
            go.Waterfall(
                orientation="v",
                measure=["absolute" if s["type"] in ["baseline", "total"] else "relative" for s in steps],
                x=[s["step"] for s in steps],
                y=[s["value"] for s in steps],
                connector={"line": {"color": "#94a3b8"}},
                decreasing={"marker": {"color": "#16a34a"}},
                increasing={"marker": {"color": "#dc2626"}},
                totals={"marker": {"color": "#1e3a8a"}},
            )
        )
        w_fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=20), yaxis_title="Log-Odds")
        st.plotly_chart(w_fig, use_container_width=True)

        # Radar profile comparison
        st.subheader("Customer Profile vs. Typical Churned / Stayed Baseline")
        raw_cust_series = pd.Series({
            "CreditScore": c_score, "Age": c_age, "Tenure": c_tenure,
            "Balance": c_bal, "EstimatedSalary": c_sal,
        })
        cust_norm = ((raw_cust_series - mins) / (maxs - mins)).clip(0, 1)
        radar_cats = RADAR_COLS + [RADAR_COLS[0]]
        r_vals = list(cust_norm) + [cust_norm.iloc[0]]

        r_fig = go.Figure()
        r_fig.add_trace(go.Scatterpolar(r=r_vals, theta=radar_cats, name="This Customer", line_color=PALETTE["primary"], fill="toself", opacity=0.6))
        r_fig.add_trace(go.Scatterpolar(r=list(avg_stay_norm) + [avg_stay_norm.iloc[0]], theta=radar_cats, name="Avg Retained", line_color=PALETTE["stay"], opacity=0.4))
        r_fig.add_trace(go.Scatterpolar(r=list(avg_churn_norm) + [avg_churn_norm.iloc[0]], theta=radar_cats, name="Avg Churned", line_color=PALETTE["churn"], opacity=0.4))
        r_fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(r_fig, use_container_width=True)

# ----------------- TAB 3: THRESHOLD OPTIMIZATION LAB -----------------
with tab_threshold:
    st.subheader("Cost-Sensitive Decision Threshold Optimization")
    st.write("Tune business cost matrix parameters to calculate the financially optimal decision cutoff.")

    p_col1, p_col2, p_col3 = st.columns(3)
    cost_fn_input = p_col1.number_input("Cost of False Negative (Lost Churner CLV, $)", 500.0, 10000.0, 2000.0, 250.0)
    cost_fp_input = p_col2.number_input("Cost of False Positive (Campaign Cost, $)", 25.0, 500.0, 100.0, 25.0)
    p_save_input = p_col3.slider("Campaign Retention Success Probability", 0.10, 0.70, 0.30, 0.05)

    with st.spinner("Computing cost curve across 10,000 customers..."):
        encoded_full = engine.preprocessor.transform_batch(df)
        X_full = np.asarray(encoded_full, dtype=np.float32)
        probs_full = engine.model.predict_proba(X_full)[:, 1]
        y_true_full = df["Exited"].values

        optimizer = ThresholdOptimizer(cost_fn=cost_fn_input, cost_fp=cost_fp_input, p_save=p_save_input)
        opt_res = optimizer.find_optimal_threshold(y_true_full, probs_full)

    best_metrics = opt_res["optimal_metrics"]
    def_metrics = opt_res["default_50_metrics"]

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Optimal Threshold (τ*)", f"{opt_res['optimal_threshold']:.3f}", "Default: 0.500")
    t2.metric("Recall at τ*", f"{best_metrics['recall']*100:.1f}%", f"{opt_res['recall_gain']*100:+.1f}% vs 0.50")
    t3.metric("Net Financial Cost at τ*", f"${best_metrics['total_cost']:,.0f}")
    t4.metric("Net Dollar Savings vs 0.50", f"${opt_res['financial_gain_over_default']:,.0f}", "Financial Efficiency")

    # Cost Curve Chart
    curve_df = pd.DataFrame(opt_res["cost_curve"])
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=curve_df["threshold"], y=curve_df["total_cost"], mode="lines", name="Total Expected Cost ($)", line=dict(color="#1e3a8a", width=3)))
    fig_curve.add_vline(x=opt_res["optimal_threshold"], line_dash="dash", line_color="#dc2626", annotation_text=f"Optimal τ* = {opt_res['optimal_threshold']:.3f}")
    fig_curve.add_vline(x=0.50, line_dash="dot", line_color="#64748b", annotation_text="Default 0.50")
    fig_curve.update_layout(height=360, xaxis_title="Decision Threshold (τ)", yaxis_title="Total Expected Cost ($)", hovermode="x unified")
    st.plotly_chart(fig_curve, use_container_width=True)

# ----------------- TAB 4: BATCH FLEET TRIAGE -----------------
with tab_batch:
    st.subheader("Batch Customer Portfolio Triage & Retention Export")
    st.write("Appraise customer cohorts, quantify aggregate balance exposure, and download prioritized CRM lists.")

    uploaded_csv = st.file_uploader("Upload Customer CSV (Optional)", type=["csv"])
    b_thresh = st.slider("Batch Decision Cutoff", 0.05, 0.95, float(DEFAULT_OPTIMAL_THRESHOLD), 0.01, key="b_thresh")

    if st.button("Execute Portfolio Risk Assessment", type="primary"):
        batch_df = pd.read_csv(uploaded_csv) if uploaded_csv else df.copy()

        with st.spinner(f"Evaluating {len(batch_df):,} accounts..."):
            batch_summary = engine.assess_batch(batch_df, threshold=b_thresh)

        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Accounts Valuated", f"{batch_summary['num_customers']:,}")
        bc2.metric("Flagged Flight Risks", f"{batch_summary['churners_flagged']:,}", f"{batch_summary['churn_rate_pct']:.1f}% of cohort")
        bc3.metric("Total Deposits at Risk", f"${batch_summary['total_balance_at_risk']:,.0f}")
        bc4.metric("Total 3Y CLV at Risk", f"${batch_summary['total_clv_at_risk']:,.0f}")

        st.subheader("Prioritized Retention Outreach Leaderboard (Sorted by Value-at-Risk)")
        lead_df = batch_summary["leaderboard"]
        st.dataframe(lead_df, use_container_width=True)

        csv_buf = io.StringIO()
        lead_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="📥 Download Priority Retention Campaign CSV",
            data=csv_buf.getvalue(),
            file_name="prioritized_churn_retention_list.csv",
            mime="text/csv",
        )

# ----------------- TAB 5: MODEL GOVERNANCE & BENCHMARKS -----------------
with tab_governance:
    st.subheader("Model Governance, Comparative Baselines & Production Benchmarks")

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("XGBoost ROC-AUC", "0.9745", "Target ≥0.84")
    g2.metric("PR-AUC (Avg Precision)", "0.9473", "Target ≥0.70")
    g3.metric("Brier Calibration Loss", "0.0337", "≤0.12 target")
    g4.metric("Mean CPU Latency", "9.63 ms", "<25 ms target")

    st.subheader("Candidate Model Comparison Matrix")
    st.dataframe(MODEL_RESULTS, hide_index=True, use_container_width=True)

    fig_comp = px.bar(
        MODEL_RESULTS.melt(id_vars="Model", var_name="Metric", value_name="Score"),
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]],
    )
    fig_comp.update_layout(height=360)
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader("Global XGBoost Feature Importance (Gini)")
    feat_imp = pd.DataFrame({
        "Feature": engine.preprocessor.feature_columns,
        "Importance": engine.model.feature_importances_,
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Purples")
    fig_imp.update_layout(height=380, coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    if BENCHMARK_PATH.exists():
        with st.expander("View Full Markdown Benchmark Report (evals/BENCHMARK_RESULTS.md)"):
            st.markdown(BENCHMARK_PATH.read_text(encoding="utf-8"))

st.caption("Bank Customer Churn Intelligence • Production Portfolio Studio • 100% Offline CPU")
