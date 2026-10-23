# RetentionOps AI: Customer Churn & Retention Risk Platform & Retention Strategy Engine

[![CI](https://github.com/samaya8/churn-retention-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/samaya8/churn-retention-intelligence/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/Tests-45%20Passed-success?style=flat&logo=pytest&logoColor=white)](tests/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **GitHub Repository**: `churn-retention-intelligence`  
> **Repository Description**: Enterprise churn risk engine featuring cost-sensitive threshold optimization ($\tau^*$), Customer Lifetime Value (CLV) revenue loss modeling, TreeSHAP waterfall feature attributions, dynamic retention action playbooks, dual Streamlit apps, production CLI, and automated benchmark evaluation suite.

An end-to-end enterprise machine learning system that transforms retail banking customer churn prediction into an actionable, value-maximizing retention engine. Beyond raw probability scoring, it models forward customer lifetime value, optimizes decision thresholds for financial payoff, attributes risk drivers via TreeSHAP, and delivers tactical retention playbooks for frontline relationship managers and executive risk leaders.

---

## Key Enterprise Features

- **Trained Gradient Boosting Pipeline**: High-performance XGBoost classification engine with exact categorical encoding and standard scaling alignment (`xgb_churn_model.pkl`, `scaler.pkl`, `feature_columns.pkl`).
- **Cost-Sensitive Threshold Optimization ($\tau^*$)**: Sweeps decision thresholds across asymmetry matrices ($C_{\text{FN}} = \$2,000$, $C_{\text{FP}} = \$100$, $P_{\text{save}} = 30\%$) to find the financially optimal cutoff ($\tau^* = 0.275$), capturing **+5.1% more churners** and saving **\$33,700** compared to the arbitrary $0.50$ default.
- **Customer Lifetime Value (CLV) & Value-at-Risk Engine**: Models annual banking margins from Net Interest Margin (NIM) on balance deposits, transactional salary velocity, and product multipliers, discounting cash flows over 3-year and 5-year horizons and computing expected dollar loss ($\text{VaR} = \text{CLV} \times P(\text{churn})$).
- **TreeSHAP Waterfall Feature Attribution**: Exact local feature attributions decomposing individual log-odds into top *Risk Accelerators* (e.g. age transition, inactivity, product friction) and *Loyalty Anchors* (e.g. balance anchoring, tenure inertia), guaranteeing mathematical additivity ($f(x) = \phi_0 + \sum \phi_i$).
- **Tactical Retention Action Advisor**: Dynamic diagnostic engine generating triage urgency tiers (Immediate Emergency, Urgent 24h, Proactive 72h, Routine), tailored intervention playbooks, estimated intervention costs, expected success rates, and projected net campaign ROI ($ROI = (\text{Saved CLV} - \text{Cost}) / \text{Cost}$).
- **Dual Streamlit Applications**:
  - `app.py`: Frontline Relationship Manager Action Portal with quick customer persona presets, interactive risk gauge, Plotly TreeSHAP waterfall charts, and custom retention playbooks.
  - `app2.py`: Executive Strategy Studio with 5 dedicated tabs: Executive Overview & KPI Cockpit, Single Customer Studio, Decision Threshold Lab, Batch Fleet Triage, and Model Governance & Audit.
- **Production CLI (`cli.py`)**: Full command-line interface supporting `predict`, `explain`, `retention`, `batch`, `threshold`, and `benchmark` commands with rich formatted terminal cards or machine-readable JSON output (`--json`).
- **Automated Benchmark Harness (`evals/runner.py`)**: Reproducible benchmarking suite computing ROC-AUC, PR-AUC, F1, Brier calibration score, Expected Calibration Error (ECE), net financial gain, and single-core CPU inference latency.
- **Comprehensive Pytest Suite**: 45 unit and integration tests achieving **87% overall coverage** (and **98.4% core library coverage**) with complete backward compatibility for legacy artifacts.

---

## System Architecture

```text
                                  [Customer Demographics & Financials]
                                                   │
                                                   ▼
                                        [ChurnPreprocessor]
                             ├── Categorical Encoding (Geography, Gender)
                             ├── Feature Alignment (feature_columns.pkl)
                             └── Standard Scaling (scaler.pkl)
                                                   │
                                                   ▼
                                       [XGBoost Classifier]
                                      (xgb_churn_model.pkl)
                                                   │
                         ┌─────────────────────────┼─────────────────────────┐
                         ▼                         ▼                         ▼
             [ThresholdOptimizer]            [CLVEngine]             [ChurnExplainer]
            - Sweeps τ ∈ [0.05, 0.95]    - Annual Margin Model      - C++ Native TreeSHAP
            - Finds τ* = 0.275           - 3Y/5Y Discounted CLV     - Additivity Verification
            - Saves $33,700 vs 0.50      - Value-at-Risk ($)        - Risk vs Anchor Drivers
                         │                         │                         │
                         └─────────────────────────┼─────────────────────────┘
                                                   ▼
                                        [RetentionAdvisor]
                                - Urgency Tier & Response SLA Target
                                - Targeted Retention Playbook
                                - Campaign Cost, Success Rate & Net ROI
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         ▼                                                   ▼
           [Dual Streamlit Dashboards]                              [Production CLI]
     ├── Relationship Manager Action Portal (app.py)          ├── cli.py predict
     └── Executive Strategy Studio (app2.py)                  ├── cli.py explain
                                                              ├── cli.py retention
                                                              ├── cli.py batch
                                                              ├── cli.py threshold
                                                              └── cli.py benchmark
```

---

## Tech Stack

- **Language & Runtime**: Python 3.10+
- **Machine Learning & Boosting**: XGBoost 2.x, Scikit-Learn, Joblib, NumPy, Pandas
- **Explainability**: TreeSHAP (Tree-based SHapley Additive exPlanations)
- **Financial & Risk Modeling**: Discounted Cash Flow (DCF), Value-at-Risk (VaR), Cost Matrix Optimization
- **Web Applications & Data Viz**: Streamlit, Plotly Express & Graph Objects
- **Quality Assurance & Verification**: Pytest, Pytest-Cov (45 tests, 87% coverage)

---

## Repository Structure

```text
.
├── evals/
│   ├── __init__.py
│   ├── runner.py                              # Automated benchmark evaluation harness
│   └── BENCHMARK_RESULTS.md                   # Verified empirical benchmark audit report
├── notebooks/
│   └── data-science-end-to-end-project.ipynb  # Exploratory analysis & baseline training
├── src/
│   ├── __init__.py                            # Package exports
│   ├── churn_engine.py                        # Unified facade orchestrating full lifecycle
│   ├── clv_engine.py                          # Customer Lifetime Value & revenue loss model
│   ├── explainer.py                           # TreeSHAP local waterfall explainer
│   ├── preprocessor.py                        # Strict tabular feature alignment & scaling
│   ├── retention_advisor.py                   # Tactical retention playbooks & ROI estimator
│   └── threshold_optimizer.py                 # Cost-sensitive decision threshold optimization
├── tests/
│   ├── conftest.py                            # Pytest fixtures and environment configuration
│   ├── test_churn_engine.py                   # ChurnEngine facade unit tests
│   ├── test_cli.py                            # CLI subcommands & formatting tests
│   ├── test_clv_engine.py                     # CLV & margin calculation tests
│   ├── test_evals.py                          # Evaluation harness & metrics tests
│   ├── test_explainer.py                      # TreeSHAP waterfall additivity tests
│   ├── test_inference.py                      # Legacy pipeline compatibility tests
│   ├── test_preprocessor.py                   # Preprocessing & schema alignment tests
│   ├── test_retention_advisor.py              # Retention playbook & ROI tests
│   └── test_threshold_optimizer.py            # Cost matrix & threshold search tests
├── app.py                                     # Frontline Relationship Manager Action Portal
├── app2.py                                    # Executive Strategy Studio Dashboard (5 tabs)
├── cli.py                                     # Production CLI with 6 subcommands
├── Churn_Modelling.csv                        # Primary tabular dataset (10,000 customers)
├── feature_columns.pkl                        # Serialized training feature schema (11 features)
├── scaler.pkl                                 # Fitted StandardScaler instance
├── xgb_churn_model.pkl                        # Trained XGBoostClassifier model
├── test.txt                                   # Curated verification profiles
├── requirements.txt                           # Pinned dependencies
├── LICENSE                                    # MIT License
├── PROJECT_DETAILS.md                         # Detailed project specifications & guide
└── README.md                                  # Repository documentation
```

---

## Setup & Quick Start

### 1. Prerequisites & Installation

```bash
git clone https://github.com/your-username/churn-retention-intelligence.git
cd churn-retention-intelligence

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Production CLI

Assess a single customer:
```bash
python cli.py predict --age 45 --balance 75000 --geography Germany --is-active 0
```

Explain churn drivers with TreeSHAP:
```bash
python cli.py explain --age 45 --balance 75000 --geography Germany --is-active 0
```

Generate tactical retention recommendations:
```bash
python cli.py retention --age 45 --balance 75000 --geography Germany --is-active 0
```

Run threshold optimization across cost matrix:
```bash
python cli.py threshold --cost-fn 2000 --cost-fp 100 --p-save 0.30
```

Triage an entire customer cohort from CSV:
```bash
python cli.py batch --input Churn_Modelling.csv --limit 100 --output triage_results.csv
```

Run the automated evaluation benchmark:
```bash
python cli.py benchmark
```

### 3. Running the Streamlit Applications

Launch the **Frontline Relationship Manager Action Portal**:
```bash
streamlit run app.py
```

Launch the **Executive Strategy Studio Dashboard**:
```bash
streamlit run app2.py
```

### 4. Running the Automated Test Suite

```bash
pytest --cov=src --cov=cli --cov=evals.runner --cov-report=term-missing tests/
```

---

## Verified Benchmark Results

Empirical results generated via `evals/runner.py` on the complete 10,000-customer dataset:

| Metric Category | Metric | Baseline (τ = 0.50) | Optimized (τ* = 0.275) | Impact / Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Discrimination** | ROC-AUC | 0.9745 | 0.9745 | High separability |
| **Precision-Recall** | PR-AUC (Avg Precision) | 0.9473 | 0.9473 | Robust on minority class |
| **Sensitivity** | Churn Recall | 87.78% | **92.93%** | **+5.15% Attrition Detected** |
| **Accuracy / F1** | F1-Score | 0.9114 | 0.8930 | Cost-calibrated tradeoff |
| **Calibration** | Brier Score Loss | 0.0337 | 0.0337 | Well-calibrated probabilities |
| **Calibration** | Expected Calibration Error | 0.0264 | 0.0264 | Low probability distortion |
| **Financial Impact** | Total Business Cost | \$3,189,500 | **\$3,155,800** | **\$33,700 Net Savings** |
| **Inference Speed** | Single-Customer Latency | 6.56 ms | 6.56 ms | Real-time ready (<10ms) |
| **Throughput** | Single-Core CPU Throughput | ~150 req/sec | ~150 req/sec | High batch efficiency |

---

## Key Risk Drivers & Insights

1. **Age Transition**: Churn risk accelerates drastically for customers aged 45–60 (TreeSHAP attribution pushes log-odds up by +0.97), indicating vulnerability during lifecycle wealth transitions or retirement planning.
2. **Product Saturation Friction**: Customers with 3 or 4 products experience severe friction, churning at rates above 80%. 2 products remains the optimal loyalty anchor.
3. **Regional Disparities**: German accounts experience more than double the churn probability of French accounts, driven by competitive digital banking alternatives.
4. **Active Engagement**: Active monthly membership acts as a strong loyalty anchor, cutting flight risk by more than half.

---

## License & Provenance

- **Dataset**: Bank Customer Churn dataset (`Churn_Modelling.csv`), 10,000 banking customer profiles.
- **License**: Released under the MIT License.
