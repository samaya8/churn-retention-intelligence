"""Unit tests for TreeSHAP local feature attribution and waterfall explainability."""

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.explainer import ChurnExplainer
from src.preprocessor import ChurnPreprocessor

PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def explainer() -> ChurnExplainer:
    model = joblib.load(PROJECT_DIR / "xgb_churn_model.pkl")
    cols = list(joblib.load(PROJECT_DIR / "feature_columns.pkl"))
    return ChurnExplainer(model=model, feature_columns=cols)


@pytest.fixture(scope="module")
def sample_encoded() -> tuple:
    pre = ChurnPreprocessor()
    cust = {
        "CreditScore": 620,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 50,
        "Tenure": 3,
        "Balance": 70000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 75000.0,
    }
    return pre.transform_single(cust), cust


def test_explainer_additivity(explainer: ChurnExplainer, sample_encoded: tuple) -> None:
    """Verify exact TreeSHAP additivity: sum(phi_i) + base_bias == final_log_odds."""
    encoded_df, raw_cust = sample_encoded
    res = explainer.explain(encoded_df, raw_customer=raw_cust)

    assert "base_log_odds" in res
    assert "final_log_odds" in res
    assert "top_drivers" in res
    assert len(res["top_drivers"]) == 13

    # Check sum of shap values + base == final
    shap_sum = sum(d["shap_value"] for d in res["top_drivers"])
    expected_margin = res["base_log_odds"] + shap_sum
    assert res["final_log_odds"] == pytest.approx(expected_margin, abs=1e-3)


def test_explainer_drivers_categorization(explainer: ChurnExplainer, sample_encoded: tuple) -> None:
    """Verify drivers are categorized into risk accelerators and loyalty anchors."""
    encoded_df, raw_cust = sample_encoded
    res = explainer.explain(encoded_df, raw_customer=raw_cust)

    accelerators = res["risk_accelerators"]
    anchors = res["loyalty_anchors"]

    for acc in accelerators:
        assert acc["shap_value"] > 0
        assert "Risk Accelerator" in acc["direction"]

    for anc in anchors:
        assert anc["shap_value"] <= 0
        assert "Loyalty Anchor" in anc["direction"]


def test_explainer_waterfall_steps(explainer: ChurnExplainer, sample_encoded: tuple) -> None:
    """Verify waterfall payload structure."""
    encoded_df, raw_cust = sample_encoded
    res = explainer.explain(encoded_df, raw_customer=raw_cust)

    steps = res["waterfall_steps"]
    assert len(steps) >= 3
    assert steps[0]["type"] == "baseline"
    assert steps[-1]["type"] == "total"


def test_explainer_missing_model() -> None:
    """Verify handling when model is None."""
    exp = ChurnExplainer(model=None)
    res = exp.explain(None)
    assert res["status"] == "error"
