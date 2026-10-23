"""Unit tests for the unified ChurnEngine facade."""

from pathlib import Path
import pandas as pd
import pytest

from src.churn_engine import ChurnEngine


@pytest.fixture(scope="module")
def churn_engine() -> ChurnEngine:
    return ChurnEngine()


def test_churn_engine_initialization(churn_engine: ChurnEngine) -> None:
    """Verify engine components load successfully."""
    assert churn_engine.model is not None
    assert churn_engine.preprocessor is not None
    assert churn_engine.clv_engine is not None
    assert churn_engine.explainer is not None
    assert churn_engine.retention_advisor is not None


def test_assess_customer_full_payload(churn_engine: ChurnEngine) -> None:
    """Verify assess_customer produces all expected schema keys and consistent math."""
    sample = {
        "CreditScore": 680,
        "Geography": "France",
        "Gender": "Male",
        "Age": 42,
        "Tenure": 4,
        "Balance": 65000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 72000.0,
    }

    res = churn_engine.assess_customer(sample)

    assert "customer_profile" in res
    assert res["churn_prediction"] in [0, 1]
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert 0.0 <= res["stay_probability"] <= 1.0
    assert pytest.approx(res["churn_probability"] + res["stay_probability"], abs=1e-3) == 1.0
    assert "risk_tier" in res
    assert "financials" in res
    assert "explanation" in res
    assert "retention_strategy" in res
    assert res["financials"]["clv_3yr"] > 0.0


def test_assess_customer_threshold_sensitivity(churn_engine: ChurnEngine) -> None:
    """Verify passing a custom threshold changes the binary churn decision."""
    sample = {
        "CreditScore": 600,
        "Geography": "Germany",
        "Gender": "Female",
        "Age": 45,
        "Tenure": 2,
        "Balance": 80000.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 0,
        "EstimatedSalary": 60000.0,
    }

    # At low threshold, should be flagged as churn
    res_low = churn_engine.assess_customer(sample, threshold=0.10)
    assert res_low["churn_prediction"] == 1

    # At high threshold, should not be flagged as churn
    res_high = churn_engine.assess_customer(sample, threshold=0.95)
    assert res_high["churn_prediction"] == 0


def test_assess_batch(churn_engine: ChurnEngine) -> None:
    """Verify batch evaluation computes aggregate risk metrics and sorted leaderboard."""
    df = pd.DataFrame([
        {
            "CustomerId": 101,
            "Surname": "Dupont",
            "CreditScore": 600,
            "Geography": "France",
            "Gender": "Male",
            "Age": 38,
            "Tenure": 5,
            "Balance": 20000.0,
            "NumOfProducts": 2,
            "HasCrCard": 1,
            "IsActiveMember": 1,
            "EstimatedSalary": 40000.0,
        },
        {
            "CustomerId": 102,
            "Surname": "Müller",
            "CreditScore": 620,
            "Geography": "Germany",
            "Gender": "Female",
            "Age": 55,
            "Tenure": 2,
            "Balance": 95000.0,
            "NumOfProducts": 1,
            "HasCrCard": 1,
            "IsActiveMember": 0,
            "EstimatedSalary": 85000.0,
        },
    ])

    batch_res = churn_engine.assess_batch(df)

    assert batch_res["num_customers"] == 2
    assert "churners_flagged" in batch_res
    assert "total_balance_at_risk" in batch_res
    assert "total_clv_at_risk" in batch_res
    assert len(batch_res["leaderboard"]) == 2
    # Check leaderboard sorting by Value_At_Risk descending
    assert batch_res["leaderboard"]["Value_At_Risk"].iloc[0] >= batch_res["leaderboard"]["Value_At_Risk"].iloc[1]


def test_churn_engine_missing_model() -> None:
    """Verify exception raised if model cannot be loaded."""
    eng = ChurnEngine(model_path="nonexistent_model.pkl")
    with pytest.raises(RuntimeError):
        eng.assess_customer({})
