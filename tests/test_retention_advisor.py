"""Unit tests for tactical banking retention strategy advisory."""

import pytest

from src.retention_advisor import RetentionAdvisor


@pytest.fixture(scope="module")
def advisor() -> RetentionAdvisor:
    return RetentionAdvisor()


def test_recommend_strategy_urgent_high_risk(advisor: RetentionAdvisor) -> None:
    """Test critical flight risk gets assigned Urgent 24-hour SLA."""
    customer = {
        "Balance": 90000.0,
        "Age": 52.0,
        "NumOfProducts": 1,
        "IsActiveMember": 0,
        "Geography": "Germany",
    }
    val = {"value_at_risk": 2500.0}

    res = advisor.recommend_strategy(customer, churn_prob=0.82, value_assessment=val)

    assert "Urgent" in res["urgency"]
    assert res["sla_hours"] == 24
    assert res["is_recommended"] is True
    assert len(res["all_triggered_playbooks"]) >= 2
    assert res["estimated_intervention_cost"] > 0.0
    assert res["expected_saved_clv"] > 0.0


def test_recommend_strategy_low_risk_routine(advisor: RetentionAdvisor) -> None:
    """Test low risk customer receives routine engagement."""
    customer = {
        "Balance": 15000.0,
        "Age": 28.0,
        "NumOfProducts": 2,
        "IsActiveMember": 1,
        "Geography": "France",
    }
    val = {"value_at_risk": 80.0}

    res = advisor.recommend_strategy(customer, churn_prob=0.08, value_assessment=val)

    assert "Routine" in res["urgency"]
    assert res["sla_hours"] == 720


def test_recommend_strategy_product_deepening(advisor: RetentionAdvisor) -> None:
    """Test single product customer triggers cross-sell stickiness playbook."""
    customer = {
        "Balance": 20000.0,
        "Age": 35.0,
        "NumOfProducts": 1,
        "IsActiveMember": 1,
        "Geography": "Spain",
    }
    val = {"value_at_risk": 600.0}

    res = advisor.recommend_strategy(customer, churn_prob=0.38, value_assessment=val)

    assert any("Product Deepening" in p for p in res["all_triggered_playbooks"])
    assert res["estimated_intervention_cost"] == 80.0
    assert res["expected_retention_success_prob"] == 0.42
