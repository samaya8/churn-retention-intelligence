"""Unit tests for Customer Lifetime Value (CLV) and financial revenue modeling."""

import pytest

from src.clv_engine import CLVEngine


@pytest.fixture(scope="module")
def clv_engine() -> CLVEngine:
    return CLVEngine(discount_rate=0.08, nim_rate=0.025)


def test_calculate_annual_margin_balance_scaling(clv_engine: CLVEngine) -> None:
    """Test annual margin increases monotonically with higher deposit balance."""
    m_low = clv_engine.calculate_annual_margin(balance=10000.0, estimated_salary=50000.0)
    m_high = clv_engine.calculate_annual_margin(balance=100000.0, estimated_salary=50000.0)
    assert m_high > m_low
    assert m_low >= 150.0  # Floor respected


def test_calculate_annual_margin_active_bonus(clv_engine: CLVEngine) -> None:
    """Test active members generate higher annual margin than inactive members."""
    m_active = clv_engine.calculate_annual_margin(
        balance=50000.0, estimated_salary=60000.0, is_active_member=1
    )
    m_inactive = clv_engine.calculate_annual_margin(
        balance=50000.0, estimated_salary=60000.0, is_active_member=0
    )
    assert m_active > m_inactive


def test_calculate_forward_clv(clv_engine: CLVEngine) -> None:
    """Test forward CLV decreases as churn probability increases."""
    margin = 2000.0
    clv_loyal = clv_engine.calculate_forward_clv(annual_margin=margin, churn_prob=0.05, years=3)
    clv_churner = clv_engine.calculate_forward_clv(annual_margin=margin, churn_prob=0.80, years=3)

    assert clv_loyal > clv_churner
    assert clv_loyal > 0.0
    assert clv_churner >= 0.0


def test_assess_customer_value_roi(clv_engine: CLVEngine) -> None:
    """Test full assessment dictionary and retention ROI calculations."""
    res = clv_engine.assess_customer_value(
        balance=80000.0,
        estimated_salary=90000.0,
        churn_prob=0.60,
        num_of_products=2,
        is_active_member=1,
        intervention_cost=100.0,
        p_save=0.35,
    )

    assert "clv_3yr" in res
    assert "clv_5yr" in res
    assert res["clv_5yr"] >= res["clv_3yr"]
    assert res["value_at_risk"] > 0.0
    assert "value_tier" in res
    assert res["expected_saved_value"] == pytest.approx(res["value_at_risk"] * 0.35, 1e-2)
    assert res["net_retention_roi"] == pytest.approx(res["expected_saved_value"] - 100.0, 1e-2)
