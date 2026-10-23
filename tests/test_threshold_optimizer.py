"""Unit tests for cost-sensitive threshold optimization."""

import numpy as np
import pytest

from src.threshold_optimizer import ThresholdOptimizer


@pytest.fixture(scope="module")
def optimizer() -> ThresholdOptimizer:
    return ThresholdOptimizer(cost_fn=2000.0, cost_fp=100.0, p_save=0.30)


def test_evaluate_threshold_extremes(optimizer: ThresholdOptimizer) -> None:
    """Test metrics at extreme threshold boundaries (0.0 and 1.0)."""
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.7])

    # Low threshold: flags all as churn (Recall = 1.0)
    low_res = optimizer.evaluate_threshold(y_true, y_prob, threshold=0.05)
    assert low_res["recall"] == 1.0
    assert low_res["tp"] == 3
    assert low_res["fn"] == 0

    # High threshold: flags none as churn (Recall = 0.0)
    high_res = optimizer.evaluate_threshold(y_true, y_prob, threshold=0.95)
    assert high_res["recall"] == 0.0
    assert high_res["tp"] == 0
    assert high_res["fn"] == 3


def test_compute_cost_curve_shape(optimizer: ThresholdOptimizer) -> None:
    """Test cost curve evaluates all points properly."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
    y_prob = np.array([0.85, 0.15, 0.75, 0.25, 0.65, 0.35, 0.90, 0.55, 0.10, 0.40])

    curve = optimizer.compute_cost_curve(y_true, y_prob, num_points=10)
    assert len(curve) == 10
    for pt in curve:
        assert "threshold" in pt
        assert "total_cost" in pt
        assert "net_savings_vs_inaction" in pt
        assert pt["total_cost"] >= 0.0


def test_find_optimal_threshold(optimizer: ThresholdOptimizer) -> None:
    """Test optimal threshold finder returns lower cost than default 0.50."""
    # Synthetic dataset with realistic churners
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.20, size=200)
    # Calibrated probabilities with noise
    y_prob = np.clip(y_true * 0.5 + np.random.uniform(0.1, 0.4, size=200), 0.0, 1.0)

    opt = optimizer.find_optimal_threshold(y_true, y_prob)

    assert 0.05 <= opt["optimal_threshold"] <= 0.95
    assert opt["optimal_metrics"]["total_cost"] <= opt["default_50_metrics"]["total_cost"]
    assert "cost_curve" in opt
    assert opt["financial_gain_over_default"] >= 0.0
