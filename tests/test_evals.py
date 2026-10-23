"""Unit tests for the evaluation benchmark harness."""

from pathlib import Path
import numpy as np
import pytest

from evals.runner import (
    calculate_ece,
    evaluate_classification,
    evaluate_financial_impact,
    benchmark_latency,
    generate_markdown_report,
    run_eval_suite,
)


def test_calculate_ece():
    """Verify Expected Calibration Error calculation."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.85, 0.15, 0.75, 0.25, 0.1, 0.05])
    ece = calculate_ece(y_true, y_prob, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_evaluate_classification():
    """Verify classification metrics evaluation dictionary structure."""
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4])

    res = evaluate_classification(y_true, y_prob, threshold=0.50)
    assert res["roc_auc"] >= 0.80
    assert res["pr_auc"] >= 0.70
    assert res["accuracy"] >= 0.80
    assert res["f1"] >= 0.80
    assert 0.0 <= res["brier_score"] <= 1.0
    assert "tp" in res
    assert "fp" in res


def test_evaluate_financial_impact():
    """Verify financial threshold tuning and balance calculations."""
    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.85, 0.15, 0.75, 0.25, 0.65, 0.35])
    balances = np.array([50000.0, 20000.0, 80000.0, 10000.0, 40000.0, 30000.0])

    res = evaluate_financial_impact(y_true, y_prob, balances)
    assert 0.05 <= res["optimal_threshold"] <= 0.95
    assert res["total_churn_balance"] == 170000.0
    assert res["caught_balance_at_optimal"] >= 0.0
    assert "cost_at_optimal" in res


def test_benchmark_latency():
    """Verify latency benchmark returns expected metrics."""
    res = benchmark_latency(n_iterations=5)
    assert res["mean_latency_ms"] > 0.0
    assert res["median_latency_ms"] > 0.0
    assert res["p95_latency_ms"] >= res["median_latency_ms"]
    assert res["throughput_req_per_sec"] > 0.0


def test_generate_markdown_report(tmp_path):
    """Verify markdown report generation correctly writes sections."""
    sample_results = {
        "classification_metrics": {
            "roc_auc": 0.9745,
            "pr_auc": 0.9473,
            "accuracy": 0.9500,
            "precision": 0.9200,
            "recall": 0.8780,
            "f1": 0.9114,
            "brier_score": 0.0337,
            "ece": 0.0377,
            "tp": 1789,
            "fp": 100,
            "fn": 248,
            "tn": 7863,
        },
        "financial_impact": {
            "cost_fn": 2000.0,
            "cost_fp": 100.0,
            "p_save": 0.30,
            "optimal_threshold": 0.275,
            "cost_at_optimal": 1000000.0,
            "cost_at_default": 1033700.0,
            "net_financial_gain": 33700.0,
            "recall_at_optimal": 0.929,
            "recall_at_default": 0.878,
            "recall_gain": 0.051,
            "total_churn_balance": 180000000.0,
            "caught_balance_at_optimal": 172970761.59,
            "caught_balance_at_default": 165125172.47,
            "balance_protection_gain": 7845589.12,
        },
        "latency_benchmark": {
            "mean_latency_ms": 9.63,
            "median_latency_ms": 9.40,
            "p95_latency_ms": 13.05,
            "p99_latency_ms": 15.00,
            "throughput_req_per_sec": 104.0,
        },
    }

    out_file = tmp_path / "test_report.md"
    generate_markdown_report(sample_results, out_file)
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert "Executive Performance Matrix" in content
    assert "Cost-Sensitive Financial Threshold Optimization" in content
    assert "0.9745" in content


def test_run_eval_suite(tmp_path):
    """Verify run_eval_suite executes end-to-end and outputs file."""
    rep_file = tmp_path / "custom_eval.md"
    res = run_eval_suite(report_path=rep_file)
    assert rep_file.exists()
    assert res["num_samples"] == 10000
    assert "classification_metrics" in res
    assert "financial_impact" in res
    assert "latency_benchmark" in res


def test_run_eval_suite_missing_data():
    """Verify FileNotFoundError is raised on missing data."""
    with pytest.raises(FileNotFoundError):
        run_eval_suite(data_path="missing_data.csv")
