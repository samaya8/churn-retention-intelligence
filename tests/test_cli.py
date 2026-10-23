"""Unit tests for the Bank Customer Churn production CLI."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure correct CLI module resolution
PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
sys.modules.pop("cli", None)

from cli import main, build_parser


def test_cli_help(capsys: pytest.CaptureFixture) -> None:
    """Test CLI help output."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "churn-cli" in captured.out or "usage:" in captured.out.lower()


def test_cli_empty_args(capsys: pytest.CaptureFixture) -> None:
    """Test running CLI without arguments prints help and exits 0."""
    code = main([])
    assert code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_cli_predict_text(capsys: pytest.CaptureFixture) -> None:
    """Test predict subcommand in plain text mode."""
    code = main(["predict", "--age", "45", "--balance", "80000", "--geography", "Germany"])
    assert code == 0
    captured = capsys.readouterr()
    assert "BANK CUSTOMER CHURN RISK ASSESSMENT" in captured.out
    assert "Churn Probability" in captured.out
    assert "3-Year Forward CLV" in captured.out


def test_cli_predict_json(capsys: pytest.CaptureFixture) -> None:
    """Test predict subcommand in JSON mode."""
    code = main(["predict", "--age", "45", "--balance", "80000", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "churn_probability" in data
    assert "financials" in data
    assert "clv_3yr" in data["financials"]


def test_cli_explain_text(capsys: pytest.CaptureFixture) -> None:
    """Test explain subcommand in plain text mode."""
    code = main(["explain", "--age", "50", "--geography", "Germany", "--is-active", "0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "TREESHAP LOCAL FEATURE ATTRIBUTION" in captured.out
    assert "TOP RISK ACCELERATORS" in captured.out


def test_cli_explain_json(capsys: pytest.CaptureFixture) -> None:
    """Test explain subcommand in JSON mode."""
    code = main(["explain", "--age", "50", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "base_log_odds" in data
    assert "waterfall_steps" in data
    assert "risk_accelerators" in data


def test_cli_retention_text(capsys: pytest.CaptureFixture) -> None:
    """Test retention subcommand in plain text mode."""
    code = main(["retention", "--age", "48", "--balance", "70000", "--is-active", "0"])
    assert code == 0
    captured = capsys.readouterr()
    assert "TACTICAL CUSTOMER RETENTION PLAYBOOK" in captured.out
    assert "Primary Playbook" in captured.out
    assert "Estimated Campaign Cost" in captured.out


def test_cli_retention_json(capsys: pytest.CaptureFixture) -> None:
    """Test retention subcommand in JSON mode."""
    code = main(["retention", "--age", "48", "--balance", "70000", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "primary_strategy" in data
    assert "estimated_intervention_cost" in data
    assert "net_expected_profit" in data


def test_cli_batch(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Test batch subcommand with limit and file output."""
    out_file = tmp_path / "batch_out.json"
    code = main(["batch", "--limit", "5", "--output", str(out_file), "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["num_customers"] == 5
    assert "total_balance_at_risk" in data
    assert out_file.exists()


def test_cli_batch_missing_file(capsys: pytest.CaptureFixture) -> None:
    """Test batch subcommand error on missing input file."""
    code = main(["batch", "--input", "nonexistent_file.csv"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: Input file not found" in captured.err


def test_cli_threshold(capsys: pytest.CaptureFixture) -> None:
    """Test threshold optimization subcommand in JSON mode."""
    code = main(["threshold", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "optimal_threshold" in data
    assert "financial_gain_over_default" in data


def test_cli_benchmark(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Test benchmark subcommand with mocked evals runner."""
    mock_run = MagicMock(return_value={
        "num_samples": 10000,
        "classification_metrics": {"roc_auc": 0.86, "pr_auc": 0.71, "f1": 0.60, "brier_score": 0.11},
        "financial_impact": {"optimal_threshold": 0.32, "recall_at_optimal": 0.70, "recall_at_default": 0.52, "net_financial_gain": 45000.0},
        "latency_benchmark": {"mean_latency_ms": 1.2, "throughput_req_per_sec": 830},
    })
    with patch("evals.runner.run_eval_suite", mock_run):
        code = main(["benchmark", "--json"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["num_samples"] == 10000
