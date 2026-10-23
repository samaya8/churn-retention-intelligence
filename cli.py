"""Production Command-Line Interface for Bank Customer Churn Intelligence."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.churn_engine import ChurnEngine
from src.threshold_optimizer import ThresholdOptimizer, DEFAULT_OPTIMAL_THRESHOLD


def build_parser() -> argparse.ArgumentParser:
    """Construct command-line argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="churn-cli",
        description="Bank Customer Churn Intelligence — Enterprise CLI",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Helper to add standard customer profile arguments
    def add_customer_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--credit-score", type=float, default=650.0, help="Customer credit score (300 - 850)")
        p.add_argument("--geography", type=str, default="France", choices=["France", "Germany", "Spain"], help="Country")
        p.add_argument("--gender", type=str, default="Female", choices=["Female", "Male"], help="Customer gender")
        p.add_argument("--age", type=float, default=40.0, help="Customer age")
        p.add_argument("--tenure", type=float, default=5.0, help="Tenure with the bank in years")
        p.add_argument("--balance", type=float, default=50000.0, help="Account deposit balance ($)")
        p.add_argument("--products", type=int, default=1, choices=[1, 2, 3, 4], help="Number of bank products held")
        p.add_argument("--has-card", type=int, default=1, choices=[0, 1], help="Has credit card (1=Yes, 0=No)")
        p.add_argument("--is-active", type=int, default=1, choices=[0, 1], help="Is active member (1=Yes, 0=No)")
        p.add_argument("--salary", type=float, default=60000.0, help="Estimated annual salary ($)")
        p.add_argument("--threshold", type=float, default=DEFAULT_OPTIMAL_THRESHOLD, help="Custom decision threshold")
        p.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 1. PREDICT
    p_predict = subparsers.add_parser("predict", help="Predict customer churn probability and risk tier")
    add_customer_args(p_predict)

    # 2. EXPLAIN
    p_explain = subparsers.add_parser("explain", help="Compute TreeSHAP local waterfall feature attributions")
    add_customer_args(p_explain)

    # 3. RETENTION
    p_retention = subparsers.add_parser("retention", help="Generate tactical retention playbook and calculate ROI")
    add_customer_args(p_retention)

    # 4. BATCH
    p_batch = subparsers.add_parser("batch", help="Batch assess customer records from CSV and build triage leaderboard")
    p_batch.add_argument("--input", type=str, default=str(PROJECT_DIR / "Churn_Modelling.csv"), help="Input CSV dataset")
    p_batch.add_argument("--limit", type=int, default=None, help="Optional max records to process")
    p_batch.add_argument("--threshold", type=float, default=DEFAULT_OPTIMAL_THRESHOLD, help="Decision threshold")
    p_batch.add_argument("--output", type=str, default=None, help="Optional output path for evaluated CSV or JSON")
    p_batch.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 5. THRESHOLD
    p_thresh = subparsers.add_parser("threshold", help="Optimize decision threshold using asymmetric business cost matrix")
    p_thresh.add_argument("--data", type=str, default=str(PROJECT_DIR / "Churn_Modelling.csv"), help="Data CSV for threshold search")
    p_thresh.add_argument("--cost-fn", type=float, default=2000.0, help="Cost of False Negative (lost customer CLV)")
    p_thresh.add_argument("--cost-fp", type=float, default=100.0, help="Cost of False Positive (wasted campaign)")
    p_thresh.add_argument("--p-save", type=float, default=0.30, help="Retention campaign success probability")
    p_thresh.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 6. BENCHMARK
    p_bench = subparsers.add_parser("benchmark", help="Run automated model evaluation benchmark and generate report")
    p_bench.add_argument("--data", type=str, default=str(PROJECT_DIR / "Churn_Modelling.csv"), help="Path to evaluation CSV")
    p_bench.add_argument("--report", type=str, default=str(PROJECT_DIR / "evals" / "BENCHMARK_RESULTS.md"), help="Report path")
    p_bench.add_argument("--json", action="store_true", help="Output results in JSON format")

    return parser


def parse_customer_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Extract standard customer dictionary from parsed CLI arguments."""
    return {
        "CreditScore": args.credit_score,
        "Geography": args.geography,
        "Gender": args.gender,
        "Age": args.age,
        "Tenure": args.tenure,
        "Balance": args.balance,
        "NumOfProducts": args.products,
        "HasCrCard": args.has_card,
        "IsActiveMember": args.is_active,
        "EstimatedSalary": args.salary,
    }


def handle_predict(args: argparse.Namespace) -> int:
    """Execute predict command."""
    engine = ChurnEngine(decision_threshold=args.threshold)
    customer = parse_customer_from_args(args)
    res = engine.assess_customer(customer, threshold=args.threshold)

    if args.json:
        payload = {
            "churn_prediction": res["churn_prediction"],
            "is_churn": res["is_churn"],
            "churn_probability": res["churn_probability"],
            "risk_tier": res["risk_tier"],
            "decision_threshold": res["decision_threshold"],
            "financials": res["financials"],
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 62)
    print("  BANK CUSTOMER CHURN RISK ASSESSMENT")
    print("=" * 62)
    print(f"Customer Profile:      {res['customer_profile']['Age']:.0f}yo {res['customer_profile']['Gender']} | {res['customer_profile']['Geography']}")
    print(f"Financial Position:    ${res['customer_profile']['Balance']:,.0f} Balance | ${res['customer_profile']['EstimatedSalary']:,.0f} Salary")
    print(f"Products / Activity:   {res['customer_profile']['NumOfProducts']} product(s) | Active: {'Yes' if res['customer_profile']['IsActiveMember'] else 'No'}")
    print("-" * 62)
    print(f"Churn Probability:     {res['churn_probability'] * 100:.1f}%")
    print(f"Decision Threshold:    {res['decision_threshold']:.3f}")
    status_label = "[AT-RISK] CHURN" if res["is_churn"] else "[STABLE] LOYAL"
    print(f"Classification:        {status_label}")
    print(f"Risk Tier:             {res['risk_tier']}")
    print("-" * 62)
    print(f"Annual Bank Margin:    ${res['financials']['annual_customer_margin']:,.2f}/year")
    print(f"3-Year Forward CLV:    ${res['financials']['clv_3yr']:,.2f}")
    print(f"Customer Value at Risk:${res['financials']['value_at_risk']:,.2f}")
    print(f"Customer Value Tier:   {res['financials']['value_tier']}")
    print("=" * 62 + "\n")
    return 0


def handle_explain(args: argparse.Namespace) -> int:
    """Execute explain command."""
    engine = ChurnEngine(decision_threshold=args.threshold)
    customer = parse_customer_from_args(args)
    res = engine.assess_customer(customer, threshold=args.threshold)
    exp = res["explanation"]

    if args.json:
        print(json.dumps(exp, indent=2))
        return 0

    print("\n" + "=" * 68)
    print("  TREESHAP LOCAL FEATURE ATTRIBUTION WATERFALL")
    print("=" * 68)
    print(f"Baseline Margin:        {exp['base_log_odds']:+.4f} log-odds")
    print(f"Final Model Margin:     {exp['final_log_odds']:+.4f} log-odds")
    print(f"Calibrated Churn Prob:  {exp['calibrated_probability'] * 100:.1f}%")
    print("-" * 68)
    print("TOP RISK ACCELERATORS (Pushing towards churn):")
    for d in exp["risk_accelerators"][:4]:
        print(f"  [+] {d['feature']:<20} {d['shap_value']:>+7.3f} log-odds | {d['comment']}")

    print("\nTOP LOYALTY ANCHORS (Holding customer with bank):")
    for d in exp["loyalty_anchors"][:4]:
        print(f"  [-] {d['feature']:<20} {d['shap_value']:>+7.3f} log-odds | {d['comment']}")
    print("=" * 68 + "\n")
    return 0


def handle_retention(args: argparse.Namespace) -> int:
    """Execute retention command."""
    engine = ChurnEngine(decision_threshold=args.threshold)
    customer = parse_customer_from_args(args)
    res = engine.assess_customer(customer, threshold=args.threshold)
    ret = res["retention_strategy"]

    if args.json:
        print(json.dumps(ret, indent=2))
        return 0

    print("\n" + "=" * 68)
    print("  TACTICAL CUSTOMER RETENTION PLAYBOOK")
    print("=" * 68)
    print(f"Action Urgency Tier:     {ret['urgency']}")
    print(f"Response SLA Target:     Within {ret['sla_hours']} hours")
    print(f"Primary Playbook:        {ret['primary_strategy']}")
    print(f"Business Rationale:      {ret['rationale']}")
    print("-" * 68)
    print("Recommended Action Items:")
    for idx, tactic in enumerate(ret["action_items"], 1):
        print(f"  {idx}. {tactic}")
    print("-" * 68)
    print(f"Estimated Campaign Cost: ${ret['estimated_intervention_cost']:,.2f}")
    print(f"Expected Success Rate:   {ret['expected_retention_success_prob'] * 100:.1f}%")
    print(f"Expected Saved CLV:      ${ret['expected_saved_clv']:,.2f}")
    print(f"Net Campaign ROI:        ${ret['net_expected_profit']:,.2f} ({ret['roi_ratio_pct']:+.1f}% ROI)")
    print("=" * 68 + "\n")
    return 0


def handle_batch(args: argparse.Namespace) -> int:
    """Execute batch command."""
    in_path = Path(args.input)
    if not in_path.exists():
        sys.stderr.write(f"Error: Input file not found at {in_path}\n")
        return 1

    df = pd.read_csv(in_path)
    if args.limit:
        df = df.head(int(args.limit)).copy()

    engine = ChurnEngine(decision_threshold=args.threshold)
    res = engine.assess_batch(df, threshold=args.threshold)

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if str(out_p).endswith(".json"):
            payload = {
                "num_customers": res["num_customers"],
                "churners_flagged": res["churners_flagged"],
                "churn_rate_pct": res["churn_rate_pct"],
                "total_balance_at_risk": res["total_balance_at_risk"],
                "total_clv_at_risk": res["total_clv_at_risk"],
                "leaderboard": res["leaderboard"].to_dict(orient="records"),
            }
            out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            res["leaderboard"].to_csv(out_p, index=False)

    if args.json:
        payload = {
            "num_customers": res["num_customers"],
            "churners_flagged": res["churners_flagged"],
            "churn_rate_pct": res["churn_rate_pct"],
            "decision_threshold": res["decision_threshold"],
            "total_balance_at_risk": res["total_balance_at_risk"],
            "total_clv_at_risk": res["total_clv_at_risk"],
            "top_at_risk_customers": res["leaderboard"].head(5).to_dict(orient="records"),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n" + "=" * 66)
    print("  BATCH PORTFOLIO CHURN APPRAISAL")
    print("=" * 66)
    print(f"Total Accounts Evaluated:  {res['num_customers']:,}")
    print(f"Decision Threshold:        {res['decision_threshold']:.3f}")
    print(f"Flagged Churn Flight Risk: {res['churners_flagged']:,} ({res['churn_rate_pct']:.1f}%)")
    print(f"Total Balance Exposure:    ${res['total_balance_at_risk']:,.2f}")
    print(f"Total 3-Year CLV At Risk:  ${res['total_clv_at_risk']:,.2f}")
    print("-" * 66)
    print("TOP 5 HIGH-RISK CUSTOMER FLIGHT CANDIDATES:")
    top5 = res["leaderboard"].head(5)
    for _, row in top5.iterrows():
        print(f"  - ID {row['CustomerId']} ({row['Surname']}): Prob={row['Churn_Probability']*100:.1f}% | Balance=${row['Balance']:,.0f} | CLV At Risk=${row['Value_At_Risk']:,.2f}")
    if args.output:
        print(f"\nSaved full triage results to: {args.output}")
    print("=" * 66 + "\n")
    return 0


def handle_threshold(args: argparse.Namespace) -> int:
    """Execute threshold optimization command."""
    data_path = Path(args.data)
    if not data_path.exists():
        sys.stderr.write(f"Error: Dataset not found at {data_path}\n")
        return 1

    df = pd.read_csv(data_path)
    engine = ChurnEngine()
    encoded = engine.preprocessor.transform_batch(df)
    X_arr = np.asarray(encoded, dtype=np.float32)
    probs = engine.model.predict_proba(X_arr)[:, 1]
    y_true = df["Exited"].values

    opt = ThresholdOptimizer(cost_fn=args.cost_fn, cost_fp=args.cost_fp, p_save=args.p_save)
    search_res = opt.find_optimal_threshold(y_true, probs)

    if args.json:
        # Don't serialize full cost_curve in compact JSON output
        out = dict(search_res)
        out.pop("cost_curve", None)
        print(json.dumps(out, indent=2))
        return 0

    best = search_res["optimal_metrics"]
    def50 = search_res["default_50_metrics"]

    print("\n" + "=" * 68)
    print("  COST-SENSITIVE DECISION THRESHOLD OPTIMIZATION")
    print("=" * 68)
    print(f"Parameters: Cost_FN=${args.cost_fn:,.0f} | Cost_FP=${args.cost_fp:,.0f} | P(Save)={args.p_save*100:.0f}%")
    print("-" * 68)
    print(f"Optimal Decision Threshold:  tau* = {search_res['optimal_threshold']:.3f} (vs Default 0.500)")
    print(f"Recall Gain:                 {search_res['recall_gain']*100:+.1f}% (from {def50['recall']*100:.1f}% to {best['recall']*100:.1f}%)")
    print(f"Targeted Customer Cohort:    {best['targeted_customers']:,} ({best['targeted_pct']:.1f}% of base)")
    print(f"Expected Net Cost at tau*:   ${best['total_cost']:,.2f}")
    print(f"Expected Net Cost at 0.50:   ${def50['total_cost']:,.2f}")
    print(f"Financial Gain over Default: ${search_res['financial_gain_over_default']:,.2f} Net Savings")
    print("=" * 68 + "\n")
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute evaluation benchmark command."""
    from evals.runner import run_eval_suite
    rep_path = Path(args.report) if args.report else None
    data_path = Path(args.data) if args.data else None

    res = run_eval_suite(report_path=rep_path, data_path=data_path)

    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    cls_m = res["classification_metrics"]
    fin_m = res["financial_impact"]
    lat_m = res["latency_benchmark"]

    print("\n" + "=" * 66)
    print("  MODEL BENCHMARK & CALIBRATION SUMMARY")
    print("=" * 66)
    print(f"Evaluated Records:      {res['num_samples']:,}")
    print(f"ROC-AUC Score:          {cls_m['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Precision): {cls_m['pr_auc']:.4f}")
    print(f"F1-Score:               {cls_m['f1']:.4f}")
    print(f"Brier Calibration:      {cls_m['brier_score']:.4f}")
    print("-" * 66)
    print(f"Optimal Threshold (tau*): {fin_m['optimal_threshold']:.3f}")
    print(f"Recall at tau*:           {fin_m['recall_at_optimal']*100:.1f}% (vs {fin_m['recall_at_default']*100:.1f}% at 0.50)")
    print(f"Net Financial Savings:    ${fin_m['net_financial_gain']:,.2f}")
    print("-" * 66)
    print(f"Mean CPU Latency:       {lat_m['mean_latency_ms']:.2f} ms")
    print(f"Single-Core Throughput: {lat_m['throughput_req_per_sec']:,.0f} evaluations/sec")
    print("=" * 66 + "\n")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if args.subcommand == "predict":
        return handle_predict(args)
    elif args.subcommand == "explain":
        return handle_explain(args)
    elif args.subcommand == "retention":
        return handle_retention(args)
    elif args.subcommand == "batch":
        return handle_batch(args)
    elif args.subcommand == "threshold":
        return handle_threshold(args)
    elif args.subcommand == "benchmark":
        return handle_benchmark(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
