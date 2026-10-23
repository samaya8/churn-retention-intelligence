"""Cost-sensitive decision threshold optimization engine for customer churn."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_COST_FN = 2000.0   # Average financial loss of a lost customer (CLV)
DEFAULT_COST_FP = 100.0    # Cost of retention outreach/incentive on a non-churner
DEFAULT_P_SAVE = 0.30      # Probability of successfully retaining a targeted churner
DEFAULT_OPTIMAL_THRESHOLD = 0.32


class ThresholdOptimizer:
    """Optimizes classification decision thresholds based on asymmetric business costs."""

    def __init__(
        self,
        cost_fn: float = DEFAULT_COST_FN,
        cost_fp: float = DEFAULT_COST_FP,
        p_save: float = DEFAULT_P_SAVE,
    ):
        self.cost_fn = max(1.0, float(cost_fn))
        self.cost_fp = max(0.0, float(cost_fp))
        self.p_save = max(0.01, min(0.99, float(p_save)))
        # Unrecovered loss for a caught churner: (1 - p_save) * cost_fn + cost_fp
        self.cost_tp_unrecovered = ((1.0 - self.p_save) * self.cost_fn) + self.cost_fp

    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float,
    ) -> Dict[str, Any]:
        """Compute confusion matrix and financial metrics for a specific decision threshold."""
        y_true_arr = np.asarray(y_true).astype(int)
        y_prob_arr = np.asarray(y_prob).astype(float)

        y_pred = (y_prob_arr >= threshold).astype(int)

        tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Financial cost calculation:
        # FN: lost completely (cost_fn)
        # FP: wasted retention cost (cost_fp)
        # TP: retention campaign delivered; (1 - p_save) lost, p_save saved
        total_cost = (fn * self.cost_fn) + (fp * self.cost_fp) + (tp * self.cost_tp_unrecovered)

        # Baseline cost with zero interventions (everyone predicted as 0):
        # All churners are lost as FNs
        baseline_do_nothing_cost = float(np.sum(y_true_arr == 1) * self.cost_fn)

        # Net financial savings vs doing nothing
        net_savings = max(0.0, baseline_do_nothing_cost - total_cost)

        # Net value saved from targeted interventions:
        # Saved CLV minus total intervention cost (TP + FP) * cost_fp
        saved_clv = tp * self.p_save * self.cost_fn
        campaign_expense = (tp + fp) * self.cost_fp
        net_retention_profit = saved_clv - campaign_expense

        return {
            "threshold": round(float(threshold), 3),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "total_cost": round(total_cost, 2),
            "net_savings_vs_inaction": round(net_savings, 2),
            "net_retention_profit": round(net_retention_profit, 2),
            "targeted_customers": tp + fp,
            "targeted_pct": round(float((tp + fp) / max(1, len(y_true_arr)) * 100.0), 1),
        }

    def compute_cost_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        num_points: int = 91,
    ) -> List[Dict[str, Any]]:
        """Compute the cost and savings curve across threshold values from 0.05 to 0.95."""
        thresholds = np.linspace(0.05, 0.95, num_points)
        curve = []
        for t in thresholds:
            res = self.evaluate_threshold(y_true, y_prob, float(t))
            curve.append(res)
        return curve

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> Dict[str, Any]:
        """Find the optimal financial decision threshold that minimizes total expected cost."""
        curve = self.compute_cost_curve(y_true, y_prob, num_points=181)
        best = min(curve, key=lambda x: x["total_cost"])

        default_50 = self.evaluate_threshold(y_true, y_prob, 0.50)
        cost_reduction = default_50["total_cost"] - best["total_cost"]
        savings_gain = best["net_savings_vs_inaction"] - default_50["net_savings_vs_inaction"]

        return {
            "optimal_threshold": best["threshold"],
            "optimal_metrics": best,
            "default_50_metrics": default_50,
            "financial_gain_over_default": round(cost_reduction, 2),
            "recall_gain": round(best["recall"] - default_50["recall"], 4),
            "cost_curve": curve,
        }
