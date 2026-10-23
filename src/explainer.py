"""TreeSHAP local feature attribution and waterfall explainability engine for XGBoost."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb


HUMAN_INSIGHTS: Dict[str, Dict[str, str]] = {
    "Age": {
        "pos": "Customer age is elevated, indicating life-stage transition risk and higher retirement/competitor vulnerability.",
        "neg": "Younger customer demographic correlates with long-term retention stability.",
    },
    "IsActiveMember": {
        "pos": "Customer is inactive; lack of monthly engagement is a critical leading indicator of attrition.",
        "neg": "Active membership status serves as a strong behavioral anchor against churn.",
    },
    "NumOfProducts": {
        "pos": "Product portfolio configuration (single product or product saturation) correlates with flight risk.",
        "neg": "Holding optimal multi-product accounts creates substantial relationship stickiness.",
    },
    "Geography_Germany": {
        "pos": "German account holders exhibit statistically elevated market churn propensity due to regional competitive pressure.",
        "neg": "Non-German account placement aligns with baseline regional retention rates.",
    },
    "Balance": {
        "pos": "Substantial deposit balance without active product utilization creates high-balance flight exposure.",
        "neg": "Well-utilized balance anchors customer relationship within the institution.",
    },
    "CreditScore": {
        "pos": "Sub-prime or constrained credit score increases price sensitivity to bank fees.",
        "neg": "Strong credit profile indicates established relationship loyalty and credit card utilization.",
    },
    "Gender_Female": {
        "pos": "Female demographic segment shows higher price/service sensitivity in historical portfolio data.",
        "neg": "Male demographic segment shows lower sensitivity in baseline portfolio.",
    },
    "Tenure": {
        "pos": "Early-tenure customer with unestablished institutional relationship.",
        "neg": "Long tenured customer showing multi-year institutional inertia.",
    },
    "EstimatedSalary": {
        "pos": "High salary customer prone to competitive private wealth solicitation.",
        "neg": "Stable salary deposit anchoring primary checking account.",
    },
    "HasCrCard": {
        "pos": "Lacks an active credit card line, weakening day-to-day payment stickiness.",
        "neg": "Active credit card usage integrates bank into daily payment routines.",
    },
}


class ChurnExplainer:
    """Computes exact TreeSHAP feature attributions and waterfall breakdowns using XGBoost C++ kernel."""

    def __init__(self, model: Optional[Any] = None, feature_columns: Optional[List[str]] = None):
        self.model = model
        self.feature_columns = feature_columns

    def explain(
        self,
        encoded_df: pd.DataFrame,
        raw_customer: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute TreeSHAP contributions and rank features into risk accelerators and loyalty anchors."""
        if self.model is None:
            return {"status": "error", "message": "Model not loaded"}

        booster = self.model.get_booster()
        feature_names = self.feature_columns or list(encoded_df.columns)

        X_arr = np.asarray(encoded_df, dtype=np.float32)
        dmatrix = xgb.DMatrix(X_arr, feature_names=feature_names)
        # XGBoost pred_contribs returns [phi_1, phi_2, ..., phi_p, bias]
        contribs = booster.predict(dmatrix, pred_contribs=True)[0]

        feature_contribs = contribs[:-1]
        base_bias = float(contribs[-1])
        margin = float(np.sum(contribs))
        prob_from_margin = float(1.0 / (1.0 + np.exp(-margin)))

        drivers: List[Dict[str, Any]] = []
        for name, val in zip(feature_names, feature_contribs):
            val_flt = float(val)
            direction = "Risk Accelerator (+)" if val_flt > 0 else "Loyalty Anchor (-)"
            abs_mag = abs(val_flt)

            insights = HUMAN_INSIGHTS.get(name, {})
            comment = insights.get("pos" if val_flt > 0 else "neg", f"Contribution score: {val_flt:+.3f}")

            drivers.append({
                "feature": name,
                "shap_value": round(val_flt, 4),
                "abs_importance": round(abs_mag, 4),
                "direction": direction,
                "is_risk_accelerator": val_flt > 0,
                "comment": comment,
            })

        # Sort by absolute impact
        drivers_sorted = sorted(drivers, key=lambda d: d["abs_importance"], reverse=True)

        risk_accelerators = [d for d in drivers_sorted if d["is_risk_accelerator"]]
        loyalty_anchors = [d for d in drivers_sorted if not d["is_risk_accelerator"]]

        # Waterfall steps preparation
        waterfall_steps = [
            {"step": "Baseline Expected Value", "value": round(base_bias, 4), "type": "baseline"}
        ]
        running_margin = base_bias
        for d in drivers_sorted[:7]:  # Top 7 drivers
            running_margin += d["shap_value"]
            waterfall_steps.append({
                "step": d["feature"],
                "value": d["shap_value"],
                "running_margin": round(running_margin, 4),
                "type": "positive" if d["is_risk_accelerator"] else "negative",
            })

        waterfall_steps.append({
            "step": "Final Log-Odds Margin",
            "value": round(margin, 4),
            "type": "total",
        })

        return {
            "base_log_odds": round(base_bias, 4),
            "final_log_odds": round(margin, 4),
            "calibrated_probability": round(prob_from_margin, 4),
            "top_drivers": drivers_sorted,
            "risk_accelerators": risk_accelerators[:5],
            "loyalty_anchors": loyalty_anchors[:5],
            "waterfall_steps": waterfall_steps,
        }
