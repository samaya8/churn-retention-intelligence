"""Unified Churn Intelligence Engine: Orchestrating inference, CLV, TreeSHAP, and retention advisory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from src.clv_engine import CLVEngine
from src.explainer import ChurnExplainer
from src.preprocessor import ChurnPreprocessor
from src.retention_advisor import RetentionAdvisor
from src.threshold_optimizer import DEFAULT_OPTIMAL_THRESHOLD

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_DIR / "xgb_churn_model.pkl"
DEFAULT_SCALER_PATH = PROJECT_DIR / "scaler.pkl"
DEFAULT_COLUMNS_PATH = PROJECT_DIR / "feature_columns.pkl"


class ChurnEngine:
    """Enterprise churn risk engine combining XGBoost inference, financial CLV, TreeSHAP, and retention."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
        feature_columns_path: Optional[Union[str, Path]] = None,
        decision_threshold: float = DEFAULT_OPTIMAL_THRESHOLD,
    ):
        m_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        s_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH
        f_path = Path(feature_columns_path) if feature_columns_path else DEFAULT_COLUMNS_PATH

        self.model = joblib.load(m_path) if m_path.exists() else None
        self.preprocessor = ChurnPreprocessor(scaler_path=s_path, feature_columns_path=f_path)
        self.clv_engine = CLVEngine()
        self.explainer = ChurnExplainer(
            model=self.model,
            feature_columns=self.preprocessor.feature_columns,
        )
        self.retention_advisor = RetentionAdvisor()
        self.decision_threshold = float(decision_threshold)

    def assess_customer(
        self,
        customer: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Perform end-to-end churn risk appraisal on an individual customer."""
        t = float(threshold if threshold is not None else self.decision_threshold)
        cleaned = self.preprocessor.clean_record(customer)
        encoded_df = self.preprocessor.transform_single(cleaned)

        if self.model is None:
            raise RuntimeError("XGBoost model is not loaded.")

        # Compute probability and decision
        X_arr = np.asarray(encoded_df, dtype=np.float32)
        probs = self.model.predict_proba(X_arr)[0]
        stay_prob = float(probs[0])
        churn_prob = float(probs[1])
        is_churn = bool(churn_prob >= t)

        # Classify risk tier
        if churn_prob >= 0.70:
            risk_tier = "Critical Flight Risk"
        elif churn_prob >= 0.45:
            risk_tier = "High Churn Propensity"
        elif churn_prob >= 0.25:
            risk_tier = "Moderate Churn Risk"
        else:
            risk_tier = "Loyal Account"

        # Financial Customer Value Modeling
        financials = self.clv_engine.assess_customer_value(
            balance=cleaned["Balance"],
            estimated_salary=cleaned["EstimatedSalary"],
            churn_prob=churn_prob,
            num_of_products=cleaned["NumOfProducts"],
            is_active_member=cleaned["IsActiveMember"],
            has_cr_card=cleaned["HasCrCard"],
        )

        # TreeSHAP Explainability
        explanation = self.explainer.explain(encoded_df=encoded_df, raw_customer=cleaned)

        # Dynamic Tactical Retention Playbook
        retention = self.retention_advisor.recommend_strategy(
            customer=cleaned,
            churn_prob=churn_prob,
            value_assessment=financials,
            shap_explanation=explanation,
        )

        return {
            "customer_profile": cleaned,
            "churn_prediction": 1 if is_churn else 0,
            "is_churn": is_churn,
            "churn_probability": round(churn_prob, 4),
            "stay_probability": round(stay_prob, 4),
            "decision_threshold": round(t, 3),
            "risk_tier": risk_tier,
            "financials": financials,
            "explanation": explanation,
            "retention_strategy": retention,
        }

    def assess_batch(
        self,
        df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Batch evaluate a portfolio DataFrame of customer records."""
        t = float(threshold if threshold is not None else self.decision_threshold)
        if self.model is None:
            raise RuntimeError("XGBoost model is not loaded.")

        encoded_batch = self.preprocessor.transform_batch(df)
        X_batch_arr = np.asarray(encoded_batch, dtype=np.float32)
        probs = self.model.predict_proba(X_batch_arr)[:, 1]
        preds = (probs >= t).astype(int)

        evaluated_rows: List[Dict[str, Any]] = []
        total_balance_at_risk = 0.0
        total_clv_at_risk = 0.0

        for idx, (_, row) in enumerate(df.iterrows()):
            p_c = float(probs[idx])
            pred_val = int(preds[idx])
            bal = float(row.get("Balance", 0.0))
            sal = float(row.get("EstimatedSalary", 50000.0))
            prods = int(row.get("NumOfProducts", 1))
            act = int(row.get("IsActiveMember", 1))
            card = int(row.get("HasCrCard", 1))

            fin = self.clv_engine.assess_customer_value(
                balance=bal,
                estimated_salary=sal,
                churn_prob=p_c,
                num_of_products=prods,
                is_active_member=act,
                has_cr_card=card,
            )

            if pred_val == 1:
                total_balance_at_risk += bal
                total_clv_at_risk += fin["value_at_risk"]

            row_summary = {
                "CustomerId": row.get("CustomerId", idx + 1),
                "Surname": str(row.get("Surname", f"Customer_{idx+1}")),
                "CreditScore": row.get("CreditScore", 650),
                "Geography": row.get("Geography", "France"),
                "Gender": row.get("Gender", "Female"),
                "Age": row.get("Age", 40),
                "Tenure": row.get("Tenure", 5),
                "Balance": bal,
                "NumOfProducts": prods,
                "IsActiveMember": act,
                "EstimatedSalary": sal,
                "Churn_Probability": round(p_c, 4),
                "Predicted_Churn": pred_val,
                "Risk_Tier": "High" if p_c >= 0.45 else ("Medium" if p_c >= 0.25 else "Low"),
                "CLV_3Yr": fin["clv_3yr"],
                "Value_At_Risk": fin["value_at_risk"],
            }
            evaluated_rows.append(row_summary)

        res_df = pd.DataFrame(evaluated_rows)
        # Sort by Value_At_Risk descending
        res_df = res_df.sort_values("Value_At_Risk", ascending=False).reset_index(drop=True)

        n_total = len(df)
        n_flagged = int(np.sum(preds))

        return {
            "num_customers": n_total,
            "churners_flagged": n_flagged,
            "churn_rate_pct": round((n_flagged / max(1, n_total)) * 100.0, 2),
            "decision_threshold": t,
            "total_balance_at_risk": round(total_balance_at_risk, 2),
            "total_clv_at_risk": round(total_clv_at_risk, 2),
            "leaderboard": res_df,
        }
