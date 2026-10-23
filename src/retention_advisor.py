"""Tactical banking retention strategy and intervention advisory engine."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RetentionAdvisor:
    """Prescribes actionable, driver-informed banking retention strategies and calculates campaign ROI."""

    def recommend_strategy(
        self,
        customer: Dict[str, Any],
        churn_prob: float,
        value_assessment: Dict[str, Any],
        shap_explanation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize customer attributes, risk probability, and value to generate an intervention playbook."""
        p_churn = float(churn_prob)
        balance = float(customer.get("Balance", 0.0))
        age = float(customer.get("Age", 40.0))
        num_products = int(customer.get("NumOfProducts", 1))
        is_active = bool(customer.get("IsActiveMember", 1))
        geography = str(customer.get("Geography", "France")).capitalize()

        # Urgency Tiering
        if p_churn >= 0.60 or (p_churn >= 0.35 and balance >= 60000.0):
            urgency = "Urgent (24-Hour SLA)"
            sla_hours = 24
        elif p_churn >= 0.35:
            urgency = "High Priority (48-Hour SLA)"
            sla_hours = 48
        elif p_churn >= 0.20:
            urgency = "Medium Priority (1 Week SLA)"
            sla_hours = 168
        else:
            urgency = "Low / Routine Engagement"
            sla_hours = 720

        # Determine Primary Retention Playbook
        actions: List[Dict[str, Any]] = []

        # Playbook 1: Inactive Account Re-activation
        if not is_active:
            actions.append({
                "playbook": "Executive Re-Engagement & Digital Activity Drive",
                "rationale": "Customer has zero active monthly touchpoints; immediate relationship manager touchpoint needed.",
                "tactics": [
                    "Direct phone call from branch relationship manager within SLA window.",
                    "Offer 6-month fee waiver on checking and international transfers.",
                    "Provide mobile app onboarding concierge session.",
                ],
                "cost": 75.0,
                "success_prob": 0.38,
            })

        # Playbook 2: High Balance Flight Protection
        if balance >= 50000.0:
            actions.append({
                "playbook": "Premier Wealth Retention & Yield Optimization",
                "rationale": f"High balance exposure (${balance:,.0f}); customer is vulnerable to competitive high-yield offers.",
                "tactics": [
                    "Offer preferential 0.75% annualized deposit yield bonus on 12-month lock.",
                    "Assign dedicated private wealth advisor.",
                    "Invite to exclusive retail banking loyalty tier.",
                ],
                "cost": 150.0,
                "success_prob": 0.35,
            })

        # Playbook 3: Single Product Cross-Sell Stickiness
        if num_products == 1:
            actions.append({
                "playbook": "Product Deepening & Relationship Stickiness",
                "rationale": "Holding a single product is a primary structural vulnerability; multi-product customers retain at 3x higher rates.",
                "tactics": [
                    "Pre-approved cashback credit card with $50 welcome statement credit.",
                    "Zero-fee automated savings sweep enrollment.",
                    "Preferred rate mortgage / consumer loan consultation.",
                ],
                "cost": 80.0,
                "success_prob": 0.42,
            })
        elif num_products >= 3:
            actions.append({
                "playbook": "Fee Rationalization & Account Consolidation",
                "rationale": f"Customer holds {num_products} products; historical data indicates fee dissatisfaction among 3+ product holders.",
                "tactics": [
                    "Perform comprehensive portfolio fee review and rebate redundant account fees.",
                    "Consolidate multiple statements into a single digital dashboard.",
                ],
                "cost": 60.0,
                "success_prob": 0.30,
            })

        # Playbook 4: German Market Regional Retention
        if geography == "Germany":
            actions.append({
                "playbook": "German Regional Market Assurance",
                "rationale": "German account cohort exhibits above-average competitive churn; require regional rate match.",
                "tactics": [
                    "Benchmark rates against top German retail competitors.",
                    "Guarantee zero maintenance fees for 24 months upon direct salary deposit.",
                ],
                "cost": 100.0,
                "success_prob": 0.32,
            })

        # Playbook 5: Mature Demographic Wealth Transition
        if age >= 48.0:
            actions.append({
                "playbook": "Retirement & Wealth Preservation Advisory",
                "rationale": "Mature life-stage transition; customer actively seeks wealth preservation and estate planning stability.",
                "tactics": [
                    "Complimentary 1-hour session with senior wealth planner.",
                    "Tax-efficient savings portfolio structuring proposal.",
                ],
                "cost": 180.0,
                "success_prob": 0.28,
            })

        # Fallback if no specific playbook triggered
        if not actions:
            actions.append({
                "playbook": "Customer Appreciation & Relationship Check-In",
                "rationale": "Routine customer maintenance to reinforce brand loyalty.",
                "tactics": [
                    "Personalized customer anniversary note and digital satisfaction survey.",
                    "Complimentary annual credit score monitoring report.",
                ],
                "cost": 25.0,
                "success_prob": 0.20,
            })

        primary_action = actions[0]
        action_cost = primary_action["cost"]
        success_p = primary_action["success_prob"]

        # Financial ROI Calculation
        clv_at_risk = value_assessment.get("value_at_risk", 500.0)
        expected_saved_clv = round(clv_at_risk * success_p, 2)
        net_expected_profit = round(expected_saved_clv - action_cost, 2)
        roi_ratio = round((net_expected_profit / max(1.0, action_cost)) * 100.0, 1)

        return {
            "urgency": urgency,
            "sla_hours": sla_hours,
            "primary_strategy": primary_action["playbook"],
            "rationale": primary_action["rationale"],
            "action_items": primary_action["tactics"],
            "all_triggered_playbooks": [a["playbook"] for a in actions],
            "estimated_intervention_cost": action_cost,
            "expected_retention_success_prob": success_p,
            "expected_saved_clv": expected_saved_clv,
            "net_expected_profit": net_expected_profit,
            "roi_ratio_pct": roi_ratio,
            "is_recommended": (net_expected_profit > 0 or p_churn >= 0.50),
        }
