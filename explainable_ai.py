"""explainable_ai.py

Single source of truth for explainable, fully traceable RL decisions.

Every decision must return:
- decision (allow/block)
- risk_score (0-100)
- explanation (human readable)
- attack_type (string)

This module is intentionally rule-based (O(1)) to avoid slowing down RL training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from utils import compute_risk_score, risk_score_to_levels

from typing import Literal





@dataclass(frozen=True)
class ExplainableDecision:
    action: str  # "ALLOW" | "MONITOR" | "THROTTLE" | "BLOCK"
    attack_type: str

    # current risk
    risk_score: int  # 0..100
    severity_level: str  # LOW|MEDIUM|HIGH|CRITICAL

    # confidence proxy
    confidence_level: str  # LOW|MEDIUM|HIGH

    # explainability fields required by Phase 3
    why_action_taken: str
    why_risk_increased: str
    why_honeypot_used: str
    why_traffic_isolated: str

    # final human readable narrative
    explanation: str




StateInput = Tuple[int, int, int, int]


def _classify_attack(*, request_rate: int, failed_logins: int, unknown_ip: int, traffic_spike: int, repeated_requests: int) -> str:
    if failed_logins >= 7 and repeated_requests >= 15:
        return "credential_stuffing"
    if failed_logins >= 6:
        return "brute_force"
    if request_rate >= 80 and traffic_spike == 1:
        return "ddos"
    if repeated_requests >= 18 and request_rate >= 45:
        return "port_scan"
    if unknown_ip == 1 and repeated_requests >= 10:
        return "unknown_behavior"
    return "normal"



def _action_to_str(action_id: int) -> str:
    mapping = {
        0: "ALLOW",
        1: "MONITOR",
        2: "THROTTLE",
        3: "BLOCK",
    }
    return mapping.get(int(action_id), "ALLOW")



def _why_risk_increased(
    *,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    traffic_spike: int,
    repeated_requests: int,
) -> str:
    parts = []
    if failed_logins >= 7:
        parts.append("failed logins strongly indicate credential attack behavior")
    elif failed_logins >= 4:
        parts.append("failed logins are elevated")

    if unknown_ip == 1:
        parts.append("unknown IP increases suspicion")

    if traffic_spike == 1:
        parts.append("traffic spike suggests volumetric/burst activity")

    if repeated_requests >= 18:
        parts.append("high repetition suggests automated probing")

    if request_rate >= 75:
        parts.append("high request rate increases load/abuse likelihood")

    if not parts:
        parts.append("signals do not show strong intrusion indicators")

    return "; ".join(parts)


def _why_action_taken(*, action_str: str) -> str:
    if action_str == "BLOCK":
        return "Action is BLOCK to mitigate the current high-risk signals."
    if action_str == "THROTTLE":
        return "Action is THROTTLE to reduce load while observing impact."
    if action_str == "MONITOR":
        return "Action is MONITOR to watch suspicious patterns with minimal disruption."
    return "Action is ALLOW because intrusion indicators are not strong enough to escalate."


def _why_honeypot_used(*, honeypot_triggered: bool, honeypot_reason: str) -> str:
    if honeypot_triggered:
        return f"Honeypot is used because: {honeypot_reason}"
    return "Honeypot is not used because redirect criteria are not met."


def _why_traffic_isolated(*, isolated: bool) -> str:
    if isolated:
        return "Traffic/session is isolated to prevent lateral impact while the system observes the attacker." 
    return "Traffic is not isolated because the isolation threshold is not reached."


def _final_explanation(
    *,
    action_id: int,
    risk_score: int,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    traffic_spike: int,
    repeated_requests: int,
    honeypot_triggered: bool,
    honeypot_reason: str,
    isolated: bool,
) -> str:

    action_str = _action_to_str(action_id)
    why_risk = _why_risk_increased(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        traffic_spike=traffic_spike,
        repeated_requests=repeated_requests,
    )

    why_action = _why_action_taken(action_str=action_str)
    why_hp = _why_honeypot_used(honeypot_triggered=honeypot_triggered, honeypot_reason=honeypot_reason)
    why_iso = _why_traffic_isolated(isolated=isolated)

    # Keep narrative concise but professional.
    if risk_score >= 80:
        risk_tag = "High"
    elif risk_score >= 50:
        risk_tag = "Moderate"
    else:
        risk_tag = "Low"

    return (
        f"{why_action} Risk is {risk_tag} ({risk_score}/100). "
        f"Why risk increased: {why_risk}. "
        f"{why_hp} {why_iso}"
    )


def _rule_based_explanation(

    *,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    traffic_spike: int,
    repeated_requests: int,
    risk_score: int,
    action_id: int,
) -> str:

    # Keep this small and O(1): fixed checks, no loops over large structures.
    reasons = []

    if failed_logins >= 7:
        reasons.append("High failed logins detected → possible brute force")
    elif failed_logins >= 4:
        reasons.append("Elevated failed logins → increased login-attack likelihood")

    if unknown_ip == 1:
        reasons.append("Traffic from unknown IP source → increased suspicion")

    if request_rate >= 75:
        reasons.append("Very high request rate → possible DDoS / high load")
    elif request_rate >= 45:
        reasons.append("Moderately high request rate → elevated activity")

    if risk_score >= 80:
        reasons.append("Overall risk is high based on the above signals")
    elif risk_score >= 50:
        reasons.append("Overall risk is moderate based on the above signals")
    else:
        reasons.append("Overall risk is low based on the above signals")

    action_str = _action_to_str(action_id)
    if action_str == "BLOCK":
        reasons.append("Decision: BLOCK to mitigate the detected risk")
    elif action_str == "THROTTLE":
        reasons.append("Decision: THROTTLE to reduce load while monitoring impact")
    elif action_str == "MONITOR":
        reasons.append("Decision: MONITOR to observe suspicious patterns")
    else:
        reasons.append("Decision: ALLOW as risk signals are not strong")

    return "; ".join(reasons)



def explain_decision(
    *,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    traffic_spike: int,
    repeated_requests: int,
    time_of_day: int,
    action_id: int,
    rl_confidence: float = 0.5,
    # Phase-3 explainability plumbing
    honeypot_triggered: bool = False,
    honeypot_reason: str = "",
    traffic_isolated: bool = False,
) -> ExplainableDecision:


    """Explain a single decision.

    This is the single source of truth called by train.py and api.py.
    """

    risk_score = compute_risk_score(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=time_of_day,
        traffic_spike=traffic_spike,
        time_window=None,
        agent_action=action_id,
    )

    attack_type = _classify_attack(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        traffic_spike=traffic_spike,
        repeated_requests=repeated_requests,
    )

    risk_label, severity_level = risk_score_to_levels(risk_score)

    if rl_confidence >= 0.66:
        confidence_level = "HIGH"
    elif rl_confidence >= 0.33:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    action_str = _action_to_str(action_id)

    why_risk = _why_risk_increased(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        traffic_spike=traffic_spike,
        repeated_requests=repeated_requests,
    )
    why_action = _why_action_taken(action_str=action_str)
    why_hp = _why_honeypot_used(
        honeypot_triggered=honeypot_triggered,
        honeypot_reason=honeypot_reason,
    )
    why_iso = _why_traffic_isolated(isolated=traffic_isolated)

    explanation = _final_explanation(
        action_id=action_id,
        risk_score=risk_score,
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        traffic_spike=traffic_spike,
        repeated_requests=repeated_requests,
        honeypot_triggered=honeypot_triggered,
        honeypot_reason=honeypot_reason,
        isolated=traffic_isolated,
    )

    return ExplainableDecision(
        action=action_str,
        attack_type=attack_type,
        risk_score=risk_score,
        confidence_level=confidence_level,
        severity_level=severity_level,
        why_action_taken=why_action,
        why_risk_increased=why_risk,
        why_honeypot_used=why_hp,
        why_traffic_isolated=why_iso,
        explanation=explanation,
    )




def explain_decision_from_state(
    state: StateInput,
    *,
    action_id: int,
    traffic_spike: int = 0,
    repeated_requests: int = 0,
    rl_confidence: float = 0.5,
    honeypot_triggered: bool = False,
    honeypot_reason: str = "",
    traffic_isolated: bool = False,
) -> ExplainableDecision:

    request_rate, failed_logins, unknown_ip, time_of_day = state
    return explain_decision(
        request_rate=int(request_rate),
        failed_logins=int(failed_logins),
        unknown_ip=int(unknown_ip),
        traffic_spike=int(traffic_spike),
        repeated_requests=int(repeated_requests),
        time_of_day=int(time_of_day),
        action_id=int(action_id),
        rl_confidence=float(rl_confidence),
        honeypot_triggered=honeypot_triggered,
        honeypot_reason=honeypot_reason,
        traffic_isolated=traffic_isolated,
    )



