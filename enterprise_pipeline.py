"""enterprise_pipeline.py

Unified enterprise prediction pipeline for the RL-Driven Autonomous Network
Intrusion Defense System.

This module centralizes:
- Near-future predictive threat intelligence
- Honeypot redirect decision
- Multi-agent security action selection
- Explainable AI narratives
- Output schema required by API + Dashboard

The pipeline is intentionally dependency-light and uses:
- predictor.predict_near_future_attack() for predicted_risk
- honeypot.maybe_use_honeypot() for honeypot_status
- multi_agent.MultiAgentSecurityPlatform for coordinated actions
- explainable_ai.explain_decision() for human-readable explanation

All downstream components should call `enterprise_predict()`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from honeypot import maybe_use_honeypot
from predictor import predict_near_future_attack
from explainable_ai import explain_decision
from multi_agent import (
    MultiAgentSecurityPlatform,
)


@dataclass(frozen=True)
class EnterprisePredictInput:
    # Required traffic features
    request_rate: int
    failed_logins: int
    unknown_ip: int
    time_of_day: int

    # Optional features (defaulted by caller)
    traffic_spike: int = 0
    session_duration: int = 0
    packet_size: int = 0
    repeated_requests: int = 0
    ip_reputation_score: int = 50
    session_id: int = -1


def _map_action_to_final(action_str: str, honeypot_triggered: bool, isolate: bool) -> str:
    # enterprise action contract
    # action_str originates from the RL tabular policy mapping (ALLOW/BLOCK).
    # isolate is handled by platform action.
    if honeypot_triggered:
        return "REDIRECT_TO_HONEYPOT"
    if isolate:
        return "ISOLATE_DEVICE"
    if action_str == "BLOCK":
        return "BLOCK"
    return "ALLOW" if action_str == "ALLOW" else "MONITOR"


def enterprise_predict(*, inp: EnterprisePredictInput, platform: Optional[MultiAgentSecurityPlatform] = None) -> Dict[str, Any]:
    """Run enterprise prediction.

    Returns a dict matching the required API schema.
    """

    t0 = time.perf_counter()

    platform = platform or MultiAgentSecurityPlatform()

    # Current risk score and severity/explanation
    # We compute explainable fields from the current signals.
    # action_id in explainable_ai is mapped for ALLOW/BLOCK only in this repo.
    state = (int(inp.request_rate), int(inp.failed_logins), int(inp.unknown_ip), int(inp.time_of_day))

    decision, final_action_id = platform.step(state, session_id=int(inp.session_id))

    # RL action_id -> coarse action string
    if int(final_action_id) == 3 or int(decision.firewall_action) == 3:
        base_action_str = "BLOCK"
    else:
        base_action_str = "ALLOW"

    # Near-future predictive threat intelligence
    pred = predict_near_future_attack(
        request_rate=int(inp.request_rate),
        failed_logins=int(inp.failed_logins),
        unknown_ip=int(inp.unknown_ip),
        traffic_spike=int(inp.traffic_spike),
        repeated_requests=int(inp.repeated_requests),
        ip_reputation_score=int(inp.ip_reputation_score),
    )

    # Current numeric risk (aligns with explainable_ai compute_risk_score)
    exp = explain_decision(
        request_rate=int(inp.request_rate),
        failed_logins=int(inp.failed_logins),
        unknown_ip=int(inp.unknown_ip),
        traffic_spike=int(inp.traffic_spike),
        repeated_requests=int(inp.repeated_requests),
        time_of_day=int(inp.time_of_day),
        action_id=int(decision.firewall_action),
        rl_confidence=0.5,
        # honeypot/isolation narrative are set after decisions below
        honeypot_triggered=False,
        honeypot_reason="",
        traffic_isolated=False,
    )

    current_risk = int(exp.risk_score)

    honeypot_dec = maybe_use_honeypot(
        predicted_risk=int(pred.predicted_risk),
        current_risk=current_risk,
        severity=str(exp.severity_level),
        predicted_severity=str(pred.predicted_severity),
        is_intrusion_likely=(int(pred.predicted_risk) >= 60),
    )

    isolate = (int(decision.isolation_action) == 4)  # ACTION_ISOLATE_DEVICE in multi_agent

    final_action = _map_action_to_final(
        base_action_str,
        honeypot_triggered=bool(honeypot_dec.honeypot_triggered),
        isolate=isolate,
    )

    # Finalize explainability with honeypot + isolation flags
    exp2 = explain_decision(
        request_rate=int(inp.request_rate),
        failed_logins=int(inp.failed_logins),
        unknown_ip=int(inp.unknown_ip),
        traffic_spike=int(inp.traffic_spike),
        repeated_requests=int(inp.repeated_requests),
        time_of_day=int(inp.time_of_day),
        action_id=int(decision.firewall_action),
        rl_confidence=0.5,
        honeypot_triggered=bool(honeypot_dec.honeypot_triggered),
        honeypot_reason=str(honeypot_dec.honeypot_reason),
        traffic_isolated=bool(isolate),
    )

    latency_ms = int((time.perf_counter() - t0) * 1000.0)

    return {
        "action": final_action,
        "attack_type": exp2.attack_type,
        "risk_score": int(exp2.risk_score),
        "predicted_risk": int(pred.predicted_risk),
        "confidence": exp2.confidence_level,
        "severity_level": str(exp2.severity_level),
        "predicted_severity": str(pred.predicted_severity),
        "predicted_attack_type": str(pred.predicted_attack_type),
        "explanation": exp2.explanation,
        "honeypot_status": bool(honeypot_dec.honeypot_triggered),
        "honeypot_reason": str(honeypot_dec.honeypot_reason),
        "latency_ms": latency_ms,
        # useful for dashboard alerts
        "early_warning": int(pred.predicted_risk) >= 75,
        "session_id": int(inp.session_id),
        "session_isolated": bool(isolate),
    }

