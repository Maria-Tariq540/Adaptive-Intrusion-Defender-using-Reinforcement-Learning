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
from explainable_ai import explain_decision

# Optional supervised model integration (real dataset)
try:
    from ml_attack_detector import load_detector, predict_from_raw_features
except Exception:  # pragma: no cover
    load_detector = None  # type: ignore
    predict_from_raw_features = None  # type: ignore

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


_CACHED_DETECTOR = {"model": None}


def _get_supervised_detector():
    """Lazy-load the supervised ML detector from models/ml_detector.joblib."""
    global _CACHED_DETECTOR

    if _CACHED_DETECTOR.get("model") is not None:
        return _CACHED_DETECTOR["model"]

    if load_detector is None:
        return None

    from pathlib import Path
    from config import get_config

    cfg = get_config()
    model_path = getattr(cfg, "ml_detector_model_path", "models/ml_detector.joblib")
    try:
        det = load_detector(Path(model_path))
    except Exception:
        det = None

    _CACHED_DETECTOR["model"] = det
    return det


def enterprise_predict(*, inp: EnterprisePredictInput, platform: Optional[MultiAgentSecurityPlatform] = None) -> Dict[str, Any]:
    """Run enterprise prediction.

    Returns a dict matching the required API schema.
    """

    t0 = time.perf_counter()

    platform = platform or MultiAgentSecurityPlatform()

    # Current RL action decision
    state = (int(inp.request_rate), int(inp.failed_logins), int(inp.unknown_ip), int(inp.time_of_day))
    decision, final_action_id = platform.step(state, session_id=int(inp.session_id))

    # RL action_id -> coarse action string
    if int(final_action_id) == 3 or int(decision.firewall_action) == 3:
        base_action_str = "BLOCK"
    else:
        base_action_str = "ALLOW"

    # -------------------------
    # Supervised classifier (real dataset)
    # -------------------------
    detector = _get_supervised_detector()
    classifier_attack_type = "normal"
    classifier_confidence = 0.0
    classifier_predicted_risk = 0
    classifier_predicted_severity = "LOW"

    if detector is not None and predict_from_raw_features is not None:
        try:
            raw_features = {
                # include the canonical RL/enterprise features; detector may ignore missing columns
                "request_rate": int(inp.request_rate),
                "failed_logins": int(inp.failed_logins),
                "unknown_ip": int(inp.unknown_ip),
                "time_of_day": int(inp.time_of_day),
                "traffic_spike": int(inp.traffic_spike),
                "repeated_requests": int(inp.repeated_requests),
                "ip_reputation_score": int(inp.ip_reputation_score),
                "session_duration": int(inp.session_duration),
                "packet_size": int(inp.packet_size),
                "destination_port": 0,
                "syn_flag_count": 0,
            }
            det_pred = predict_from_raw_features(detector=detector, raw_features=raw_features)
            classifier_attack_type = str(det_pred.attack_type).strip()
            classifier_confidence = float(det_pred.confidence)
            classifier_predicted_risk = int(round(det_pred.confidence * 100.0))
        except Exception:
            classifier_attack_type = "normal"
            classifier_confidence = 0.0
            classifier_predicted_risk = 0

    # Risk/severity narrative from existing explainable_ai
    exp = explain_decision(
        request_rate=int(inp.request_rate),
        failed_logins=int(inp.failed_logins),
        unknown_ip=int(inp.unknown_ip),
        traffic_spike=int(inp.traffic_spike),
        repeated_requests=int(inp.repeated_requests),
        time_of_day=int(inp.time_of_day),
        action_id=int(decision.firewall_action),
        rl_confidence=0.5,
        honeypot_triggered=False,
        honeypot_reason="",
        traffic_isolated=False,
    )

    # Calibrate severity from classifier_predicted_risk (preferred)
    if classifier_predicted_risk < 35:
        classifier_predicted_severity = "LOW"
    elif classifier_predicted_risk < 60:
        classifier_predicted_severity = "MEDIUM"
    elif classifier_predicted_risk < 85:
        classifier_predicted_severity = "HIGH"
    else:
        classifier_predicted_severity = "CRITICAL"

    current_risk = int(exp.risk_score)

    honeypot_dec = maybe_use_honeypot(
        predicted_risk=int(classifier_predicted_risk),
        current_risk=current_risk,
        severity=str(exp.severity_level),
        predicted_severity=str(classifier_predicted_severity),
        is_intrusion_likely=(int(classifier_predicted_risk) >= 60),
    )

    isolate = (int(decision.isolation_action) == 4)

    final_action = _map_action_to_final(
        base_action_str,
        honeypot_triggered=bool(honeypot_dec.honeypot_triggered),
        isolate=isolate,
    )

    synthetic_action_id = int(decision.firewall_action)
    if bool(honeypot_dec.honeypot_triggered):
        synthetic_action_id = 3
    elif bool(isolate):
        synthetic_action_id = 2

    exp2 = explain_decision(
        request_rate=int(inp.request_rate),
        failed_logins=int(inp.failed_logins),
        unknown_ip=int(inp.unknown_ip),
        traffic_spike=int(inp.traffic_spike),
        repeated_requests=int(inp.repeated_requests),
        time_of_day=int(inp.time_of_day),
        action_id=synthetic_action_id,
        rl_confidence=0.5,
        honeypot_triggered=bool(honeypot_dec.honeypot_triggered),
        honeypot_reason=str(honeypot_dec.honeypot_reason),
        traffic_isolated=bool(isolate),
    )

    latency_ms = int((time.perf_counter() - t0) * 1000.0)

    return {
        "action": final_action,
        "attack_type": str(classifier_attack_type),
        "classifier_confidence": float(classifier_confidence),
        "risk_score": int(exp2.risk_score),
        "predicted_risk": int(classifier_predicted_risk),
        "confidence": exp2.confidence_level,
        "severity_level": str(exp2.severity_level),
        "predicted_severity": str(classifier_predicted_severity),
        "predicted_attack_type": str(classifier_attack_type),
        "explanation": exp2.explanation,
        "honeypot_status": bool(honeypot_dec.honeypot_triggered),
        "honeypot_reason": str(honeypot_dec.honeypot_reason),
        "latency_ms": latency_ms,
        "early_warning": int(classifier_predicted_risk) >= 75,
        "session_id": int(inp.session_id),
        "session_isolated": bool(isolate),
    }


