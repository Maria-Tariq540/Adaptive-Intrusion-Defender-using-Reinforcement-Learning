"""predictor.py

Lightweight predictive risk scoring for near-future attack likelihood.

This is rule-based (O(1)) to keep the SOC pipeline responsive.

Output:
- predicted_risk: int 0..100
- predicted_severity: LOW|MEDIUM|HIGH|CRITICAL
- predicted_attack_type: string label
- reasons: short human readable phrases
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    predicted_risk: int
    predicted_severity: str
    predicted_attack_type: str
    reasons: str


def predict_near_future_attack(
    *,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    traffic_spike: int,
    repeated_requests: int,
    ip_reputation_score: int,
) -> Prediction:
    # Risk components
    risk = 0.0

    # Volumetric / spike
    if traffic_spike == 1:
        risk += 35.0
    risk += (float(request_rate) / 100.0) * 25.0

    # Credential signals
    risk += (float(failed_logins) / 10.0) * 25.0
    risk += (float(repeated_requests) / 50.0) * 15.0

    # Source quality
    if unknown_ip == 1:
        risk += 10.0
    # Low reputation tends to indicate automated/hostile
    risk += (1.0 - (float(ip_reputation_score) / 100.0)) * 10.0

    predicted_risk = int(max(0, min(100, round(risk))))

    # Severity + attack type (coarse)
    if failed_logins >= 7 and repeated_requests >= 15:
        attack_type = "credential_stuffing"
    elif failed_logins >= 6:
        attack_type = "brute_force"
    elif request_rate >= 80 and traffic_spike == 1:
        attack_type = "ddos"
    elif repeated_requests >= 18 and request_rate >= 45:
        attack_type = "port_scan"
    elif unknown_ip == 1 and repeated_requests >= 10:
        attack_type = "unknown_behavior"
    elif repeated_requests >= 12 and ip_reputation_score <= 55:
        attack_type = "insider_threat"
    else:
        attack_type = "normal"


    if predicted_risk < 35:
        severity = "LOW"
    elif predicted_risk < 60:
        severity = "MEDIUM"
    elif predicted_risk < 85:
        severity = "HIGH"
    else:
        severity = "CRITICAL"

    reasons = []
    if traffic_spike == 1:
        reasons.append("traffic spike detected")
    if failed_logins >= 7:
        reasons.append("high failed logins")
    if repeated_requests >= 18:
        reasons.append("high request repetition")
    if unknown_ip == 1:
        reasons.append("unknown IP source")
    if request_rate >= 75:
        reasons.append("high request rate")

    if not reasons:
        reasons.append("signals appear stable")

    return Prediction(
        predicted_risk=predicted_risk,
        predicted_severity=severity,
        predicted_attack_type=attack_type,
        reasons=", ".join(reasons),
    )

