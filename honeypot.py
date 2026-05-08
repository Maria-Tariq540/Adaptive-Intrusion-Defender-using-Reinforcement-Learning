"""honeypot.py

Simulated honeypot defense module.

The platform uses the honeypot to redirect suspicious/likely-attack traffic
into an isolated sandbox to collect signals and reduce risk.

This repository is a synthetic platform; honeypot behavior is represented as:
- honeypot_triggered (bool)
- honeypot_reason (human readable)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HoneypotDecision:
    honeypot_triggered: bool
    honeypot_reason: str


def maybe_use_honeypot(
    *,
    predicted_risk: int,
    current_risk: int,
    severity: str,
    predicted_severity: str,
    is_intrusion_likely: bool,
) -> HoneypotDecision:
    # Trigger honeypot when near-future risk is high and current signals are suspicious.
    trigger = False
    reason_parts = []

    if predicted_risk >= 75:
        trigger = True
        reason_parts.append(f"predicted risk is high ({predicted_risk}/100)")

    if current_risk >= 60:
        trigger = True
        reason_parts.append(f"current risk is elevated ({current_risk}/100)")

    if severity in ("HIGH", "CRITICAL") or predicted_severity in ("HIGH", "CRITICAL"):
        trigger = True
        reason_parts.append("severity threshold exceeded")

    if is_intrusion_likely:
        trigger = True
        reason_parts.append("attack likelihood indicates imminent activity")

    if not reason_parts:
        return HoneypotDecision(
            honeypot_triggered=False,
            honeypot_reason="Honeypot not needed; traffic does not meet redirect criteria.",
        )

    if trigger:
        return HoneypotDecision(
            honeypot_triggered=True,
            honeypot_reason="Redirecting to honeypot to observe behavior and reduce exposure: "
            + "; ".join(reason_parts),
        )

    return HoneypotDecision(
        honeypot_triggered=False,
        honeypot_reason="Honeypot criteria not met strictly: " + "; ".join(reason_parts),
    )

