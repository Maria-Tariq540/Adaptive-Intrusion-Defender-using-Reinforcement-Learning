"""multi_agent.py

Simulated multi-agent security platform built on top of the existing
RL-based intrusion detection components.

Agents:
- MonitoringAgent: monitors traffic record, classifies attack behavior,
  computes initial threat level.
- FirewallAgent: chooses defensive network action from:
    0=ALLOW, 1=MONITOR, 2=THROTTLE, 3=BLOCK
- IsolationAgent: chooses session/device isolation actions:
    4=ISOLATE_DEVICE (plus recovery management)

Design goals:
- Modular responsibilities
- Logical communication via a small shared state object
- Stable execution for dashboard + API integration

Note:
- Current repository RL/Q-learning is tabular and designed for 4 actions.
  For Phase 1 stability we treat RL as the FirewallAgent policy over actions
  {ALLOW, MONITOR, THROTTLE, BLOCK}. Isolation is handled deterministically
  inside IsolationAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from agent import RLIntrusionAgent
from utils import risk_score_to_levels


State = Tuple[int, int, int, int]  # (request_rate, failed_logins, unknown_ip, time_of_day)


ACTION_ALLOW = 0
ACTION_MONITOR = 1
ACTION_THROTTLE = 2
ACTION_BLOCK = 3
ACTION_ISOLATE_DEVICE = 4


def _severity_to_action_set(severity: str) -> Tuple[int, ...]:
    # Mapping to help IsolationAgent decide quickly.
    if severity in ("CRITICAL",):
        return (ACTION_BLOCK,)
    if severity in ("HIGH",):
        return (ACTION_BLOCK, ACTION_ISOLATE_DEVICE)
    if severity in ("MEDIUM",):
        return (ACTION_MONITOR, ACTION_THROTTLE)
    return (ACTION_ALLOW, ACTION_MONITOR)


@dataclass
class ThreatSnapshot:
    """MonitoringAgent output."""

    request_rate: int
    failed_logins: int
    unknown_ip: int
    time_of_day: int

    # Derived
    threat_score: int  # 0..100 (risk score)
    severity: str  # LOW/MEDIUM/HIGH/CRITICAL
    attack_type: str  # string label


@dataclass
class AgentDecision:
    firewall_action: int  # 0..3
    isolation_action: int  # 0 or 4
    final_action: int  # 0..4 (combined)

    # for dashboard visibility
    severity: str
    threat_score: int
    attack_type: str


@dataclass
class IsolationRecord:
    session_id: int
    remaining_ticks: int


@dataclass
class SecurityState:
    """Holds platform-level state shared across agents."""

    tick: int = 0

    isolated_sessions: Dict[int, IsolationRecord] = field(default_factory=dict)

    recovered_sessions_count: int = 0
    isolated_devices_count: int = 0

    # lightweight log for dashboard/API
    last_recovery: Optional[Dict[str, Any]] = None


class MonitoringAgent:
    """Monitors a traffic record and produces a threat snapshot."""

    @staticmethod
    def _classify_attack(request_rate: int, failed_logins: int, unknown_ip: int) -> str:
        # Keep in sync with explainable_ai-ish heuristics at a coarse level.
        if failed_logins >= 7:
            return "credential_stuffing"
        if failed_logins >= 6:
            return "brute_force"
        if request_rate >= 80:
            return "ddos"
        if unknown_ip == 1 and request_rate >= 45:
            return "unknown_behavior"
        return "normal"

    @staticmethod
    def _threat_score(request_rate: int, failed_logins: int, unknown_ip: int, time_of_day: int) -> int:
        # Use existing single-source-of-truth mapping.
        # utils.risk_score_to_levels is severity mapping, but we need score.
        # For Phase 1 we compute score using the same compute_risk_score formula.
        from utils import compute_risk_score

        return compute_risk_score(
            request_rate=request_rate,
            failed_logins=failed_logins,
            unknown_ip=unknown_ip,
            time_of_day=time_of_day,
            agent_action=None,
        )

    def analyze(self, state: State, session_features: Optional[Dict[str, int]] = None) -> ThreatSnapshot:
        request_rate, failed_logins, unknown_ip, time_of_day = state
        threat_score = self._threat_score(request_rate, failed_logins, unknown_ip, time_of_day)
        _risk_label, severity = risk_score_to_levels(threat_score)
        attack_type = self._classify_attack(request_rate, failed_logins, unknown_ip)

        return ThreatSnapshot(
            request_rate=int(request_rate),
            failed_logins=int(failed_logins),
            unknown_ip=int(unknown_ip),
            time_of_day=int(time_of_day),
            threat_score=int(threat_score),
            severity=str(severity),
            attack_type=str(attack_type),
        )


class FirewallAgent:
    """Decides network action using the existing RLIntrusionAgent as policy."""

    def __init__(self, rl_agent: Optional[RLIntrusionAgent] = None) -> None:
        self.rl_agent = rl_agent or RLIntrusionAgent(config=None)

    def decide_firewall_action(self, snapshot: ThreatSnapshot, state: State) -> int:
        # RL policy over 4 actions.
        action = self.rl_agent.choose_action(state)
        # Safety: clamp to 0..3
        return int(action) if int(action) in (0, 1, 2, 3) else ACTION_ALLOW


class IsolationAgent:
    """Isolates suspicious sessions/devices and performs autonomous recovery."""

    def __init__(self, *, isolation_duration_ticks: int = 5, isolate_severity: str = "HIGH") -> None:
        self.isolation_duration_ticks = int(isolation_duration_ticks)
        self.isolate_severity = str(isolate_severity)

    @staticmethod
    def _severity_rank(sev: str) -> int:
        mapping = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
        return mapping.get(str(sev).upper(), 0)

    def should_isolate(self, snapshot: ThreatSnapshot) -> bool:
        # Simple rule: isolate at HIGH/CRITICAL by default.
        return self._severity_rank(snapshot.severity) >= self._severity_rank(self.isolate_severity)

    def decide_isolation_action(
        self,
        snapshot: ThreatSnapshot,
        state: State,
        session_id: int,
        platform_state: SecurityState,
    ) -> int:
        # If already isolated, keep isolation (no further action).
        if int(session_id) in platform_state.isolated_sessions:
            return ACTION_ISOLATE_DEVICE

        if self.should_isolate(snapshot):
            platform_state.isolated_sessions[int(session_id)] = IsolationRecord(
                session_id=int(session_id),
                remaining_ticks=self.isolation_duration_ticks,
            )
            platform_state.isolated_devices_count += 1
            return ACTION_ISOLATE_DEVICE

        return 0

    def tick_recovery(self, platform_state: SecurityState, snapshot: Optional[ThreatSnapshot] = None) -> None:
        """Decrement timers and recover any expired isolations."""
        platform_state.last_recovery = None

        if not platform_state.isolated_sessions:
            return

        expired = []
        for sid, rec in list(platform_state.isolated_sessions.items()):
            rec.remaining_ticks -= 1
            if rec.remaining_ticks <= 0:
                expired.append(sid)

        if not expired:
            return

        # Recover sessions.
        for sid in expired:
            platform_state.isolated_sessions.pop(sid, None)
            platform_state.recovered_sessions_count += 1

        platform_state.last_recovery = {
            "recovered_session_ids": expired,
            "recovered_count": len(expired),
        }


class MultiAgentSecurityPlatform:
    """Coordinates all agents and produces the final action."""

    def __init__(
        self,
        *,
        rl_agent: Optional[RLIntrusionAgent] = None,
        isolation_duration_ticks: int = 5,
        isolate_severity: str = "HIGH",
    ) -> None:
        self.monitoring_agent = MonitoringAgent()
        self.firewall_agent = FirewallAgent(rl_agent=rl_agent)
        self.isolation_agent = IsolationAgent(
            isolation_duration_ticks=isolation_duration_ticks,
            isolate_severity=isolate_severity,
        )
        self.platform_state = SecurityState()

    def step(
        self,
        state: State,
        *,
        session_id: int,
        isolation_enabled: bool = True,
    ) -> Tuple[AgentDecision, int]:
        """Produce decisions for this tick.

        Returns:
            (AgentDecision, final_action_id)
        """

        self.platform_state.tick += 1

        snapshot = self.monitoring_agent.analyze(state)

        firewall_action = self.firewall_agent.decide_firewall_action(snapshot, state)

        isolation_action = 0
        if isolation_enabled:
            isolation_action = self.isolation_agent.decide_isolation_action(
                snapshot,
                state,
                session_id=session_id,
                platform_state=self.platform_state,
            )

        # Combine: isolation dominates final action when active.
        final_action = ACTION_ISOLATE_DEVICE if isolation_action == ACTION_ISOLATE_DEVICE else firewall_action

        decision = AgentDecision(
            firewall_action=int(firewall_action),
            isolation_action=int(isolation_action),
            final_action=int(final_action),
            severity=snapshot.severity,
            threat_score=int(snapshot.threat_score),
            attack_type=str(snapshot.attack_type),
        )

        # Autonomous recovery tick occurs every platform step.
        self.isolation_agent.tick_recovery(self.platform_state, snapshot=snapshot)

        return decision, decision.final_action

