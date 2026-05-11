"""environment.py

Reinforcement-learning based intrusion detection environment.

State vector (tabular Q-learning):
    [request_rate, failed_logins, unknown_ip, time_of_day]

Actions:
    0 = ALLOW
    1 = MONITOR
    2 = THROTTLE
    3 = BLOCK

Reward:
    Uses ground-truth intrusion heuristic + severity to shape rewards.

Backward compatibility:
- Existing code passes simulator-generated traffic records.
- For real dataset RL (Phase A/D), callers may inject a dataset ground-truth label
  via record keys:
    - record["_gt"] OR record["ground_truth_label"]
  When provided, the environment uses it instead of heuristic inference.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

Action = int

# Discretized tabular RL state.
# Format: (request_rate_bin, failed_logins_bin, unknown_ip, time_of_day_bin,
#          traffic_spike, repeated_requests_bin, ip_reputation_bin, session_duration_bin)
State = Tuple[int, int, int, int, int, int, int, int]


class IntrusionDetectionEnvironment:
    """Episodic environment over a fixed traffic stream."""

    # action mapping escalation order for reward shaping
    ACTION_ALLOW = 0
    ACTION_MONITOR = 1
    ACTION_THROTTLE = 2
    ACTION_BLOCK = 3
    ACTION_ISOLATE_DEVICE = 4
    ACTION_REDIRECT_TO_HONEYPOT = 5

    def __init__(
        self,
        traffic_data: Sequence[Dict[str, int]] | None = None,
        *,
        simulator: object | None = None,
        n_samples: int = 1000,
        seed: int | None = 42,
        config: object | None = None,
    ) -> None:
        self.config = config

        if traffic_data is None:
            if simulator is None:
                raise ValueError("Either traffic_data or simulator must be provided")
            self.traffic_data: List[Dict[str, int]] = list(
                simulator.generate_traffic(n_samples, as_dataframe=False, seed=seed)
            )
        else:
            self.traffic_data = list(traffic_data)

        if not self.traffic_data:
            raise ValueError("traffic_data is empty")

        self._idx = 0

    @staticmethod
    def _bin(v: int, *, bins: List[int]) -> int:
        """Map integer value to a small discrete bin id using upper bounds."""
        for i, ub in enumerate(bins):
            if v <= ub:
                return i
        return len(bins)

    @classmethod
    def _record_to_state(cls, record: Dict[str, int]) -> State:
        rr = int(record["request_rate"])
        fl = int(record["failed_logins"])
        ui = int(record["unknown_ip"])
        tod = int(record["time_of_day"])

        traffic_spike = int(record.get("traffic_spike", 0))
        repeated = int(record.get("repeated_requests", 0))
        ip_rep = int(record.get("ip_reputation_score", 50))
        session_duration = int(record.get("session_duration", 0))

        rr_bin = cls._bin(rr, bins=[15, 30, 45, 60, 80])
        fl_bin = cls._bin(fl, bins=[1, 3, 5, 7, 9])
        tod_bin = cls._bin(tod, bins=[4, 9, 14, 19, 23])
        rep_bin = cls._bin(repeated, bins=[3, 8, 15, 22, 30])
        ip_rep_bin = cls._bin(ip_rep, bins=[25, 40, 55, 70, 85])
        sess_bin = cls._bin(session_duration, bins=[80, 170, 300, 450, 650])

        return (
            rr_bin,
            fl_bin,
            ui,
            tod_bin,
            traffic_spike,
            rep_bin,
            ip_rep_bin,
            sess_bin,
        )

    @staticmethod
    def _severity_and_is_intrusion(record: Dict[str, int]) -> tuple[bool, int, str]:
        """Return (is_intrusion, severity_level_1_to_3, attack_type).

        If a dataset ground-truth label is present in the record, it is used.
        Otherwise we fall back to heuristic inference based on simulator-like signals.
        """

        rr = int(record.get("request_rate", 0))
        fl = int(record.get("failed_logins", 0))
        ui = int(record.get("unknown_ip", 0))
        traffic_spike = int(record.get("traffic_spike", 0))
        repeated = int(record.get("repeated_requests", 0))
        session_duration = int(record.get("session_duration", 0))
        packet_size = int(record.get("packet_size", 0))
        ip_rep = int(record.get("ip_reputation_score", 50))

        gt = record.get("_gt") or record.get("ground_truth_label")
        if gt is not None:
            gt_str = str(gt).strip()
            intrusion_classes = {
                "DDoS",
                "Brute Force",
                "Port Scan",
                "Botnet",
                "Web Attack",
                "Infiltration",
            }
            if gt_str in intrusion_classes:
                is_intrusion = True
                # severity tiering heuristic
                severity = 3 if gt_str in {"DDoS", "Brute Force"} else 2
                attack_type = gt_str.lower().replace(" ", "_")
            else:
                is_intrusion = False
                severity = 1
                attack_type = "benign"

            return is_intrusion, severity, attack_type

        # --- Heuristic fallback (legacy simulator) ---
        if fl >= 7 and repeated >= 15:
            attack_type = "credential_stuffing"
        elif fl >= 6 and repeated >= 10:
            attack_type = "brute_force"
        elif rr >= 80 and traffic_spike == 1:
            attack_type = "ddos"
        elif repeated >= 18 and rr >= 45 and packet_size <= 8:
            attack_type = "port_scan"
        elif rr >= 35 and fl <= 3 and ip_rep <= 35 and session_duration >= 200:
            attack_type = "bot_traffic"
        elif session_duration >= 500 and repeated >= 12 and ip_rep <= 55 and fl <= 4:
            attack_type = "insider_threat"
        elif ui == 1 and (rr >= 40 or repeated >= 10) and ip_rep >= 35:
            attack_type = "unknown_behavior"
        else:
            attack_type = "normal"

        severity = 1
        is_intrusion = False

        if attack_type in {"ddos", "credential_stuffing"}:
            if rr >= 75 or fl >= 7 or traffic_spike == 1:
                severity = 3
                is_intrusion = True

        if attack_type in {"brute_force", "bot_traffic", "port_scan"}:
            if fl >= 5 or rr >= 55 or repeated >= 18:
                severity = 2 if (ip_rep >= 20 and traffic_spike == 0) else 3
                is_intrusion = True

        if attack_type == "insider_threat":
            if session_duration >= 550 or repeated >= 16:
                severity = 2
                is_intrusion = True
            if traffic_spike == 1 or ip_rep <= 30:
                severity = 3
                is_intrusion = True

        if attack_type == "unknown_behavior":
            if ui == 1 and (rr >= 50 or repeated >= 12):
                severity = 2
                is_intrusion = True

        if attack_type == "normal":
            if rr >= 90 and ip_rep <= 30 and traffic_spike == 1:
                is_intrusion = True
                severity = 3
            else:
                is_intrusion = False
                severity = 1

        return is_intrusion, severity, attack_type

    @staticmethod
    def _required_action(severity: int) -> int:
        if severity <= 1:
            return IntrusionDetectionEnvironment.ACTION_ALLOW
        if severity == 2:
            return IntrusionDetectionEnvironment.ACTION_THROTTLE
        return IntrusionDetectionEnvironment.ACTION_BLOCK

    def reset(self) -> State:
        self._idx = 0
        return self._record_to_state(self.traffic_data[self._idx])

    def step(self, action: Action) -> tuple[State, float, bool, dict]:
        if action not in (0, 1, 2, 3, 4, 5):
            raise ValueError(
                "action must be one of 0..5 (ALLOW/MONITOR/THROTTLE/BLOCK/ISOLATE_DEVICE/REDIRECT_TO_HONEYPOT)"
            )

        if self._idx >= len(self.traffic_data):
            raise RuntimeError("step() called after episode ended; call reset()")

        record = self.traffic_data[self._idx]
        state = self._record_to_state(record)

        is_intrusion, severity, attack_type = self._severity_and_is_intrusion(record)

        ideal = self._required_action(severity)

        # Reward shaping
        reward = 0.0

        suspicious_case = (not is_intrusion) and (severity == 1)
        if suspicious_case and action in (self.ACTION_MONITOR, self.ACTION_THROTTLE):
            reward += 1.0

        false_positive = (not is_intrusion) and (action != self.ACTION_ALLOW)
        missed_attack = is_intrusion and (action < ideal)
        over_escalation = (not is_intrusion) and (action > self.ACTION_ALLOW)

        if is_intrusion:
            if action == ideal:
                reward += 10.0 if severity == 3 else 5.0
            elif missed_attack:
                reward -= 10.0 if severity == 3 else 6.0
            else:
                reward -= 2.0 if severity == 3 else 1.0
        else:
            if action == self.ACTION_ALLOW:
                reward += 3.0
            else:
                reward -= 5.0 if action == self.ACTION_BLOCK else 3.0

        if (not is_intrusion) and severity == 1 and action == self.ACTION_MONITOR:
            reward += 1.0

        correct = (is_intrusion and action >= ideal) or ((not is_intrusion) and action == self.ACTION_ALLOW)

        self._idx += 1
        done = self._idx >= len(self.traffic_data)

        next_state = state if done else self._record_to_state(self.traffic_data[self._idx])

        info = {
            "is_intrusion": is_intrusion,
            "severity": severity,
            "attack_type": attack_type,
            "ideal_action": ideal,
            "correct": correct,
            "missed_attack": missed_attack,
            "false_positive": false_positive or over_escalation,
            "idx": self._idx,
            "taken_action": action,
        }

        return next_state, float(reward), done, info

