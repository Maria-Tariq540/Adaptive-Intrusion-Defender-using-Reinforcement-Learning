"""simulator.py

Purpose:
    Advanced synthetic cybersecurity traffic simulator.

Notes:
    - Generates realistic, dynamic, scenario-based traffic features.
    - Designed to be dependency-light; pandas is optional.

Public API:
    - generate_traffic(n_samples, as_dataframe=False, seed=None)

Run:
    python simulator.py
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None


def _clamp_int(x: float | int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(x))))


def _clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _sample_time_of_day(rng: random.Random) -> int:
    # Weighted day/night pattern: more activity during daytime.
    # 0-23 hours
    h = rng.random()
    if h < 0.65:
        return int(rng.gauss(14, 4)) % 24
    return int(rng.gauss(2, 3)) % 24


def _seasonality_multiplier(time_of_day: int) -> float:
    # Daytime multiplier.
    if 9 <= time_of_day <= 18:
        return 1.2
    if 0 <= time_of_day <= 5:
        return 0.9
    return 1.0


@dataclass(frozen=True)
class TrafficRecord:
    # Required base signals
    request_rate: int
    failed_logins: int
    unknown_ip: int
    time_of_day: int

    # Required extra signals
    traffic_spike: int
    session_duration: int
    packet_size: int
    repeated_requests: int
    ip_reputation_score: int
    session_id: int

    def to_dict(self) -> Dict[str, int]:
        return {
            "request_rate": int(self.request_rate),
            "failed_logins": int(self.failed_logins),
            "unknown_ip": int(self.unknown_ip),
            "time_of_day": int(self.time_of_day),
            "traffic_spike": int(self.traffic_spike),
            "session_duration": int(self.session_duration),
            "packet_size": int(self.packet_size),
            "repeated_requests": int(self.repeated_requests),
            "ip_reputation_score": int(self.ip_reputation_score),
            "session_id": int(self.session_id),
        }


def _base_session_features(rng: random.Random, session_id: int) -> tuple[int, int, int, int, int]:
    """Return shared features: traffic_spike, session_duration, packet_size, repeated_requests, ip_reputation_score."""
    # traffic_spike is 0/1; rare in benign, more common in attacks.
    traffic_spike = 1 if rng.random() < 0.12 else 0

    # Session duration in seconds; longer sessions for benign/browsing, shorter for automated attacks.
    session_duration = _clamp_int(rng.gauss(220, 120), 5, 1200)

    # Packet size in KB (synthetic), moderate for normal and varied for attacks.
    packet_size = _clamp_int(rng.gauss(6.5, 1.8), 1, 20)

    # Repeated requests count within a time window.
    repeated_requests = _clamp_int(rng.gauss(3, 2.5), 0, 50)

    # IP reputation score: 0..100 (higher = better)
    ip_reputation_score = _clamp_int(rng.gauss(65, 20), 0, 100)

    return traffic_spike, session_duration, packet_size, repeated_requests, ip_reputation_score


def _sample_normal(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Benign: modest request rates and very low failed logins.
    request_rate = _clamp_int(rng.gauss(18, 10) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(1.0, 1.2), 0, 10)
    unknown_ip = _clamp_int(1 if rng.random() < 0.03 else 0, 0, 1)

    # Benign typically has low repeated_requests.
    repeated_requests = _clamp_int(min(repeated_requests, rng.gauss(4, 2)), 0, 50)
    # Benign IPs usually have higher reputation.
    ip_rep = _clamp_int(max(ip_rep, rng.gauss(55, 15)), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_brute_force(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Brute force: high failed_logins; moderate request_rate; lots of repeats.
    failed_logins = _clamp_int(rng.gauss(8.5, 1.0), 0, 10)
    request_rate = _clamp_int(rng.gauss(38, 14) * mult, 0, 100)
    unknown_ip = _clamp_int(1 if rng.random() < 0.35 else 0, 0, 1)

    repeated_requests = _clamp_int(rng.gauss(18, 8), 0, 50)
    traffic_spike = 1 if rng.random() < 0.45 else traffic_spike

    # Automated logins often have poor reputation.
    ip_rep = _clamp_int(rng.gauss(22, 18), 0, 100)
    # Sessions tend to be shorter.
    session_duration = _clamp_int(rng.gauss(95, 45), 5, 600)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=_clamp_int(rng.gauss(5.5, 1.2), 1, 20),
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_ddos(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # DDoS: very high request_rate; failed logins can be low-medium.
    request_rate = _clamp_int(rng.gauss(88, 9) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(2.0, 1.7), 0, 10)
    unknown_ip = _clamp_int(1 if rng.random() < 0.7 else 0, 0, 1)

    traffic_spike = 1
    repeated_requests = _clamp_int(rng.gauss(26, 10), 0, 50)
    ip_rep = _clamp_int(rng.gauss(18, 18), 0, 100)

    # DDoS sessions can be long but with bursts.
    session_duration = _clamp_int(rng.gauss(420, 170), 20, 1200)
    packet_size = _clamp_int(rng.gauss(7.8, 2.5), 1, 20)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_port_scan(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Port scan: moderate-high request_rate, low failed_logins, high repeats and suspicion.
    request_rate = _clamp_int(rng.gauss(55, 18) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(1.8, 1.2), 0, 10)
    unknown_ip = 1

    repeated_requests = _clamp_int(rng.gauss(22, 9), 0, 50)
    traffic_spike = 1 if rng.random() < 0.35 else 0
    session_duration = _clamp_int(rng.gauss(160, 70), 10, 900)
    packet_size = _clamp_int(rng.gauss(4.2, 1.0), 1, 20)
    ip_rep = _clamp_int(rng.gauss(30, 22), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_credential_stuffing(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Credential stuffing: high failed_logins + high repeats; request_rate often high.
    failed_logins = _clamp_int(rng.gauss(7.5, 1.4), 0, 10)
    request_rate = _clamp_int(rng.gauss(65, 16) * mult, 0, 100)
    unknown_ip = _clamp_int(1 if rng.random() < 0.55 else 0, 0, 1)

    repeated_requests = _clamp_int(rng.gauss(30, 10), 0, 50)
    traffic_spike = 1 if rng.random() < 0.5 else 0
    session_duration = _clamp_int(rng.gauss(180, 85), 10, 1000)
    packet_size = _clamp_int(rng.gauss(5.8, 1.6), 1, 20)
    ip_rep = _clamp_int(rng.gauss(24, 18), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_bot_traffic(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Bot traffic: moderate-high request_rate, low failed logins, low reputation.
    request_rate = _clamp_int(rng.gauss(45, 20) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(1.4, 1.0), 0, 10)
    unknown_ip = _clamp_int(1 if rng.random() < 0.6 else 0, 0, 1)

    traffic_spike = 1 if rng.random() < 0.28 else 0
    repeated_requests = _clamp_int(rng.gauss(14, 7), 0, 50)
    session_duration = _clamp_int(rng.gauss(320, 160), 20, 1200)
    packet_size = _clamp_int(rng.gauss(6.3, 1.9), 1, 20)
    ip_rep = _clamp_int(rng.gauss(35, 25), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_insider_threat(rng: random.Random, session_id: int) -> TrafficRecord:
    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Insider threat: can look like legitimate traffic but with anomalous patterns:
    # - moderate request_rate with elevated session duration
    # - higher repeated_requests than normal
    # - unknown_ip can be 0 (internal) but ip_reputation lower
    request_rate = _clamp_int(rng.gauss(42, 14) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(2.5, 1.8), 0, 10)
    unknown_ip = _clamp_int(0 if rng.random() < 0.85 else 1, 0, 1)

    repeated_requests = _clamp_int(rng.gauss(16, 9), 0, 50)
    traffic_spike = 1 if rng.random() < 0.18 else 0
    session_duration = _clamp_int(rng.gauss(650, 220), 50, 1400)
    packet_size = _clamp_int(rng.gauss(7.2, 2.1), 1, 20)
    # Reputation is not necessarily terrible, but often slightly degraded for insiders.
    ip_rep = _clamp_int(rng.gauss(45, 22), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_unknown_behavior(rng: random.Random, session_id: int) -> TrafficRecord:

    tod = _sample_time_of_day(rng)
    mult = _seasonality_multiplier(tod)
    traffic_spike, session_duration, packet_size, repeated_requests, ip_rep = _base_session_features(rng, session_id)

    # Unknown behavior: a noisy blend that is suspicious but not as strongly indicative.
    request_rate = _clamp_int(rng.gauss(55, 25) * mult, 0, 100)
    failed_logins = _clamp_int(rng.gauss(4.0, 2.8), 0, 10)
    unknown_ip = 1

    traffic_spike = 1 if rng.random() < 0.45 else 0
    repeated_requests = _clamp_int(rng.gauss(12, 12), 0, 50)
    session_duration = _clamp_int(rng.gauss(260, 170), 5, 1200)
    packet_size = _clamp_int(rng.gauss(6.0, 3.0), 1, 20)
    ip_rep = _clamp_int(rng.gauss(40, 25), 0, 100)

    return TrafficRecord(
        request_rate=request_rate,
        failed_logins=failed_logins,
        unknown_ip=unknown_ip,
        time_of_day=tod,
        traffic_spike=traffic_spike,
        session_duration=session_duration,
        packet_size=packet_size,
        repeated_requests=repeated_requests,
        ip_reputation_score=ip_rep,
        session_id=session_id,
    )


def _sample_scenario(rng: random.Random, session_id: int) -> TrafficRecord:
    # Realistic-ish mix; attacks are less frequent than normal.
    p = rng.random()
    if p < 0.55:
        return _sample_normal(rng, session_id)
    if p < 0.64:
        return _sample_port_scan(rng, session_id)
    if p < 0.75:
        return _sample_bot_traffic(rng, session_id)
    if p < 0.83:
        return _sample_brute_force(rng, session_id)
    if p < 0.91:
        return _sample_credential_stuffing(rng, session_id)
    if p < 0.955:
        return _sample_insider_threat(rng, session_id)
    if p < 0.988:
        return _sample_ddos(rng, session_id)
    return _sample_unknown_behavior(rng, session_id)



def generate_traffic(
    n_samples: int, *, as_dataframe: bool = False, seed: Optional[int] = None
) -> List[Dict[str, int]] | "pd.DataFrame":
    """Generate advanced synthetic cybersecurity traffic logs.

    Each record includes:
      request_rate, failed_logins, unknown_ip, time_of_day,
      traffic_spike, session_duration, packet_size, repeated_requests,
      ip_reputation_score, session_id
    """

    if n_samples < 0:
        raise ValueError("n_samples must be >= 0")

    rng = random.Random(seed)
    rows: List[Dict[str, int]] = []

    for i in range(n_samples):
        rec = _sample_scenario(rng, session_id=100000 + i)
        rows.append(rec.to_dict())

    if as_dataframe:
        if pd is None:
            raise ImportError("pandas is required for as_dataframe=True")
        return pd.DataFrame(rows)

    return rows


if __name__ == "__main__":
    dataset = generate_traffic(20, as_dataframe=False, seed=42)
    print(f"Generated {len(dataset)} traffic records")
    for i, row in enumerate(dataset[:10]):
        print(f"{i+1}: {row}")


