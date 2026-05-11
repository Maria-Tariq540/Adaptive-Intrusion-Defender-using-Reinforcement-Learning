"""rl_state_builder.py

Bridges supervised ML detector outputs and engineered traffic features
into the existing RL tabular state representation used by environment.py.

Current RL state in environment.py is:
  (request_rate_bin, failed_logins_bin, unknown_ip, time_of_day_bin,
   traffic_spike, repeated_requests_bin, ip_reputation_bin, session_duration_bin)

This module provides best-effort conversion from raw engineered features
(typical CIC-IDS / flow features) into the numeric signals expected by the
bins.

Important:
- This does NOT alter RL internals.
- It just maps values to approximate ranges so Q-learning can operate.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def build_rl_state_from_ml_features(
    *,
    features: Dict[str, Any],
) -> Tuple[int, int, int, int, int, int, int, int]:
    """Convert a feature dict into RL environment state tuple.

    Because environment.py discretization is internal, we approximate the
    raw values and then let environment.py discretize when needed.

    Here we produce raw-ish values and then apply the same binning
    logic used in environment.py by calling equivalent discretization.

    To avoid importing environment.py (circular), we replicate bin cutoffs.
    """

    # --- Raw-ish proxy extraction ---
    flow_duration = float(features.get("flow_duration", features.get("Flow Duration", 0.0)) or 0.0)
    total_fwd_packets = float(features.get("total_fwd_packets", features.get("Total Fwd Packets", 0.0)) or 0.0)
    total_bwd_packets = float(features.get("total_backward_packets", features.get("Total Backward Packets", 0.0)) or 0.0)
    pkt_len_mean = float(features.get("packet_length_mean", features.get("Packet Length Mean", 0.0)) or 0.0)
    flow_bytes_per_second = float(features.get("flow_bytes_per_second", features.get("Flow Bytes/s", 0.0)) or 0.0)
    destination_port = float(features.get("destination_port", features.get("Destination Port", 0.0)) or 0.0)

    syn_flag_count = float(features.get("syn_flag_count", features.get("SYN Flag Count", 0.0)) or 0.0)

    # unknown_ip and failed logins might not exist in real dataset -> derive best-effort
    # failed_logins proxy: if any auth-related failed login feature exists.
    failed_logins = 0
    for k in [
        "failed_logins",
        "Failed Login",
        "Sub. category",  # sometimes used in demo/synthetic
        "login_failures",
        "failed_login",
    ]:
        if k in features:
            failed_logins = _safe_int(features.get(k), 0)
            break

    unknown_ip = _safe_int(features.get("unknown_ip", features.get("Unknown IP", 0)), 0)

    time_of_day = _safe_int(features.get("time_of_day", features.get("Time of Day", 0)), 0)

    traffic_spike = 1 if (syn_flag_count >= 10) or (flow_bytes_per_second > 1_000_000) else 0

    repeated_requests = _safe_int(features.get("repeated_requests", features.get("repeated_requests_count", 0)), 0)

    ip_reputation_score = _safe_int(features.get("ip_reputation_score", features.get("IP reputation", 50)), 50)

    session_duration = int(max(0, round(flow_duration)))

    # request_rate proxy (normalized into 0..100)
    # Use fwd packet rate proxy as a stand-in.
    # request_rate grows with total_fwd_packets and flow_bytes_per_second.
    rr = 0.0
    if flow_duration > 0:
        rr = (total_fwd_packets / max(1.0, flow_duration)) * 60.0
    else:
        rr = total_fwd_packets * 0.5

    request_rate = int(max(0, min(100, round(rr))))

    # --- Discretization cutoffs mirrored from environment.py ---
    def _bin(v: int, *, bins):
        for i, ub in enumerate(bins):
            if v <= ub:
                return i
        return len(bins)

    rr_bin = _bin(request_rate, bins=[15, 30, 45, 60, 80])
    fl_bin = _bin(failed_logins, bins=[1, 3, 5, 7, 9])
    tod_bin = _bin(time_of_day, bins=[4, 9, 14, 19, 23])
    rep_bin = _bin(repeated_requests, bins=[3, 8, 15, 22, 30])
    ip_rep_bin = _bin(ip_reputation_score, bins=[25, 40, 55, 70, 85])
    sess_bin = _bin(session_duration, bins=[80, 170, 300, 450, 650])

    return (
        int(rr_bin),
        int(fl_bin),
        int(unknown_ip),
        int(tod_bin),
        int(traffic_spike),
        int(rep_bin),
        int(ip_rep_bin),
        int(sess_bin),
    )

