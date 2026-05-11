"""feature_engineering.py

Feature engineering and feature selection helpers for real dataset ingestion.

The project ultimately needs consistent feature columns for training the
supervised attack classifier.

This module provides:
- feature whitelisting based on common CIC-IDS2017 flow-derived fields
- fallback behavior when columns differ

Important:
- We do NOT change RL environment/discretization yet; this module is for
  ML feature engineering.
"""

from __future__ import annotations

from typing import List, Optional


# Common CIC-IDS2017 / flow-like features (best-effort names)
CIC_COMMON_FEATURES: List[str] = [
    # Basic flow duration
    "Flow Duration",
    "flow_duration",
    # Packet counts
    "Total Fwd Packets",
    "total_fwd_packets",
    "Total Backward Packets",
    "total_backward_packets",
    # Byte/Rate
    "Total Length of Fwd Packets",
    "total_fwd_bytes",
    "Flow Bytes/s",
    "flow_bytes_per_second",
    # Packet length-ish
    "Packet Length Mean",
    "packet_length_mean",
    "Packet Length Std",
    "packet_length_std",
    # Ports / flags
    "Destination Port",
    "destination_port",
    # SYN-related
    "Fwd Header Length.1",
    "SYN Flag Count",
    "syn_flag_count",
    # Login failures proxy
    "Sub. category",
    # Request/response rates proxies (varies by dataset)
    "Fwd Packet Length Mean",
    "bwd_packet_length_max",
    # Misc
    "Avg Packet Size",
    "avg_packet_size",
]


def choose_feature_columns(
    df_columns: List[str],
    *,
    requested: Optional[List[str]] = None,
    fallback_to_all_numeric: bool = True,
) -> List[str]:
    """Pick feature columns from available columns.

    If requested provided, use intersection with df_columns.
    Otherwise use a best-effort whitelist of flow-like fields.

    If resulting list is empty and fallback_to_all_numeric is True,
    caller should handle numeric-only columns.
    """

    cols_set = set(df_columns)

    if requested:
        picked = [c for c in requested if c in cols_set]
        return picked

    # whitelist intersection
    picked = []
    for c in CIC_COMMON_FEATURES:
        if c in cols_set:
            picked.append(c)

    return picked


def normalize_attack_class_order() -> List[str]:
    """Stable class order used for metrics/outputs."""
    return [
        "DDoS",
        "Brute Force",
        "Port Scan",
        "Botnet",
        "Web Attack",
        "Infiltration",
        "Benign Traffic",
    ]

