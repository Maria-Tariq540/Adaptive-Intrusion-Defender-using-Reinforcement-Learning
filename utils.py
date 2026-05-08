"""utils.py

Purpose:
    Shared utility functions used across the project.

Notes:
    - Seed helpers
    - CSV logging helpers for predictions
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional




def risk_score_to_levels(risk_score: int) -> tuple[str, str]:
    """Map 0..100 risk to (risk_label, severity_level).

    Severity tiers required by Phase 1:
      - LOW
      - MEDIUM
      - HIGH
      - CRITICAL

    Returns:
        (risk_label, severity_level)
    """
    rs = int(max(0, min(100, risk_score)))
    if rs <= 30:
        return ("Safe", "LOW")
    if rs <= 60:
        return ("Suspicious", "MEDIUM")
    if rs <= 85:
        return ("Dangerous", "HIGH")
    return ("Critical", "CRITICAL")




def compute_risk_score(
    *,
    request_rate: int,
    failed_logins: int,
    unknown_ip: int,
    time_of_day: Optional[int] = None,
    traffic_spike: Optional[int] = None,
    time_window: Optional[int] = None,
    agent_action: Optional[int] = None,
) -> int:
    """Compute numeric risk score (0-100) as a single source of truth.

    Backward compatible: works whether the caller only knows legacy features
    (request_rate, failed_logins, unknown_ip, time_of_day) or also has newer
    features (traffic_spike, time_window).
    """

    risk = 0.0

    # Failed logins is the strongest brute-force indicator.
    risk += (float(failed_logins) / 10.0) * 55.0  # up to 55

    # High request rate contributes to volumetric abuse / load.
    risk += (float(request_rate) / 100.0) * 30.0  # up to 30

    # Unknown IP adds suspicion.
    risk += (1.0 if int(unknown_ip) == 1 else 0.0) * 20.0  # up to 20

    # Optional additional features.
    if traffic_spike is not None:
        risk += (1.0 if int(traffic_spike) == 1 else 0.0) * 10.0

    # time_window can represent repeating patterns; keep small bias.
    if time_window is not None:
        risk += (float(int(time_window)) / 10.0) * 5.0

    # Small bias if model decided to block (confidence proxy from policy).
    if agent_action is not None:
        risk += 5.0 if int(agent_action) == 1 else 0.0

    # Clamp 0..100
    risk_int = int(max(0, min(100, round(risk))))
    return risk_int



def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility (best-effort)."""
    random.seed(seed)



def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


def append_dicts_to_csv(
    rows: Iterable[Mapping[str, Any]],
    *,
    csv_path: str | os.PathLike,
    fieldnames: List[str],
) -> None:
    """Append multiple dict rows to a CSV file.

    - Creates the file and writes header if missing.
    - Missing keys are written as empty string.
    """
    path = Path(csv_path)
    _ensure_parent_dir(path)

    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def safe_append_prediction_to_csv(
    row: Mapping[str, Any],
    *,
    csv_path: str | os.PathLike,
    fieldnames: List[str],
    buffer: list[Mapping[str, Any]] | None = None,
    flush_every: int = 25,
) -> None:
    """Training-safe CSV logger.

    Constraints:
    - never raises (swallows all exceptions)
    - supports buffering to avoid per-step disk I/O
    - flushes when buffer reaches flush_every

    If buffer is None: performs a single append using the existing implementation,
    still swallowing any exceptions.
    """

    try:
        if buffer is None:
            append_prediction_to_csv(row, csv_path=csv_path, fieldnames=fieldnames)
            return

        # Normalize all values to single-field CSV-friendly strings/ints to
        # avoid accidental extra columns when values contain separators.
        safe_row = {k: ("" if v is None else str(v).replace("\n", " ") ) for k, v in dict(row).items()}
        buffer.append(safe_row)

        if len(buffer) >= int(flush_every):
            append_dicts_to_csv(buffer, csv_path=csv_path, fieldnames=fieldnames)
            buffer.clear()
    except Exception:
        # Logging must NEVER crash or slow training.
        return


def flush_prediction_buffer(
    buffer: list[Mapping[str, Any]] | None,
    *,
    csv_path: str | os.PathLike,
    fieldnames: List[str],
) -> None:
    """Flush buffered prediction rows (best-effort, never raises)."""
    if not buffer:
        return
    try:
        append_dicts_to_csv(buffer, csv_path=csv_path, fieldnames=fieldnames)
        buffer.clear()
    except Exception:
        return


def append_prediction_to_csv(
    row: Mapping[str, Any],
    *,
    csv_path: str | os.PathLike,
    fieldnames: List[str],
) -> None:
    """Append a single prediction row to a CSV file."""
    append_dicts_to_csv([row], csv_path=csv_path, fieldnames=fieldnames)



