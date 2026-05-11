"""soc_visuals.py

Lightweight visualization helpers for the enterprise SOC dashboard.

Constraints:
- Keep dependencies minimal.
- Return Streamlit/Plotly objects without starting background loops.
- Used only by dashboard.py (no module re-wiring).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


def rolling_avg(values: Sequence[float], window: int) -> List[float]:
    """Compute rolling average with a small warm-up."""
    if not values:
        return []
    w = max(1, int(window))
    out: List[float] = []
    for i in range(len(values)):
        start = max(0, i - w + 1)
        chunk = values[start : i + 1]
        out.append(float(sum(chunk) / len(chunk)))
    return out


def quantize_bucket(v: float, lo: float, hi: float, buckets: int) -> int:
    """Map v into bucket index [0..buckets-1]."""
    if hi <= lo:
        return 0
    x = max(lo, min(hi, float(v)))
    t = (x - lo) / (hi - lo)
    return min(buckets - 1, int(t * buckets))


def build_heatmap_matrix(intensity: Sequence[float], buckets: int = 10, width: int = 60) -> np.ndarray:
    """Convert rolling intensity history into a heatmap-like matrix.

    Matrix shape: [buckets, width]
    - y-axis: risk bucket
    - x-axis: time (latest values)

    Each column marks the intensity bucket for that time.
    """
    if not intensity:
        return np.zeros((buckets, width), dtype=float)

    vals = list(intensity)[-width:]
    if len(vals) < width:
        vals = [vals[0]] * (width - len(vals)) + vals

    mat = np.zeros((buckets, width), dtype=float)
    for x, v in enumerate(vals):
        b = quantize_bucket(v, lo=0.0, hi=1.0, buckets=buckets)
        mat[b, x] = 1.0
    return mat

