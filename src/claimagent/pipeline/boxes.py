"""Axis-aligned boxes (x0, y0, x1, y1) in pixels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def box_iou(a: Sequence[float] | np.ndarray[Any, Any], b: Sequence[float] | np.ndarray[Any, Any]) -> float:
    """Intersection over union of two boxes; 0 when either box is missing or empty."""
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
