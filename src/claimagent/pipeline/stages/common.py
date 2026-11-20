"""Helpers shared by the stages."""

from __future__ import annotations

from claimagent.capture.store import FrameMeta
from claimagent.pipeline import geometry as geo

MARKER_SIDE_CM_DEFAULT = 10.0


def camera_of(m: FrameMeta) -> geo.Camera | None:
    """Metric camera of an AR frame; None for camera-only frames and frames whose AR pose was dropped."""
    if not m.pose or not m.intrinsics:
        return None
    return geo.Camera.from_meta(m.pose, m.intrinsics)
