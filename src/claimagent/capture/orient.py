"""Turn each captured frame upright, however the phone was held.

A phone held sideways (screen rotation locked) gives frames on their side: standing books look like lying
ones, the text runs vertically, and every rule that relies on "up" goes wrong. Gravity says where up is: in AR
mode from the camera pose (ARCore's orientation comes from the gyroscope and stays right even when its
position and depth are not), in camera-only mode from the phone's motion sensor.

Frames are turned in quarter turns (numpy.rot90, counter-clockwise). The camera geometry turns with them:
intrinsics, the depth grid and the pose's camera axes, so pixel <-> 3D relations stay exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

MIN_TILT = 0.35  # looking almost straight down or up, "up" is undefined in the image: keep the last turn


def turns_from_pose(pose: Sequence[float]) -> int | None:
    """Quarter turns (counter-clockwise) that bring world-up to the top of the image. WebXR camera space:
    +x image right, +y image up, looking down -z; pose is camera-to-world, column-major."""
    rot = np.array(pose, np.float64).reshape(4, 4).T[:3, :3]
    up = rot.T @ np.array([0.0, 1.0, 0.0])  # world up in camera coordinates
    return _turns(float(up[0]), float(up[1]))


def turns_from_gravity(g: Sequence[float]) -> int | None:
    """Same from the phone's accelerationIncludingGravity (device axes: +x screen right, +y screen top); at rest
    it points away from the ground, so it is world-up in screen coordinates."""
    v = np.array([np.nan if x is None else x for x in g[:3]], np.float64)
    if v.size < 3 or not np.all(np.isfinite(v)):
        return None  # a sensor that reports no value on an axis gives no usable "up"
    norm = float(np.linalg.norm(v)) or 1.0
    return _turns(float(v[0]) / norm, float(v[1]) / norm)


def _turns(right: float, up: float) -> int | None:
    if max(abs(right), abs(up)) < MIN_TILT:
        return None
    if abs(up) >= abs(right):
        return 0 if up >= 0 else 2
    return 1 if right > 0 else 3  # up is at the image's right: one counter-clockwise turn brings it to the top


# Camera axes after one counter-clockwise image turn, in the old camera's coordinates (columns): the new
# right is the old down (-y), the new up is the old right (+x), the viewing axis is unchanged.
_QUARTER = np.array([[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def rotate_frame(
    img: np.ndarray,
    k: int,
    depth: np.ndarray | None = None,
    pose: Sequence[float] | None = None,
    intrinsics: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[float] | None, dict[str, Any] | None]:
    k %= 4
    if k == 0:
        return img, depth, list(pose) if pose is not None else None, intrinsics
    img = np.ascontiguousarray(np.rot90(img, k))
    if depth is not None:
        depth = np.ascontiguousarray(np.rot90(depth, k))
    new_pose = None
    if pose is not None:
        m = np.array(pose, np.float64).reshape(4, 4).T
        for _ in range(k):
            m = m @ _QUARTER
        new_pose = m.T.reshape(-1).tolist()
    if intrinsics is not None:
        intr = dict(intrinsics)
        for _ in range(k):
            w = float(intr["width"])
            intr = {
                **intr,
                "fx": intr["fy"],
                "fy": intr["fx"],
                "cx": intr["cy"],
                "cy": w - float(intr["cx"]),
                "width": intr["height"],
                "height": intr["width"],
            }
        intrinsics = intr
    return img, depth, new_pose, intrinsics
