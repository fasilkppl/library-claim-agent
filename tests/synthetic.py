"""Synthetic WebXR data for tests: a phone walking inside a room of known size.

Depth grids are ray-cast analytically against the room's floor, ceiling and walls, using the same
conventions the browser sends (camera-to-world pose, column-major; camera looks down -Z).
"""

from __future__ import annotations

import numpy as np


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = target - position
    fwd /= np.linalg.norm(fwd)
    z = -fwd
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, position
    return M


def raycast_depth(
    M: np.ndarray,
    polygon: list[tuple[float, float]],
    height: float,
    fx: float,
    fy: float,
    w: int,
    h: int,
    gw: int = 64,
    gh: int = 48,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    u = (jj + 0.5) / gw * w
    v = (ii + 0.5) / gh * h
    dc = np.stack([(u - w / 2) / fx, -(v - h / 2) / fy, -np.ones_like(u)], -1)  # z-depth 1 along -Z
    dw = dc @ M[:3, :3].T
    o = M[:3, 3]
    best = np.full(u.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for plane_y in (0.0, height):
            t = (plane_y - o[1]) / dw[..., 1]
            best = np.where((t > 0) & (t < best), t, best)
        n = len(polygon)
        for k in range(n):
            (x0, z0), (x1, z1) = polygon[k], polygon[(k + 1) % n]
            ex, ez = x1 - x0, z1 - z0
            den = dw[..., 0] * ez - dw[..., 2] * ex
            t = ((x0 - o[0]) * ez - (z0 - o[2]) * ex) / den
            s = ((x0 - o[0]) * dw[..., 2] - (z0 - o[2]) * dw[..., 0]) / den
            hit_y = o[1] + t * dw[..., 1]
            ok = (t > 0) & (s >= 0) & (s <= 1) & (hit_y >= 0) & (hit_y <= height)
            best = np.where(ok & (t < best), t, best)
    depth = best  # t is already z-depth because the camera-space ray has z = -1
    if noise and rng is not None:
        depth = depth * (1 + rng.normal(0, noise, depth.shape))
    return depth.astype(np.float32)


def walk(
    polygon: list[tuple[float, float]], height: float, n: int = 120, seed: int = 0, noise: float = 0.01
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Phone held at 1.4 m: turn once on the spot at each of a few stations, walking between them at a
    sweep's pace (at most STEP_M per keyframe), with some up/down tilt."""
    step_m = 0.3  # per 0.7 s keyframe: about 0.4 m/s, a slow walk
    rng = np.random.default_rng(seed)
    poly = np.array(polygon)
    centre = poly.mean(axis=0)
    stations = [centre + (p - centre) * 0.45 for p in poly] + [centre]
    positions: list[tuple[np.ndarray, float]] = []  # (xz position, heading)
    turn = max(4, n // len(stations))
    for i, st in enumerate(stations):
        if i:
            prev = stations[i - 1]
            steps = int(np.ceil(np.linalg.norm(st - prev) / step_m))
            for s in range(1, steps):
                positions.append((prev + (st - prev) * s / steps, float(np.arctan2(*(st - prev)[::-1]))))
        for k in range(turn):
            positions.append((st, 2 * np.pi * k / turn + float(rng.normal(0, 0.05))))
    out = []
    fx = fy = 480.0
    for k, (xz, ang) in enumerate(positions):
        pos = np.array([xz[0], 1.4, xz[1]])
        pitch = [-0.6, 0.0, 0.9][k % 3]
        target = pos + np.array([np.cos(ang), pitch, np.sin(ang)])
        M = look_at(pos, target)
        out.append((M, raycast_depth(M, polygon, height, fx, fy, 640, 480, noise=noise, rng=rng)))
    return out


INTRINSICS = {"fx": 480.0, "fy": 480.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}
