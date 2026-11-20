"""Metric geometry from WebXR (ARCore) data: camera pose, intrinsics and a view-aligned depth grid.

Conventions (WebXR):
- pose = camera-to-world 4x4, column-major, in the 'local-floor' reference space (metres, +Y up).
- camera looks down -Z; image x grows right, image y grows down.
- depth grid: depth[i, j] = metres along the camera axis at normalised view coords ((j+.5)/W, (i+.5)/H),
  origin top-left (the browser samples XRCPUDepthInformation.getDepthInMeters on this grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

M2_TO_FT2 = 10.7639


@dataclass
class Camera:
    pose: np.ndarray  # 4x4 camera-to-world
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_meta(cls, pose: list[float], intr: dict[str, float]) -> Camera:
        M = np.array(pose, dtype=np.float64).reshape(4, 4).T  # column-major -> row-major
        return cls(M, intr["fx"], intr["fy"], intr["cx"], intr["cy"], int(intr["width"]), int(intr["height"]))

    @property
    def position(self) -> np.ndarray:
        return self.pose[:3, 3]

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (u - self.cx) / self.fx * z
        y = -(v - self.cy) / self.fy * z
        pts = np.stack([x, y, -z, np.ones_like(z)], axis=-1)
        world: np.ndarray = pts @ self.pose.T
        return world[..., :3]


def sample_depth(depth: np.ndarray, cam: Camera, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear depth at image pixels (u, v)."""
    gh, gw = depth.shape
    gx = np.clip(u / cam.width * gw - 0.5, 0, gw - 1.001)
    gy = np.clip(v / cam.height * gh - 0.5, 0, gh - 1.001)
    x0, y0 = gx.astype(int), gy.astype(int)
    fx, fy = gx - x0, gy - y0
    d00, d01 = depth[y0, x0], depth[y0, x0 + 1]
    d10, d11 = depth[y0 + 1, x0], depth[y0 + 1, x0 + 1]
    out: np.ndarray = d00 * (1 - fx) * (1 - fy) + d01 * fx * (1 - fy) + d10 * (1 - fx) * fy + d11 * fx * fy
    return out


def depth_in_polygon(depth: np.ndarray, cam: Camera, corners: list[list[float]]) -> float | None:
    """Robust depth of a flat object (a spine, a frame) from grid cells inside its image polygon."""
    gh, gw = depth.shape
    poly = np.array([[x / cam.width * gw, y / cam.height * gh] for x, y in corners], np.float32)
    mask = np.zeros((gh, gw), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    vals = depth[(mask > 0) & (depth > 0.15) & (depth < 8.0)]
    if vals.size == 0:
        cx, cy = poly.mean(axis=0)
        d = float(sample_depth(depth, cam, np.array([cx / gw * cam.width]), np.array([cy / gh * cam.height]))[0])
        return d if 0.15 < d < 8.0 else None
    # The front face is the near mode; background behind thin spines leaks into far cells.
    return float(np.percentile(vals, 35))


def metric_extent(cam: Camera, z: float, p0: tuple[float, float], p1: tuple[float, float]) -> float:
    """Distance in metres between two image points lying on a fronto-parallel plane at depth z."""
    a = cam.backproject(np.array([p0[0]]), np.array([p0[1]]), np.array([z]))[0]
    b = cam.backproject(np.array([p1[0]]), np.array([p1[1]]), np.array([z]))[0]
    return float(np.linalg.norm(a - b))


# ----------------------------------------------------------------------------- room
@dataclass
class RoomResult:
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    wall_area_m2: float | None = None
    perimeter_m: float | None = None
    shape_method: str = ""
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)
    points_used: int = 0
    polygon: list[list[float]] = field(default_factory=list)


def frame_points(cam: Camera, depth: np.ndarray, near: float = 0.3, far: float = 7.0) -> np.ndarray:
    gh, gw = depth.shape
    jj, ii = np.meshgrid(np.arange(gw), np.arange(gh))
    u = (jj + 0.5) / gw * cam.width
    v = (ii + 0.5) / gh * cam.height
    z = depth
    ok = (z > near) & (z < far) & np.isfinite(z)
    return cam.backproject(u[ok], v[ok], z[ok])


def _peak(values: np.ndarray, lo: float, hi: float, bin_m: float = 0.04) -> tuple[float | None, int]:
    sel = values[(values >= lo) & (values <= hi)]
    if sel.size < 50:
        return None, int(sel.size)
    bins = np.arange(lo, hi + bin_m, bin_m)
    hist, edges = np.histogram(sel, bins)
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2), int(hist[k])


def _dominant_angle(xz: np.ndarray, cell: float = 0.05) -> float:
    """Rotation (radians) that best aligns walls with the axes (rooms are mostly rectilinear)."""
    best, best_score = 0.0, -1.0
    for deg in np.arange(0, 90, 1.0):
        t = np.radians(deg)
        rot = xz @ np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        score = 0.0
        for k in range(2):
            h, _ = np.histogram(rot[:, k], bins=np.arange(rot[:, k].min(), rot[:, k].max() + cell, cell))
            p = h / max(1, h.sum())
            score += float((p**2).sum())
        if score > best_score:
            best, best_score = float(t), score
    return best


def measure_room(points: np.ndarray, cam_positions: np.ndarray, cell: float = 0.05) -> RoomResult:
    res = RoomResult(points_used=len(points))
    if len(points) < 2000:
        res.notes.append("too few depth points for a room measurement")
        return res
    y = points[:, 1]
    floor_y, floor_n = _peak(y, float(np.percentile(y, 0.5)) - 0.1, float(np.percentile(y, 0.5)) + 0.6)
    if floor_y is None:
        floor_y = 0.0
        res.notes.append("floor plane not seen; used ARCore local-floor height 0")
    top = float(np.percentile(y, 99.7))
    ceil_y, ceil_n = _peak(y, max(floor_y + 1.8, top - 0.6), top + 0.1)
    ceiling_seen = ceil_y is not None and ceil_n > 150
    if not ceiling_seen:
        ceil_y = float(np.percentile(y, 99.5))
        res.notes.append("ceiling not captured; height is the highest wall point seen (a lower bound)")
    assert ceil_y is not None
    height = ceil_y - floor_y

    band = points[(y > floor_y + 0.25) & (y < ceil_y - 0.25)]
    xz = band[:, [0, 2]]
    if len(xz) < 1000:
        res.notes.append("walls not captured")
        res.height_m = round(height, 2)
        return res
    t = _dominant_angle(xz[:: max(1, len(xz) // 20000)])
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    pts = xz @ R
    cams = cam_positions[:, [0, 2]] @ R
    lo = np.percentile(pts, 1.0, axis=0) - 0.3
    hi = np.percentile(pts, 99.0, axis=0) + 0.3
    nx, nz = (np.ceil((hi - lo) / cell)).astype(int) + 1
    grid = np.zeros((nz, nx), np.int32)
    ij = np.floor((pts - lo) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < nz)
    np.add.at(grid, (ij[ok, 1], ij[ok, 0]), 1)
    # Depth noise spreads each wall into a band of cells; the wall surface is the dense ridge of that band.
    g = grid.astype(np.float32)
    local_max = cv2.dilate(g, np.ones((7, 7), np.uint8))
    min_count = max(2, int(np.percentile(grid[grid > 0], 40)))
    walls = ((g >= 0.6 * local_max) & (grid >= min_count)).astype(np.uint8)
    dil = 2
    walls = cv2.dilate(walls, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    free = (1 - walls).astype(np.uint8)
    filled = np.zeros_like(free)
    for c in cams:
        j, i = np.floor((c - lo) / cell).astype(int)
        if 0 <= i < nz and 0 <= j < nx and free[i, j] and not filled[i, j]:
            mask = np.zeros((nz + 2, nx + 2), np.uint8)
            tmp = free.copy()
            cv2.floodFill(tmp, mask, (int(j), int(i)), 2)
            filled |= (tmp == 2).astype(np.uint8)
    if filled.sum() == 0:
        res.notes.append("camera path not inside a closed outline")
        res.height_m = round(height, 2)
        return res
    touches = filled[0].any() or filled[-1].any() or filled[:, 0].any() or filled[:, -1].any()
    if touches:
        res.notes.append("outline open (doorway or unseen wall); clipped to the extent of seen walls")
    # Grow back over the dilation ring onto the wall cells; the contour then runs through the centres
    # of the wall cells, which is where the wall surface points are.
    grow = dil + 1
    filled = cv2.dilate(filled, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 2.0, True).reshape(-1, 2).astype(np.float64)
    poly_m = (poly + 0.5) * cell
    area = float(cv2.contourArea(poly_m.astype(np.float32)))
    perim = float(cv2.arcLength(poly_m.astype(np.float32), True))
    (_, _), (a, b), _ = cv2.minAreaRect(poly_m.astype(np.float32))
    rectangularity = area / max(1e-6, a * b)
    res.length_m, res.width_m = round(max(a, b), 2), round(min(a, b), 2)
    res.height_m = round(height, 2)
    res.floor_area_m2 = round(area, 2)
    res.perimeter_m = round(perim, 2)
    res.wall_area_m2 = round(perim * height, 2)
    res.polygon = [[round(float(x), 2), round(float(z), 2)] for x, z in poly_m]
    if rectangularity >= 0.93:
        res.shape_method = "rectangular: floor = outline area of the depth point cloud (length x width shown)"
    else:
        res.shape_method = (
            f"non-rectangular ({len(poly)} corners): floor area from the traced outline polygon "
            "(shoelace), length/width are its bounding rectangle; wall area = perimeter x height"
        )
    conf = 0.9
    if not ceiling_seen:
        conf -= 0.25
    if touches:
        conf -= 0.2
    if floor_n < 200:
        conf -= 0.1
    res.confidence = round(max(0.1, conf), 2)
    return res
