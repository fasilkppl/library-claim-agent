import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from claimagent.pipeline.geometry import Camera, frame_points, measure_room, metric_extent
from synthetic import INTRINSICS, walk


def _measure(poly, h, noise=0.02):
    pts, cams = [], []
    for M, d in walk(poly, h, noise=noise):
        cam = Camera.from_meta(M.T.reshape(-1).tolist(), INTRINSICS)
        pts.append(frame_points(cam, d))
        cams.append(cam.position)
    return measure_room(np.concatenate(pts), np.array(cams))


def _area(poly):
    xs = np.array(poly)
    return 0.5 * abs(np.dot(xs[:, 0], np.roll(xs[:, 1], 1)) - np.dot(xs[:, 1], np.roll(xs[:, 0], 1)))


def test_rectangular_room_within_bars():
    poly = [(0, 0), (4.2, 0), (4.2, 3.5), (0, 3.5)]
    r = _measure(poly, 2.6)
    assert abs(r.floor_area_m2 / _area(poly) - 1) < 0.05
    assert abs(r.wall_area_m2 / (15.4 * 2.6) - 1) < 0.05
    assert abs(r.height_m - 2.6) < 0.05 and r.shape_method.startswith("rectangular")


def test_l_shaped_room_uses_polygon():
    poly = [(0, 0), (5, 0), (5, 2.5), (2.5, 2.5), (2.5, 4), (0, 4)]
    r = _measure(poly, 2.7)
    assert abs(r.floor_area_m2 / _area(poly) - 1) < 0.05
    assert r.shape_method.startswith("non-rectangular")


def test_metric_extent_at_known_depth():
    cam = Camera(np.eye(4), 500, 500, 320, 240, 640, 480)
    assert abs(metric_extent(cam, 2.0, (320, 140), (320, 340)) - 0.8) < 1e-6
