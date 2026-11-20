"""Capture-quality measurements. Plain image statistics, no model, so hints are never invented."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FrameQuality:
    sharpness: float  # detail sharpness: 90th percentile of tile Laplacian variances (see measure)
    glare: float  # share of near-white clipped pixels
    brightness: float  # mean grey 0-255

    def is_blurry(self, blur_min: float) -> bool:
        return self.sharpness < blur_min

    def has_glare(self, glare_max: float) -> bool:
        return self.glare > glare_max

    def is_dark(self) -> bool:
        return self.brightness < 45


def grey_640(img: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = 640 / grey.shape[1]
    if scale < 1:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return grey


def measure(img: np.ndarray, region: tuple[int, int, int, int] | None = None) -> FrameQuality:
    """region = (x0, y0, x1, y1) in image pixels, for example the union of detected spines."""
    if region is not None:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = region
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)  # boxes may extend past the frame
        if x1 - x0 > 32 and y1 - y0 > 32:
            img = img[y0:y1, x0:x1]
    grey = grey_640(img)
    sharpness = detail_sharpness(grey)
    glare = float(np.mean(grey >= 250))
    return FrameQuality(sharpness=sharpness, glare=glare, brightness=float(grey.mean()))


TILE_PX = 80  # tiles of the 640 px image
DETAIL_PERCENTILE = 90


def detail_sharpness(grey: np.ndarray) -> float:
    """Sharpness where the picture has detail. A plain wall or desk has a low Laplacian variance however well
    it is focused, so a whole-frame variance calls a frame with a lot of wall "blurry". Measured per tile, the
    detailed tiles (book edges, print) show the focus: the 90th percentile of the tile variances.
    Calibrated on real phone frames: in-focus frames 60-230, the same frames blurred (Gaussian sigma 1.5) 8-17."""
    lap = cv2.Laplacian(grey, cv2.CV_64F)
    h, w = lap.shape
    ny, nx = max(1, h // TILE_PX), max(1, w // TILE_PX)
    th, tw = h // ny, w // nx
    tiles = [lap[i * th : (i + 1) * th, j * tw : (j + 1) * tw].var() for i in range(ny) for j in range(nx)]
    return float(np.percentile(tiles, DETAIL_PERCENTILE))


def decode_jpeg(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode JPEG")
    out: np.ndarray = img
    return out


def thumb(img: np.ndarray, width: int = 320) -> np.ndarray:
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h = round(grey.shape[0] * width / grey.shape[1])
    small: np.ndarray = cv2.resize(grey, (width, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return small


def view_shift(a: np.ndarray, b: np.ndarray) -> float:
    """How far the view moved between two thumbnails, as a fraction of the frame width (phase correlation).
    Returns 1.0 when the frames are unrelated or differ in size."""
    if a.shape != b.shape:
        return 1.0
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    if response < 0.05:
        return 1.0
    return float(np.hypot(dx, dy) / a.shape[1])
