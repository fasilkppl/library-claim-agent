"""Spine segmentation: one oriented box per visible book in a frame.

Two pretrained models, fused in code:
- FastSAM (class-agnostic segmentation) cuts tightly packed spines apart, including thin ones.
- YOLO11 'book' boxes are coarse (neighbours often merge) but say *where* books are, which rejects
  elongated non-book segments (shelf boards, table legs, door frames).
A spine = an elongated FastSAM segment that lies inside the YOLO book region.
"""

from __future__ import annotations

import itertools
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from claimagent.config import get_settings
from claimagent.pipeline.vision import Box, clean_spines


@dataclass
class Spine:
    cx: float
    cy: float
    length: float  # pixels along the spine (book height for an upright book)
    width: float  # pixels across the spine (book thickness)
    angle: float  # degrees of the long axis from vertical, -90..90
    orientation: str  # "upright" | "flat"
    conf: float
    source: str  # "sam+yolo" | "yolo"
    touches_edge: bool = False
    corners: list[list[float]] = field(default_factory=list)

    @property
    def box(self) -> Box:
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return Box(min(xs), min(ys), max(xs), max(ys), self.conf, "book")

    def to_dict(self) -> dict[str, object]:
        return {
            "cx": round(self.cx, 1),
            "cy": round(self.cy, 1),
            "length": round(self.length, 1),
            "width": round(self.width, 1),
            "angle": round(self.angle, 1),
            "orientation": self.orientation,
            "conf": round(self.conf, 3),
            "source": self.source,
            "touches_edge": self.touches_edge,
            "bbox": self.box.as_list(),
            "corners": [[round(x, 1), round(y, 1)] for x, y in self.corners],
        }


class Segmenter:
    _lock = threading.Lock()

    def __init__(self, model: str = "FastSAM-x.pt", device: str | None = None) -> None:
        from ultralytics import FastSAM

        s = get_settings()
        self.device = device or s.device
        self.model = FastSAM(str(Path(s.models_dir) / model))

    def masks(self, img: np.ndarray) -> list[np.ndarray]:
        with self._lock:
            r = self.model.predict(
                img, device=self.device, retina_masks=True, imgsz=1280, conf=0.25, iou=0.7, verbose=False
            )[0]
        if r.masks is None:
            return []
        return [m.astype(bool) for m in r.masks.data.cpu().numpy()]


def _rect_from_mask(
    mask: np.ndarray, step: int
) -> tuple[tuple[float, float], tuple[float, float], float, float] | None:
    """Oriented rectangle of a (downsampled) mask, returned in full-resolution pixels."""
    pts = cv2.findNonZero(mask.view(np.uint8))
    if pts is None or len(pts) * step * step < 150:
        return None
    (cx, cy), (a, b), ang = cv2.minAreaRect(pts)
    fill = len(pts) / max(1.0, a * b)
    s = float(step)
    return (cx * s + s / 2, cy * s + s / 2), (a * s, b * s), ang, fill


def _to_spine(
    center: tuple[float, float], size: tuple[float, float], ang: float, conf: float, source: str, shape: tuple[int, int]
) -> Spine:
    a, b = size
    corners = cv2.boxPoints((center, (a, b), ang)).tolist()
    # Long axis direction in degrees from vertical.
    long_is_a = a >= b
    theta = ang if long_is_a else ang + 90.0  # angle of the long side from the x-axis
    from_vertical = ((theta - 90.0 + 90.0) % 180.0) - 90.0
    length, width = (a, b) if long_is_a else (b, a)
    orient = "upright" if abs(from_vertical) <= 45 else "flat"
    h, w = shape
    margin = 3
    touches = any(x <= margin or y <= margin or x >= w - margin or y >= h - margin for x, y in corners)
    return Spine(
        cx=center[0],
        cy=center[1],
        length=length,
        width=width,
        angle=from_vertical,
        orientation=orient,
        conf=conf,
        source=source,
        touches_edge=touches,
        corners=corners,
    )


def book_region(yolo_books: list[Box], shape: tuple[int, int], grow: float = 0.06) -> np.ndarray:
    h, w = shape
    region = np.zeros((h, w), bool)
    for b in yolo_books:
        gx, gy = b.w * grow + 4, b.h * grow + 4
        x0, y0 = int(max(0, b.x0 - gx)), int(max(0, b.y0 - gy))
        x1, y1 = int(min(w, b.x1 + gx)), int(min(h, b.y1 + gy))
        region[y0:y1, x0:x1] = True
    return region


def segment_spines(masks: list[np.ndarray], yolo_books: list[Box], shape: tuple[int, int]) -> list[Spine]:
    """Fuse FastSAM masks with YOLO book boxes into one Spine per book."""
    h, w = shape
    # Geometry on downsampled masks: the oriented box changes by less than a pixel, the cost drops ~4x.
    step = max(1, int(np.ceil(max(h, w) / 960)))
    masks = [np.ascontiguousarray(m[::step, ::step]) for m in masks]
    region = book_region(yolo_books, shape)[::step, ::step]
    yolo_clean = clean_spines(yolo_books)
    heights = [b.h for b in yolo_clean if b.h > b.w] or [0.25 * h]
    typical = float(np.median(heights))

    candidates: list[tuple[Spine, np.ndarray]] = []
    for m in masks:
        rect = _rect_from_mask(m, step)
        if rect is None:
            continue
        center, size, ang, fill = rect
        long_side, short_side = max(size), min(size)
        if short_side < 4 or fill < 0.55:
            continue
        if long_side / short_side < 2.0:
            continue
        if long_side < 0.35 * typical or long_side > 1.8 * typical:
            continue  # shelf boards, wall edges and specks
        inside = float(region[m].mean())
        if inside < 0.6:
            continue
        sp = _to_spine(center, size, ang, 0.5 + 0.5 * inside, "sam+yolo", shape)
        # Thickness = the mask's mean width (area / length). The bounding rectangle's short side is the
        # widest point, which tilt, perspective and a ragged mask edge all inflate.
        sp.width = short_side * fill
        candidates.append((sp, m))

    spines = _merge_collinear(_dedupe(candidates), typical)
    # YOLO boxes that no segment explains are kept: FastSAM sometimes misses a low-contrast spine.
    for b in yolo_clean:
        if b.conf < 0.35 or max(b.w, b.h) / max(1.0, min(b.w, b.h)) < 2.0:
            continue
        if max(b.w, b.h) > 1.8 * typical:
            continue
        explained = any(_overlap_frac(s.box, b) > 0.3 or _overlap_frac(b, s.box) > 0.3 for s in spines)
        if not explained:
            size = (b.w, b.h)
            spines.append(_to_spine((b.cx, b.cy), size, 0.0, b.conf, "yolo", shape))
    return sorted(spines, key=lambda s: (round(s.cy / max(1.0, typical)), s.cx))


def _overlap_frac(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    return ix * iy / max(1.0, a.area)


def _dedupe(cands: list[tuple[Spine, np.ndarray]]) -> list[Spine]:
    """Nested or duplicate masks: keep the finest partition.

    A mask that covers two or more narrower accepted masks side by side is a merge of neighbours and is
    dropped. Two masks covering mostly the same pixels are the same spine; the larger one is kept.
    """
    cands = sorted(cands, key=lambda c: int(c[1].sum()))  # small first
    kept: list[tuple[Spine, np.ndarray, int]] = []
    for sp, m in cands:
        area = int(m.sum())
        inside = [k for k in kept if int((k[1] & m).sum()) > 0.6 * k[2]]
        if len(inside) >= 2 and _side_by_side([k[0] for k in inside], sp):
            continue  # this mask spans several neighbouring spines
        if inside:
            # Fragments of this spine (top/bottom pieces, a label patch): the full mask replaces them.
            for k in inside:
                kept.remove(k)
            kept.append((sp, m, area))
            continue
        if any(int((k[1] & m).sum()) > 0.5 * area for k in kept):
            continue
        kept.append((sp, m, area))
    return [k[0] for k in kept]


def _merge_collinear(spines: list[Spine], typical: float) -> list[Spine]:
    """Join pieces of one spine split along its length (a label band or a colour change can cut a mask):
    same orientation, nearly the same span across the spine, similar width, small gap along it."""
    spines = list(spines)
    merged = True
    while merged:
        merged = False
        for i in range(len(spines)):
            for j in range(i + 1, len(spines)):
                a, b = spines[i], spines[j]
                if a.orientation != b.orientation or abs(a.angle - b.angle) > 8:
                    continue
                ba, bb = a.box, b.box
                if a.orientation == "upright":
                    across = max(0.0, min(ba.x1, bb.x1) - max(ba.x0, bb.x0)) / max(1.0, min(ba.w, bb.w))
                    gap = max(ba.y0, bb.y0) - min(ba.y1, bb.y1)
                else:
                    across = max(0.0, min(ba.y1, bb.y1) - max(ba.y0, bb.y0)) / max(1.0, min(ba.h, bb.h))
                    gap = max(ba.x0, bb.x0) - min(ba.x1, bb.x1)
                if across < 0.7 or gap > 0.08 * typical or max(a.width, b.width) > 1.5 * min(a.width, b.width):
                    continue
                if a.length + b.length > 1.8 * typical:
                    continue
                pts = np.array(a.corners + b.corners, np.float32)
                (cx, cy), size, ang = cv2.minAreaRect(pts)
                joined = _to_spine((cx, cy), size, ang, max(a.conf, b.conf), a.source, (10**6, 10**6))
                joined.touches_edge = a.touches_edge or b.touches_edge
                spines[i] = joined
                del spines[j]
                merged = True
                break
            if merged:
                break
    return spines


def _side_by_side(parts: list[Spine], whole: Spine) -> bool:
    """True when the parts sit next to each other across the spine axis (different books)."""

    def across(s: Spine) -> tuple[float, float]:
        b = s.box
        return (b.x0, b.x1) if whole.orientation == "upright" else (b.y0, b.y1)

    spans = sorted(across(p) for p in parts)
    for (a0, a1), (b0, b1) in itertools.pairwise(spans):
        overlap = max(0.0, min(a1, b1) - max(a0, b0))
        if overlap < 0.5 * min(a1 - a0, b1 - b0):
            return True
    return False
