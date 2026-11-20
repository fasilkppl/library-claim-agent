"""Vision primitives: text recognition (Apple Vision) and object detection (YOLO).

Both sit behind small Protocols so stages can be tested with fakes and swapped for other models.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from claimagent.config import get_settings
from claimagent.pipeline.household import COCO_ITEMS, ITEM_VOCAB


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 1.0
    label: str = ""

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    def as_list(self) -> list[float]:
        return [round(self.x0, 1), round(self.y0, 1), round(self.x1, 1), round(self.y1, 1)]

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)

    def contains_point(self, x: float, y: float, pad: float = 0.0) -> bool:
        return self.x0 - pad <= x <= self.x1 + pad and self.y0 - pad <= y <= self.y1 + pad


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x1, b.x1) - max(a.x0, b.x0))
    iy = max(0.0, min(a.y1, b.y1) - max(a.y0, b.y0))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


@dataclass
class TextLine:
    text: str
    conf: float
    box: Box
    vertical: bool  # text runs top-bottom or bottom-top (an upright spine)
    words: list[tuple[str, Box]] = field(default_factory=list)  # per-word boxes, to split cross-spine lines


class TextReader(Protocol):
    def read(self, img: np.ndarray) -> list[TextLine]: ...


class Detector(Protocol):
    def books(self, img: np.ndarray) -> list[Box]: ...

    def objects(self, img: np.ndarray) -> list[Box]: ...


# --------------------------------------------------------------------------- OCR
class AppleVisionReader:
    """macOS Vision framework text recognition. Runs on the Neural Engine, ~0.2 s per 1080p frame."""

    def __init__(self, languages: list[str] | None = None) -> None:
        import Vision  # noqa: F401  (fails fast off macOS)

        self.languages = languages

    def read(self, img: np.ndarray) -> list[TextLine]:
        import Vision
        from Foundation import NSData

        h, w = img.shape[:2]
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return []
        data = NSData.dataWithBytes_length_(buf.tobytes(), len(buf))
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(False)
        req.setMinimumTextHeight_(0.008)
        if self.languages:
            req.setRecognitionLanguages_(self.languages)
        handler.performRequests_error_([req], None)
        lines: list[TextLine] = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if not cand:
                continue
            c = cand[0]
            text = str(c.string()).strip()
            if not text:
                continue
            # Vision corners are normalised with the origin at the bottom-left.
            corners = [obs.topLeft(), obs.topRight(), obs.bottomRight(), obs.bottomLeft()]
            pts = [(p.x * w, (1 - p.y) * h) for p in corners]
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
            angle = abs(math.degrees(math.atan2(dy, dx))) % 180
            vertical = 45 < angle < 135
            words: list[tuple[str, Box]] = []
            for m in re.finditer(r"\S+", text):
                rect, _ = c.boundingBoxForRange_error_((m.start(), m.end() - m.start()), None)
                if rect is None:
                    continue
                wc = [rect.topLeft(), rect.topRight(), rect.bottomRight(), rect.bottomLeft()]
                wx, wy = [q.x * w for q in wc], [(1 - q.y) * h for q in wc]
                words.append((m.group(0), Box(min(wx), min(wy), max(wx), max(wy))))
            lines.append(
                TextLine(
                    text=text,
                    conf=float(c.confidence()),
                    box=Box(min(xs), min(ys), max(xs), max(ys)),
                    vertical=vertical,
                    words=words,
                )
            )
        return lines


# --------------------------------------------------------------------------- detection
# Open-vocabulary prompts for things in a home library that are not books.


class YoloDetector:
    """YOLO11x (COCO 'book' class) for spines, YOLO-World for open-vocabulary items. Apple MPS."""

    _lock = threading.Lock()

    def __init__(
        self, books_model: str | None = None, world_model: str | None = None, device: str | None = None
    ) -> None:
        from ultralytics import YOLO, YOLOWorld

        s = get_settings()
        models_dir = Path(s.models_dir)
        self.device = device or s.device
        self.books_model = YOLO(str(models_dir / (books_model or s.yolo_books_model)))
        self.world = YOLOWorld(str(models_dir / (world_model or s.yolo_world_model)))
        self.world.set_classes(ITEM_VOCAB)

    def books(self, img: np.ndarray) -> list[Box]:
        with self._lock:
            r = self.books_model.predict(img, conf=0.15, imgsz=1600, device=self.device, verbose=False, max_det=600)[0]
        out: list[Box] = []
        for (x0, y0, x1, y1), c, k in zip(
            r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist(), strict=True
        ):
            name = r.names[int(k)]
            if name == "book":
                out.append(Box(x0, y0, x1, y1, float(c), "book"))
        return out

    def objects(self, img: np.ndarray) -> list[Box]:
        with self._lock:
            rw = self.world.predict(img, conf=0.12, imgsz=1280, device=self.device, verbose=False)[0]
            rc = self.books_model.predict(img, conf=0.30, imgsz=1280, device=self.device, verbose=False)[0]
        out: list[Box] = []
        for (x0, y0, x1, y1), c, k in zip(
            rw.boxes.xyxy.tolist(), rw.boxes.conf.tolist(), rw.boxes.cls.tolist(), strict=True
        ):
            out.append(Box(x0, y0, x1, y1, float(c), ITEM_VOCAB[int(k)]))
        for (x0, y0, x1, y1), c, k in zip(
            rc.boxes.xyxy.tolist(), rc.boxes.conf.tolist(), rc.boxes.cls.tolist(), strict=True
        ):
            name = rc.names[int(k)]
            if name in COCO_ITEMS:
                out.append(Box(x0, y0, x1, y1, float(c), name))
        return out


# --------------------------------------------------------------------------- spine clean-up
def orientation(b: Box) -> str:
    return "flat" if b.w > 1.6 * b.h else "upright"


def clean_spines(boxes: list[Box]) -> list[Box]:
    """Turn raw 'book' detections into one box per spine.

    The detector returns fragments of one spine (top half, bottom half) and sometimes one box over
    two neighbours. Spines sit side by side, so the test is overlap along the stacking axis:
    upright spines are compared on x, flat (stacked) spines on y.
    """
    boxes = sorted(boxes, key=lambda b: -b.conf)
    # 1. Drop boxes that span two or more narrower, side-by-side spines.
    keep: list[Box] = []
    for b in boxes:
        axis_x = orientation(b) == "upright"
        inner = [
            o
            for o in boxes
            if o is not b
            and orientation(o) == orientation(b)
            and _inside(o, b, 0.85)
            and (o.w if axis_x else o.h) < 0.7 * (b.w if axis_x else b.h)
        ]
        inner = _non_overlapping(inner, axis_x)
        span = b.w if axis_x else b.h
        covered = sum((o.w if axis_x else o.h) for o in inner)
        if len(inner) >= 2 and covered >= 0.6 * span:
            continue
        keep.append(b)
    # 2. Merge fragments of the same spine.
    merged: list[Box] = []
    for b in sorted(keep, key=lambda b: -b.area):
        axis_x = orientation(b) == "upright"
        for i, m in enumerate(merged):
            if orientation(m) != orientation(b) and min(m.area, b.area) > 0.5 * max(m.area, b.area):
                continue
            if axis_x:
                along = overlap_1d(b.x0, b.x1, m.x0, m.x1) / max(1.0, min(b.w, m.w))
                across = overlap_1d(b.y0, b.y1, m.y0, m.y1) / max(1.0, min(b.h, m.h))
                gap_ok = across > 0.0 or abs(b.cy - m.cy) < 0.6 * max(b.h, m.h)
            else:
                along = overlap_1d(b.y0, b.y1, m.y0, m.y1) / max(1.0, min(b.h, m.h))
                across = overlap_1d(b.x0, b.x1, m.x0, m.x1) / max(1.0, min(b.w, m.w))
                gap_ok = across > 0.0 or abs(b.cx - m.cx) < 0.6 * max(b.w, m.w)
            if (along >= 0.6 and gap_ok and iou(b, m) > 0.0) or (along >= 0.8 and across > 0.5):
                merged[i] = Box(
                    min(m.x0, b.x0), min(m.y0, b.y0), max(m.x1, b.x1), max(m.y1, b.y1), max(m.conf, b.conf), "book"
                )
                break
        else:
            merged.append(b)
    return sorted(merged, key=lambda b: (round(b.cy / 200), b.cx))


def _inside(o: Box, b: Box, frac: float) -> bool:
    ix = overlap_1d(o.x0, o.x1, b.x0, b.x1)
    iy = overlap_1d(o.y0, o.y1, b.y0, b.y1)
    return o.area > 0 and ix * iy >= frac * o.area


def _non_overlapping(boxes: list[Box], axis_x: bool) -> list[Box]:
    chosen: list[Box] = []
    for o in sorted(boxes, key=lambda b: -b.conf):
        if all(
            (overlap_1d(o.x0, o.x1, c.x0, c.x1) if axis_x else overlap_1d(o.y0, o.y1, c.y0, c.y1))
            < 0.3 * min(o.w if axis_x else o.h, c.w if axis_x else c.h)
            for c in chosen
        ):
            chosen.append(o)
    return chosen


def nms_items(boxes: list[Box], thr: float = 0.5) -> list[Box]:
    """Class-agnostic NMS; the higher-confidence label wins."""
    out: list[Box] = []
    for b in sorted(boxes, key=lambda b: -b.conf):
        if all(iou(b, o) < thr for o in out):
            out.append(b)
    return out
