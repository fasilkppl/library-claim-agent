"""Per-frame analysis, shared by the live sweep and the offline pipeline.

For one keyframe: capture quality, spines (FastSAM + YOLO), text lines (Apple Vision), non-book items
(YOLO-World + YOLO11), and the optional ArUco reference marker. The result is saved as
stages/detect/<seq>.json so the post-sweep pipeline never recomputes a frame the live sweep already did.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np

from claimagent.capture import quality
from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import get_settings
from claimagent.pipeline.spines import Segmenter, segment_spines
from claimagent.pipeline.vision import (
    AppleVisionReader,
    Box,
    Detector,
    TextLine,
    TextReader,
    YoloDetector,
    nms_items,
)

ARUCO_DICT = cv2.aruco.DICT_4X4_50


@dataclass
class Models:
    detector: Detector
    segmenter: Segmenter | None
    reader: TextReader


_models: Models | None = None
_models_lock = threading.Lock()


def load_models() -> Models:
    global _models
    with _models_lock:
        if _models is None:
            import torch

            torch.set_num_threads(get_settings().torch_threads)
            _models = Models(YoloDetector(), Segmenter(), AppleVisionReader())
        return _models


def detect_marker(img: np.ndarray) -> dict[str, Any] | None:
    """ArUco 4x4 id 0 printed at a known size (docs/marker.pdf). Returns its corners and side in px."""
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(grey)
    if ids is None:
        return None
    for c, i in zip(corners, ids.flatten(), strict=True):
        if int(i) == 0:
            pts = c.reshape(4, 2)
            sides = [float(np.linalg.norm(pts[k] - pts[(k + 1) % 4])) for k in range(4)]
            return {
                "corners": pts.round(1).tolist(),
                "side_px": float(np.mean(sides)),
                "side_spread": float((max(sides) - min(sides)) / np.mean(sides)),
            }
    return None


def _best_spine(polys: list[np.ndarray], b: Box) -> tuple[int, float]:
    rect = np.array([[b.x0, b.y0], [b.x1, b.y0], [b.x1, b.y1], [b.x0, b.y1]], np.float32)
    area = max(1.0, b.area)
    best, best_frac = -1, 0.0
    for k, p in enumerate(polys):
        inter, _ = cv2.intersectConvexConvex(rect, p)
        frac = float(inter) / area
        if frac > best_frac:
            best, best_frac = k, frac
    return best, best_frac


def assign_text(spines: list[dict[str, Any]], lines: list[TextLine]) -> list[list[dict[str, Any]]]:
    """A text line belongs to the spine that contains most of its area (at least half).

    Spine text runs along a thin strip, so a line from a neighbour can have its centre near our edge;
    area overlap with the oriented spine box rejects it. A line that runs across several spines (the same
    author printed at the same height on neighbouring volumes) is split word by word at spine borders.
    """
    out: list[list[dict[str, Any]]] = [[] for _ in spines]
    polys = [np.array(s["corners"], np.float32) for s in spines]
    for ln in lines:
        best, frac = _best_spine(polys, ln.box)
        if frac >= 0.8 or not ln.words:
            pieces = [(best, frac, ln.text, ln.box)]
        else:
            pieces = []
            for word, wb in ln.words:
                k, f = _best_spine(polys, wb)
                if pieces and pieces[-1][0] == k:
                    _, pf, ptext, pb = pieces[-1]
                    pieces[-1] = (
                        k,
                        min(pf, f),
                        f"{ptext} {word}",
                        Box(min(pb.x0, wb.x0), min(pb.y0, wb.y0), max(pb.x1, wb.x1), max(pb.y1, wb.y1)),
                    )
                else:
                    pieces.append((k, f, word, wb))
        for k, f, text, box in pieces:
            if k >= 0 and f >= 0.5:
                out[k].append(
                    {
                        "text": text,
                        "conf": round(ln.conf, 3),
                        "vertical": ln.vertical,
                        "box": box.as_list(),
                        "inside": round(f, 2),
                    }
                )
    for group, sp in zip(out, spines, strict=True):
        # Reading order: along the spine for stacked lines, across it for side-by-side lines.
        if sp["orientation"] == "upright":
            group.sort(key=lambda t: (t["box"][0], t["box"][1]))
        else:
            group.sort(key=lambda t: (t["box"][1], t["box"][0]))
    return out


class FrameAnalyzer:
    def __init__(self, store: SweepStore, models: Models | None = None, items_every: int = 1) -> None:
        self.store = store
        self.models = models or load_models()
        self.items_every = items_every
        self.dir = store.stages_dir / "detect"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def path(self, meta: FrameMeta) -> Any:
        return self.dir / f"{meta.name}.json"

    def cached(self, meta: FrameMeta) -> dict[str, Any] | None:
        p = self.path(meta)
        return json.loads(p.read_text()) if p.exists() else None

    def analyze(self, meta: FrameMeta, img: np.ndarray | None = None, with_items: bool | None = None) -> dict[str, Any]:
        do_items = with_items if with_items is not None else self._count % self.items_every == 0
        hit = self.cached(meta)
        if hit is not None and (hit.get("items_checked", True) or not do_items):
            return hit
        t0 = time.perf_counter()
        if img is None:
            img = cv2.imread(str(self.store.frame_path(meta)))
            if img is None:
                raise ValueError(f"frame {meta.frame_ref} could not be read")
        timings: dict[str, float] = {}
        q = quality.measure(img)
        t = time.perf_counter()
        books = self.models.detector.books(img)
        timings["yolo_books"] = time.perf_counter() - t
        t = time.perf_counter()
        masks = self.models.segmenter.masks(img) if self.models.segmenter else []
        timings["fastsam"] = time.perf_counter() - t
        spines = [s.to_dict() for s in segment_spines(masks, books, img.shape[:2])]
        t = time.perf_counter()
        lines = self.models.reader.read(img)
        timings["ocr"] = time.perf_counter() - t
        texts = assign_text(spines, lines)
        for sp, tx in zip(spines, texts, strict=True):
            sp["text"] = tx

        if spines:
            boxes: list[list[float]] = [cast(list[float], s["bbox"]) for s in spines]
            xs = [v for b in boxes for v in (b[0], b[2])]
            ys = [v for b in boxes for v in (b[1], b[3])]
            region = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            q_books = quality.measure(img, region)
        else:
            q_books = q
        items: list[dict[str, Any]] = []
        if do_items:
            t = time.perf_counter()
            raw = nms_items(self.models.detector.objects(img), 0.5)
            timings["items"] = time.perf_counter() - t
            items = [_item_dict(b, lines) for b in raw if b.label != "book"]
        self._count += 1
        out = {
            "seq": meta.seq,
            "t_ms": meta.t_ms,
            "width": img.shape[1],
            "height": img.shape[0],
            "quality": {
                "sharpness": round(q.sharpness, 1),
                "glare": round(q.glare, 4),
                "brightness": round(q.brightness, 1),
                "book_sharpness": round(q_books.sharpness, 1),
                "book_glare": round(q_books.glare, 4),
            },
            "spines": spines,
            "text_lines": len(lines),
            "items": items,
            "items_checked": do_items,
            "marker": detect_marker(img),
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }
        self.path(meta).write_text(json.dumps(out))
        return out


CROP_MIN_LENGTH_PX = 120
CROP_UPSCALE = 2.0
CROP_MAX_PX = 2400


def read_spine_crop(img: np.ndarray, sp: dict[str, Any], reader: TextReader) -> list[dict[str, Any]]:
    """Close reading of one spine: the spine alone, straightened so its text runs horizontally, enlarged and
    read both ways up (spine text runs either direction). Small print on thin or angled spines is often
    missed or garbled in the whole-frame pass."""
    pts = np.array(sp["corners"], np.float32)
    # Order: the two long edges become top and bottom of a horizontal strip.
    d01, d12 = np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2])
    if d01 < d12:
        pts = np.roll(pts, -1, axis=0)
    length, width = max(d01, d12), min(d01, d12)
    scale = min(CROP_UPSCALE, CROP_MAX_PX / max(1.0, length))
    w_out, h_out = int(length * scale), max(16, int(width * scale))
    dst = np.array([[0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1]], np.float32)
    strip = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (w_out, h_out), flags=cv2.INTER_CUBIC)
    pad = cv2.copyMakeBorder(strip, h_out // 2, h_out // 2, 8, 8, cv2.BORDER_REPLICATE)
    best: list[TextLine] = []
    for candidate in (pad, cv2.rotate(pad, cv2.ROTATE_180)):
        found = [ln for ln in reader.read(candidate) if len(ln.text.strip()) >= 2]
        if sum(len(ln.text) * ln.conf for ln in found) > sum(len(ln.text) * ln.conf for ln in best):
            best = found
    return [
        {"text": ln.text, "conf": round(ln.conf, 3), "vertical": True, "box": sp["bbox"], "inside": 1.0, "crop": True}
        for ln in best
    ]


def _item_dict(b: Box, lines: list[TextLine]) -> dict[str, Any]:
    inside = [ln.text for ln in lines if b.contains_point(ln.box.cx, ln.box.cy)]
    return {"label": b.label, "conf": round(b.conf, 3), "bbox": b.as_list(), "text": inside[:10]}
