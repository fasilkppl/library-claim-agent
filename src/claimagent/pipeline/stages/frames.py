"""Frame stages: keyframe selection, AR tracking check, per-frame analysis, cross-frame registration."""

from __future__ import annotations

import bisect
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import cv2
import numpy as np

from claimagent.capture import quality, tracking
from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings
from claimagent.pipeline.analyze import FrameAnalyzer
from claimagent.pipeline.register import Features, Registrar, frame_features

# ============================================================================ select
KEYFRAME_SHIFT = 0.12  # keep a frame once the view has moved this share of its width (~88% overlap)
KEYFRAME_MAX_GAP_MS = 2000  # but never leave more than this without a keyframe

log = logging.getLogger(__name__)


def is_keyframe(prev: np.ndarray | None, cur: np.ndarray, gap_ms: int) -> tuple[bool, float]:
    if prev is None:
        return True, 1.0
    shift = quality.view_shift(prev, cur)
    return shift >= KEYFRAME_SHIFT or gap_ms >= KEYFRAME_MAX_GAP_MS, shift


def select_frames(store: SweepStore, s: Settings) -> dict[str, Any]:
    """Keep sharp keyframes: a new one each time the view has moved enough. Blurry frames are dropped;
    the next sharp frame takes their place."""
    out: list[dict[str, Any]] = []
    prev: np.ndarray | None = None
    prev_t = 0
    for m in store.frames():
        img = cv2.imread(str(store.frame_path(m)), cv2.IMREAD_GRAYSCALE)
        if img is None:
            out.append({"seq": m.seq, "keep": False, "reason": "unreadable"})
            continue
        sharp = quality.measure(img).sharpness
        th = quality.thumb(img)
        keep, shift = is_keyframe(prev, th, m.t_ms - prev_t)
        reason = "" if keep else f"view moved only {shift:.0%}"
        if sharp < 0.5 * s.blur_min:
            keep, reason = False, f"blurry (sharpness {sharp:.0f})"
        if keep:
            prev, prev_t = th, m.t_ms
        out.append({"seq": m.seq, "t_ms": m.t_ms, "keep": keep, "reason": reason, "sharpness": round(sharp, 1)})
    data = {"frames": out, "kept": sum(bool(f["keep"]) for f in out), "total": len(out)}
    store.write_stage("select", data)
    return data


# ============================================================================ tracking
def check_tracking(store: SweepStore, metas: list[FrameMeta]) -> dict[str, Any]:
    """Drops the AR pose of every frame whose tracking is not believable (see capture.tracking). The
    metas are changed in place: a frame without a pose is handled like a camera-only frame."""
    report = tracking.assess(metas, store.load_depth)
    for m in metas:
        if m.seq in report.rejected:
            m.pose = None
    data = {
        "ar_frames": len(report.trusted) + len(report.rejected),
        "trusted": sorted(report.trusted),
        "rejected": {str(k): v for k, v in sorted(report.rejected.items())},
        "lost_for_good": report.lost_for_good,
        "note": report.note,
    }
    store.write_stage("tracking", data)
    return data


# ============================================================================ analyze
def analyze_frames(
    store: SweepStore,
    selected: list[FrameMeta],
    analyzer: FrameAnalyzer | None = None,
    progress: Callable[[int, int], None] | None = None,
    budget_s: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Frames analysed during the walk are reused; the rest are analysed now, the biggest gap in time between
    analysed frames first, until the time budget is spent (the packet is due within 5 minutes). Frames left
    over are close in time to analysed ones, so what they show is mostly seen already; they are logged."""
    analyzer = analyzer or FrameAnalyzer(store)
    out: dict[int, dict[str, Any]] = {}
    todo: list[FrameMeta] = []
    for m in selected:
        hit = analyzer.cached(m)
        if hit is not None:
            out[m.seq] = hit
        else:
            todo.append(m)
    t_of = {m.seq: m.t_ms for m in selected}
    t0 = time.perf_counter()
    done = len(out)
    while todo:
        if budget_s is not None and time.perf_counter() - t0 > budget_s:
            log.info("analysis budget spent: %s of %s frames left unanalysed", len(todo), len(selected))
            break
        times = sorted(t_of[seq] for seq in out)
        gaps = [_gap(f.t_ms, times) for f in todo]
        m = todo[gaps.index(max(gaps))]
        todo.remove(m)
        out[m.seq] = analyzer.analyze(m)
        done += 1
        if progress:
            progress(done, len(selected))
    return out


def _gap(t_ms: int, analysed: list[int]) -> float:
    """Time to the nearest analysed frame (sorted list); infinite when none is analysed yet."""
    if not analysed:
        return float("inf")
    k = bisect.bisect_left(analysed, t_ms)
    return float(min(abs(t_ms - analysed[j]) for j in (k - 1, k) if 0 <= j < len(analysed)))


# ============================================================================ register
REGISTER_WORKERS = 4  # frames' features in parallel (OpenCV runs outside the Python lock); chaining stays in order


def register_tracks(
    store: SweepStore, selected: list[FrameMeta], det: dict[int, dict[str, Any]]
) -> tuple[Registrar, dict[str, Any]]:
    def features(m: FrameMeta) -> Features | None:
        img = cv2.imread(str(store.frame_path(m)))
        return None if img is None else frame_features(img)

    with ThreadPoolExecutor(max_workers=REGISTER_WORKERS) as pool:
        found = list(pool.map(features, selected))
    reg = Registrar()
    for m, f in zip(selected, found, strict=True):
        if f is None:
            continue
        reg.add_frame(m.seq, None, det[m.seq]["spines"], features=f)
    summary = reg.summary()
    store.write_stage("register", summary)
    return reg, summary
