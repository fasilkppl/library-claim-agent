"""Frame stages: keyframe selection, AR tracking check, per-frame analysis, cross-frame registration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from claimagent.capture import quality, tracking
from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings
from claimagent.pipeline.analyze import FrameAnalyzer
from claimagent.pipeline.register import Registrar

# ============================================================================ select
KEYFRAME_SHIFT = 0.12  # keep a frame once the view has moved this share of its width (~88% overlap)
KEYFRAME_MAX_GAP_MS = 2000  # but never leave more than this without a keyframe


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
) -> dict[int, dict[str, Any]]:
    analyzer = analyzer or FrameAnalyzer(store)
    out: dict[int, dict[str, Any]] = {}
    for i, m in enumerate(selected):
        out[m.seq] = analyzer.analyze(m)
        if progress:
            progress(i + 1, len(selected))
    return out


# ============================================================================ register
def register_tracks(
    store: SweepStore, selected: list[FrameMeta], det: dict[int, dict[str, Any]]
) -> tuple[Registrar, dict[str, Any]]:
    reg = Registrar()
    for m in selected:
        img = cv2.imread(str(store.frame_path(m)))
        if img is None:
            continue
        reg.add_frame(m.seq, img, det[m.seq]["spines"])
    summary = reg.summary()
    store.write_stage("register", summary)
    return reg, summary
