"""Capture coaching during the sweep. Rules in code, measured on each frame; the model only voices them.

The coach decides what to say and how often (one hint at a time, each kind not repeated too soon). The
session decides whether it may speak right now (still sweeping, not already talking) and speaks.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.capture.tracking import TrackingMonitor
from claimagent.config import Settings
from claimagent.pipeline.household import CATEGORY

log = logging.getLogger(__name__)

AR_STARTING = "Move the phone slowly left and right for a few seconds, so the AR tracking can start."
AR_LOST = (
    "I've lost track of where the phone is. Please move more slowly. "
    "I'll measure the books from the catalogue instead, and the room may not be measured."
)
TURN_TOO_FAST_DEG_S = 70
UNREADABLE_MIN_LENGTH_PX = 120


@dataclass
class _State:
    last_spoken: float = 0.0
    last_by_kind: dict[str, float] = field(default_factory=dict)
    blurry_run: int = 0
    glare_run: int = 0
    asked_art: set[int] = field(default_factory=set)
    last_progress: float = 0.0
    unreadable_run: int = 0
    top_cut: int = 0
    bottom_cut: int = 0
    pitches: list[float] = field(default_factory=list)
    yaws: list[float] = field(default_factory=list)
    prev_pose: np.ndarray | None = None
    prev_t: float = 0.0
    tracking: TrackingMonitor = field(default_factory=TrackingMonitor)
    tracking_warned: bool = False


class Coach:
    def __init__(
        self,
        s: Settings,
        store: SweepStore,
        can_speak: Callable[[], bool],
        speak: Callable[[str, str], None],
        elapsed_ms: Callable[[], int],
    ) -> None:
        self.s, self.store = s, store
        self._can_speak, self._speak, self._elapsed_ms = can_speak, speak, elapsed_ms
        self.st = _State()
        self.step_of: Callable[[], str] = lambda: "books"  # the guide's current step

    def reset(self) -> None:
        self.st = _State(last_progress=time.monotonic())

    # ------------------------------------------------------------------ output
    def hint(self, kind: str, text: str, gap: float = 7.0, per_kind: float = 20.0) -> None:
        now = time.monotonic()
        if now - self.st.last_spoken < gap or now - self.st.last_by_kind.get(kind, 0) < per_kind:
            log.debug("hint %s held back (spoke too recently): %s", kind, text)
            return
        if not self._can_speak():
            log.debug("hint %s held back (talking or not sweeping): %s", kind, text)
            return
        log.info("HINT %s: %s", kind, text)
        self.st.last_spoken = now
        self.st.last_by_kind[kind] = now
        self.store.log_event("hint", self._elapsed_ms(), hint=kind, text=text)
        self._speak(kind, text)

    # ------------------------------------------------------------------ rules
    def on_blur(self, blurry: bool) -> None:
        self.st.blurry_run = self.st.blurry_run + 1 if blurry else 0
        if self.st.blurry_run >= 2:
            self.hint("blur", "Slow down a little, the picture is blurry.")

    def on_dark(self) -> None:
        self.hint("dark", "It's quite dark here. Could you switch on a light?")

    def on_ar_lost(self) -> None:
        self.hint("tracking", AR_LOST, per_kind=60)

    def on_motion(self, meta: FrameMeta) -> None:
        """Pose-based rules: AR tracking health, turning speed, which directions were covered."""
        if not meta.pose:
            return
        ok, reason = self.st.tracking.check(meta, self.store.load_depth(meta))
        if not ok and self.st.tracking.lost_for_good and not self.st.tracking_warned:
            self.st.tracking_warned = True
            log.info("AR tracking lost at frame %s: %s", meta.seq, reason)
            self.store.log_event("tracking_lost", self._elapsed_ms(), seq=meta.seq, reason=reason)
            self.hint("tracking", AR_LOST, gap=0.0, per_kind=60)
        if not ok:
            return  # a runaway pose says nothing about where the camera points
        M = np.array(meta.pose, np.float64).reshape(4, 4).T
        fwd = -M[:3, 2]
        self.st.pitches.append(math.degrees(math.asin(max(-1.0, min(1.0, fwd[1])))))
        self.st.yaws.append(math.degrees(math.atan2(fwd[0], -fwd[2])))
        # Capture time, not processing time: frames can reach this rule in bursts.
        client_t = (meta.extra or {}).get("client_t")
        now = float(client_t) / 1000 if isinstance(client_t, (int, float)) else time.monotonic()
        if self.st.prev_pose is not None:
            dt = max(1e-3, now - self.st.prev_t)
            f0 = -self.st.prev_pose[:3, 2]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(f0, fwd))))))
            if ang / dt > TURN_TOO_FAST_DEG_S:
                self.hint("fast", "You're turning quickly. Slow down so I can register every book.")
        self.st.prev_pose, self.st.prev_t = M, now

    def on_frame(self, d: dict[str, Any], spines_logged: int, spines_readable: Callable[[], int]) -> None:
        """Rules on one analysed keyframe: glare, unreadable or cut-off shelves, art, progress, coverage."""
        st = self.st
        step = self.step_of()
        # Only advice for the current part of the walk: spine advice on the shelves, room advice in the room.
        spines = d["spines"] if step == "books" else []
        qd = d["quality"]
        st.glare_run = st.glare_run + 1 if qd["book_glare"] > self.s.glare_max and spines else 0
        if st.glare_run >= 2:
            self.hint("glare", "There's glare on the spines. Tilt the phone a little to avoid the reflection.")
        # Spines that stay unreadable while the picture is sharp are too small in the frame: step closer.
        unreadable = [sp for sp in spines if not sp.get("text") and sp["length"] >= UNREADABLE_MIN_LENGTH_PX]
        sharp = qd["book_sharpness"] > self.s.blur_min
        st.unreadable_run = st.unreadable_run + 1 if sharp and len(unreadable) >= 2 else 0
        if st.unreadable_run >= 3:
            self.hint("closer", "Step a little closer to these books, I can't read some of the spines yet.")
        if len(spines) >= 6:
            top = sum(1 for sp in spines if sp["bbox"][1] < 4)
            bottom = sum(1 for sp in spines if sp["bbox"][3] > d["height"] - 4)
            st.top_cut = st.top_cut + 1 if top >= 3 else 0
            st.bottom_cut = st.bottom_cut + 1 if bottom >= 3 else 0
            if st.top_cut >= 3:
                self.hint("top", "Tilt up a little, the top of the shelf is cut off.")
            if st.bottom_cut >= 3:
                self.hint("bottom", "Tilt down a little, the bottom shelf is cut off.")
        for it in d.get("items", []):
            if CATEGORY.get(it["label"]) == "art" and it["conf"] > 0.3:
                key = d["seq"]
                if not any(abs(key - k) < 8 for k in st.asked_art):
                    st.asked_art.add(key)
                    self.hint("art", "I can see a framed picture. Is it an original or a print?", gap=4, per_kind=15)
        # No spoken running counts: live counts still include spines seen twice, so they would mislead.
        elapsed = self._elapsed_ms() / 1000
        if step == "room" and st.pitches and elapsed > 60:
            if max(st.pitches) < 35:
                self.hint("ceiling", "Please point the camera up at the ceiling for a moment.", per_kind=45)
            elif min(st.pitches) > -35:
                self.hint("floor", "Please point the camera down at the floor for a moment.", per_kind=45)
            elif st.yaws:
                bins = {int(((y + 360) % 360) // 90) for y in st.yaws}
                if len(bins) < 4 and elapsed > 90:
                    self.hint("coverage", "Turn around slowly, I haven't seen every wall yet.", per_kind=40)
