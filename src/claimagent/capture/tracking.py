"""Is the phone's AR tracking (ARCore pose and depth) believable?

ARCore can lose its metric scale, typically when the phone turns fast in the first seconds or sees only
far, featureless surfaces. Its reported position then runs away (12 m in 10 s was seen on a real
sweep), and the depth grows with it. Nothing downstream may use those numbers: one runaway pose turns a
2 m room into a 7 m one and spreads one object over many positions.

A frame is rejected when
- the browser says the position is emulated (WebXR `emulatedPosition`: tracking lost, rotation only),
- the camera moved faster than a person walks during a sweep,
- the phone is higher above the floor than a hand can hold it, or
- the depth grid is beyond indoor range.
Once tracking has failed in a sustained way, ARCore does not repair the scale, so every later frame is
rejected as well. A single bad frame on its own is dropped and tracking continues.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from claimagent.capture.store import FrameMeta

MAX_SPEED_MPS = 1.0  # sweeps are walked slowly; ARCore scale failures show up as 1.5-4 m/s jumps
MAX_INDOOR_DEPTH_M = 8.0  # median of a frame's depth grid; home rooms are rarely deeper
PHONE_HEIGHT_M = (0.05, 2.4)  # a hand-held phone above the floor (local-floor reference space)
SUSTAINED = 2  # this many bad frames out of the last 3 means tracking is gone for good


def _position(pose: Sequence[float]) -> np.ndarray:
    return np.array(pose[12:15], dtype=np.float64)  # column-major 4x4: translation is elements 12..14


def _time_s(m: FrameMeta) -> float:
    client = (m.extra or {}).get("client_t")
    return float(client if isinstance(client, (int, float)) else m.t_ms) / 1000.0


@dataclass
class TrackingMonitor:
    """Incremental check, fed one AR frame at a time (live and after the sweep)."""

    lost_for_good: bool = False
    first_failure: str = ""
    recent: list[bool] = field(default_factory=list)
    last_pos: np.ndarray | None = None
    last_t: float = 0.0

    def check(self, m: FrameMeta, depth: np.ndarray | None) -> tuple[bool, str]:
        """(pose usable, reason when not)."""
        if not m.pose:
            return False, "no AR pose"
        if self.lost_for_good:
            return False, f"AR tracking lost earlier ({self.first_failure}); later poses are not trusted"
        reason = ""
        if (m.extra or {}).get("emulated"):
            reason = "AR tracking lost (position emulated by the browser)"
        pos, t = _position(m.pose), _time_s(m)
        if not reason and self.last_pos is not None and t > self.last_t:
            speed = float(np.linalg.norm(pos - self.last_pos)) / (t - self.last_t)
            if speed > MAX_SPEED_MPS:
                reason = f"implausible camera speed {speed:.1f} m/s"
        if not reason and not PHONE_HEIGHT_M[0] <= pos[1] <= PHONE_HEIGHT_M[1]:
            reason = f"implausible phone height {pos[1]:.1f} m above the floor"
        if not reason and depth is not None:
            finite = depth[np.isfinite(depth) & (depth > 0)]
            if finite.size and float(np.median(finite)) > MAX_INDOOR_DEPTH_M:
                reason = f"implausible depth (median {float(np.median(finite)):.1f} m)"
        ok = not reason
        self.recent = [*self.recent, ok][-3:]
        if ok:
            self.last_pos, self.last_t = pos, t
        elif self.recent.count(False) >= SUSTAINED:
            self.lost_for_good = True
            self.first_failure = self.first_failure or reason
        else:
            self.first_failure = self.first_failure or reason
        return ok, reason


@dataclass
class TrackingReport:
    trusted: set[int]
    rejected: dict[int, str]
    lost_for_good: bool
    first_failure: str

    @property
    def note(self) -> str:
        if not self.rejected:
            return ""
        n_bad, n_all = len(self.rejected), len(self.rejected) + len(self.trusted)
        head = f"AR tracking not trusted on {n_bad} of {n_all} frames"
        return f"{head}: {self.first_failure}" if self.first_failure else head


def assess(metas: Sequence[FrameMeta], load_depth: Callable[[FrameMeta], np.ndarray | None]) -> TrackingReport:
    mon = TrackingMonitor()
    trusted: set[int] = set()
    rejected: dict[int, str] = {}
    for m in sorted(metas, key=lambda m: m.seq):
        if not m.pose:
            continue
        ok, reason = mon.check(m, load_depth(m))
        if ok:
            trusted.add(m.seq)
        else:
            rejected[m.seq] = reason
    # Frames accepted just before a sustained failure already carry the drift that triggered it.
    if mon.lost_for_good and rejected:
        first_bad = min(rejected)
        for s in sorted(trusted):
            if first_bad - 2 <= s < first_bad:
                trusted.discard(s)
                rejected[s] = "just before AR tracking was lost"
    return TrackingReport(trusted, rejected, mon.lost_for_good, mon.first_failure)
