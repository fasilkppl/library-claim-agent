"""Cross-frame registration: every physical book counted once.

A shelf front is close to a plane, so consecutive frames of it are related by a homography. Frames are
chained into *segments* (one continuous view of a shelving wall). Each spine observation is mapped into
its segment's coordinates, and observations that land on the same place in different frames become one
track (= one book). When the chain breaks (the camera left the shelf), the next frame is matched against
earlier segments so a re-capture of the same shelf merges instead of double counting.
"""

from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from claimagent.pipeline.boxes import box_iou

WORK_WIDTH = 1280
Features = tuple[list[cv2.KeyPoint], np.ndarray | None, float]
_local = threading.local()  # one SIFT detector per thread: frames' features are computed in parallel after the sweep


def frame_features(img: np.ndarray) -> Features:
    """SIFT keypoints and descriptors of a frame at working width, and that width's scale."""
    sift = getattr(_local, "sift", None)
    if sift is None:
        sift = _local.sift = cv2.SIFT_create(nfeatures=2500)
    scale = min(1.0, WORK_WIDTH / img.shape[1])
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        grey = cv2.resize(grey, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kps, desc = sift.detectAndCompute(grey, None)
    return list(kps), desc, scale


@dataclass
class FrameLink:
    seq: int
    segment: int
    H: np.ndarray  # frame pixels -> segment coordinates


@dataclass
class Observation:
    seq: int
    index: int  # spine index within the frame
    corners: np.ndarray  # 4x2 in segment coordinates
    spine: dict[str, object]

    @property
    def box(self) -> tuple[float, float, float, float]:
        xs, ys = self.corners[:, 0], self.corners[:, 1]
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


@dataclass
class Track:
    id: int
    segment: int
    observations: list[Observation] = field(default_factory=list)

    @property
    def last(self) -> Observation:
        return self.observations[-1]

    def box(self) -> tuple[float, float, float, float]:
        boxes = np.array([o.box for o in self.observations])
        return tuple(float(v) for v in np.median(boxes, axis=0))  # type: ignore[return-value]

    @property
    def orientation(self) -> str:
        votes = Counter(str(o.spine["orientation"]) for o in self.observations)
        return max(sorted(votes), key=lambda v: votes[v])  # a tie goes the same way on every run


def _same_spine(a: tuple[float, float, float, float], b: tuple[float, float, float, float], upright: bool) -> float:
    """Similarity in [0, 1]: overlap across the stacking axis matters most for thin spines."""
    if upright:
        along = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1.0, min(a[2] - a[0], b[2] - b[0]))
        other = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(1.0, min(a[3] - a[1], b[3] - b[1]))
    else:
        along = max(0.0, min(a[3], b[3]) - max(a[1], b[1])) / max(1.0, min(a[3] - a[1], b[3] - b[1]))
        other = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) / max(1.0, min(a[2] - a[0], b[2] - b[0]))
    if along < 0.5 or other < 0.4:
        return 0.0
    return 0.6 * along + 0.4 * box_iou(a, b)


class Registrar:
    """Incremental: call add_frame for every selected frame in time order (used live and offline)."""

    def __init__(self, min_inliers: int = 30, lookback: int = 16) -> None:
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.min_inliers = min_inliers
        self.lookback = lookback
        self.links: dict[int, FrameLink] = {}
        self.features: dict[int, Features] = {}
        self.order: list[int] = []
        self.tracks: list[Track] = []
        self.segments = 0
        self.segment_parent: dict[int, int] = {}

    # ---------------------------------------------------------------- frames
    def _homography(self, a: int, b: int) -> tuple[np.ndarray | None, int]:
        """H mapping frame b pixels to frame a pixels (full resolution)."""
        ka, da, sa = self.features[a]
        kb, db, sb = self.features[b]
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return None, 0
        pairs = self.matcher.knnMatch(db, da, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < self.min_inliers:
            return None, len(good)
        src = np.array([kb[m.queryIdx].pt for m in good], np.float32) / sb
        dst = np.array([ka[m.trainIdx].pt for m in good], np.float32) / sa
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if H is None or inliers < self.min_inliers or not _plausible(H):
            return None, inliers
        return H, inliers

    def add_frame(
        self, seq: int, img: np.ndarray | None, spines: list[dict[str, object]], features: Features | None = None
    ) -> FrameLink:
        """`features`: computed beforehand (frame_features), then `img` is not needed."""
        if features is None:
            assert img is not None
            features = frame_features(img)
        self.features[seq] = features
        link: FrameLink | None = None
        if self.order:
            prev = self.order[-1]
            H, _ = self._homography(prev, seq)
            if H is not None:
                pl = self.links[prev]
                link = FrameLink(seq, pl.segment, pl.H @ H)
            else:
                link = self._relocalise(seq)
        if link is None:
            link = FrameLink(seq, self.segments, np.eye(3))
            self.segments += 1
        self.links[seq] = link
        self.order.append(seq)
        self._associate(link, spines)
        return link

    def _relocalise(self, seq: int) -> FrameLink | None:
        """Chain broken: try earlier frames (a re-capture of a shelf already seen).

        Bounded cost: the last few frames plus an evenly spread sample of each earlier segment.
        """
        recent = self.order[-2::-1][:6]
        by_seg: dict[int, list[int]] = {}
        for s in self.order[:-1]:
            by_seg.setdefault(self.links[s].segment, []).append(s)
        spread: list[int] = []
        for frames in by_seg.values():
            k = max(1, len(frames) // 3)
            spread += frames[::k][:3]
        candidates = list(dict.fromkeys(recent + spread))[: self.lookback]
        best: tuple[int, np.ndarray, int] | None = None
        for other in candidates:
            H, inl = self._homography(other, seq)
            if H is not None and (best is None or inl > best[2]):
                best = (other, H, inl)
        if best is None:
            return None
        ol = self.links[best[0]]
        return FrameLink(seq, ol.segment, ol.H @ best[1])

    # ---------------------------------------------------------------- tracks
    def _associate(self, link: FrameLink, spines: list[dict[str, object]]) -> None:
        obs = []
        for i, sp in enumerate(spines):
            pts = np.array(sp["corners"], np.float64).reshape(-1, 1, 2)
            mapped = cv2.perspectiveTransform(pts, link.H).reshape(-1, 2)
            obs.append(Observation(link.seq, i, mapped, sp))
        live = [t for t in self.tracks if t.segment == link.segment]
        scores: list[tuple[float, int, int]] = []
        for oi, o in enumerate(obs):
            upright = o.spine["orientation"] == "upright"
            for ti, t in enumerate(live):
                s = _same_spine(o.box, t.last.box, upright)
                if s > 0:
                    scores.append((s, oi, ti))
        used_o: set[int] = set()
        used_t: set[int] = set()
        for _, oi, ti in sorted(scores, reverse=True):
            if oi in used_o or ti in used_t:
                continue
            live[ti].observations.append(obs[oi])
            used_o.add(oi)
            used_t.add(ti)
        for oi, o in enumerate(obs):
            if oi not in used_o:
                self.tracks.append(Track(len(self.tracks), link.segment, [o]))

    # ---------------------------------------------------------------- results
    def scale_at(self, seq: int, x: float, y: float) -> float:
        """Local linear scale (segment px per frame px) of a frame's homography at a pixel."""
        H = self.links[seq].H
        J = _jacobian(H, x, y)
        return float(np.sqrt(abs(np.linalg.det(J))))

    def summary(self) -> dict[str, object]:
        return {
            "segments": self.segments,
            "frames": [
                {"seq": s, "segment": self.links[s].segment, "H": self.links[s].H.round(6).tolist()} for s in self.order
            ],
            "tracks": [
                {
                    "id": t.id,
                    "segment": t.segment,
                    "orientation": t.orientation,
                    "box": [round(v, 1) for v in t.box()],
                    "observations": [
                        {"seq": o.seq, "index": o.index, "corners": o.corners.round(1).tolist()} for o in t.observations
                    ],
                }
                for t in self.tracks
            ],
        }


def _plausible(H: np.ndarray) -> bool:
    H = H / H[2, 2]
    det = float(np.linalg.det(H[:2, :2]))
    if not 0.33 < det < 3.0:
        return False
    return bool(abs(H[2, 0]) < 0.003 and abs(H[2, 1]) < 0.003)


def _jacobian(H: np.ndarray, x: float, y: float) -> np.ndarray:
    p = H @ np.array([x, y, 1.0])
    w = p[2]
    u, v = p[0] / w, p[1] / w
    return np.array(
        [
            [(H[0, 0] - u * H[2, 0]) / w, (H[0, 1] - u * H[2, 1]) / w],
            [(H[1, 0] - v * H[2, 0]) / w, (H[1, 1] - v * H[2, 1]) / w],
        ]
    )
