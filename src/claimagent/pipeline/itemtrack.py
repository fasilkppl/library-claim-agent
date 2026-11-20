"""Follows each detected object from keyframe to keyframe so it is counted once.

Between two consecutive keyframes the camera motion is estimated from the images alone (ORB features and
a RANSAC homography). The previous frame's boxes are carried into the new frame with it and matched to
the new detections of the same kind by overlap. No AR pose is needed, so this works in camera-only
sweeps and after AR tracking was lost. A homography is exact only for a plane; for small moves between
keyframes it is close enough to keep boxes on the same object.

Revisits (the same object seen again after the camera left it) are joined later by 3D position when AR
tracking is trusted; without it they stay separate and the item count can run high.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np

WORK_WIDTH = 480  # images are matched at this width
MIN_INLIERS = 25
MATCH_IOU = 0.25
CROSS_LABEL_IOU = 0.5  # the detector often renames one object between frames (pan / bowl / cup)
MAX_MISSES = 3  # a track survives this many keyframes without a matching detection

# Detector labels that are often confused for the same object; they are matched as one kind.
SAME_KIND = {
    "laptop": "screen",
    "television": "screen",
    "tv": "screen",
    "monitor": "screen",
    "computer monitor": "screen",
    "desk": "table",
    "dining table": "table",
    "table": "table",
    "coffee table": "table",
    "couch": "sofa",
    "sofa": "sofa",
    "potted plant": "plant",
    "plant": "plant",
    "framed picture": "picture",
    "painting": "picture",
    "poster": "picture",
    "bottle": "bottle",
    "water bottle": "bottle",
    "saucepan": "pan",
    "frying pan": "pan",
    "box": "box",
    "cardboard box": "box",
    "shelf": "shelf",
    "bookshelf": "shelf",
    "rug": "rug",
    "doormat": "rug",
}
# A detection of the first kind lying inside one of the second kind is part of it, not a separate item.
PART_OF = {"keyboard": "screen"}


def kind(label: str) -> str:
    return SAME_KIND.get(label, label)


@dataclass
class Sighting:
    seq: int
    index: int
    label: str
    bbox: tuple[float, float, float, float]
    conf: float


@dataclass
class _Track:
    id: int
    kind: str
    box: np.ndarray  # x0, y0, x1, y1 in the current frame's work coordinates
    misses: int = 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def inside_fraction(a: Sequence[float], b: Sequence[float]) -> float:
    """Share of box a that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return ix * iy / area if area > 0 else 0.0


def drop_parts(sightings: list[Sighting]) -> list[Sighting]:
    out = []
    for s in sightings:
        whole = PART_OF.get(s.label)
        if whole and any(
            o is not s and kind(o.label) == whole and inside_fraction(s.bbox, o.bbox) > 0.7 for o in sightings
        ):
            continue
        out.append(s)
    return out


class _Motion:
    def __init__(self) -> None:
        self.orb = cv2.ORB_create(1500)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def features(self, gray: np.ndarray) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        kp, des = self.orb.detectAndCompute(gray, None)
        return list(kp), des

    def homography(
        self, a: tuple[list[cv2.KeyPoint], np.ndarray | None], b: tuple[list[cv2.KeyPoint], np.ndarray | None]
    ) -> np.ndarray | None:
        (ka, da), (kb, db) = a, b
        if da is None or db is None or len(ka) < MIN_INLIERS or len(kb) < MIN_INLIERS:
            return None
        matches = self.matcher.match(da, db)
        if len(matches) < MIN_INLIERS:
            return None
        src = np.array([ka[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 1, 2)
        dst = np.array([kb[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask is None or int(mask.sum()) < MIN_INLIERS:
            return None
        return np.asarray(H, dtype=np.float64)


def _warp_box(box: np.ndarray, H: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    pts = np.array([[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], np.float64)
    w = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return np.array([w[:, 0].min(), w[:, 1].min(), w[:, 0].max(), w[:, 1].max()])


def link(frames: Sequence[tuple[int, np.ndarray]], sightings: Sequence[Sighting]) -> dict[tuple[int, int], int]:
    """Track id for every sighting. `frames` are (seq, grayscale image) in capture order."""
    by_seq: dict[int, list[Sighting]] = {}
    for s in sightings:
        by_seq.setdefault(s.seq, []).append(s)
    motion = _Motion()
    out: dict[tuple[int, int], int] = {}
    active: list[_Track] = []
    prev_feat = None
    next_id = 0
    for seq, gray in frames:
        scale = WORK_WIDTH / gray.shape[1]
        small = cv2.resize(gray, (WORK_WIDTH, max(1, round(gray.shape[0] * scale))))
        feat = motion.features(small)
        H = motion.homography(prev_feat, feat) if prev_feat is not None else None
        prev_feat = feat
        if H is not None:
            for t in active:
                t.box = _warp_box(t.box, H)
        # Without a motion estimate the boxes stay where they were; keyframes are close in time, so a
        # same-kind detection overlapping the old box is still most likely the same object.
        cands = [(s, np.array(s.bbox, np.float64) * scale) for s in by_seq.get(seq, [])]
        # Same kind: a loose overlap is enough. Different label: only a strong overlap, the same spot.
        pairs = sorted(
            (
                (iou(t.box, b), t.kind == kind(s.label), ti, ci)
                for ti, t in enumerate(active)
                for ci, (s, b) in enumerate(cands)
            ),
            key=lambda x: (x[1], x[0]),
            reverse=True,
        )
        used_t: set[int] = set()
        used_c: set[int] = set()
        for ov, same_kind, ti, ci in pairs:
            if ov < (MATCH_IOU if same_kind else CROSS_LABEL_IOU) or ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t, (s, b) = active[ti], cands[ci]
            t.box, t.misses = b, 0
            out[(s.seq, s.index)] = t.id
        for ti, t in enumerate(active):
            if ti not in used_t:
                t.misses += 1
        active = [t for t in active if t.misses <= MAX_MISSES]
        for ci, (s, b) in enumerate(cands):
            if ci not in used_c:
                active.append(_Track(next_id, kind(s.label), b))
                out[(s.seq, s.index)] = next_id
                next_id += 1
    return out


REID_MIN_INLIERS = 15
REID_CROP = 320
REID_HIST_CORR = 0.85  # colour histograms this alike (and a similar shape) count as the same object
REID_ASPECT_TOL = 0.35


def _crop(gray: np.ndarray, box: Sequence[float]) -> np.ndarray | None:
    x0, y0, x1, y1 = (round(v) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 24 or y1 - y0 < 24:
        return None
    c = gray[y0:y1, x0:x1]
    s = REID_CROP / max(c.shape)
    return cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s)))) if s < 1 else c


def _signature(colour: np.ndarray, box: Sequence[float]) -> tuple[np.ndarray, float] | None:
    x0, y0, x1, y1 = (max(0, round(v)) for v in box)
    c = colour[y0:y1, x0:x1]
    if c.size == 0 or min(c.shape[:2]) < 16:
        return None
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist, (x1 - x0) / max(1, y1 - y0)


def same_object_groups(
    tracks: dict[int, list[Sighting]], images: dict[int, np.ndarray], colours: dict[int, np.ndarray] | None = None
) -> list[list[int]]:
    """Joins tracks of the same object seen again later (a revisit). Two tracks are joined when they are
    of the same kind, never appear in the same frame (two objects seen together are two objects), and
    their clearest crops match: by image features, or for plain objects with few features (a steel cup)
    by colour histogram and shape. Two identical objects never seen together can be joined by mistake."""
    motion = _Motion()
    feats: dict[int, tuple[list[cv2.KeyPoint], np.ndarray | None]] = {}
    sigs: dict[int, list[tuple[np.ndarray, float]]] = {}
    for tid, ss in tracks.items():
        best = max(ss, key=lambda s: s.conf)
        crop = _crop(images[best.seq], best.bbox) if best.seq in images else None
        if crop is not None:
            feats[tid] = motion.features(crop)
        if colours:
            # Crops cut off by the frame edge have the wrong shape and colours: prefer whole ones.
            whole = [s for s in ss if s.seq in colours and not _at_edge(s.bbox, colours[s.seq].shape)]
            for s in sorted(whole or ss, key=lambda s: -s.conf)[:3]:
                sig = _signature(colours[s.seq], s.bbox) if s.seq in colours else None
                if sig is not None:
                    sigs.setdefault(tid, []).append(sig)
    parent = {tid: tid for tid in tracks}

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seqs = {tid: {s.seq for s in ss} for tid, ss in tracks.items()}
    ids = sorted(tracks)
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            ra, rb = root(a), root(b)
            if ra == rb or kind(tracks[a][0].label) != kind(tracks[b][0].label):
                continue
            members_a = [t for t in ids if root(t) == ra]
            members_b = [t for t in ids if root(t) == rb]
            if any(seqs[x] & seqs[y] for x in members_a for y in members_b):
                continue
            if (a in feats and b in feats and _inliers(motion, feats[a], feats[b]) >= REID_MIN_INLIERS) or (
                a in sigs and b in sigs and any(_alike(x, y) for x in sigs[a] for y in sigs[b])
            ):
                parent[rb] = ra
    groups: dict[int, list[int]] = {}
    for tid in ids:
        groups.setdefault(root(tid), []).append(tid)
    return list(groups.values())


def _inliers(
    motion: _Motion, a: tuple[list[cv2.KeyPoint], np.ndarray | None], b: tuple[list[cv2.KeyPoint], np.ndarray | None]
) -> int:
    (ka, da), (kb, db) = a, b
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return 0
    matches = motion.matcher.match(da, db)
    if len(matches) < 8:
        return 0
    src = np.array([ka[m.queryIdx].pt for m in matches], np.float32).reshape(-1, 1, 2)
    dst = np.array([kb[m.trainIdx].pt for m in matches], np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return int(mask.sum()) if H is not None and mask is not None else 0


def _alike(a: tuple[np.ndarray, float], b: tuple[np.ndarray, float]) -> bool:
    corr = float(cv2.compareHist(a[0], b[0], cv2.HISTCMP_CORREL))
    ra, rb = a[1], b[1]
    return corr >= REID_HIST_CORR and abs(ra - rb) / max(ra, rb) <= REID_ASPECT_TOL


def _at_edge(box: Sequence[float], shape: tuple[int, ...], margin: int = 4) -> bool:
    return box[0] < margin or box[1] < margin or box[2] > shape[1] - margin or box[3] > shape[0] - margin
