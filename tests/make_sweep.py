"""Build a camera-mode test sweep from still shelf photos.

A virtual camera pans across each photo (overlapping crops with small perspective jitter, a few blurred
frames), so registration must merge repeated sightings of the same spine. An ArUco reference marker of a
known size is pasted on the first photo so the marker scale path is exercised end to end.

Ground truth that follows from the construction:
- the photo is assumed to have PX_PER_CM pixels per centimetre, so a spine L px long is L / PX_PER_CM cm;
- the marker is pasted MARKER_PX pixels wide = MARKER_PX / PX_PER_CM cm.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from claimagent.capture.store import SweepStore, utc_now

PX_PER_CM = 40.0
MARKER_PX = 200
FIXTURES = Path(__file__).parent / "fixtures" / "shelves"
# The main shelf row of each photo (the cut-off row above is dropped so the true count is unambiguous:
# philosophy 23 books, german 22 books).
MAIN_ROW = {"philosophy.jpg": (120, 1256), "german.jpg": (80, 888)}


def load_photo(name: str) -> np.ndarray:
    img = cv2.imread(str(FIXTURES / name))
    y0, y1 = MAIN_ROW[name]
    return img[y0:y1]


def marker_image(px: int) -> np.ndarray:
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    m = cv2.aruco.generateImageMarker(d, 0, px)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def paste_marker(img: np.ndarray, x: int, y: int, px: int = MARKER_PX) -> np.ndarray:
    out = img.copy()
    pad = px // 6
    cv2.rectangle(out, (x - pad, y - pad), (x + px + pad, y + px + pad), (255, 255, 255), -1)
    out[y : y + px, x : x + px] = marker_image(px)
    return out


def pan_frames(img: np.ndarray, n: int, out_w: int = 1280, seed: int = 0, blur_every: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    # Full shelf height, 60% of its width; the output keeps the crop's aspect so x and y scale equally.
    crop_h, crop_w = h, int(0.6 * w)
    out_h = round(out_w * crop_h / crop_w)
    y_top = 0
    xs = np.linspace(0, w - crop_w, n)
    frames = []
    for k, x in enumerate(xs):
        x = int(x)
        src = np.float32([[x, y_top], [x + crop_w, y_top], [x + crop_w, y_top + crop_h], [x, y_top + crop_h]])
        jit = rng.normal(0, 0.012 * crop_w, (4, 2)).astype(np.float32)
        dst = np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]]) + jit * out_w / crop_w
        M = cv2.getPerspectiveTransform(src, dst)
        f = cv2.warpPerspective(img, M, (out_w, out_h), borderMode=cv2.BORDER_REPLICATE)
        if blur_every and k % blur_every == blur_every - 1:
            f = cv2.blur(f, (15, 1))
        frames.append(f)
    return frames


def build(root: Path, sweep_id: str, frames_per_photo: int = 9) -> SweepStore:
    store = SweepStore(root, sweep_id)
    photos = [paste_marker(load_photo("philosophy.jpg"), 420, 55), load_photo("german.jpg")]
    t = 0
    for i, ph in enumerate(photos):
        for f in pan_frames(ph, frames_per_photo, seed=i, blur_every=5):
            ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            assert ok
            store.add_frame(buf.tobytes(), t, f.shape[1], f.shape[0])
            t += 700
        # A turn away from the shelves between the two units breaks the homography chain.
        blank = np.full((720, 1280, 3), 180, np.uint8)
        cv2.putText(blank, "wall", (560, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (90, 90, 90), 3)
        ok, buf = cv2.imencode(".jpg", blank)
        store.add_frame(buf.tobytes(), t, 1280, 720)
        t += 700
    store.update_session(
        country="IN",
        currency="INR",
        device="synthetic pan of still photos",
        capture_mode="camera",
        captured_at=utc_now(),
        duration_s=t / 1000,
        marker_side_cm=MARKER_PX / PX_PER_CM,
        started_monotonic=time.monotonic(),
    )
    (store.dir / "construction.json").write_text(json.dumps({"px_per_cm": PX_PER_CM, "marker_px": MARKER_PX}))
    return store


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).parents[1] / "sweeps"))
    ap.add_argument("--id", default="test-pan")
    a = ap.parse_args()
    s = build(Path(a.root), a.id)
    print(s.dir)


def build_xr(root: Path, sweep_id: str) -> SweepStore:
    """WebXR-mode sweep: the shelf pan frames get a camera pose, intrinsics and a metric depth grid that
    is consistent with PX_PER_CM, followed by a walk around a 4.2 x 3.5 x 2.6 m room for the room stage."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from synthetic import look_at, walk

    store = SweepStore(root, sweep_id)
    t = 0
    ph = load_photo("philosophy.jpg")
    crop_w = int(0.6 * ph.shape[1])
    frames = pan_frames(ph, 9, seed=3, blur_every=0)
    for k, f in enumerate(frames):
        h, w = f.shape[:2]
        fx = fy = 0.9 * w
        px_per_cm = PX_PER_CM * w / crop_w  # this frame's pixels per centimetre on the shelf plane
        z = fx / (px_per_cm * 100.0)  # metres: a 1 cm object at depth z spans px_per_cm pixels
        pose = look_at(np.array([0.5 + 0.03 * k, 1.2, 1.0]), np.array([0.5 + 0.03 * k, 1.2, 1.0 - z]))
        depth = np.full((72, 96), z, np.float32)
        _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        store.add_frame(
            buf.tobytes(),
            t,
            w,
            h,
            pose=pose.T.reshape(-1).tolist(),
            intrinsics={"fx": fx, "fy": fy, "cx": w / 2, "cy": h / 2, "width": w, "height": h},
            depth=depth,
        )
        t += 700
    poly = [(0.0, 0.0), (4.2, 0.0), (4.2, 3.5), (0.0, 3.5)]
    blank = np.full((480, 640, 3), 170, np.uint8)
    _, buf = cv2.imencode(".jpg", blank)
    for M, d in walk(poly, 2.6, n=90, seed=1):
        store.add_frame(
            buf.tobytes(),
            t,
            640,
            480,
            pose=M.T.reshape(-1).tolist(),
            intrinsics={"fx": 480.0, "fy": 480.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
            depth=d,
        )
        t += 700
    store.update_session(
        country="IN",
        currency="INR",
        device="synthetic WebXR",
        capture_mode="webxr",
        captured_at=utc_now(),
        duration_s=t / 1000,
        started_monotonic=time.monotonic(),
    )
    (store.dir / "construction.json").write_text(json.dumps({"px_per_cm": PX_PER_CM, "room": poly, "height": 2.6}))
    return store
