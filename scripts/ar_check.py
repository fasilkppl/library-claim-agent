"""Is ARCore healthy on this phone, and is our AR capture as healthy as a native AR app?

Two checks, run in the same room within minutes of each other:

1. Record ARCore's own log while an AR app runs (phone on USB, adb authorised):
       uv run python scripts/ar_check.py record native 45     # use a native AR app (Measure, AR Ruler) for 45 s
       uv run python scripts/ar_check.py record ours 60       # then our app's room step in AR for 60 s
   Each run prints how often ARCore lost tracking, reset its motion tracking (VIO), lost camera/IMU sync and
   how far the camera-to-IMU clock offset strayed (ARCore's limit is 5 ms). Same numbers for both apps mean
   the phone decides; clean native numbers next to bad ones for our app mean our capture is at fault.

2. Summarise the AR frames our app recorded in a sweep:
       uv run python scripts/ar_check.py sweep <sweep_id>
   Per tracking segment: frames, trusted and rejected (with reasons), camera path length and depth medians.
"""

from __future__ import annotations

import collections
import itertools
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# ARCore's log lines worth counting (tag CELog and the native session); matched case-insensitively.
EVENTS = {
    "vio_reset": r"VIO_RESET",
    "cam_imu_desync": r"CAM_IMU_DESYNC",
    "features_insufficient": r"FEATURES_TRACKS_INSUFFICIENT",
    "ransac_insufficient": r"RANSAC_INLIER_INSUFFICIENT",
    "tracking_failure": r"TrackingFailureReason|tracking (?:failure|lost)",
    "excessive_motion": r"EXCESSIVE_MOTION",
    "insufficient_light": r"INSUFFICIENT_LIGHT",
}
KEEP = re.compile(r"CELog|ARCore|native_session|arcore|Vio|VIO|camera.*imu|clock offset", re.I)
OFFSET = re.compile(r"Camera to IMU clock offset \((-?[\d.]+)\s*ms\)", re.I)


def adb(*args: str) -> str:
    return subprocess.run(["adb", *args], capture_output=True, text=True, timeout=30).stdout


def record(label: str, seconds: int) -> Path:
    if "device" not in adb("get-state"):
        sys.exit("no phone on adb: connect it over USB and allow USB debugging")
    out = ROOT / "sweeps" / "ar-check" / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    version = re.search(r"versionName=(\S+)", adb("shell", "dumpsys", "package", "com.google.ar.core"))
    adb("logcat", "-c")
    print(f"recording ARCore's log for {seconds} s: use the AR app now ({label})")
    proc = subprocess.Popen(["adb", "logcat", "-v", "time"], stdout=subprocess.PIPE, text=True)
    kept: list[str] = []
    t_end = time.time() + seconds
    assert proc.stdout is not None
    while time.time() < t_end:
        line = proc.stdout.readline()
        if line and KEEP.search(line):
            kept.append(line.rstrip())
    proc.terminate()
    header = f"# {label} · ARCore {version.group(1) if version else '?'} · {seconds} s\n"
    out.write_text(header + "\n".join(kept) + "\n")
    print(summary(out))
    return out


def summary(path: Path) -> str:
    text = path.read_text(errors="ignore")
    counts = {k: len(re.findall(rx, text, re.I)) for k, rx in EVENTS.items()}
    offsets = [abs(float(x)) for x in OFFSET.findall(text)]
    lines = [text.splitlines()[0] if text else str(path)]
    lines += [f"  {k}: {v}" for k, v in counts.items()]
    if offsets:
        over = sum(o > 5 for o in offsets)
        lines.append(
            f"  camera-to-IMU offset: median {statistics.median(offsets):.1f} ms, max {max(offsets):.1f} ms, "
            f"{over} of {len(offsets)} reports over ARCore's 5 ms limit"
        )
    else:
        lines.append("  camera-to-IMU offset: no reports (ARCore logs them only when they drift)")
    return "\n".join(lines)


def sweep(sweep_id: str) -> str:
    from claimagent.capture.store import SweepStore
    from claimagent.capture.tracking import assess, segment_of
    from claimagent.config import get_settings

    store = SweepStore(get_settings().sweeps_dir, sweep_id)
    metas = [m for m in store.frames() if m.pose]
    if not metas:
        return f"{sweep_id}: no AR frames (camera-only, or AR never reached 'ready')"
    report = assess(metas, store.load_depth)
    by_seg: dict[int, list] = collections.defaultdict(list)
    for m in metas:
        by_seg[segment_of(m)].append(m)
    out = [f"{sweep_id}: {len(metas)} AR frames in {len(by_seg)} tracking segment(s); {report.note or 'all trusted'}"]
    for seg, ms in sorted(by_seg.items()):
        pos = [m.pose[12:15] for m in ms if m.pose]
        path = sum(sum((a - b) ** 2 for a, b in zip(p, q, strict=True)) ** 0.5 for p, q in itertools.pairwise(pos))
        meds = []
        for m in ms:
            d = store.load_depth(m)
            if d is not None:
                v = d[(d > 0) & (d == d)]
                if v.size:
                    meds.append(float(statistics.median(v.ravel().tolist())))
        trusted = sum(m.seq in report.trusted for m in ms)
        reasons = collections.Counter(report.rejected[m.seq].split(" (")[0] for m in ms if m.seq in report.rejected)
        depth = (
            f"depth median per frame {min(meds):.2f}-{max(meds):.2f} m (middle {statistics.median(meds):.2f})"
            if meds
            else "no depth (depth-sensing not granted?)"
        )
        out.append(
            f"  segment {seg}: {len(ms)} frames, {trusted} trusted, path {path:.1f} m, {depth}; "
            f"rejected: {dict(reasons) or 'none'}"
        )
    room = store.dir / "stages" / "room.json"
    if room.exists():
        r = json.loads(room.read_text())
        keys = ("measured", "floor_area_m2", "wall_area_m2", "length_m", "width_m", "height_m")
        out.append(f"  room stage: { ({k: r.get(k) for k in keys}) } notes={r.get('notes', [])}")
    return "\n".join(out)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "record":
        record(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 45)
    elif len(sys.argv) == 3 and sys.argv[1] == "summary":
        print(summary(Path(sys.argv[2])))
    elif len(sys.argv) == 3 and sys.argv[1] == "sweep":
        print(sweep(sys.argv[2]))
    else:
        print(__doc__)
