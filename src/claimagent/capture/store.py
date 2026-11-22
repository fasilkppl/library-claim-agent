"""On-disk sweep layout. Everything the packet cites lives here, so every frame_ref can be opened.

sweeps/<id>/
  session.json          locale, device, capture mode, start/end
  frames/000001.jpg     keyframes (full resolution)
  frames/000001.json    t_ms, camera pose, intrinsics, depth file name
  depth/000001.npy      metric depth (float32 metres), when WebXR depth is available
  events.jsonl          transcript turns, tool calls (corrections), capture hints
  stages/<stage>.json   output of each pipeline stage
  cache/                raw HTTP responses (price and catalogue evidence)
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_sweep_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


@dataclass
class FrameMeta:
    seq: int
    t_ms: int
    width: int
    height: int
    pose: list[float] | None = None  # camera-to-world 4x4, column-major (WebXR convention)
    intrinsics: dict[str, float] | None = None
    depth_file: str | None = None
    depth_uv_matrix: list[float] | None = None  # normalized view uv -> normalized depth-buffer uv, column-major
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.seq:06d}"

    @property
    def frame_ref(self) -> str:
        return f"frames/{self.name}.jpg"


class SweepStore:
    def __init__(self, root: Path, sweep_id: str) -> None:
        self.id = sweep_id
        self.dir = root / sweep_id
        self.frames_dir = self.dir / "frames"
        self.depth_dir = self.dir / "depth"
        self.stages_dir = self.dir / "stages"
        self.cache_dir = self.dir / "cache"
        for d in (self.frames_dir, self.depth_dir, self.stages_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = len(list(self.frames_dir.glob("*.jpg")))

    # session -------------------------------------------------------------
    @property
    def session_path(self) -> Path:
        return self.dir / "session.json"

    def read_session(self) -> dict[str, Any]:
        if not self.session_path.exists():
            return {}
        data: dict[str, Any] = json.loads(self.session_path.read_text())
        return data

    def update_session(self, **values: Any) -> dict[str, Any]:
        with self._lock:
            data = self.read_session()
            data.update(values)
            tmp = self.session_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2))
            os.replace(tmp, self.session_path)  # readers never see a half-written file
            return data

    # frames --------------------------------------------------------------
    def add_frame(
        self,
        jpeg: bytes,
        t_ms: int,
        width: int,
        height: int,
        pose: list[float] | None = None,
        intrinsics: dict[str, float] | None = None,
        depth: np.ndarray | None = None,
        depth_uv_matrix: list[float] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> FrameMeta:
        if not jpeg.startswith(b"\xff\xd8"):
            raise ValueError("frame must be a JPEG")
        with self._lock:
            self._seq += 1
            seq = self._seq
        meta = FrameMeta(
            seq=seq,
            t_ms=t_ms,
            width=width,
            height=height,
            pose=pose,
            intrinsics=intrinsics,
            depth_uv_matrix=depth_uv_matrix,
            extra=extra or {},
        )
        (self.frames_dir / f"{meta.name}.jpg").write_bytes(jpeg)
        if depth is not None:
            meta.depth_file = f"depth/{meta.name}.npy"
            np.save(self.dir / meta.depth_file, depth.astype(np.float32))
        (self.frames_dir / f"{meta.name}.json").write_text(json.dumps(meta.__dict__))
        return meta

    def frames(self) -> list[FrameMeta]:
        metas = [FrameMeta(**json.loads(p.read_text())) for p in sorted(self.frames_dir.glob("*.json"))]
        return sorted(metas, key=lambda m: m.seq)

    def frame_path(self, meta: FrameMeta) -> Path:
        return self.dir / meta.frame_ref

    def load_depth(self, meta: FrameMeta) -> np.ndarray | None:
        if not meta.depth_file:
            return None
        path = self.dir / meta.depth_file
        return np.load(path) if path.exists() else None

    # events --------------------------------------------------------------
    def save_audio(self, audio: np.ndarray, rate: int) -> str:
        """Saves one spoken utterance (mono float32) as a 16-bit WAV in audio/; returns its path in the sweep."""
        import wave

        folder = self.dir / "audio"
        folder.mkdir(exist_ok=True)
        with self._lock:
            name = f"audio/utt_{len(list(folder.glob('utt_*.wav'))) + 1:03d}.wav"
        with wave.open(str(self.dir / name), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        return name

    def log_event(self, kind: str, t_ms: int | None = None, **data: Any) -> dict[str, Any]:
        event = {"kind": kind, "t_ms": t_ms if t_ms is not None else self.elapsed_ms(), "at": utc_now(), **data}
        with self._lock, (self.dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        path = self.dir / "events.jsonl"
        if not path.exists():
            return []
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return [r for r in rows if kind is None or r["kind"] == kind]

    def elapsed_ms(self) -> int:
        started = self.read_session().get("started_monotonic")
        return int((time.monotonic() - started) * 1000) if started else 0

    # stage outputs -------------------------------------------------------
    def write_stage(self, name: str, data: Any) -> Path:
        path = self.stages_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=1, default=_json_default))
        return path

    def read_stage(self, name: str) -> Any:
        return json.loads((self.stages_dir / f"{name}.json").read_text())


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")
