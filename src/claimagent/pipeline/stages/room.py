"""Room size from the AR depth point cloud."""

from __future__ import annotations

from typing import Any

import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.pipeline import geometry as geo
from claimagent.pipeline.stages.common import camera_of

MIN_ROOM_FRAMES = 8  # a room outline needs depth from several directions


# ============================================================================ room
def measure_room(
    store: SweepStore,
    selected_all: list[FrameMeta],
    items: dict[str, Any],
    tracking_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pts, cams, refs = [], [], []
    for m in selected_all:
        cam = camera_of(m)
        depth = store.load_depth(m)
        if cam is None or depth is None:
            continue
        pts.append(geo.frame_points(cam, depth))
        cams.append(cam.position)
        refs.append(m.frame_ref)
    tracking_note = (tracking_data or {}).get("note", "")
    if len(pts) < MIN_ROOM_FRAMES:
        if tracking_note:
            why = f"{tracking_note}. Room size not measured: too few frames with reliable AR depth."
        elif pts:
            why = f"only {len(pts)} frames with AR depth: room size not measured."
        else:
            why = (
                "no metric depth in this sweep (camera-only mode): room size not measured. Use the Android WebXR "
                "capture for room dimensions."
            )
        data: dict[str, Any] = {"measured": False, "notes": [why], "frame_refs": refs[:12]}
        store.write_stage("room", data)
        return data
    r = geo.measure_room(np.concatenate(pts), np.array(cams))
    shelved = 0.0
    shelf_items = [i for i in items["items"] if i["category"] == "shelving" and i.get("dims_cm")]
    for i in shelf_items:
        shelved += i["dims_cm"]["w"] * i["dims_cm"]["h"] / 10000
    data = {
        "measured": r.floor_area_m2 is not None,
        **r.__dict__,
        "frame_refs": refs[:: max(1, len(refs) // 12)],
        "shelved_wall_area_m2": round(shelved, 2) if shelf_items else None,
        "shelving_units_measured": len(shelf_items),
    }
    if tracking_note:
        data["notes"] = [*data.get("notes", []), tracking_note + "; those frames were left out of the room"]
    store.write_stage("room", data)
    return data
