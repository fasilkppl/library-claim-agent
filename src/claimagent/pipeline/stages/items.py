"""Non-book items: detections linked into one entry per object."""

from __future__ import annotations

import statistics
from typing import Any

import cv2
import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import get_settings
from claimagent.pipeline import geometry as geo
from claimagent.pipeline import itemtrack, vlm
from claimagent.pipeline.household import CATEGORY
from claimagent.pipeline.register import Registrar
from claimagent.pipeline.stages.common import camera_of

# ============================================================================ items
SINGLE_SIGHTING_MIN_CONF = 0.6
EDGE_FRAGMENT_MIN_CONF = 0.8
DEDUPE_RADIUS_M = {"shelving": 1.0, "seating": 0.7, "table": 0.8, "rug": 1.2}
MIN_CONF = 0.25


def collect_items(
    store: SweepStore, reg: Registrar, det: dict[int, dict[str, Any]], selected: list[FrameMeta]
) -> dict[str, Any]:
    metas = {m.seq: m for m in selected}
    obs = []
    for m in selected:
        d = det[m.seq]
        for k, it in enumerate(d.get("items", [])):
            if it["conf"] < MIN_CONF and it["label"] not in ("framed picture", "painting", "rug", "floor lamp"):
                continue
            cat = CATEGORY.get(it["label"], "other")
            x0, y0, x1, y1 = it["bbox"]
            entry = {
                "seq": m.seq,
                "k": k,
                "label": it["label"],
                "category": cat,
                "conf": it["conf"],
                "bbox": it["bbox"],
                "text": it.get("text", []),
                "world": None,
                "dims_cm": None,
            }
            cam, depth = camera_of(m), store.load_depth(m)
            if cam is not None and depth is not None:
                corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                z = geo.depth_in_polygon(depth, cam, corners)
                if z is not None:
                    c = cam.backproject(np.array([(x0 + x1) / 2]), np.array([(y0 + y1) / 2]), np.array([z]))[0]
                    entry["world"] = c.round(3).tolist()
                    entry["dims_cm"] = {
                        "w": round((x1 - x0) * z / cam.fx * 100, 0),
                        "h": round((y1 - y0) * z / cam.fy * 100, 0),
                    }
            entry["touches_edge"] = x0 < 4 or y0 < 4 or x1 > det[m.seq]["width"] - 4 or y1 > det[m.seq]["height"] - 4
            obs.append(entry)
    checked = [m for m in selected if det[m.seq].get("items_checked", True)]
    clusters = _item_clusters(store, checked, obs)
    items = []
    for i, c in enumerate(clusters):
        best = max(c, key=lambda e: e["conf"] * (0.5 if e["touches_edge"] else 1.0))
        dims = [e["dims_cm"] for e in c if e["dims_cm"] and not e["touches_edge"]]
        votes: dict[str, tuple[int, float]] = {}
        for e in c:
            n, w = votes.get(e["label"], (0, 0.0))
            votes[e["label"]] = (n + 1, w + e["conf"])
        label = max(votes, key=lambda lb: (votes[lb], lb))  # most sightings, then summed confidence
        conf = min(0.99, max(e["conf"] for e in c) * (1 + 0.1 * min(len(c) - 1, 3)))
        # One sighting that the detector is unsure of is noise (often far background) more often than an
        # object in the room.
        if len(c) == 1 and best["conf"] < SINGLE_SIGHTING_MIN_CONF and best["category"] not in ("art", "rug"):
            continue
        # Seen only cut by the frame edge, once or twice: a part of an object seen whole elsewhere (a pan
        # handle, the rim of a bucket) more often than an object of its own.
        if len(c) <= 2 and all(e["touches_edge"] for e in c) and best["conf"] < EDGE_FRAGMENT_MIN_CONF:
            continue
        items.append(
            {
                "id": f"I{i + 1:03d}",
                "label": label,
                "category": CATEGORY.get(label, best["category"]),  # follows the label shown
                "frame_ref": metas[best["seq"]].frame_ref,
                "frame_seq": best["seq"],
                "bbox": best["bbox"],
                "dims_cm": {k: statistics.median([d[k] for d in dims]) for k in ("w", "h")} if dims else None,
                "text": sorted({t for e in c for t in e["text"]})[:10],
                "observations": len(c),
                "confidence": round(conf, 2),
                "world": best["world"],
            }
        )
    removed: list[dict[str, Any]] = []
    if get_settings().item_vision_check and items:
        images = {}
        for it in items:
            img = cv2.imread(str(store.frame_path(metas[it["frame_seq"]])))
            if img is not None:
                images[it["frame_seq"]] = img
        removed = vlm.name_items(items, images)
    data = {"items": items, "observations": len(obs), "removed_fixtures": removed}
    store.write_stage("items", data)
    return data


def _item_clusters(
    store: SweepStore, selected: list[FrameMeta], obs: list[dict[str, Any]]
) -> list[list[dict[str, Any]]]:
    """Sightings of one object: linked frame to frame by image motion, then tracks joined by 3D position
    when AR tracking is trusted (the same object seen again later in the sweep)."""
    sightings = [itemtrack.Sighting(e["seq"], e["k"], e["label"], tuple(e["bbox"]), e["conf"]) for e in obs]
    keep = {
        (s.seq, s.index)
        for seq in {s.seq for s in sightings}
        for s in itemtrack.drop_parts([x for x in sightings if x.seq == seq])
    }
    obs = [e for e in obs if (e["seq"], e["k"]) in keep]
    frames, colours = [], {}
    for m in sorted(selected, key=lambda m: m.seq):
        img = cv2.imread(str(store.frame_path(m)))
        if img is not None:
            colours[m.seq] = img
            frames.append((m.seq, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
    linked = [s for s in sightings if (s.seq, s.index) in keep]
    ids = itemtrack.link(frames, linked)
    # Sightings in frames that could not be read have no track: they are left out, not a crash.
    linked = [s for s in linked if (s.seq, s.index) in ids]
    obs = [e for e in obs if (e["seq"], e["k"]) in ids]
    by_track: dict[int, list[itemtrack.Sighting]] = {}
    for s in linked:
        by_track.setdefault(ids[(s.seq, s.index)], []).append(s)
    # Revisits: tracks of one object, joined by appearance (image-only) ...
    group_of = {}
    for g, members in enumerate(itemtrack.same_object_groups(by_track, dict(frames), colours)):
        for tid in members:
            group_of[tid] = g
    tracks: dict[int, list[dict[str, Any]]] = {}
    for e in obs:
        tracks.setdefault(group_of[ids[(e["seq"], e["k"])]], []).append(e)
    # ... and by 3D position when AR tracking is trusted.
    clusters: list[list[dict[str, Any]]] = []
    centres: list[np.ndarray | None] = []
    for c in sorted(tracks.values(), key=lambda c: -max(e["conf"] for e in c)):
        worlds = [e["world"] for e in c if e["world"] is not None]
        centre = np.median(np.array(worlds), axis=0) if worlds else None
        k = itemtrack.kind(c[0]["label"])
        seen_in = {e["seq"] for e in c}
        for j, other in enumerate(clusters):
            oc = centres[j]
            if centre is None or oc is None or itemtrack.kind(other[0]["label"]) != k:
                continue
            if seen_in & {e["seq"] for e in other}:  # both in one frame: two objects, not one seen twice
                continue
            if float(np.linalg.norm(centre - oc)) < DEDUPE_RADIUS_M.get(c[0]["category"], 0.45):
                other.extend(c)
                break
        else:
            clusters.append(list(c))
            centres.append(centre)
    return clusters
