"""Non-book items: detections linked into one entry per object."""

from __future__ import annotations

import logging
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

log = logging.getLogger(__name__)

# ============================================================================ items
# Recall first: in a continuous pan most objects are seen in one or two analysed frames, at low detector
# confidence and often under the wrong name (an iron called "mouse"). Every kept item is named again by the vision
# model, which also removes building parts, so the detector only has to find candidates. Measured on three real
# sweeps: items found 2/6/11 of 18 -> 6/7/15 of 18 with these values (previously 0.25 / 0.6 / 0.8).
SINGLE_SIGHTING_MIN_CONF = 0.35
EDGE_FRAGMENT_MIN_CONF = 0.6
DEDUPE_RADIUS_M = {"shelving": 1.0, "seating": 0.7, "table": 0.8, "rug": 1.2}
MIN_CONF = 0.15


def guided_items(store: SweepStore) -> list[dict[str, Any]]:
    """Items confirmed one at a time by the guide: exactly one entry each, in the order shown."""
    frames = {m.seq: m for m in store.frames()}
    out: list[dict[str, Any]] = []
    for ev in store.events("guide_item"):
        seq = int(ev["seqs"][-1])  # the frame the item was confirmed on; the box is from that frame
        if seq not in frames:
            continue
        label = str(ev["label"])
        out.append(
            {
                "id": f"I{len(out) + 1:03d}",
                "label": label,
                "category": CATEGORY.get(label, "other"),
                "frame_ref": frames[seq].frame_ref,
                "frame_seq": seq,
                "bbox": ev["bbox"],
                "dims_cm": None,
                "text": [],
                "observations": len(ev["seqs"]),
                "confidence": float(ev.get("conf", 0.5)),
                "world": None,
                "guided": int(ev["n"]),
                "forced": bool(ev.get("forced", False)),
            }
        )
    return out


def collect_items(
    store: SweepStore,
    reg: Registrar,
    det: dict[int, dict[str, Any]],
    selected: list[FrameMeta],
    vlm_budget_s: float = vlm.BUDGET_S,
) -> dict[str, Any]:
    guided = guided_items(store)
    if guided:
        # A guided sweep: the items shown one by one are the items. Detections elsewhere (a pan in the corner
        # of a shelf frame, the same bucket seen again) are not added, so nothing is counted twice.
        log.info("items: %s confirmed by the guide", len(guided))
        return _finish_items(
            store, guided, {m.seq: m for m in store.frames()}, observations=len(guided), vlm_budget_s=vlm_budget_s
        )
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
                "seqs": sorted({e["seq"] for e in c}),
                "boxes": {str(e["seq"]): list(e["bbox"]) for e in c},
            }
        )
    return _finish_items(store, items, metas, observations=len(obs), vlm_budget_s=vlm_budget_s)


def _finish_items(
    store: SweepStore,
    items: list[dict[str, Any]],
    metas: dict[int, FrameMeta],
    observations: int,
    vlm_budget_s: float = vlm.BUDGET_S,
) -> dict[str, Any]:
    """Names each item from its picture (vision model) and removes building fixtures and books."""
    removed: list[dict[str, Any]] = []
    for it in items:
        m = metas.get(it["frame_seq"])
        it["room_step"] = bool(m is not None and (m.extra or {}).get("step") == "room")
    if get_settings().item_vision_check and items:
        images = {}
        for it in items:
            img = cv2.imread(str(store.frame_path(metas[it["frame_seq"]])))
            if img is not None:
                images[it["frame_seq"]] = img
        removed = vlm.name_items(items, images, vlm_budget_s)
    merged = _drop_repeats(store, items, metas)
    data = {"items": items, "observations": observations, "removed_fixtures": removed, "merged_repeats": merged}
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


# Colour-histogram correlation at which two items of the same name are one object seen again (measured on a real
# sweep: the same box from other angles 0.57-0.99, different objects at most 0.33).
SAME_ITEM_HIST = 0.5
SEPARATE_IOU = 0.1  # boxes in one picture overlapping less than this are two different objects


def _item_hist(img: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    x0, y0, x1, y1 = (int(v) for v in bbox)
    crop = img[max(0, y0) : y1, max(0, x0) : x1]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    out: np.ndarray = cv2.normalize(h, h).flatten()
    return out


def _drop_repeats(store: SweepStore, items: list[dict[str, Any]], metas: dict[int, FrameMeta]) -> list[dict[str, Any]]:
    """Final check: no object listed twice. Two entries with the same final name and the same look are one
    object seen twice; the better-seen entry is kept and the merge is recorded."""
    hists: dict[str, np.ndarray | None] = {}
    for it in items:
        m = metas.get(it["frame_seq"])
        img = cv2.imread(str(store.frame_path(m))) if m is not None else None
        hists[it["id"]] = _item_hist(img, it["bbox"]) if img is not None else None
    merged: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for it in sorted(items, key=lambda x: -x["observations"]):
        twin = None
        for k in kept:
            if k["label"] != it["label"]:
                continue
            if "seqs" in it and "seqs" in k:
                # Two objects in one picture are two objects; the same name seen at different moments of the
                # walk is one object seen again (unless its colours clearly differ while both are in view).
                together = set(it["seqs"]) & set(k["seqs"])
                # In a shared picture they are two objects only if they sit in different places; boxes that
                # overlap there are one object detected twice.
                apart = [
                    seq
                    for seq in together
                    if _box_iou(it["boxes"].get(str(seq), []), k["boxes"].get(str(seq), [])) < SEPARATE_IOU
                ]
                ha, hb = hists[k["id"]], hists[it["id"]]
                looks_same = (
                    ha is not None
                    and hb is not None
                    and float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)) >= SAME_ITEM_HIST
                )
                if not apart or looks_same:
                    twin = k
                    break
                continue
            if "guided" in it and "guided" in k:
                # Shown one by one: the same name is the same object shown again, unless the policyholder said it
                # is a different one when it was recorded.
                if not (it.get("forced") or k.get("forced")):
                    twin = k
                    break
                continue
            ha, hb = hists[k["id"]], hists[it["id"]]
            if (
                ha is not None
                and hb is not None
                and float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)) >= SAME_ITEM_HIST
            ):
                twin = k
                break
        if twin is None:
            kept.append(it)
            continue
        twin["observations"] += it["observations"]
        merged.append({"kept": twin["id"], "merged": it["id"], "label": it["label"]})
        log.info("ITEMS repeat: %s is the same %s as %s", it["id"], it["label"], twin["id"])
    keep_ids = {k["id"] for k in kept}
    items[:] = [it for it in items if it["id"] in keep_ids]
    return merged


def _box_iou(a: list[float], b: list[float]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0
