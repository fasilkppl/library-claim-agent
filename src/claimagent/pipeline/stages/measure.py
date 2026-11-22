"""Spine size: catalogue heights of identified books as rulers in each frame (or metric depth, a marker or a
stated shelf width), the photo's width for thickness, combined with the catalogue page count."""

from __future__ import annotations

import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings
from claimagent.pipeline import geometry as geo
from claimagent.pipeline.register import Registrar, Track
from claimagent.pipeline.resolve import Resolver
from claimagent.pipeline.stages.common import MARKER_SIDE_CM_DEFAULT, camera_of
from claimagent.pipeline.stages.read import WIDTH_PERCENTILE, assign_rows, central_track, nearest_frame

log = logging.getLogger(__name__)


RULER_BUDGET_S = 30.0


RULER_WORKERS = 6


# Photo and page-count thickness err in opposite directions: the photo outline takes in a strip of cover or a
# neighbour's edge (too wide), the page count assumes thin paper (most Indian paperbacks are bulkier). Both
# errors are proportional, so the two are combined as a geometric mean. A photo more than twice or under half
# the page-count figure is a wrong outline (two spines merged, or a fragment): the page count alone is used.
THICK_RATIO_MAX = 2.0


# Scale sources that measure the picture itself (a printed marker of known size, AR depth, a shelf width said
# aloud). With them the photo is a measurement: on a marker-scaled test shelf photo widths were 2-7% off while
# page counts of other editions were 30-50% off. Catalogue-height rulers come from records that are often another
# edition, so a photo scaled by them is combined with the page count instead.
METRIC_SCALES = {"reference-marker", "webxr-depth", "stated-shelf-width"}


def fuse_thickness(photo: float | None, catalogue: tuple[float, str], metric_scale: bool = False) -> tuple[float, str]:
    """One thickness from the photo and the page-count estimate (see THICK_RATIO_MAX), with the reason."""
    cat, where = catalogue
    if photo is None or not 1 / THICK_RATIO_MAX <= photo / cat <= THICK_RATIO_MAX:
        measured = "none" if photo is None else f"{photo:.1f} cm"
        return cat, f"thickness from the catalogue ({where}); the photo gave {measured}"
    if metric_scale:
        return photo, f"thickness from the photo on a metric scale; catalogue {cat:.1f} cm ({where}) agrees within 2x"
    return (photo * cat) ** 0.5, f"thickness: photo {photo:.1f} cm and catalogue {cat:.1f} cm ({where}) combined"


def _spine_extent(t: Track, det: dict[int, dict[str, Any]], reg: Registrar) -> tuple[float, float]:
    """Spine length and width in segment pixels.

    Median over observations where the spine is fully in view (a spine cut by the frame edge is shorter
    and thinner than the book); each observation is mapped into segment pixels first. Measurement errors
    here are truncations (a fragment, a coarse detector box over half a spine), rarely over-extensions, so
    length uses a high percentile. Width errors go the other way: a spine seen at an angle also shows a
    strip of the cover, so width uses a low percentile of near-whole observations. Segmentation-based
    observations are preferred over coarse detector-only boxes."""

    def spine_of(o: Any) -> dict[str, Any]:
        return det[o.seq]["spines"][o.index]  # type: ignore[no-any-return]

    full = [o for o in t.observations if not spine_of(o)["touches_edge"]] or t.observations
    fine = [o for o in full if spine_of(o)["source"] != "yolo"] or full
    lens: list[float] = []
    wids: list[float] = []
    for o in fine:
        sp = spine_of(o)
        to_seg = reg.scale_at(o.seq, sp["cx"], sp["cy"])
        lens.append(sp["length"] * to_seg)
        wids.append(sp["width"] * to_seg)
    whole = [w for ln, w in zip(lens, wids, strict=True) if ln >= 0.6 * max(lens)]
    return float(np.percentile(lens, 80)), float(np.percentile(whole, WIDTH_PERCENTILE))


def measure_books(
    store: SweepStore,
    reg: Registrar,
    det: dict[int, dict[str, Any]],
    selected: list[FrameMeta],
    read: dict[str, Any],
    resolved: dict[str, Any],
    resolver: Resolver,
    s: Settings,
    ruler_budget_s: float = 30.0,
) -> dict[str, Any]:
    metas = {m.seq: m for m in selected}
    tracks = {t.id: t for t in reg.tracks}
    marker_cm = float(store.read_session().get("marker_side_cm") or MARKER_SIDE_CM_DEFAULT)
    depth_cache: dict[int, np.ndarray | None] = {}

    def depth(seq: int) -> np.ndarray | None:
        if seq not in depth_cache:
            depth_cache[seq] = store.load_depth(metas[seq])
        return depth_cache[seq]

    # --- per-observation metric size from WebXR depth
    xr: dict[int, tuple[float, float]] = {}
    for b in read["books"]:
        t = tracks[b["track"]]
        ls, ws = [], []
        for o in t.observations:
            cam, d = camera_of(metas[o.seq]), depth(o.seq)
            sp = det[o.seq]["spines"][o.index]
            if cam is None or d is None or sp["touches_edge"] or sp["source"] == "yolo":
                continue
            z = geo.depth_in_polygon(d, cam, sp["corners"])
            if z is None:
                continue
            f = (cam.fx * cam.fy) ** 0.5
            ls.append(sp["length"] * z / f * 100)
            ws.append(sp["width"] * z / f * 100)
        if ls:
            xr[t.id] = (float(np.percentile(ls, 80)), statistics.median(ws))

    # --- segment scales (segment px per cm)
    seg_scales: dict[int, dict[str, Any]] = {}
    for m in selected:
        mk = det[m.seq].get("marker")
        if not mk or m.seq not in reg.links or mk["side_spread"] > 0.25:
            continue
        c = np.array(mk["corners"]).mean(axis=0)
        px_per_cm = mk["side_px"] * reg.scale_at(m.seq, float(c[0]), float(c[1])) / marker_cm
        seg_scales.setdefault(reg.links[m.seq].segment, {}).setdefault("marker", []).append(px_per_cm)
    for ev in store.events("shelf_dimension"):
        fm = nearest_frame(selected, int(ev["t_ms"]))
        ct = central_track(reg, det, fm.seq) if fm else None
        if ct is None or not ev.get("width_cm"):
            continue
        rows = assign_rows(reg)
        label = rows.get(ct.id)
        kept = {b["track"] for b in read["books"]}  # rejected or excluded tracks must not widen the row
        same = [tracks[k] for k, r in rows.items() if label and r.label == label.label and k in kept]
        if not same:
            continue
        x0 = min(t.box()[0] for t in same)
        x1 = max(t.box()[2] for t in same)
        seg_scales.setdefault(ct.segment, {}).setdefault("voice", []).append((x1 - x0) / float(ev["width_cm"]))
    extents = {b["track"]: _spine_extent(tracks[b["track"]], det, reg) for b in read["books"]}
    resolver.deadline = time.monotonic() + min(RULER_BUDGET_S, ruler_budget_s)  # catalogue heights for rulers
    ruler_h: dict[int, float] = {}
    res_by_track = {r["track"]: r for r in resolved["books"]}
    # Review-level matches may serve as rulers: at least three must agree within 15% (below). The catalogue
    # lookups run in parallel: one after another they did not fit the packet's time budget.
    candidates = [
        (b, res_by_track[b["track"]])
        for b in read["books"]
        if res_by_track.get(b["track"], {}).get("status") == "identified"
        and res_by_track[b["track"]].get("confidence", 0) >= s.id_review
    ]

    def catalogue_size(br: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float | None, tuple[float, str] | None]:
        # The thickness comes from the records the height lookup has just read, so it is asked right after it
        # (a later round hit the stage's deadline and was skipped).
        r = br[1]
        h = resolver.catalogue_height_cm(
            r.get("volume_id", ""), r.get("title", ""), r.get("author", ""), r.get("url", "")
        )
        return h, resolver.catalogue_thickness_cm(
            r.get("volume_id", ""),
            r.get("url", ""),
            check_deadline=False,
            pages_hint=r.get("pages"),
            title=r.get("title", ""),
            author=r.get("author", ""),
        )

    with ThreadPoolExecutor(max_workers=RULER_WORKERS) as pool:
        sizes = list(pool.map(catalogue_size, candidates))
    heights = [h for h, _ in sizes]
    thick_cat = {b["track"]: th for (b, _), (_, th) in zip(candidates, sizes, strict=True) if th is not None}
    height_cat = {b["track"]: h for (b, _), h in zip(candidates, heights, strict=True) if h and 10 <= h <= 45}
    for (b, r), h_cm in zip(candidates, heights, strict=True):
        t = tracks[b["track"]]
        log.info("ruler candidate %s %r: catalogue height %s cm (%s)", t.id, r.get("title"), h_cm, t.orientation)
        if h_cm and 10 <= h_cm <= 45 and t.orientation == "upright":
            seg_scales.setdefault(t.segment, {}).setdefault("rulers", []).append(extents[t.id][0] / h_cm)
            ruler_h[t.id] = h_cm

    scale_report: dict[int, dict[str, Any]] = {}
    for seg, srcs in seg_scales.items():
        rep: dict[str, Any] = {}
        for k, vals in srcs.items():
            if k == "rulers":
                if len(vals) < 3:
                    continue
                med = statistics.median(vals)
                mad = statistics.median([abs(v - med) for v in vals])
                if mad / med > 0.15:
                    rep["rulers_rejected"] = f"{len(vals)} catalogue rulers disagree (MAD {mad / med:.0%})"
                    continue
            rep[k] = round(statistics.median(vals), 4)
        scale_report[seg] = rep

    frame_sizes = _frame_ruler_sizes(read, tracks, det, ruler_h, assign_rows(reg))

    out: list[dict[str, Any]] = []
    for b in read["books"]:
        t = tracks[b["track"]]

        seg_len, seg_wid = extents[t.id]
        rep = scale_report.get(t.segment, {})
        height = thick = None
        source = ""
        checks: list[str] = []
        if t.id in xr:
            height, thick = xr[t.id]
            source = "webxr-depth"
        elif t.id in frame_sizes and "marker" not in rep and "voice" not in rep:
            # Rulers in the same picture and shelf row: a local scale, so it beats one averaged over a segment
            # in which the distance to the shelf changes.
            height, thick = frame_sizes[t.id]
            source = "catalogue-heights-in-frame"
        else:
            for name in ("marker", "voice", "rulers"):
                if name in rep:
                    height, thick = seg_len / rep[name], seg_wid / rep[name]
                    source = {
                        "marker": "reference-marker",
                        "voice": "stated-shelf-width",
                        "rulers": "catalogue-book-heights",
                    }[name]
                    break

        # Cross-check against the other available sources.
        if height is not None:
            for name in ("marker", "voice", "rulers"):
                if (
                    name in rep
                    and source
                    != {
                        "marker": "reference-marker",
                        "voice": "stated-shelf-width",
                        "rulers": "catalogue-book-heights",
                    }[name]
                ):
                    alt = seg_len / rep[name]
                    diff = abs(alt - height) / height
                    if diff > s.scale_disagreement:
                        checks.append(f"{name} scale disagrees by {diff:.0%}")
        if height is None and t.id in height_cat:
            # No scale where this spine was seen (too few rulers in its frames and pass): its edition's height.
            height = height_cat[t.id]
            source = source or "catalogue-height"
            checks.append("height from the catalogue; the photo gave none (no scale where this spine was seen)")
        cat = thick_cat.get(t.id)
        photo_thick = thick
        if cat is not None:
            thick, why = fuse_thickness(thick, cat, source in METRIC_SCALES)
            checks.append(why)
        if b["only_partial"]:
            checks.append("spine only seen cut off at the frame edge; size is a lower bound")
        out.append(
            {
                "track": t.id,
                "spine_height_cm": None if height is None else round(height, 1),
                "spine_thickness_cm": None if thick is None else round(thick, 1),
                "photo_thickness_cm": None if photo_thick is None else round(photo_thick, 2),
                "catalogue_thickness_cm": cat[0] if cat else None,
                "catalogue_height_cm": ruler_h.get(t.id),
                "scale_source": source,
                "seg_len_px": round(seg_len, 1),
                "seg_wid_px": round(seg_wid, 1),
                "checks": checks,
            }
        )

    # Shelf run: union of book extents along each shelf (stacks share one footprint).
    by_track = {m["track"]: m for m in out}
    runs: dict[str, float] = {}
    run_complete = True
    for label in sorted({b["shelf"] for b in read["books"]}):
        row = [b for b in read["books"] if b["shelf"] == label]
        intervals: list[tuple[float, float]] = []
        for b in row:
            mm = by_track[b["track"]]
            if mm["spine_thickness_cm"] is None:
                run_complete = False
                continue
            t = tracks[b["track"]]
            bx = t.box()
            cm_per_px = (
                (mm["spine_thickness_cm"] / mm["seg_wid_px"])
                if t.orientation == "upright"
                else (mm["spine_height_cm"] / mm["seg_len_px"] if mm["spine_height_cm"] else 0)
            )
            intervals.append((bx[0] * cm_per_px, bx[2] * cm_per_px))
        intervals.sort()
        total = 0.0
        cur: list[float] | None = None
        for lo, hi in intervals:
            if cur is None or lo > cur[1]:
                if cur:
                    total += cur[1] - cur[0]
                cur = [lo, hi]
            else:
                cur[1] = max(cur[1], hi)
        if cur:
            total += cur[1] - cur[0]
        if intervals:  # a shelf with no metric scale has an unknown run, not a zero run
            runs[label] = round(total / 100, 3)
    data = {
        "books": out,
        "segment_scales": {str(k): v for k, v in scale_report.items()},
        "xr_measured": len(xr),
        "shelf_run_m": round(sum(runs.values()), 2) if runs else None,
        "shelf_run_by_shelf": runs,
        "shelf_run_complete": run_complete,
    }
    store.write_stage("measure", data)
    return data


FRAME_RULER_SPREAD = 0.15  # rulers in one frame row that disagree by more than this (MAD/median) are not used


def _frame_ruler_sizes(
    read: dict[str, Any],
    tracks: dict[int, Any],
    det: dict[int, dict[str, Any]],
    ruler_h: dict[int, float],
    rows: dict[int, Any],
) -> dict[int, tuple[float, float]]:
    """Sizes from rulers in the same frame: a book of known printed height standing in the same shelf row of
    one picture gives that picture's pixels per centimetre for its neighbours. Unlike the per-segment scale it
    needs no unbroken tracking, only one identified book next to the others. A ruler book itself is measured
    from the other rulers in its frames. Each book: median over its frames."""
    # A book's tracks: its own and those merged into it (the same book seen on another pass).
    members_of = {
        int(b["track"]): [int(b["track"]), *[int(m) for m in b.get("merged_tracks", []) if int(m) in tracks]]
        for b in read["books"]
    }
    per_frame: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for tid, h in ruler_h.items():
        for m in members_of.get(tid, [tid]):
            for o in tracks[m].observations:
                sp = det[o.seq]["spines"][o.index]
                if not sp["touches_edge"] and m in rows:
                    per_frame.setdefault((o.seq, rows[m].label), []).append((tid, float(sp["length"]) / h))
    sizes: dict[int, tuple[float, float]] = {}
    for b in read["books"]:
        tid = int(b["track"])
        if tracks[tid].orientation != "upright":
            continue
        members = members_of[tid]
        hs, ws = [], []
        for m in members:
            if m not in rows:
                continue
            for o in tracks[m].observations:
                sp = det[o.seq]["spines"][o.index]
                if sp["touches_edge"]:
                    continue
                scales = [v for r, v in per_frame.get((o.seq, rows[m].label), []) if r != tid]
                if not scales:
                    continue
                med = statistics.median(scales)
                if len(scales) >= 2 and statistics.median(abs(v - med) for v in scales) / med > FRAME_RULER_SPREAD:
                    continue
                hs.append(float(sp["length"]) / med)
                ws.append(float(sp["width"]) / med)
        if hs:
            # Width as in the per-segment measure: a low percentile, since mask edges bleed into neighbours.
            # Median height over the frames (a higher percentile was tried and measured worse on 2 of 4 sweeps).
            # Width from a low percentile: mask edges bleed into neighbours, never shrink.
            sizes[tid] = (statistics.median(hs), float(np.percentile(ws, WIDTH_PERCENTILE)))
    log.info("in-frame rulers sized %s books (%s rulers)", len(sizes), len(ruler_h))
    return sizes
