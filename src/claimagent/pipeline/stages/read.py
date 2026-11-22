"""Reading stage: one text per spine track, from every frame's OCR, close re-reads of its clearest views
and, for spines the OCR could not read, the local vision model's copy."""

from __future__ import annotations

import logging
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.pipeline import itemtrack
from claimagent.pipeline.analyze import CROP_MIN_LENGTH_PX, read_spine_crop
from claimagent.pipeline.boxes import box_iou
from claimagent.pipeline.register import Registrar, Track
from claimagent.pipeline.vision import TextReader

log = logging.getLogger(__name__)


def _obs_text(sp: dict[str, Any]) -> tuple[str, float]:
    lines = sp.get("text", [])
    text = " | ".join(t["text"] for t in lines)
    score = sum(len(re.findall(r"[A-Za-zÀ-ÿ]", t["text"])) * t["conf"] for t in lines)
    return text, score


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def consensus_text(obs_lines: list[list[dict[str, Any]]], widths: list[float] | None = None) -> str:
    """Lines read on a spine across frames. A line seen in only one of many frames is usually a
    neighbour's text that leaked in; the clearest reading of each repeated line is kept.

    Observations whose spine width is far from the track's median were segmented wrongly (merged with a
    neighbour, or a fragment) and do not vote."""
    from rapidfuzz import fuzz

    if widths and len(widths) == len(obs_lines) and len(widths) >= 3:
        med = statistics.median(widths)
        obs_lines = [ls for ls, w in zip(obs_lines, widths, strict=True) if abs(w - med) <= 0.4 * med]
    with_text = [ls for ls in obs_lines if ls]
    if not with_text:
        return ""
    clusters: list[dict[str, Any]] = []
    for k, ls in enumerate(with_text):
        for ln in ls:
            key = _key(ln["text"])
            if len(key) < 2:
                continue
            for c in clusters:
                if fuzz.ratio(key, c["key"]) >= 75 or (len(key) > 6 and fuzz.partial_ratio(key, c["key"]) >= 90):
                    c["frames"].add(k)
                    if ln["conf"] * len(key) > c["best"]["conf"] * len(c["key"]):
                        c["best"], c["key"] = ln, key
                    break
            else:
                clusters.append({"key": key, "best": ln, "frames": {k}, "order": len(clusters)})
    need = 1 if len(with_text) <= 2 else max(2, round(0.3 * len(with_text)))
    kept = [c for c in clusters if len(c["frames"]) >= need]
    if not kept:
        kept = sorted(clusters, key=lambda c: -len(c["frames"]))[:1]
    kept.sort(key=lambda c: c["order"])
    return " | ".join(c["best"]["text"] for c in kept)


def nearest_frame(selected: list[FrameMeta], t_ms: int) -> FrameMeta | None:
    return min(selected, key=lambda m: abs(m.t_ms - t_ms)) if selected else None


def central_track(reg: Registrar, det: dict[int, dict[str, Any]], seq: int) -> Track | None:
    d = det.get(seq)
    if not d or not d["spines"]:
        return None
    cx, cy = d["width"] / 2, d["height"] / 2
    idx = min(
        range(len(d["spines"])), key=lambda i: (d["spines"][i]["cx"] - cx) ** 2 + (d["spines"][i]["cy"] - cy) ** 2
    )
    for t in reg.tracks:
        if any(o.seq == seq and o.index == idx for o in t.observations):
            return t
    return None


@dataclass
class RowInfo:
    label: str
    position: int


def assign_rows(reg: Registrar) -> dict[int, RowInfo]:
    """Shelf rows inside each segment: upright books on one board share their bottom edge."""
    out: dict[int, RowInfo] = {}
    for seg in range(reg.segments):
        tracks = [t for t in reg.tracks if t.segment == seg]
        if not tracks:
            continue
        lengths = [max(t.box()[2] - t.box()[0], t.box()[3] - t.box()[1]) for t in tracks]
        tol = 0.35 * statistics.median(lengths)
        upright = sorted((t for t in tracks if t.orientation == "upright"), key=lambda t: t.box()[3])
        rows: list[list[Track]] = []
        for t in upright:
            if rows and abs(t.box()[3] - statistics.median([r.box()[3] for r in rows[-1]])) <= tol:
                rows[-1].append(t)
            else:
                rows.append([t])
        for t in (t for t in tracks if t.orientation != "upright"):
            cy = (t.box()[1] + t.box()[3]) / 2
            best = min(
                range(len(rows)), key=lambda k: abs(statistics.median([r.box()[3] for r in rows[k]]) - cy), default=None
            )
            if best is None or abs(statistics.median([r.box()[3] for r in rows[best]]) - t.box()[3]) > 2 * tol:
                rows.append([t])
            else:
                rows[best].append(t)
        rows.sort(key=lambda r: statistics.median([t.box()[3] for t in r]))
        for k, row in enumerate(rows):
            for p, t in enumerate(sorted(row, key=lambda t: t.box()[0]), start=1):
                out[t.id] = RowInfo(f"unit {seg + 1} / shelf {k + 1}", p)
    return out


WIDTH_PERCENTILE = 10


SHARP_ENOUGH = 100.0  # detail sharpness above which a view is not penalised when picking the clearest views


NOT_BOOK_IOU = 0.6  # a spine box this much covered by another object's box is that object


NOT_BOOK_SHARE = 0.3  # in this share of the spine's frames (detectors miss objects in some frames)


NOISE_MAX_OBS = 2


ODD_LENGTH = 1.4


GLANCE_FRAMES = 4  # a registration segment shorter than this is a glance, not a look at a shelf


NOT_BOOK_CONF = 0.5  # object detections below this confidence do not overrule the book detector on their own


# Labels a book is easily mistaken for: on their own they do not prove the spine is not a book.
BOOK_LIKE = {"box", "cardboard box", "gift box", "storage box", "trunk", "toolbox", "lunch box"}


def _not_books(reg: Registrar, det: dict[int, dict[str, Any]]) -> dict[int, str]:
    """Spine tracks that are not books (each reason is kept in the read stage):
    - the object detector sees another object on the same spot: a bottle, a phone; for box-like objects, which a
      book resembles, only when the spine is also out of place in its row;
    - a textless sliver cut by the frame edge, seen once or twice, lying on a better-seen counted book;
    - a strip seen once or twice at the frame edge lying across a row of standing books;
    - a textless detection seen once during a brief glance."""
    upright = sum(1 for t in reg.tracks if t.orientation == "upright")
    majority = "upright" if upright * 2 >= len(reg.tracks) else "flat"
    lengths = {t.id: max(det[o.seq]["spines"][o.index]["length"] for o in t.observations) for t in reg.tracks}
    frames_in_segment: dict[int, int] = {}
    for link in reg.links.values():
        frames_in_segment[link.segment] = frames_in_segment.get(link.segment, 0) + 1
    # The typical length comes from spines seen properly: one-frame slivers at the frame edge would pull it down
    # and make the tallest book look out of place.
    seen = [
        lengths[t.id]
        for t in reg.tracks
        if t.orientation == majority
        and len(t.observations) > NOISE_MAX_OBS
        and not all(det[o.seq]["spines"][o.index]["touches_edge"] for o in t.observations)
    ]
    typical = statistics.median(seen or list(lengths.values())) if lengths else 0.0
    out: dict[int, str] = {}
    for t in reg.tracks:
        odd_shape = t.orientation != majority or lengths[t.id] > ODD_LENGTH * typical
        texts = [det[o.seq]["spines"][o.index].get("text") for o in t.observations]
        hits: dict[str, int] = {}
        strong: set[str] = set()
        for o in t.observations:
            box = det[o.seq]["spines"][o.index]["bbox"]
            for it in det[o.seq].get("items", []):
                if box_iou(box, it["bbox"]) >= NOT_BOOK_IOU:
                    hits[it["label"]] = hits.get(it["label"], 0) + 1
                    if it["conf"] >= NOT_BOOK_CONF:
                        strong.add(it["label"])
                    break
        label = max(hits, key=lambda k: hits[k]) if hits else ""
        confirmed = label and hits[label] >= max(1, NOT_BOOK_SHARE * len(t.observations))
        # Box-like objects only count with an out-of-place shape; anything else (a bottle) must be a confident
        # detection on a spine that shows no text (printed words on it mean a book the detector mislabelled).
        if confirmed and (odd_shape or (label not in BOOK_LIKE and label in strong and not any(texts))):
            out[t.id] = f"the object detector sees a {label} here, not a book"
            continue
        partial = all(det[o.seq]["spines"][o.index]["touches_edge"] for o in t.observations)
        if len(t.observations) <= NOISE_MAX_OBS and partial and t.orientation != majority:
            out[t.id] = f"seen only cut off at the frame edge, in {len(t.observations)} frame(s), across the row"
            continue
        # A textless sliver cut by the frame edge, seen once or twice, lying on a book already counted (and
        # better seen) is a partial view of that book; one standing where no other book is counted may be a real
        # book only ever seen at the edge, and is kept. (A thin side face next to a book cannot be told apart
        # from a thin book, so it is kept too: counting a real book matters more.)
        if len(t.observations) <= NOISE_MAX_OBS and partial and not any(texts):
            owner = next(
                (
                    o
                    for o in reg.tracks
                    if o.id != t.id
                    and o.segment == t.segment
                    and len(o.observations) > len(t.observations)
                    and itemtrack.inside_fraction(t.box(), o.box()) >= 0.5
                ),
                None,
            )
            if owner is not None:
                out[t.id] = f"a partial view of a book already counted (track {owner.id})"
                continue
        if len(t.observations) == 1 and not any(texts) and frames_in_segment.get(t.segment, 0) < GLANCE_FRAMES:
            out[t.id] = "seen once, without text, during a brief glance"
    return out


CLOSE_READS_PER_SPINE = 3


# A spine the OCR read nothing on in this many frames is a real spine whose print it cannot see (gold on dark
# cloth): the local vision model copies its text from its two closest views (about 6 s each).
VISION_READ_MIN_FRAMES = 3


VISION_READS_PER_SPINE = 2


VISION_READ_BUDGET_S = 30.0


def _close_readings(
    store: SweepStore, t: Track, det: dict[int, dict[str, Any]], metas: dict[int, FrameMeta], reader: TextReader
) -> list[list[dict[str, Any]]]:
    """Close readings of a spine on its clearest views (whole, sharp, longest), one list of lines each."""

    def rank(o: Any) -> float:
        sp = det[o.seq]["spines"][o.index]
        q = float(det[o.seq]["quality"]["sharpness"])
        return float(sp["length"]) * (0.3 if sp["touches_edge"] else 1.0) * min(1.0, q / SHARP_ENOUGH)

    out = []
    for o in sorted(t.observations, key=rank, reverse=True)[:CLOSE_READS_PER_SPINE]:
        sp = det[o.seq]["spines"][o.index]
        img = cv2.imread(str(store.frame_path(metas[o.seq])))
        if img is not None and sp["length"] >= CROP_MIN_LENGTH_PX:
            out.append(read_spine_crop(img, sp, reader))
    return out


READ_WORKERS = 4  # spines read closely in parallel: Apple Vision and the crops run outside the Python lock


def _all_close_readings(
    store: SweepStore,
    tracks: list[Track],
    det: dict[int, dict[str, Any]],
    metas: dict[int, FrameMeta],
    reader: TextReader,
) -> dict[int, list[list[dict[str, Any]]]]:
    """Close readings of every spine, a few spines at a time (one OCR pass is ~50 ms; 120 spines in a row
    are most of the reading stage)."""

    def one(t: Track) -> list[list[dict[str, Any]]]:
        try:
            return _close_readings(store, t, det, metas, reader)
        except (cv2.error, ValueError, RuntimeError):  # a degenerate crop: this spine keeps its frame readings
            log.warning("close reading failed for track %s", t.id, exc_info=True)
            return []

    with ThreadPoolExecutor(max_workers=READ_WORKERS) as pool:
        return dict(zip([t.id for t in tracks], pool.map(one, tracks), strict=True))


def _spine_strip(img: np.ndarray, corners: Any) -> np.ndarray:
    """The spine alone, straightened to a horizontal strip."""
    pts = np.array(corners, np.float32)
    d01, d12 = np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[1] - pts[2])
    if d01 < d12:
        pts = np.roll(pts, -1, axis=0)
    length, width = int(max(d01, d12)), max(16, int(min(d01, d12)))
    dst = np.array([[0, 0], [length - 1, 0], [length - 1, width - 1], [0, width - 1]], np.float32)
    strip: np.ndarray = cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts, dst), (length, width))
    return strip


def vision_readings(
    store: SweepStore, t: Track, det: dict[int, dict[str, Any]], metas: dict[int, FrameMeta], deadline: float
) -> list[str]:
    """The vision model's copies of a spine's text on its closest whole views (empty when it read nothing)."""
    from claimagent.pipeline.vlm import copy_spine_text

    def rank(o: Any) -> float:
        sp = det[o.seq]["spines"][o.index]
        return float(sp["length"]) * (0.3 if sp["touches_edge"] else 1.0)

    out = []
    for o in sorted(t.observations, key=rank, reverse=True)[:VISION_READS_PER_SPINE]:
        left = deadline - time.monotonic()
        img = cv2.imread(str(store.frame_path(metas[o.seq]))) if o.seq in metas else None
        if left < 3 or img is None:
            break
        try:
            text = copy_spine_text(_spine_strip(img, det[o.seq]["spines"][o.index]["corners"]), timeout=left)
        except (httpx.HTTPError, OSError, ValueError, cv2.error):
            log.warning("vision read failed for track %s", t.id, exc_info=True)
            break
        log.info("vision read track %s frame %s: %r", t.id, o.seq, text)
        if text:
            out.append(text)
    return out


def read_tracks(
    store: SweepStore,
    reg: Registrar,
    det: dict[int, dict[str, Any]],
    selected: list[FrameMeta],
    reader: TextReader | None = None,
) -> dict[str, Any]:
    rows = assign_rows(reg)
    metas = {m.seq: m for m in selected}
    # Exclusions said by voice ("skip that shelf, those are not mine"): the shelf row in view at that moment.
    excluded: dict[int, str] = {}
    exclusions = []
    for ev in store.events("skip_region"):
        fm = nearest_frame(selected, int(ev["t_ms"]))
        if fm is None:
            continue
        ct = central_track(reg, det, fm.seq)
        frames = [fm.frame_ref]
        if ct is not None and ct.id in rows:
            label = rows[ct.id].label
            for t in reg.tracks:
                if t.id in rows and rows[t.id].label == label:
                    excluded[t.id] = (
                        f'excluded by policyholder at {ev["t_ms"] / 1000:.0f}s: "{ev.get("statement", "")}"'
                    )
        exclusions.append({"t_s": round(ev["t_ms"] / 1000, 1), "statement": ev.get("statement", ""), "frames": frames})
    # Book notes ("that is a first edition", "that one is signed"): the book in the centre of the view.
    notes: dict[int, list[dict[str, Any]]] = {}
    for ev in store.events("book_note"):
        fm = nearest_frame(selected, int(ev["t_ms"]))
        ct = central_track(reg, det, fm.seq) if fm else None
        if ct is not None:
            notes.setdefault(ct.id, []).append(ev)

    rejected = _not_books(reg, det)
    kept = [t for t in reg.tracks if t.id not in excluded and t.id not in rejected]
    close_by_track = _all_close_readings(store, kept, det, metas, reader) if reader is not None else {}
    books = []
    vision_deadline = time.monotonic() + VISION_READ_BUDGET_S
    for t in kept:
        best_text, best_score = "", 0.0
        readings: dict[str, float] = {}
        for o in t.observations:
            sp = det[o.seq]["spines"][o.index]
            text, sc = _obs_text(sp)
            if text:
                readings[text] = max(sc, readings.get(text, 0.0))
            if sc > best_score:
                best_score = sc
        obs_lines = [det[o.seq]["spines"][o.index].get("text", []) for o in t.observations]
        widths = [float(det[o.seq]["spines"][o.index]["width"]) for o in t.observations]
        if reader is not None:
            close = close_by_track.get(t.id, [])
            for lines in close:
                text = " | ".join(ln["text"] for ln in lines)
                if text:
                    readings[text] = max(readings.get(text, 0.0), _obs_text({"text": lines})[1])
            # Close readings vote like extra frames; they carry no width outlier risk.
            obs_lines += close
            widths += [statistics.median(widths)] * len(close)
        best_text = consensus_text(obs_lines, widths)
        text_source = "ocr"
        if reader is not None and not readable(best_text) and len(t.observations) >= VISION_READ_MIN_FRAMES:
            copied = [c for c in vision_readings(store, t, det, metas, vision_deadline) if readable(c)]
            if copied:
                best_text = max(copied, key=lambda c: len(re.sub(r"[^A-Za-zÀ-ÿ]", "", c)))
                readings.update({c: max(readings.get(c, 0.0), 1.0) for c in copied})
                text_source = "vision-model"

        def obs_rank(o: Any) -> float:
            sp = det[o.seq]["spines"][o.index]
            q = float(det[o.seq]["quality"]["sharpness"])
            return float(sp["length"]) * (0.3 if sp["touches_edge"] else 1.0) * min(1.0, q / SHARP_ENOUGH)

        rep = max(t.observations, key=obs_rank)
        row = rows.get(t.id, RowInfo(f"unit {t.segment + 1}", 0))
        # The shelf the guide was on when this spine was seen (guided sweeps): the most common one.
        shelf_tags = [(metas[o.seq].extra or {}).get("shelf") for o in t.observations if o.seq in metas]
        shelves = [int(x) for x in shelf_tags if x is not None]
        guide_shelf = max(sorted(set(shelves)), key=shelves.count) if shelves else None
        books.append(
            {
                "track": t.id,
                "segment": t.segment,
                "shelf": row.label,
                "position": row.position,
                "orientation": t.orientation,
                "guide_shelf": guide_shelf,
                "frame_ref": metas[rep.seq].frame_ref,
                "frame_seq": rep.seq,
                "bbox": det[rep.seq]["spines"][rep.index]["bbox"],
                "spine_text": best_text,
                "text_source": text_source,
                # clearest single-frame readings first
                "text_variants": sorted(readings, key=lambda x: -readings[x])[:5],
                "text_score": round(best_score, 1),
                "observations": len(t.observations),
                "only_partial": all(det[o.seq]["spines"][o.index]["touches_edge"] for o in t.observations),
                "notes": [
                    {
                        "statement": n.get("statement", ""),
                        "first_edition": n.get("first_edition", False),
                        "signed": n.get("signed", False),
                        "t_s": round(n["t_ms"] / 1000, 1),
                    }
                    for n in notes.get(t.id, [])
                ],
            }
        )
    books.sort(key=lambda b: (b["shelf"], b["position"]))
    data = {"books": books, "excluded_tracks": excluded, "rejected_tracks": rejected, "exclusions": exclusions}
    store.write_stage("read", data)
    return data


def readable(text: str) -> bool:
    return len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)) >= 1 and len(re.sub(r"[^A-Za-zÀ-ÿ]", "", text)) >= 5
