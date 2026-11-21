"""Book stages: read each spine track, identify it in the catalogues, measure it."""

from __future__ import annotations

import logging
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any

import cv2
import httpx
import numpy as np

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings
from claimagent.pipeline import geometry as geo
from claimagent.pipeline import itemtrack
from claimagent.pipeline.analyze import CROP_MIN_LENGTH_PX, read_spine_crop
from claimagent.pipeline.dedupe import same_word
from claimagent.pipeline.dedupe import tokens as dd_tokens
from claimagent.pipeline.register import Registrar, Track
from claimagent.pipeline.resolve import Resolver
from claimagent.pipeline.stages.common import MARKER_SIDE_CM_DEFAULT, camera_of
from claimagent.pipeline.vision import TextReader


# ============================================================================ read
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


def _nearest_frame(selected: list[FrameMeta], t_ms: int) -> FrameMeta | None:
    return min(selected, key=lambda m: abs(m.t_ms - t_ms)) if selected else None


def _central_track(reg: Registrar, det: dict[int, dict[str, Any]], seq: int) -> Track | None:
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


log = logging.getLogger(__name__)

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
                if itemtrack.iou(box, it["bbox"]) >= NOT_BOOK_IOU:
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
RESOLVE_KEYS = (
    "status",
    "confidence",
    "title",
    "author",
    "publisher",
    "edition",
    "isbn",
    "year",
    "source",
    "url",
    "volume_id",
    "pages",
)


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


def _vision_readings(
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
        fm = _nearest_frame(selected, int(ev["t_ms"]))
        if fm is None:
            continue
        ct = _central_track(reg, det, fm.seq)
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
        fm = _nearest_frame(selected, int(ev["t_ms"]))
        ct = _central_track(reg, det, fm.seq) if fm else None
        if ct is not None:
            notes.setdefault(ct.id, []).append(ev)

    rejected = _not_books(reg, det)
    books = []
    vision_deadline = time.monotonic() + VISION_READ_BUDGET_S
    for t in reg.tracks:
        if t.id in excluded or t.id in rejected:
            continue
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
            try:
                close = _close_readings(store, t, det, metas, reader)
            except (cv2.error, ValueError, RuntimeError):  # a degenerate crop: this spine keeps its frame readings
                log.warning("close reading failed for track %s", t.id, exc_info=True)
                close = []
            for lines in close:
                text = " | ".join(ln["text"] for ln in lines)
                if text:
                    readings[text] = max(readings.get(text, 0.0), _obs_text({"text": lines})[1])
            # Close readings vote like extra frames; they carry no width outlier risk.
            obs_lines += close
            widths += [statistics.median(widths)] * len(close)
        best_text = consensus_text(obs_lines, widths)
        text_source = "ocr"
        if reader is not None and not _readable(best_text) and len(t.observations) >= VISION_READ_MIN_FRAMES:
            copied = [c for c in _vision_readings(store, t, det, metas, vision_deadline) if _readable(c)]
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


# ============================================================================ resolve
def resolve_books(
    store: SweepStore, read: dict[str, Any], resolver: Resolver, s: Settings, budget_s: float = 100.0
) -> dict[str, Any]:
    """Each spine is identified from its cross-frame consensus text, and also from its clearest single-frame
    readings (angled spines are often read whole in only one or two frames). A match found only through a
    single-frame reading is never confident: it can carry a neighbour's leaked line, so it is capped below
    the acceptance level (sent to review) and dropped if a neighbour's consensus already names that book."""

    def consensus(b: dict[str, Any]) -> str:
        return b["spine_text"] if _readable(b["spine_text"]) else ""

    def singles(b: dict[str, Any]) -> list[str]:
        good = [v for v in b.get("text_variants", []) if _readable(v) and v != b["spine_text"]]
        return good[:SINGLE_READINGS]

    resolver.deadline = time.monotonic() + min(RESOLVE_BUDGET_S, budget_s)

    def resolve_all(texts: list[str]) -> dict[str, Any]:
        # Hard stop at the budget: lookups still waiting are abandoned (their spines stay unidentified).
        pool = ThreadPoolExecutor(max_workers=6)
        futures = {pool.submit(resolver.resolve, t, s.id_review): t for t in texts}
        remaining = max(0.0, (resolver.deadline or time.monotonic()) - time.monotonic()) + HARD_STOP_GRACE_S
        done, _ = wait(futures, timeout=remaining)
        pool.shutdown(wait=False, cancel_futures=True)
        out: dict[str, Any] = {}
        for f in done:
            try:
                out[futures[f]] = f.result()
            except Exception:  # a lookup that failed is a spine left unidentified, not a failed packet
                log.exception("resolve failed for %r", futures[f])
        return out

    # Consensus first; single-frame readings only for spines it did not identify confidently.
    results = resolve_all(sorted({consensus(b) for b in read["books"] if consensus(b)}))
    pending = [
        b for b in read["books"] if consensus(b) not in results or results[consensus(b)].confidence < s.id_accept
    ]
    if resolver.deadline is not None and time.monotonic() > resolver.deadline:
        log.info("resolve budget spent on consensus readings; %s spines get no single-frame lookup", len(pending))
    results.update(resolve_all(sorted({v for b in pending for v in singles(b)} - set(results))))
    by_consensus = {
        b["track"]: results[consensus(b)]
        for b in read["books"]
        if consensus(b) in results and results[consensus(b)].title
    }
    cap = round(s.id_accept - 0.01, 3)
    named_spines = {b["track"]: b["spine_text"] for b in read["books"] if b["track"] in by_consensus}
    out = []
    for b in read["books"]:
        r = results.get(consensus(b)) if consensus(b) else None
        claimed = {_norm_title(x.title) for k, x in by_consensus.items() if k != b["track"]}
        # Only for a spine its own consensus did not name: when it did, a frame naming the same book is
        # corroborated, and one naming another book still has to beat the consensus match.
        own_named = r is not None and r.status == "identified"
        others = [] if own_named else [t for k, t in named_spines.items() if k != b["track"]]
        from_single = ""
        for v in singles(b):
            rv = results.get(v)
            if rv is None:
                continue
            if not rv.title or _norm_title(rv.title) in claimed:
                continue
            if holds_spine(v, others):
                continue
            # One frame is weak evidence: its match must name the author or explain more than half the reading
            # (seen: "TWO SYPTES | A TAIL OR" named "Two Tails on the Trail", a different book, priced in the claim).
            parts = rv.evidence.get("score_parts", {})
            if not parts.get("author_hit") and float(parts.get("spine_cov", 0.0)) <= 0.5:
                continue
            conf = min(rv.confidence, cap)
            if r is None or conf > r.confidence:
                r = replace(rv, confidence=conf, status="identified" if conf >= s.id_review else "unidentified")
                from_single = v
        vision = b.get("text_source") == "vision-model"
        if vision and r is not None and not from_single and r.confidence > cap:
            # Text copied by the vision model is one frame's reading: never confident, always reviewed.
            r = replace(r, confidence=cap)
        entry: dict[str, Any] = {"track": b["track"], "readable": r is not None}
        if r is not None:
            entry.update({k: getattr(r, k) for k in RESOLVE_KEYS})
            entry["evidence"] = dict(r.evidence)
            if from_single:
                entry["evidence"]["matched_on_single_frame_reading"] = from_single
            if vision:
                entry["evidence"]["spine_text_from"] = "local vision model (the OCR read no text on this spine)"
        out.append(entry)
    data = {"books": out, "queries": len(results)}
    store.write_stage("resolve", data)
    return data


def second_look(
    store: SweepStore,
    reg: Registrar,
    det: dict[int, dict[str, Any]],
    selected: list[FrameMeta],
    read: dict[str, Any],
    resolved: dict[str, Any],
    resolver: Resolver,
    s: Settings,
    budget_s: float = VISION_READ_BUDGET_S,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """A spine seen in several frames that the catalogue could not name from its OCR text (garbled gold print:
    "IWO CTMPS | A TADEON | A TAIL OR") gets the vision model's copy of its text from its closest views, and that
    copy is looked up once more. The same rules as for a single-frame reading apply: the match is capped below
    the acceptance level, and a copy that names a book identified on another track, or holds another track's
    whole spine, saw that neighbour and is not used."""
    tracks = {t.id: t for t in reg.tracks}
    metas = {m.seq: m for m in selected}
    res_by = {r["track"]: r for r in resolved["books"]}
    named = {r["track"]: r for r in resolved["books"] if r.get("title")}
    spines = {b["track"]: b["spine_text"] for b in read["books"] if b["track"] in named}
    cap = round(s.id_accept - 0.01, 3)
    deadline = time.monotonic() + budget_s
    resolver.deadline = deadline
    for b in read["books"]:
        t = tracks.get(b["track"])
        r = res_by.get(b["track"], {})
        if t is None or r.get("title") or b.get("text_source") == "vision-model":
            continue
        if len(t.observations) < VISION_READ_MIN_FRAMES or time.monotonic() > deadline - 8:
            continue
        others_spines = [x for k, x in spines.items() if k != b["track"]]
        claimed = {_norm_title(str(x["title"])) for k, x in named.items() if k != b["track"]}
        for copy in _vision_readings(store, t, det, metas, deadline):
            if not _readable(copy) or holds_spine(copy, others_spines):
                continue
            try:
                rv = resolver.resolve(copy, s.id_review)
            except Exception:  # one lookup failing must not cost the packet
                log.warning("second look lookup failed for track %s", b["track"], exc_info=True)
                continue
            if rv.status != "identified" or not rv.title or _norm_title(rv.title) in claimed:
                continue
            rv = replace(rv, confidence=min(rv.confidence, cap))
            b["spine_text"], b["text_source"] = copy, "vision-model"
            b["text_variants"] = list(dict.fromkeys([copy, *b.get("text_variants", [])]))[:5]
            entry: dict[str, Any] = {"track": b["track"], "readable": True}
            entry.update({k: getattr(rv, k) for k in RESOLVE_KEYS})
            entry["evidence"] = {
                **dict(rv.evidence),
                "spine_text_from": "local vision model, second look (the OCR text matched no catalogue record)",
            }
            res_by[b["track"]] = entry
            named[b["track"]] = entry
            log.info("second look: track %s read %r -> %r (%.2f)", b["track"], copy, rv.title, rv.confidence)
            break
    resolved = {**resolved, "books": [res_by.get(b["track"], {"track": b["track"]}) for b in read["books"]]}
    store.write_stage("read", read)
    store.write_stage("resolve", resolved)
    return read, resolved


SINGLE_READINGS = 3
RESOLVE_BUDGET_S = 100.0  # the packet is due within 5 minutes of the sweep
HARD_STOP_GRACE_S = 10.0
RULER_BUDGET_S = 30.0
RULER_WORKERS = 6
# Photo and page-count thickness err in opposite directions: the photo outline takes in a strip of cover or a
# neighbour's edge (too wide), the page count assumes thin paper (most Indian paperbacks are bulkier). Both
# errors are proportional, so the two are combined as a geometric mean. A photo more than twice or under half
# the page-count figure is a wrong outline (two spines merged, or a fragment): the page count alone is used.
THICK_RATIO_MAX = 2.0


def fuse_thickness(photo: float | None, catalogue: tuple[float, str]) -> tuple[float, str]:
    """One thickness from the photo and the page-count estimate (see THICK_RATIO_MAX), with the reason."""
    cat, where = catalogue
    if photo is None or not 1 / THICK_RATIO_MAX <= photo / cat <= THICK_RATIO_MAX:
        measured = "none" if photo is None else f"{photo:.1f} cm"
        return cat, f"thickness from the catalogue ({where}); the photo gave {measured}"
    return (photo * cat) ** 0.5, f"thickness: photo {photo:.1f} cm and catalogue {cat:.1f} cm ({where}) combined"


def holds_spine(reading: str, spines: list[str]) -> bool:
    """The reading holds the whole of one of these spine readings (two words or more, one slip per word): the
    frame saw that spine too (an outline spanning two books). Seen on a real sweep: a frame of "A Tale of Two
    Cities" also read "THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSON" and named the Tale "The Almanack"."""
    got = dd_tokens(reading)
    for spine in spines:
        words = dd_tokens(spine)
        if len(words) >= 2 and all(any(same_word(w, g) for g in got) for w in words):
            return True
    return False


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower().split(":")[0])


def _readable(text: str) -> bool:
    return len(re.findall(r"[A-Za-zÀ-ÿ]{3,}", text)) >= 1 and len(re.sub(r"[^A-Za-zÀ-ÿ]", "", text)) >= 5


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
        fm = _nearest_frame(selected, int(ev["t_ms"]))
        ct = _central_track(reg, det, fm.seq) if fm else None
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
            thick, why = fuse_thickness(thick, cat)
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
