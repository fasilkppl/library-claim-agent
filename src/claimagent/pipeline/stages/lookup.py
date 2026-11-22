"""Title lookups: each spine's text identified in the catalogues, then a second look by the vision model at
spines still unnamed."""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from typing import Any

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings
from claimagent.pipeline.dedupe import same_word
from claimagent.pipeline.dedupe import tokens as dd_tokens
from claimagent.pipeline.register import Registrar
from claimagent.pipeline.resolve import Resolver
from claimagent.pipeline.stages.read import (
    VISION_READ_BUDGET_S,
    VISION_READ_MIN_FRAMES,
    readable,
    vision_readings,
)

log = logging.getLogger(__name__)


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


def resolve_books(
    store: SweepStore, read: dict[str, Any], resolver: Resolver, s: Settings, budget_s: float = 100.0
) -> dict[str, Any]:
    """Each spine is identified from its cross-frame consensus text, and also from its clearest single-frame
    readings (angled spines are often read whole in only one or two frames). A match found only through a
    single-frame reading is never confident: it can carry a neighbour's leaked line, so it is capped below
    the acceptance level (sent to review) and dropped if a neighbour's consensus already names that book."""

    def consensus(b: dict[str, Any]) -> str:
        return b["spine_text"] if readable(b["spine_text"]) else ""

    def singles(b: dict[str, Any]) -> list[str]:
        good = [v for v in b.get("text_variants", []) if readable(v) and v != b["spine_text"]]
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
        for copy in vision_readings(store, t, det, metas, deadline):
            if not readable(copy) or holds_spine(copy, others_spines):
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
