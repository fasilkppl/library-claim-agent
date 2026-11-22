"""Runs the post-sweep stages, times each one, and writes the packet, report and metrics.

The items need only the frame analysis, so they are named (local vision model) on a second thread while the
books are registered, read and looked up (mostly network waits); the room and the prices wait for both."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from claimagent.capture.store import FrameMeta, SweepStore
from claimagent.config import Settings, get_settings
from claimagent.contract import ClaimPacket
from claimagent.pipeline import stages
from claimagent.pipeline.analyze import FrameAnalyzer
from claimagent.pipeline.assemble import price_and_assemble
from claimagent.pipeline.dedupe import dedupe_books, duplicate_pairs
from claimagent.pipeline.http import CachedHttp
from claimagent.pipeline.price.engine import PriceEngine
from claimagent.pipeline.price.sources import (
    EbayBrowse,
    Frankfurter,
    GoogleBooksSale,
    SerpApiShopping,
    SerperShopping,
)
from claimagent.pipeline.resolve import Resolver
from claimagent.pipeline.verify import verify_books
from claimagent.report.pdf import html_to_pdf
from claimagent.report.render import render_report

log = logging.getLogger(__name__)
SERPAPI_USD_PER_SEARCH = 0.015  # SerpAPI Developer plan: 75 USD / 5000 searches
SERPER_USD_PER_SEARCH = 0.002  # Serper: 2 credits per shopping search, 1 USD / 1000 credits after the free 2500


def build_engine(http: CachedHttp, s: Settings) -> PriceEngine:
    shopping = SerperShopping(http, s.serper_key) if s.serper_key else SerpApiShopping(http, s.serpapi_key)
    return PriceEngine(
        shopping,
        EbayBrowse(http, s.ebay_client_id, s.ebay_client_secret),
        Frankfurter(http),
        GoogleBooksSale(http, s.google_books_key),
    )


# Target: the packet about 3.5 minutes after the sweep (the brief allows 5). Measured on real sweeps: registration
# 25-38 s, reading 12 s, items 15-37 s, prices 12-35 s, report 5 s; the rest goes to frame analysis and lookups.
PACKET_BUDGET_S = 200.0
RESERVE_AFTER_ANALYZE = 145.0  # registration, reading, title lookups, rulers, items, prices and the report
RESERVE_AFTER_RESOLVE = 80.0  # rulers, items, prices and the report after the title lookups
RESERVE_AFTER_VERIFY = 70.0  # rulers, items, prices and the report after the author check
RESERVE_AFTER_RULERS = 65.0
RESERVE_AFTER_ITEMS = 40.0


def _left(t0: float, reserve: float, floor: float, cap: float) -> float:
    """Seconds a stage may spend: what remains of the packet budget after the later stages' reserve."""
    left = PACKET_BUDGET_S - (time.perf_counter() - t0) - reserve
    budget = max(floor, min(cap, left))
    log.info("time budget: %.0f s left after reserve %.0f s -> stage gets %.0f s", left, reserve, budget)
    return budget


def books_only(det: dict[int, dict[str, Any]], frames: dict[int, Any]) -> dict[int, dict[str, Any]]:
    """In a guided sweep, spines count only in frames taken while the guide was on the books: a glance at the
    shelf while showing an item or the room would add the same books again."""
    guided = any((m.extra or {}).get("step") for m in frames.values())
    if not guided:
        return det
    out = {}
    for seq, d in det.items():
        step_name = (frames[seq].extra or {}).get("step") if seq in frames else None
        out[seq] = d if step_name in (None, "books") else {**d, "spines": [], "text_lines": []}
    dropped = sum(1 for seq in det if out[seq] is not det[seq])
    log.info("guided sweep: spines ignored in %s frames taken outside the books step", dropped)
    return out


def _verify(
    store: SweepStore, read: dict[str, Any], resolved: dict[str, Any], resolver: Resolver, budget_s: float
) -> dict[str, Any]:
    """Second-catalogue check of each identified book's authors (disagreements go to the review queue)."""
    checked, report = verify_books(read, resolved, resolver, budget_s)
    store.write_stage("verify", report)
    return checked


def _dedupe(
    store: SweepStore, read: dict[str, Any], resolved: dict[str, Any], boxes: dict[int, dict[int, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One entry per physical book; the check after it must find no pair left."""
    read2, resolved2, report = dedupe_books(read, resolved, boxes)
    report["duplicates_left"] = duplicate_pairs(read2, resolved2, boxes)
    if report["duplicates_left"]:
        log.error("duplicate books left after dedupe: %s", report["duplicates_left"])
    store.write_stage("dedupe", report)
    return read2, resolved2


def run_pipeline(
    store: SweepStore,
    s: Settings | None = None,
    offline: bool = False,
    analyzer: FrameAnalyzer | None = None,
    progress: Callable[[str, dict[str, Any]], None] | None = None,
) -> ClaimPacket:
    s = s or get_settings()
    http = CachedHttp(store.cache_dir, offline=offline)
    resolver = Resolver(http, s.google_books_key)
    engine = build_engine(http, s)
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    def step(name: str, fn: Callable[[], Any]) -> Any:
        if progress:
            progress("stage", {"stage": name})
        t = time.perf_counter()
        out = fn()
        timings[name] = round(time.perf_counter() - t, 2)
        log.info("stage %s %.2fs", name, timings[name])
        return out

    sel = step("select", lambda: stages.select_frames(store, s))
    frames = {m.seq: m for m in store.frames()}
    # Poses from lost AR tracking are removed here, so every later stage falls back to camera-only methods.
    tracking = step("tracking", lambda: stages.check_tracking(store, list(frames.values())))
    selected = [frames[f["seq"]] for f in sel["frames"] if f["keep"]]
    frame_analyzer = analyzer or FrameAnalyzer(store)
    det = step(
        "analyze",
        lambda: stages.analyze_frames(
            store,
            selected,
            frame_analyzer,
            (lambda i, n: progress("analyze", {"done": i, "total": n})) if progress else None,
            _left(t_all, RESERVE_AFTER_ANALYZE, 20, 60),
        ),
    )
    selected = [m for m in selected if m.seq in det]  # frames left unanalysed (time budget) are not used
    # 60-90 s for naming the items from their pictures: with 25 s the first check timed out and every item kept
    # the detector's guess; with 45 s a live sweep named 11 of 26. Run beside the book stages, not after them.
    side = ThreadPoolExecutor(max_workers=1, thread_name_prefix="items")
    items_job = side.submit(
        step, "items", lambda: stages.collect_items(store, det, selected, _left(t_all, RESERVE_AFTER_ITEMS, 60, 90))
    )
    try:
        book_parts = _book_stages(store, s, frames, selected, det, frame_analyzer, resolver, step, t_all)
        t_wait = time.perf_counter()
        items = items_job.result()
        timings["items_wait"] = round(time.perf_counter() - t_wait, 2)  # the book stages' lead over the items
    finally:
        side.shutdown(wait=True)
    read, resolved, measured = book_parts
    room = step("room", lambda: stages.measure_room(store, list(frames.values()), items, tracking))

    session = store.read_session()
    analyze_live = sum(float(d.get("elapsed_s", 0)) for d in det.values())
    metrics: dict[str, Any] = {
        "latency_s": timings,
        "frames_total": sel["total"],
        "frames_selected": sel["kept"],
        "frame_analysis_total_s": round(analyze_live, 1),
        "frame_analysis_per_frame_s": round(analyze_live / max(1, len(det)), 2),
    }
    packet: ClaimPacket = step(
        "price_assemble", lambda: price_and_assemble(store, s, engine, read, resolved, measured, items, room, metrics)
    )
    timings["total_post_sweep"] = round(time.perf_counter() - t_all, 2)
    serp = http.paid_calls.get("serpapi", 0)
    serper = http.paid_calls.get("serper", 0)
    shopping_usd = round(serp * SERPAPI_USD_PER_SEARCH + serper * SERPER_USD_PER_SEARCH, 3)
    packet.metrics.update(
        {
            "latency_s": timings,
            "api_calls": {"network": http.calls, "cache_hits": http.cache_hits, "paid": http.paid_calls},
            "sources_disabled": http.disabled,
            "cost_usd": {
                "serpapi": round(serp * SERPAPI_USD_PER_SEARCH, 3),
                "serper": round(serper * SERPER_USD_PER_SEARCH, 3),
                "local_models": 0.0,
                "total": shopping_usd,
            },
            "sweep_ended_to_packet_s": timings["total_post_sweep"],
            "live_voice": session.get("voice_metrics", {}),
        }
    )
    write_packet(store, packet)
    if progress:
        progress("done", {"packet": f"/sweeps/{store.id}/claim_packet.json"})
    return packet


def _book_stages(
    store: SweepStore,
    s: Settings,
    frames: dict[int, FrameMeta],
    selected: list[FrameMeta],
    det: dict[int, dict[str, Any]],
    frame_analyzer: FrameAnalyzer,
    resolver: Resolver,
    step: Callable[[str, Callable[[], Any]], Any],
    t_all: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Registration, reading, title lookups, dedupe, author check and measurement of the books."""
    det_books = books_only(det, frames)
    reg, _ = step("register", lambda: stages.register_tracks(store, selected, det_books))
    read = step("read", lambda: stages.read_tracks(store, reg, det_books, selected, frame_analyzer.models.reader))
    # One time budget for the packet: the slow lookups get what is left, so the packet is always due in time.
    resolved = step(
        "resolve", lambda: stages.resolve_books(store, read, resolver, s, _left(t_all, RESERVE_AFTER_RESOLVE, 20, 70))
    )
    read, resolved = step(
        "second_look",
        lambda: stages.second_look(
            store,
            reg,
            det_books,
            selected,
            read,
            resolved,
            resolver,
            s,
            min(30.0, _left(t_all, RESERVE_AFTER_RESOLVE, 0, 30)),
        ),
    )
    # Where each spine was in each frame: two spines side by side in one frame are two books.
    boxes = {t.id: {o.seq: o.corners.tolist() for o in t.observations} for t in reg.tracks}
    read, resolved = step("dedupe", lambda: _dedupe(store, read, resolved, boxes))
    resolved = step(
        "verify", lambda: _verify(store, read, resolved, resolver, _left(t_all, RESERVE_AFTER_VERIFY, 5, 15))
    )
    measured = step(
        "measure",
        lambda: stages.measure_books(
            store, reg, det_books, selected, read, resolved, resolver, s, _left(t_all, RESERVE_AFTER_RULERS, 5, 20)
        ),
    )
    return read, resolved, measured


def write_packet(store: SweepStore, packet: ClaimPacket) -> None:
    (store.dir / "claim_packet.json").write_text(packet.model_dump_json(indent=2))
    (store.dir / "metrics.json").write_text(json.dumps(packet.metrics, indent=2))
    (store.dir / "report.html").write_text(render_report(packet))
    html_to_pdf(store.dir / "report.html", store.dir / "report.pdf")


def reprice(store: SweepStore, country: str, n: int = 10, s: Settings | None = None) -> dict[str, Any]:
    """Price the first n priced books of a finished sweep for another country (locale is a setting)."""
    from claimagent.pipeline.price.sources import COUNTRY_CURRENCY

    s = s or get_settings()
    packet = ClaimPacket.model_validate_json((store.dir / "claim_packet.json").read_text())
    http = CachedHttp(store.cache_dir)
    engine = build_engine(http, s)
    resolved = {r["track"]: r for r in store.read_stage("resolve")["books"]}
    read = store.read_stage("read")["books"]
    vol = {f"B{k:03d}": resolved.get(b["track"], {}).get("volume_id", "") for k, b in enumerate(read, start=1)}
    rows = []
    for b in [b for b in packet.books if b.status == "identified"][:n]:
        new, used = engine.book(b.title, b.author, b.isbn, vol.get(b.id, ""), country)
        rows.append(
            {
                "id": b.id,
                "title": b.title,
                "primary": {
                    "country": packet.sweep.country,
                    "replacement": b.replacement_cost.amount,
                    "used": b.used_value.amount,
                    "currency": packet.sweep.currency,
                },
                "second": {
                    "country": country,
                    "replacement": new.amount,
                    "used": used.amount,
                    "currency": COUNTRY_CURRENCY.get(country, ""),
                    "replacement_source": new.source,
                    "replacement_url": new.url,
                    "used_source": used.source,
                    "used_url": used.url,
                    "retrieved_at": new.retrieved_at or used.retrieved_at,
                    "converted": new.converted or used.converted,
                    "note": new.note or used.note,
                },
            }
        )
    out = {"sweep": store.id, "second_country": country, "books": rows}
    (store.dir / f"reprice_{country}.json").write_text(json.dumps(out, indent=2))
    return out
