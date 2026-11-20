"""Runs the post-sweep stages in order, times each one, and writes the packet, report and metrics."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from claimagent.capture.store import SweepStore
from claimagent.config import Settings, get_settings
from claimagent.contract import ClaimPacket
from claimagent.pipeline import stages
from claimagent.pipeline.analyze import FrameAnalyzer
from claimagent.pipeline.assemble import price_and_assemble
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
        ),
    )
    reg, _ = step("register", lambda: stages.register_tracks(store, selected, det))
    read = step("read", lambda: stages.read_tracks(store, reg, det, selected, frame_analyzer.models.reader))
    resolved = step("resolve", lambda: stages.resolve_books(store, read, resolver, s))
    measured = step("measure", lambda: stages.measure_books(store, reg, det, selected, read, resolved, resolver, s))
    items = step("items", lambda: stages.collect_items(store, reg, det, selected))
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
