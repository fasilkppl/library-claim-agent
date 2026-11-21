"""Price, flag and assemble the claim packet. Totals are computed here, in code, from the lines."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import cv2

from claimagent.capture.store import SweepStore, utc_now
from claimagent.config import Settings
from claimagent.contract import (
    Book,
    BookPrice,
    BookStatus,
    ClaimPacket,
    Dimensions,
    Exclusion,
    Item,
    ItemPrice,
    ItemStatus,
    ReviewEntry,
    Room,
    Sweep,
    Totals,
)
from claimagent.pipeline.geometry import M2_TO_FT2
from claimagent.pipeline.price.engine import PriceEngine, PriceFigure
from claimagent.pipeline.resolve import english_words

RARE_WORDS = re.compile(r"\b(first edition|1st edition|signed|limited edition|numbered|inscribed|antiquarian)\b", re.I)
ASSUMPTIONS = [
    "Book count = distinct spines after cross-frame registration; books seen only cut off at a frame edge are "
    "counted and flagged.",
    "A title is reported only when the catalogue match is explained by text read on the spine; otherwise the "
    "book is 'unidentified' with its dimensions.",
    "Edition and ISBN are filled only when the spine shows the publisher of a catalogued edition.",
    "Replacement cost = median of new offers in the policyholder's country; used value = median of used offers "
    "(eBay fixed-price listings or second-hand Google Shopping offers), condition 'good' assumed.",
    "Shelf run = union of book extents along each shelf (stacked books share one footprint).",
    "Wall area = room perimeter x ceiling height; doors and windows are not subtracted.",
    "Shelved wall area = front area of detected shelving units.",
    "Items whose brand/model is not legible get a range (25th-75th percentile of comparable new offers).",
]

log = logging.getLogger(__name__)
PRICE_WORKERS = 8  # price lookups in parallel: each waits on a shopping search, not on this machine


def _book_price(fig: PriceFigure, used: bool) -> BookPrice:
    return BookPrice(
        amount=fig.amount,
        currency=fig.currency,
        source=fig.source,
        url=fig.url,
        retrieved_at=fig.retrieved_at,
        converted=fig.converted,
        converted_from=fig.converted_from,
        fx_rate=fig.fx_rate,
        fx_date=fig.fx_date,
        condition_assumed=fig.condition_assumed or ("used, good" if used else "new"),
        offers_used=fig.offers_used,
        note=fig.note,
    )


def item_price_status(it: dict[str, Any], vision_check_on: bool) -> str:
    """Returns "unpriced" when nothing confirms what the item is, else "range" (art is decided afterwards)."""
    if it.get("vision_check") == "unknown":
        # The vision model could not tell what it is: pricing the detector's guess (a pressure cooker called a
        # "cup", a dark corner an "electric kettle") would put a wrong product in the claim.
        return "unpriced"
    if vision_check_on and "vision_check" not in it:
        # The picture check did not reach this item in its time: the detector's name alone (a gift box called a
        # "laptop", a switchboard a "chandelier") is not a product to price.
        return "unpriced"
    return "range"


def _reads_as_words(text: str) -> bool:
    """Printed text is used as a brand or model only when it reads as words: OCR noise on an object ("12 8bun
    ong", "C perch serch youeг") would search for something else entirely."""
    if re.search(r"[^\x00-\u024f]", text):
        return False
    words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3]
    if not words:
        return False
    known = sum(1 for w in words if w.lower() in english_words())
    return known / len(words) >= 0.6


def threshold_local(engine: PriceEngine, s: Settings, currency: str) -> float | None:
    r = engine.fx.rate("USD", currency)
    return s.appraisal_threshold_usd * r[0] if r else None


def price_and_assemble(
    store: SweepStore,
    s: Settings,
    engine: PriceEngine,
    read: dict[str, Any],
    resolved: dict[str, Any],
    measured: dict[str, Any],
    items: dict[str, Any],
    room_data: dict[str, Any],
    metrics: dict[str, Any],
) -> ClaimPacket:
    session = store.read_session()
    country = session.get("country", s.country)
    currency = session.get("currency", s.currency)
    res = {r["track"]: r for r in resolved["books"]}
    meas = {m["track"]: m for m in measured["books"]}
    review: list[ReviewEntry] = []
    limit = threshold_local(engine, s, currency)

    # ------------------------------------------------------------ books
    plan = []
    for n, b in enumerate(read["books"], start=1):
        r, m = res.get(b["track"], {}), meas.get(b["track"], {})
        bid = f"B{n:03d}"
        status = "unidentified"
        conf = float(r.get("confidence", 0.0)) if r.get("readable") else 0.0
        if r.get("status") == "identified" and conf >= s.id_review:
            status = "identified"
        rare_reason = ""
        for note in b["notes"]:
            if note["first_edition"] or note["signed"]:
                rare_reason = f'policyholder said: "{note["statement"]}"'
        if RARE_WORDS.search(b["spine_text"]):
            rare_reason = rare_reason or f'spine text suggests a special copy: "{b["spine_text"][:60]}"'
        if r.get("year") and r.get("isbn") and int(r["year"]) < s.rare_before_year:
            rare_reason = rare_reason or f"catalogued edition from {r['year']}"
        if rare_reason and status == "identified":
            status = "needs_appraisal"
        plan.append((bid, b, r, m, status, conf, rare_reason))

    def do_price(p: tuple[Any, ...]) -> tuple[PriceFigure, PriceFigure] | None:
        _, _, r, _, status, _, _ = p
        if status != "identified":
            return None
        try:
            author = r.get("author_catalogue") or r.get("author", "")
            return engine.book(r["title"], author, r.get("isbn", ""), r.get("volume_id", ""), country)
        except Exception:  # one book's price lookup failing must not cost the whole packet
            log.exception("price lookup failed for %s", r.get("title"))
            return None

    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        prices = list(pool.map(do_price, plan))

    books: list[Book] = []
    for (bid, b, r, m, status, conf, rare_reason), pr in zip(plan, prices, strict=True):
        new_fig, used_fig = pr if pr else (PriceFigure(currency=currency), PriceFigure(currency=currency))
        top = max((f.amount for f in (new_fig, used_fig) if f.amount), default=None)
        if status == "identified" and top is not None and limit is None:
            # No exchange rate for the threshold: the price stands, but it is not known to be under it.
            new_fig.note = (
                new_fig.note + "; " if new_fig.note else ""
            ) + "appraisal threshold not checked (no FX rate)"
        if status == "identified" and limit is not None and top is not None and top > limit:
            status = "needs_appraisal"
            rare_reason = (
                f"price {top:.0f} {currency} is above the appraisal threshold "
                f"({limit:.0f} {currency}); found price kept as a note only"
            )
        if status == "needs_appraisal":
            for fig in (new_fig, used_fig):
                if fig.amount is not None:
                    fig.note = f"indicative only, not auto-priced: {fig.amount:.0f} {fig.currency} ({fig.url})"
                fig.amount = None
        book = Book(
            id=bid,
            shelf=b["shelf"],
            position=b["position"],
            frame_ref=b["frame_ref"],
            status=cast(BookStatus, status),
            title=r.get("title", "") if status != "unidentified" else "",
            author=r.get("author", "") if status != "unidentified" else "",
            publisher=r.get("publisher", "") if status != "unidentified" else "",
            edition=r.get("edition", "") if status != "unidentified" else "",
            isbn=r.get("isbn", "") if status != "unidentified" else "",
            year=r.get("year") if status != "unidentified" and r.get("isbn") else None,
            spine_height_cm=m.get("spine_height_cm"),
            spine_thickness_cm=m.get("spine_thickness_cm"),
            orientation=b["orientation"],
            id_confidence=round(conf, 3),
            spine_text=b["spine_text"],
            match_source=r.get("source", "") if status != "unidentified" else "",
            match_url=r.get("url", "") if status != "unidentified" else "",
            scale_source=m.get("scale_source", ""),
            bbox=b["bbox"],
            observations=b["observations"],
            replacement_cost=_book_price(new_fig, False),
            used_value=_book_price(used_fig, True),
        )
        books.append(book)
        if status == "unidentified":
            why = r.get("demoted") or (
                "spine not legible"
                if not r.get("readable")
                else (
                    f'spine text "{b["spine_text"][:50]}" did not match a catalogue record with enough confidence'
                    f" ({conf:.2f})"
                )
            )
            review.append(ReviewEntry(ref_id=bid, reason=f"unidentified: {why}; not priced, excluded from totals"))
        elif status == "needs_appraisal":
            review.append(ReviewEntry(ref_id=bid, reason=f"needs human appraisal: {rare_reason}"))
        else:
            if conf < s.id_accept:
                review.append(
                    ReviewEntry(
                        ref_id=bid,
                        reason=f'low-confidence title match ({conf:.2f}); check "{book.title}" against frame',
                    )
                )
            check = r.get("check") or {}
            if check.get("review"):
                review.append(ReviewEntry(ref_id=bid, reason=f"author check: {check['review']}"))
            if new_fig.amount is None:
                review.append(ReviewEntry(ref_id=bid, reason=f"no replacement price: {new_fig.note}"))
            elif new_fig.converted:
                review.append(
                    ReviewEntry(
                        ref_id=bid, reason=f"replacement price converted from {new_fig.converted_from}: {new_fig.note}"
                    )
                )
            elif new_fig.offers_used == 1:
                review.append(ReviewEntry(ref_id=bid, reason="replacement price from a single offer"))
            if used_fig.amount is None:
                review.append(ReviewEntry(ref_id=bid, reason=f"no used value: {used_fig.note}"))
        for c in m.get("checks", []):
            review.append(ReviewEntry(ref_id=bid, reason=f"measurement: {c}"))
        if book.spine_height_cm is None:
            review.append(
                ReviewEntry(
                    ref_id=bid,
                    reason="dimensions unknown: no metric scale for this shelf "
                    "(no depth, marker, stated width or catalogue rulers)",
                )
            )

    # ------------------------------------------------------------ items
    art_answers = store.events("art_answer")
    plan_items = []
    for it in items["items"]:
        status, query = "range", it["label"]
        statement = ""
        brand = ""
        if it["text"]:
            words = " ".join(it["text"])
            if len(re.sub(r"[^A-Za-z]", "", words)) >= 3 and len(words.split()) <= 4 and _reads_as_words(words):
                brand = words  # text printed on the object itself (a logo or a model plate)
        status = item_price_status(it, s.item_vision_check)
        if it["category"] == "art":
            status = "needs_appraisal"
            ans = _nearest_answer(art_answers, store, it)
            if ans is not None:
                statement = ans.get("statement", "")
                if ans.get("is_print"):
                    status, query = "range", "framed art print"
        plan_items.append((it, status, query, brand, statement))

    def do_item(p: tuple[Any, ...]) -> tuple[str, PriceFigure] | None:
        _, status, query, brand, _ = p
        if status == "needs_appraisal":
            return None
        if status == "unpriced":
            return "unpriced", PriceFigure(
                currency=currency, query=query, note="object not recognised clearly: not priced, for review"
            )
        return engine.item(query, brand, country)

    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as pool:
        item_prices = list(pool.map(do_item, plan_items))

    out_items: list[Item] = []
    for (it, status, query, brand, statement), ipr in zip(plan_items, item_prices, strict=True):
        fig = PriceFigure(currency=currency, query=query)
        if ipr is not None:
            status, fig = ipr
        dims = it.get("dims_cm") or {}
        item = Item(
            id=it["id"],
            category=it["category"],
            description=it["label"],
            brand_model=brand,
            frame_ref=it["frame_ref"],
            bbox=it["bbox"],
            dimensions_cm=Dimensions(w=dims.get("w"), h=dims.get("h"), d=None),
            status=cast(ItemStatus, status),
            replacement_cost=ItemPrice(
                low=fig.low,
                high=fig.high,
                currency=fig.currency,
                source=fig.source,
                url=fig.url,
                retrieved_at=fig.retrieved_at,
                converted=fig.converted,
                offers_used=fig.offers_used,
                query=fig.query,
                note=fig.note,
            ),
            confidence=it["confidence"],
            user_statement=statement,
            observations=it["observations"],
        )
        out_items.append(item)
        if status == "needs_appraisal":
            review.append(
                ReviewEntry(
                    ref_id=item.id, reason="art/portrait: needs appraisal (policyholder did not state it is a print)"
                )
            )
        elif status == "unpriced":
            review.append(ReviewEntry(ref_id=item.id, reason=f"no price: {fig.note}"))
        if item.confidence < 0.4:
            review.append(
                ReviewEntry(ref_id=item.id, reason=f"low detection confidence ({item.confidence:.2f}); check category")
            )

    # ------------------------------------------------------------ room
    room = Room(
        length_m=room_data.get("length_m"),
        width_m=room_data.get("width_m"),
        height_m=room_data.get("height_m"),
        floor_area_m2=room_data.get("floor_area_m2"),
        wall_area_m2=room_data.get("wall_area_m2"),
        shelved_wall_area_m2=room_data.get("shelved_wall_area_m2"),
        floor_area_ft2=_ft2(room_data.get("floor_area_m2")),
        wall_area_ft2=_ft2(room_data.get("wall_area_m2")),
        shelved_wall_area_ft2=_ft2(room_data.get("shelved_wall_area_m2")),
        scale_method="ARCore metric camera pose + depth (WebXR)" if room_data.get("measured") else "",
        shape_method=room_data.get("shape_method", ""),
        confidence=room_data.get("confidence", 0.0),
        frame_refs=room_data.get("frame_refs", []),
        notes=room_data.get("notes", []),
    )
    if not room.floor_area_m2:
        review.append(ReviewEntry(ref_id="room", reason="room not measured: " + "; ".join(room.notes)))
    elif room.confidence < 0.7:
        review.append(ReviewEntry(ref_id="room", reason="room measurement low confidence: " + "; ".join(room.notes)))

    if measured.get("shelf_run_m") is None:
        review.append(ReviewEntry(ref_id="shelf_run", reason="shelf run unknown: no metric scale for any shelf"))
    elif not measured.get("shelf_run_complete", True):
        review.append(
            ReviewEntry(ref_id="shelf_run", reason="shelf run is a lower bound: some shelves had no metric scale")
        )
    totals = compute_totals(books, out_items, measured.get("shelf_run_m"), currency)
    duration = float(session.get("duration_s", 0.0))
    packet = ClaimPacket(
        sweep=Sweep(
            id=store.id,
            captured_at=session.get("captured_at", utc_now()),
            device=session.get("device", ""),
            duration_s=round(duration, 1),
            country=country,
            currency=currency,
            capture_mode="webxr" if session.get("capture_mode") == "webxr" else "camera",
        ),
        room=room,
        books=books,
        items=out_items,
        totals=totals,
        review_queue=review,
        exclusions=[Exclusion(**e) for e in read.get("exclusions", [])],
        metrics=metrics,
        assumptions=ASSUMPTIONS
        + ([f"Appraisal threshold: {limit:.0f} {currency} (= {s.appraisal_threshold_usd:.0f} USD)."] if limit else []),
    )
    write_crops(store, packet)
    return packet


def _ft2(m2: float | None) -> float | None:
    return None if m2 is None else round(m2 * M2_TO_FT2, 1)


def _nearest_answer(answers: list[dict[str, Any]], store: SweepStore, it: dict[str, Any]) -> dict[str, Any] | None:
    ref_t = next((m.t_ms for m in store.frames() if m.seq == it["frame_seq"]), None)
    if ref_t is None or not answers:
        return None
    best = min(answers, key=lambda a: abs(a["t_ms"] - ref_t))
    return best if abs(best["t_ms"] - ref_t) < 60_000 else None


def compute_totals(books: list[Book], items: list[Item], shelf_run_m: float | None, currency: str) -> Totals:
    t = Totals(currency=currency)
    t.book_count = len(books)
    t.books_identified = sum(b.status == "identified" for b in books)
    t.books_unidentified = sum(b.status == "unidentified" for b in books)
    t.books_needs_appraisal = sum(b.status == "needs_appraisal" for b in books)
    t.shelf_run_m = shelf_run_m
    excluded = 0
    for b in books:
        r, u = b.replacement_cost.amount, b.used_value.amount
        if r is not None:
            t.books_replacement_cost += r
        if u is not None:
            t.books_used_value += u
        if r is None or u is None:
            excluded += 1
    for i in items:
        lo, hi = i.replacement_cost.low, i.replacement_cost.high
        if lo is not None and hi is not None and i.status in ("priced", "range"):
            t.items_replacement_cost_low += lo
            t.items_replacement_cost_high += hi
        else:
            excluded += 1
    t.books_replacement_cost = round(t.books_replacement_cost, 2)
    t.books_used_value = round(t.books_used_value, 2)
    t.items_replacement_cost_low = round(t.items_replacement_cost_low, 2)
    t.items_replacement_cost_high = round(t.items_replacement_cost_high, 2)
    t.excluded_from_totals = excluded
    return t


def write_crops(store: SweepStore, packet: ClaimPacket) -> None:
    """Small crops of every line, so the report shows exactly what was seen next to each figure."""
    crops = store.dir / "crops"
    crops.mkdir(exist_ok=True)
    cache: dict[str, Any] = {}
    lines: list[Book | Item] = [*packet.books, *packet.items]
    for line in lines:
        if not line.bbox:
            continue
        img = cache.get(line.frame_ref)
        if img is None:
            img = cv2.imread(str(store.dir / line.frame_ref))
            cache[line.frame_ref] = img
        if img is None:
            continue
        x0, y0, x1, y1 = (int(v) for v in line.bbox)
        pad = 6
        crop = img[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad]
        if crop.size == 0:
            continue
        scale = 220 / max(crop.shape[:2])
        if scale < 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(crops / f"{line.id}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
