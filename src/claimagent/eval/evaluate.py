"""Compare a claim packet with hand-collected ground truth and score every pass bar from the brief.

Ground truth folder (templates in ground_truth/template/):
  books.csv   book_no, shelf, position, title, author, legible (yes/no), spine_height_cm, spine_thickness_cm
  prices.csv  title, replacement_price, replacement_source_type, replacement_url, used_price, used_source_type,
              used_url, checked_on
  items.csv   item_no, category, description
  room.csv    key, value   (true_book_count, length_m, width_m, height_m, floor_area_m2, wall_area_m2,
                            shelved_wall_area_m2)
Every book on the shelves is a row in books.csv (legible = no when a human cannot read the spine).
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz

from claimagent.capture.store import SweepStore
from claimagent.config import get_settings
from claimagent.contract import Book, ClaimPacket
from claimagent.pipeline.household import CATEGORY
from claimagent.pipeline.resolve import content_tokens, volume_numbers

ITEM_CATEGORY_ALIASES = {
    "shelf": "shelving",
    "bookcase": "shelving",
    "bookshelf": "shelving",
    "shelving": "shelving",
    "sofa": "seating",
    "chair": "seating",
    "armchair": "seating",
    "stool": "seating",
    "seating": "seating",
    "table": "table",
    "desk": "table",
    "lamp": "lighting",
    "light": "lighting",
    "lighting": "lighting",
    "portrait": "art",
    "painting": "art",
    "picture": "art",
    "print": "art",
    "art": "art",
    "poster": "art",
    "rug": "rug",
    "carpet": "rug",
    "coffee machine": "appliance",
    "appliance": "appliance",
    "tv": "electronics",
    "television": "electronics",
    "laptop": "electronics",
    "speaker": "electronics",
    "electronics": "electronics",
    "decor": "decor",
    "vase": "decor",
    "plant": "decor",
    "clock": "decor",
}


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def _num(v: str | None) -> float | None:
    try:
        return float(str(v).replace(",", "")) if v not in (None, "") else None
    except ValueError:
        return None


def title_match(a: str, b: str) -> bool:
    """Strict on purpose (a lenient matcher would inflate accuracy). A catalogue title is compared whole, and
    also by its main part and its subtitle on their own ("Wings of Fire: An Autobiography" is "Wings of Fire";
    "Mahatma Gandhi: The Story of My Experiments with Truth" is "My Experiments with Truth")."""
    va, vb = volume_numbers(a), volume_numbers(b)
    if va and vb and not (va & vb):
        return False  # volume II is a different book from volume I
    parts_a = [a, *[x for x in a.split(":", 1) if ":" in a]]
    parts_b = [b, *[x for x in b.split(":", 1) if ":" in b]]
    return any(_same_title(x, y) for x in parts_a for y in parts_b)


def _same_title(a: str, b: str) -> bool:
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta or not tb:
        return False
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    def found(t: str, words: list[str]) -> bool:
        return any(fuzz.ratio(t, u) >= 85 for u in words)

    # Every distinctive word of the shorter title must be there: "Principles of Building AI Agents" is not
    # "Patterns for Building AI Agents", though three of their four words match.
    if not all(found(t, long_) for t in short if len(t) >= 5):
        return False
    if sum(1 for t in long_ if found(t, short)) / len(long_) >= 0.7:
        return True
    # The shorter title (2+ meaningful words) fully inside the longer one: a series name or prefix added.
    return len(short) >= 2 and all(found(t, long_) for t in short)


def _pct(sys_v: float | None, true_v: float | None) -> float | None:
    if sys_v is None or not true_v:
        return None
    return abs(sys_v - true_v) / true_v


def match_books(gt: list[dict[str, str]], books: list[Book]) -> dict[int, Book]:
    """GT row index -> packet book, one-to-one, by title."""
    used: set[str] = set()
    out: dict[int, Book] = {}
    for i, row in enumerate(gt):
        if not row.get("title"):
            continue
        for b in books:
            if b.id not in used and b.title and title_match(row["title"], b.title):
                out[i] = b
                used.add(b.id)
                break
    return out


def match_by_position(gt: list[dict[str, str]], books: list[Book], done: dict[int, Book]) -> dict[int, Book]:
    """Books not paired by title (unidentified spines still get measured) are paired by their order on the
    shelf, but only when the truth and the packet list one shelf with the same number of books: then the
    n-th spine from the left is the n-th book in the sheet."""
    if len({r.get("shelf", "") for r in gt}) != 1 or len({b.shelf for b in books}) != 1 or len(gt) != len(books):
        return done
    out = dict(done)
    used = {b.id for b in done.values()}
    gt_order = sorted(range(len(gt)), key=lambda i: _num(gt[i].get("position")) or 0)
    sys_order = sorted(books, key=lambda b: b.position)
    for i, b in zip(gt_order, sys_order, strict=True):
        if i not in out and b.id not in used:
            out[i] = b
            used.add(b.id)
    return out


def evaluate(store: SweepStore, truth: Path) -> dict[str, Any]:
    s = get_settings()
    p = ClaimPacket.model_validate_json((store.dir / "claim_packet.json").read_text())
    books_gt = _rows(truth / "books.csv")
    prices_gt = _rows(truth / "prices.csv")
    items_gt = _rows(truth / "items.csv")
    room_gt = {r["key"]: _num(r["value"]) for r in _rows(truth / "room.csv")}
    res: dict[str, Any] = {"sweep": p.sweep.id, "bars": []}

    def bar(name: str, bar_text: str, value: str, passed: bool | None, detail: Any = None) -> None:
        res["bars"].append({"measure": name, "pass_bar": bar_text, "result": value, "pass": passed, "detail": detail})

    # Book count
    true_n = room_gt.get("true_book_count") or (len(books_gt) or None)
    if true_n:
        err = abs(p.totals.book_count - true_n) / true_n
        bar("Book count", "within 5% of true count", f"{p.totals.book_count} vs {true_n:.0f} ({err:.1%})", err <= 0.05)
    else:
        bar("Book count", "within 5% of true count", "no ground truth", None)

    # Titles
    legible = [r for r in books_gt if r.get("legible", "").lower() in ("yes", "y", "1", "true") and r.get("title")]
    named = [b for b in p.books if b.status in ("identified", "needs_appraisal") and b.title]
    m = match_books(legible, named)
    confident = [b for b in named if b.id_confidence >= s.id_accept]
    all_titles = [r["title"] for r in books_gt if r.get("title")]
    wrong = [b for b in confident if not any(title_match(t, b.title) for t in all_titles)]
    if legible:
        rate = len(m) / len(legible)
        wrong_rate = len(wrong) / max(1, len(legible))
        bar(
            "Title identification",
            ">= 70% of legible spines correct, <= 3% confidently wrong",
            f"{rate:.0%} correct ({len(m)}/{len(legible)}); {wrong_rate:.1%} confidently wrong ({len(wrong)})",
            rate >= 0.70 and wrong_rate <= 0.03,
            {
                "missed": [r["title"] for i, r in enumerate(legible) if i not in m],
                "confidently_wrong": [{"id": b.id, "title": b.title, "spine": b.spine_text} for b in wrong],
            },
        )
    else:
        bar("Title identification", ">= 70% correct, <= 3% confidently wrong", "no ground truth", None)

    # Spine dimensions
    # Pair over the whole sheet (shelf order needs every book), then grade only the hand-measured rows.
    paired = match_by_position(books_gt, list(p.books), match_books(books_gt, [b for b in p.books if b.title]))
    measured_idx = [i for i, r in enumerate(books_gt) if _num(r.get("spine_height_cm"))]
    measured = [books_gt[i] for i in measured_idx]
    mm = {i: paired[i] for i in measured_idx if i in paired}
    herr, terr, rows = [], [], []
    for i, b in mm.items():
        r = books_gt[i]
        eh, et = (
            _pct(b.spine_height_cm, _num(r["spine_height_cm"])),
            _pct(b.spine_thickness_cm, _num(r.get("spine_thickness_cm"))),
        )
        # A paired book the system did not measure is a miss, not a skipped row.
        herr.append(1.0 if eh is None else eh)
        if _num(r.get("spine_thickness_cm")):
            terr.append(1.0 if et is None else et)
        rows.append(
            {
                "title": r["title"],
                "true_h": r["spine_height_cm"],
                "sys_h": b.spine_height_cm,
                "true_t": r.get("spine_thickness_cm"),
                "sys_t": b.spine_thickness_cm,
            }
        )
    if measured:
        within = sum(e <= 0.15 for e in herr + terr) / max(1, len(herr + terr))
        mh = statistics.mean(herr) if herr else None
        mt = statistics.mean(terr) if terr else None
        bar(
            "Spine dimensions",
            "within 15% on a hand-measured sample of 20",
            f"height MAPE {mh:.1%}, thickness MAPE {mt:.1%}; {within:.0%} of values within 15%; "
            f"{len(mm)}/{len(measured)} matched"
            if mh is not None and mt is not None
            else "no matched measurements",
            (mh is not None and mt is not None and mh <= 0.15 and mt <= 0.15 and len(mm) == len(measured)),
            rows,
        )
    else:
        bar("Spine dimensions", "within 15% on 20 books", "no ground truth", None)

    # Prices
    perr, prow = [], []
    priced_ids: set[str] = set()  # one packet book per truth row
    for r in prices_gt:
        hit = next(
            (b for b in p.books if b.id not in priced_ids and b.title and title_match(r["title"], b.title)), None
        )
        if hit:
            priced_ids.add(hit.id)
        true_r = _num(r.get("replacement_price"))
        sys_r = hit.replacement_cost.amount if hit else None
        e = _pct(sys_r, true_r)
        if e is not None:
            perr.append(e)
        prow.append(
            {
                "title": r["title"],
                "true": true_r,
                "system": sys_r,
                "source": hit.replacement_cost.source if hit else "",
                "error": None if e is None else round(e, 3),
            }
        )
    # Books with no reference price (not sold in the country) cannot be graded; the app's blank there is right.
    graded = [r for r in prow if r["true"] is not None]
    if graded:
        ok = sum(e <= 0.25 for e in perr)
        bar(
            "Book prices",
            "within 25% of hand-checked price on 15 books",
            f"{ok}/{len(graded)} within 25% (unpriced count as misses; "
            f"{len(prow) - len(graded)} without a reference price not graded)",
            ok == len(graded),
            prow,
        )
    else:
        bar("Book prices", "within 25% on 15 books", "no ground truth", None)

    # Items
    if items_gt:
        sys_cats = [i.category for i in p.items]
        found = 0
        for r in items_gt:
            raw = r["category"].strip().lower()
            cat = ITEM_CATEGORY_ALIASES.get(raw) or CATEGORY.get(raw, raw)
            if cat in sys_cats:
                sys_cats.remove(cat)
                found += 1
        rate = found / len(items_gt)
        bar(
            "Non-book items",
            ">= 80% found with correct category",
            f"{rate:.0%} ({found}/{len(items_gt)})",
            rate >= 0.8,
            {"extra_system_items": len(sys_cats)},
        )
    else:
        bar("Non-book items", ">= 80% found", "no ground truth", None)

    # Room
    for key, name, tol in (("floor_area_m2", "Floor area", 0.10), ("wall_area_m2", "Wall area", 0.15)):
        true_v = room_gt.get(key)
        sys_v = getattr(p.room, key)
        e = _pct(sys_v, true_v)
        bar(
            name,
            f"within {tol:.0%} of tape-measured",
            "no ground truth"
            if true_v is None
            else ("not measured" if sys_v is None else f"{sys_v} vs {true_v} m² ({e:.1%})"),
            None if true_v is None else (e is not None and e <= tol),
        )

    t_raw = p.metrics.get("sweep_ended_to_packet_s")
    t = float(t_raw) if isinstance(t_raw, (int, float)) else None
    bar(
        "Time to packet",
        "under 5 minutes after the sweep ends",
        f"{t} s" if t is not None else "unknown",
        None if t is None else t < 300,
    )
    return res


def render_markdown(res: dict[str, Any]) -> str:
    lines = [
        f"# Results vs ground truth — sweep {res['sweep']}",
        "",
        "| Measure | Pass bar | Result | Pass |",
        "|---|---|---|---|",
    ]
    for b in res["bars"]:
        mark = {True: "PASS", False: "FAIL", None: "n/a"}[b["pass"]]
        lines.append(f"| {b['measure']} | {b['pass_bar']} | {b['result']} | {mark} |")
    for b in res["bars"]:
        if b["detail"]:
            lines += ["", f"## {b['measure']} detail", "", "```", str(b["detail"])[:4000], "```"]
    return "\n".join(lines) + "\n"
