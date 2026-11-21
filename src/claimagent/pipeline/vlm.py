"""A second look at each non-book item with the local vision-language model (Qwen3-VL through Ollama).

The fast detector names objects from a fixed list and confuses look-alikes (a clothes iron called an electric
kettle, a pressure cooker a helmet, a desk a bed). The item's name decides its price query, so a wrong name
means a wrong price. The vision-language model looks at the item's crop and names it; it is not shown the
detector's guess, so it does not simply agree. Parts of the building (a switch board, a door) are not contents
and are removed. The model only names objects: it never sees or produces a price.
"""

from __future__ import annotations

import base64
import logging
import re
import time
from typing import Any

import cv2
import httpx
import numpy as np
from rapidfuzz import fuzz

from claimagent.config import get_settings
from claimagent.pipeline.household import CATEGORY, ITEM_VOCAB

log = logging.getLogger(__name__)

PROMPT = (
    "The image is a grid of numbered photos, each showing one object in a home. For each number, name the "
    'object in one to three common words, for example "pressure cooker", "clothes iron", "bucket", '
    '"light switch". Reply with one line per number, like "1: bucket". If you cannot tell, write "unknown".'
)
# Parts of the building, not contents: they are not claimed as items.
FIXTURES = (
    "switch",
    "switches",
    "switchboard",
    "socket",
    "outlet",
    "plug point",
    "wall",
    "floor",
    "ceiling",
    "door",
    "window",
    "tile",
    "tiles",
    "pipe",
    "sink",
)
# Whole words only, and only when the answer is not a contents word itself ("floor lamp", "ceiling fan").
FIXTURE_RE = re.compile(r"\b(" + "|".join(re.escape(f) for f in FIXTURES) + r")\b")
UNSURE = r"(unknown|unclear|not sure|cannot tell|can ?t tell|unsure|none|nothing)\b"
BUDGET_S = 60.0
CROP_PAD = 0.12
TILE = 256  # px per photo in the grid; the model reads an image at a fixed cost, so one grid serves nine items
GRID = 3
# A crop flatter than this at tile size (Laplacian variance) shows nothing nameable: a dark corner, a reflection,
# a hole in a table. The vision model still names such crops confidently ("coffee maker", "coaster"), so they are
# not asked. On three practice sweeps every real item scored 32 or more and 6 of 15 such crops scored under 12.
MIN_TILE_DETAIL = 15.0


def _mosaic(crops: list[np.ndarray]) -> np.ndarray:
    """Up to GRID x GRID crops on a white grid, each numbered in its corner."""
    rows = (len(crops) + GRID - 1) // GRID
    sheet = np.full((rows * TILE, GRID * TILE, 3), 255, np.uint8)
    for n, crop in enumerate(crops):
        s = (TILE - 16) / max(crop.shape[:2])
        c = cv2.resize(
            crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA
        )
        y, x = (n // GRID) * TILE + 8, (n % GRID) * TILE + 8
        sheet[y : y + c.shape[0], x : x + c.shape[1]] = c
        cv2.rectangle(sheet, (x - 6, y - 6), (x + 34, y + 30), (255, 255, 255), -1)
        cv2.putText(sheet, str(n + 1), (x - 2, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return sheet


def tile_detail(crop: np.ndarray) -> float:
    """Laplacian variance of the crop at the size the model sees it (its tile in the grid)."""
    s = (TILE - 16) / max(crop.shape[:2])
    c = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _ask(crops: list[np.ndarray], timeout: float) -> list[str]:
    """One answer per crop (empty when the model gave none for that number)."""
    s = get_settings()
    ok, buf = cv2.imencode(".jpg", _mosaic(crops), [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return [""] * len(crops)
    r = httpx.post(
        f"{s.ollama_url}/api/chat",
        timeout=timeout,
        json={
            "model": s.llm_model,
            "stream": False,
            "think": False,
            "keep_alive": "60m",  # as the live calls: the model stays loaded for the next sweep
            "options": {"temperature": 0, "num_predict": 12 * len(crops)},
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(buf).decode()]}],
        },
    )
    r.raise_for_status()
    answers = [""] * len(crops)
    for line in str(r.json().get("message", {}).get("content", "")).splitlines():
        m = re.match(r"\s*(\d+)\s*[:.)-]\s*(.+)", line)
        if m and 1 <= int(m.group(1)) <= len(crops):
            answers[int(m.group(1)) - 1] = re.sub(r"[^a-z \-]", "", m.group(2).lower()).strip()
    return answers


def _contents(text: str) -> tuple[str, str, bool] | None:
    """The contents word the text spells, with a space or a letter off allowed ("door mat"); None otherwise."""
    if FIXTURE_RE.fullmatch(text):
        return None
    if text in CATEGORY:
        return text, CATEGORY[text], False
    best = max(ITEM_VOCAB, key=lambda v: fuzz.ratio(text, v))
    return (best, CATEGORY[best], False) if fuzz.ratio(text, best) >= 90 else None


def canonical(answer: str) -> tuple[str, str, bool] | None:
    """(label, category, is_fixture) for the model's answer; None when it could not tell."""
    words = answer.split()
    if not words or len(words) > 4 or re.match(UNSURE, answer):
        return None
    if answer in CATEGORY:  # a known contents word first: "floor lamp", "wall clock" are not fixtures
        return answer, CATEGORY[answer], False
    if FIXTURE_RE.search(answer):
        # A building word in front of a contents word names the contents: "door mat", "wall mirror".
        named = _contents(answer) or _contents(words[-1])
        return named if named is not None else (answer, "fixture", True)
    if re.fullmatch(r"(a )?(stack of |pile of )?books?", answer):
        return answer, "book", True  # books are counted in the book list, not as items
    best = max(ITEM_VOCAB, key=lambda v: fuzz.token_set_ratio(answer, v))
    if fuzz.token_set_ratio(answer, best) >= 90:
        return best, CATEGORY[best], False
    return answer, "other", False


FURNITURE = {"desk", "dining table", "table", "coffee table"}  # things books stand on


def _area(b: Any) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


HOLDERS = {"table", "shelving", "seating", "rug"}  # categories things stand on or in: inside them is not part of them


def parts_of_other_items(items: list[dict[str, Any]]) -> dict[str, str]:
    """Item id -> the item it is a part of: its box lies (80%+) inside a box at least twice its size that another
    item has in the same frame. A print on a gift box ("kite", "lantern") or a pan's handle is not an object; a
    padlock on a table is (furniture and mats hold things, so they are never the owner)."""
    out: dict[str, str] = {}
    for a in items:
        ab = a["bbox"]
        for b in items:
            bb = (b.get("boxes") or {}).get(str(a["frame_seq"]))
            if b is a or bb is None or b.get("category") in HOLDERS or _area(bb) < 2 * _area(ab):
                continue
            ix = max(0.0, min(ab[2], bb[2]) - max(ab[0], bb[0]))
            iy = max(0.0, min(ab[3], bb[3]) - max(ab[1], bb[1]))
            if _area(ab) > 0 and ix * iy / _area(ab) >= 0.8:
                out[a["id"]] = b["label"]
                break
    return out


def spine_cover(box: Any, spine_boxes: list[Any]) -> float:
    """Share of the box covered by detected book spines (overlaps summed, at most 1)."""
    a = _area(box)
    if a <= 0:
        return 0.0
    hit = 0.0
    for s in spine_boxes:
        ix = max(0.0, min(box[2], s[2]) - max(box[0], s[0]))
        iy = max(0.0, min(box[3], s[3]) - max(box[1], s[1]))
        hit += ix * iy
    return min(1.0, hit / a)


def name_items(
    items: list[dict[str, Any]],
    images: dict[int, np.ndarray],
    budget_s: float = BUDGET_S,
    spines: dict[int, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Renames items in place from the model's answers; returns the items removed (building fixtures, books, parts
    of other items). `spines`: the spine boxes detected in each frame, the evidence for an answer "book"."""
    removed: list[dict[str, Any]] = []
    for iid, owner in parts_of_other_items(items).items():
        it = next(x for x in items if x["id"] == iid)
        items.remove(it)
        removed.append({"id": iid, "detector_label": it["label"], "seen_as": f"part of the {owner}"})
    jobs: list[tuple[dict[str, Any], np.ndarray]] = []
    for it in items:
        img = images.get(it["frame_seq"])
        if img is None:
            continue
        x0, y0, x1, y1 = it["bbox"]
        px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
        crop = img[max(0, int(y0 - py)) : int(y1 + py), max(0, int(x0 - px)) : int(x1 + px)]
        if crop.size and min(crop.shape[:2]) >= 24:
            if tile_detail(crop) < MIN_TILE_DETAIL:
                it["vision_check"] = "unknown"  # too little detail to name: listed, not priced, sent to review
                continue
            jobs.append((it, crop))
    deadline = time.monotonic() + min(BUDGET_S, budget_s)
    batch = GRID * GRID
    for start in range(0, len(jobs), batch):
        if time.monotonic() > deadline:
            log.info("item check budget reached; %s items keep the detector's name", len(jobs) - start)
            break
        chunk = jobs[start : start + batch]
        try:
            answers = _ask([c for _, c in chunk], timeout=max(5.0, deadline - time.monotonic()))
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("item check unavailable (%s); detector names kept", exc)
            break
        for (it, _), answer in zip(chunk, answers, strict=True):
            it["vision_check"] = answer or "unknown"
            named = canonical(answer)
            if named is None:
                continue
            label, category, fixture = named
            if fixture and it["label"] in FURNITURE and (category == "book" or label == "floor"):
                # The detector found a table and the model saw the books standing on it (or its top, like a
                # floor): the table is still there.
                it["vision_check"] = f"{answer} (kept: books stand on the {it['label']})"
                continue
            if fixture and category == "book":
                cover = None if spines is None else spine_cover(it["bbox"], spines.get(it["frame_seq"], []))
                if (cover is None and it.get("room_step")) or (cover is not None and cover < 0.5):
                    # Called a book where no spine was detected (a water bottle next to the shelf): not a book,
                    # but not named either, so it is listed unpriced for review.
                    it["vision_check"] = "unknown"
                    continue
            if fixture:  # a part of the building, or a book (counted in the book list)
                items.remove(it)
                removed.append({"id": it["id"], "detector_label": it["label"], "seen_as": label})
            elif label != it["label"]:
                # Art keeps its category (it is flagged for appraisal) unless the model names another art kind.
                if it.get("category") == "art" and category != "art":
                    category = "art"
                it["detector_label"], it["label"], it["category"] = it["label"], label, category
    return removed
