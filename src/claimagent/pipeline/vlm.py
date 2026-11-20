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


def canonical(answer: str) -> tuple[str, str, bool] | None:
    """(label, category, is_fixture) for the model's answer; None when it could not tell."""
    words = answer.split()
    if not words or len(words) > 4 or re.match(UNSURE, answer):
        return None
    if answer in CATEGORY:  # a known contents word first: "floor lamp", "wall clock" are not fixtures
        return answer, CATEGORY[answer], False
    if FIXTURE_RE.search(answer):
        return answer, "fixture", True
    if re.fullmatch(r"(a )?(stack of |pile of )?books?", answer):
        return answer, "book", True  # books are counted in the book list, not as items
    best = max(ITEM_VOCAB, key=lambda v: fuzz.token_set_ratio(answer, v))
    if fuzz.token_set_ratio(answer, best) >= 90:
        return best, CATEGORY[best], False
    return answer, "other", False


def name_items(items: list[dict[str, Any]], images: dict[int, np.ndarray]) -> list[dict[str, Any]]:
    """Renames items in place from the model's answers; returns the items removed as building fixtures."""
    jobs: list[tuple[dict[str, Any], np.ndarray]] = []
    for it in items:
        img = images.get(it["frame_seq"])
        if img is None:
            continue
        x0, y0, x1, y1 = it["bbox"]
        px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
        crop = img[max(0, int(y0 - py)) : int(y1 + py), max(0, int(x0 - px)) : int(x1 + px)]
        if crop.size and min(crop.shape[:2]) >= 24:
            jobs.append((it, crop))
    removed: list[dict[str, Any]] = []
    deadline = time.monotonic() + BUDGET_S
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
            if fixture:  # a part of the building, or a book (counted in the book list)
                items.remove(it)
                removed.append({"id": it["id"], "detector_label": it["label"], "seen_as": label})
            elif label != it["label"]:
                # Art keeps its category (it is flagged for appraisal) unless the model names another art kind.
                if it.get("category") == "art" and category != "art":
                    category = "art"
                it["detector_label"], it["label"], it["category"] = it["label"], label, category
    return removed
