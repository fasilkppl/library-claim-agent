"""One entry per physical book: spine tracks that are the same book are merged.

Registration links a spine from frame to frame by image motion, and across a revisit only when AR tracking is
trusted. Without trusted AR, a second pass over a shelf (after looking at the ceiling, say) starts new tracks
for the same books, and a spine detected as two halves becomes two neighbouring tracks. Both would be counted
twice. This step merges them, in three passes:

1. Revisits: each shelf row is aligned, in left-to-right order, against the rows already kept (sequence
   alignment). When at least two spines agree in order, the rows are the same shelf: aligned spines merge, and
   spines seen only in the revisit are added at their place.
2. Neighbours: spines up to two places apart in one row that read the same, or name the same book, merge.
3. Final check: across the whole sweep no two entries may name the same catalogue work or carry near-identical
   readings. Any pair left is merged and logged; the packet never lists a book twice.

Every merge is recorded with its reason in the stage output, so a wrong merge can be seen and argued with.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import cv2
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from claimagent.pipeline.resolve import content_tokens, same_work, volume_numbers

log = logging.getLogger(__name__)

MIN_TOKEN = 4  # shorter words ("the", "of", OCR crumbs) say nothing about which book it is
TOKEN_MATCH = 0.75  # word similarity (edit distance) for two words to be one word misread ("FIRR" ~ "FIRE")
SAME_READING = 0.5  # similarity at which two readings are taken as the same spine (when order supports it)
SAME_ANYWHERE = 0.75  # similarity at which two readings are the same spine wherever they are
NEIGHBOUR_SPAN = 2  # positions apart within one row for the neighbour pass
MIN_ANCHORS = 2  # agreeing spines needed to call two rows one shelf
GAP = -0.4
SPANNED_TITLE = 0.6  # share of an identified book's title words found in an unread detection's reading
GLIMPSE_FRAMES = 2  # an unread entry seen in at most this many frames, placed on no shelf row: a glimpse
SLOT_MISMATCH = -0.3  # on one guided shelf: pairing an unread spine with the spine at its place (vs two gaps -0.8)
# Shared area (of the smaller outline) at which two spine detections in one frame are one spine; neighbours
# stay below 0.08, one spine detected twice reaches 0.34 to 0.89 (measured on real sweeps).
SAME_SPINE_OVERLAP = 0.15
# Words printed on many different spines: publishers and generic words. They must not make two books "the same".
COMMON = {
    "with",
    "from",
    "that",
    "this",
    "your",
    "book",
    "books",
    "edition",
    "novel",
    "story",
    "penguin",
    "harper",
    "harpercollins",
    "collins",
    "rupa",
    "vintage",
    "random",
    "house",
    "press",
    "publishing",
    "publishers",
    "classics",
    "jaico",
    "hachette",
    "macmillan",
    "bloomsbury",
    "scholastic",
    "westland",
    "fingerprint",
    "simon",
    "schuster",
    "portfolio",
    "india",
}


def tokens(text: str) -> list[str]:
    words = re.findall(r"[^\W\d_]+", text.lower())  # letters of any alphabet ("erzählungen" stays one word)
    return [w for w in dict.fromkeys(words) if len(w) >= MIN_TOKEN and w not in COMMON]


def same_word(w: str, x: str) -> bool:
    """One word misread: OCR swaps letters ("ALMAMACE" ~ "ALMANACK") and adds or drops one ("CKHALED")."""
    if w == x:
        return True
    return max(Levenshtein.normalized_similarity(w, x), fuzz.ratio(w, x) / 100) >= TOKEN_MATCH


def _matched(a: list[str], b: list[str]) -> list[str]:
    return [w for w in a if any(same_word(w, x) for x in b)]


def reading_similarity(a: list[str], b: list[str]) -> float:
    """0..1: how surely two spine readings are the same spine. The shorter reading must be mostly covered and
    the longer one at least half, with one matched word of five letters or more ("DAN BROWN" leaking onto a
    neighbour's long reading is not enough)."""
    if not a or not b:
        return 0.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    m = _matched(short, long_)
    if not any(len(w) >= 5 for w in m):
        return 0.0
    cov_short, cov_long = len(m) / len(short), len(m) / len(long_)
    if cov_short < 0.5 or cov_long < 0.5:
        return 0.0
    return round((cov_short + cov_long) / 2, 3)


Box = list[list[float]]  # a spine's 4 corners in one frame


def volumes(text: str) -> set[int]:
    """Volume numbers, standalone ("Band 2", "II") or glued to a word by OCR ("Erzählungen1")."""
    glued = {int(n) for n in re.findall(r"(?<=[^\W\d_])(\d{1,2})\b", text)}
    return volume_numbers(text) | glued


class _Book:
    """A track with what is known about it: its reading and, when resolved, the catalogue work."""

    def __init__(
        self, read: dict[str, Any], res: dict[str, Any], boxes: dict[int, Box] | None = None, upright: bool = True
    ) -> None:
        self.read, self.res = read, res
        self.track = int(read["track"])
        self.boxes = boxes or {}  # frame seq -> spine box in that frame
        self.upright = upright
        self.volumes = volumes(f"{read.get('spine_text', '')} {res.get('title', '')}")
        # The consensus reading only: single-frame variants often carry a neighbour's leaked line, and a merge
        # made on a leak would remove a real book.
        self.readings = [t for t in (tokens(read.get("spine_text", "")),) if t]
        self.tokens = list(dict.fromkeys(w for r in self.readings for w in r))
        self.identified = res.get("status") in ("identified", "needs_appraisal") and bool(res.get("title"))
        self.title = str(res.get("title", "")) if self.identified else ""
        self.work_tokens = tokens(f"{self.title} {res.get('author', '')}") if self.identified else []

    def label(self) -> str:
        return f"track {self.track} [{self.title or (self.read.get('spine_text') or 'no text')[:40]}]"


def distinct(a: _Book, b: _Book) -> str:
    """Why two tracks cannot be one book (empty when they can). Two volumes of a set are different books, and
    two spines seen side by side in one frame are two physical books. A spine detected twice in one frame lies
    on top of itself instead, so that case is still allowed."""
    if a.volumes and b.volumes and not (a.volumes & b.volumes):
        return f"different volumes {sorted(a.volumes)} / {sorted(b.volumes)}"
    if a.boxes.keys() & b.boxes.keys() and not on_top(a, b):
        return "side by side in the frames where both were seen"
    return ""


def overlap_share(p: Box, q: Box) -> float:
    """Area two spine outlines (4 corners each, tilt included) share, over the smaller one."""
    pa = cv2.convexHull(np.array(p, np.float32)).reshape(-1, 2)
    qa = cv2.convexHull(np.array(q, np.float32)).reshape(-1, 2)
    smaller = float(min(cv2.contourArea(pa), cv2.contourArea(qa)))
    if smaller <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(pa, qa)
    return float(inter) / smaller


def on_top(a: _Book, b: _Book) -> bool:
    """One spine detected twice: in the frames where both were seen, their outlines lie on each other (in two of
    them, or the only one). Two neighbouring books, even leaning ones, barely touch (measured: at most 0.08)."""
    shared = sorted(a.boxes.keys() & b.boxes.keys())
    if not shared:
        return False
    hits = sum(1 for seq in shared if overlap_share(a.boxes[seq], b.boxes[seq]) >= SAME_SPINE_OVERLAP)
    return hits >= min(2, len(shared))


GLIMPSE_LETTERS = 70  # letter match (spaces ignored) of a glimpse's reading to an identified title and author


def _letters(text: str) -> str:
    """Letters only, accents dropped: "WATCHMÄNT | HARPERLEE" -> "watchmantharperlee"."""
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower())


def _title_inside(a: _Book, b: _Book) -> bool:
    """One catalogue title holds all the words (two or more) of the other, by the same author: "My Experiments
    with Truth" and "Mahatma Gandhi: The Story of My Experiments with Truth" are one book. One word is not
    enough ("Dune" in "Dune Messiah")."""
    ta, tb = (set(content_tokens(re.sub(r"\s+by\s+.*$", "", x.title, flags=re.I))) for x in (a, b))
    short, long_ = sorted((ta, tb), key=len)
    if len(short) < 2 or not short <= long_:
        return False
    aa, ab = (set(content_tokens(str(x.res.get("author", "")))) for x in (a, b))
    return not (aa and ab) or bool(aa & ab)


def similarity(a: _Book, b: _Book) -> float:
    if distinct(a, b):
        return 0.0
    if a.identified and b.identified:
        return 1.0 if same_work(a.title, b.title) or _title_inside(a, b) else 0.0
    if a.identified != b.identified:
        known, other = (a, b) if a.identified else (b, a)
        if not other.readings:
            return 0.0
        # The unidentified readings against the identified book's title and author, and against its readings.
        best = 0.0
        for r in other.readings:
            cov = len(_matched(r, known.work_tokens)) / len(r)
            if cov >= SAME_READING and any(len(w) >= 5 for w in _matched(r, known.work_tokens)):
                best = max(best, round(cov, 3))
        # A brief glimpse (a quick look at the shelf before the slow pass) is often read with words glued or split
        # ("ND HARPERLEE", "GO SETA") and short title words drop out of the word match; its letters still spell
        # the book. Glimpses only: a well-seen spine's reading can carry a neighbour's leaked lines.
        if int(other.read.get("observations", 0)) <= GLIMPSE_FRAMES:
            spelled = _letters(str(other.read.get("spine_text", "")))
            name = _letters(f"{known.title} {known.res.get('author', '')}")
            if len(spelled) >= 8 and fuzz.partial_ratio(spelled, name) >= GLIMPSE_LETTERS:
                best = max(best, SAME_ANYWHERE)
        # Reading against reading only when nearly identical: spine readings can carry a neighbour's leaked
        # lines ("WATCHMAN HARPER LEE" read onto two spines), and a partial overlap of leaks is not the same book.
        own = _best_reading(known, other)
        return max(best, own if own >= SAME_ANYWHERE else 0.0)
    return _best_reading(a, b)


def _best_reading(a: _Book, b: _Book) -> float:
    return max((reading_similarity(x, y) for x in a.readings for y in b.readings), default=0.0)


def _pair_score(a: _Book, b: _Book) -> float:
    if not a.tokens and not a.identified and not b.tokens and not b.identified:
        return 0.3  # two unreadable spines at the same place in order: likely the same book
    if (not a.tokens and not a.identified) or (not b.tokens and not b.identified):
        return -0.2
    s = similarity(a, b)
    return 2 * s if s >= SAME_READING else (-1.0 if s == 0 else 0.0)


def _slot_score(a: _Book, b: _Book, same_shelf: bool) -> float:
    """Pair score for the alignment. On one guided shelf, an unread spine at the same place as another spine is
    likely the same book (its text came out garbled), so pairing them costs less than two gaps."""
    sc = _pair_score(a, b)
    if same_shelf and sc <= 0 and not (a.identified and b.identified) and not distinct(a, b):
        return max(sc, SLOT_MISMATCH)
    return sc


def align(row_a: list[_Book], row_b: list[_Book], same_shelf: bool = False) -> list[tuple[int | None, int | None]]:
    """Global sequence alignment (Needleman-Wunsch) of two left-to-right rows; pairs of indices, None = gap."""
    n, m = len(row_a), len(row_b)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[""] * (m + 1) for _ in range(n + 1)]  # the step taken into each cell: "d" pair, "u" gap in b, "l" gap in a
    for i in range(1, n + 1):
        score[i][0], move[i][0] = i * GAP, "u"
    for j in range(1, m + 1):
        score[0][j], move[0][j] = j * GAP, "l"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                (score[i - 1][j - 1] + _slot_score(row_a[i - 1], row_b[j - 1], same_shelf), "d"),
                (score[i - 1][j] + GAP, "u"),
                (score[i][j - 1] + GAP, "l"),
            )
            score[i][j], move[i][j] = max(options, key=lambda o: o[0])
    out: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = move[i][j]
        if step == "d":
            out.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == "u":
            out.append((i - 1, None))
            i -= 1
        else:
            out.append((None, j - 1))
            j -= 1
    return out[::-1]


class _Groups:
    """Union-find over track ids, remembering why each merge happened."""

    def __init__(self, ids: list[int]) -> None:
        self.parent = {t: t for t in ids}
        self.log: list[dict[str, Any]] = []

    def find(self, t: int) -> int:
        while self.parent[t] != t:
            self.parent[t] = self.parent[self.parent[t]]
            t = self.parent[t]
        return t

    def best(self, root: int, books: dict[int, _Book]) -> _Book:
        """The group's best-known member: the one compared and kept (identified, most confident, most seen)."""
        return max((b for t, b in books.items() if self.find(t) == root), key=_rank)

    def union(self, a: _Book, b: _Book, reason: str, score: float) -> None:
        ra, rb = self.find(a.track), self.find(b.track)
        if ra == rb:
            return
        self.parent[rb] = ra
        entry = {"kept": a.track, "merged": b.track, "reason": reason, "score": score, "a": a.label(), "b": b.label()}
        self.log.append(entry)
        log.info("DEDUPE %s: %s == %s (score %.2f)", reason, a.label(), b.label(), score)


def dedupe_books(
    read: dict[str, Any],
    resolved: dict[str, Any],
    boxes: dict[int, dict[int, Box]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Returns (read, resolved, report) with one entry per physical book. `boxes`: track -> frame -> spine box,
    for the side-by-side rule (two spines in one frame are two books)."""
    res_by = {int(r["track"]): r for r in resolved["books"]}
    boxes = boxes or {}
    books = {
        int(b["track"]): _Book(
            b, res_by.get(int(b["track"]), {}), boxes.get(int(b["track"])), b.get("orientation", "upright") != "flat"
        )
        for b in read["books"]
    }
    groups = _Groups(list(books))

    # Rows in left-to-right order.
    rows: dict[str, list[_Book]] = {}
    for b in sorted(read["books"], key=lambda x: (x["shelf"], x["position"])):
        rows.setdefault(b["shelf"], []).append(books[int(b["track"])])

    # The guide's shelf for each row (guided sweeps): the most common shelf among its spines.
    guide_shelf: dict[str, int] = {}
    for label, row in rows.items():
        tags = [int(b.read["guide_shelf"]) for b in row if b.read.get("guide_shelf") is not None]
        if tags:
            guide_shelf[label] = max(sorted(set(tags)), key=tags.count)

    # 1. Neighbours within one pass: a spine detected as two pieces, or read twice side by side.
    for label, row in rows.items():
        for i, a in enumerate(row):
            for b in row[i + 1 : i + 1 + NEIGHBOUR_SPAN]:
                s = similarity(a, b)
                # A weaker match needs proof of one spine detected twice (the boxes lie on each other in the
                # same frames); otherwise neighbours sharing only an author's name would be merged.
                # The weaker match also needs a catalogue title on one side: two unread spines that share some
                # leaked words are not proof.
                weak_ok = s >= SAME_READING and on_top(a, b) and (a.identified or b.identified)
                if s >= SAME_ANYWHERE or weak_ok:
                    groups.union(a, b, f"neighbours on shelf {label} read the same", s)

    def reps_of(row: list[_Book]) -> list[_Book]:
        out: list[_Book] = []
        for b in row:
            r = groups.best(groups.find(b.track), books)
            if not out or out[-1] is not r:
                out.append(r)
        return out

    # 2. Revisits: the best-read row of a shelf first, every other row aligned against the rows kept so far.
    order = sorted(rows, key=lambda k: -sum(1 for x in rows[k] if x.tokens or x.identified))
    kept_rows: list[tuple[str, list[_Book]]] = []
    for label in order:
        row = reps_of(rows[label])
        best: tuple[int, list[tuple[int | None, int | None]], int] | None = None
        shelf = guide_shelf.get(label)
        for k, (kept_label, kept) in enumerate(kept_rows):
            other = guide_shelf.get(kept_label)
            if shelf is not None and other is not None and shelf != other:
                continue  # the guide had moved to another shelf: these are different books
            pairs = align(kept, row, same_shelf=shelf is not None and shelf == other)
            anchors = sum(
                1 for i, j in pairs if i is not None and j is not None and similarity(kept[i], row[j]) >= SAME_READING
            )
            # On the same guided shelf one agreeing spine is enough: it is that shelf seen again.
            needed = 1 if shelf is not None and shelf == other else MIN_ANCHORS
            if anchors >= needed and (best is None or anchors > best[2]):
                best = (k, pairs, anchors)
        if best is None:
            kept_rows.append((label, row))
            continue
        k, pairs, anchors = best
        kept_label, kept = kept_rows[k]
        same_shelf = shelf is not None and shelf == guide_shelf.get(kept_label)
        merged_row: list[_Book] = []
        for n, (ia, jb) in enumerate(pairs):
            if ia is not None and jb is not None and _pair_score(kept[ia], row[jb]) > 0:
                reason = f"revisit of shelf {kept_label} ({anchors} spines agree in order)"
                groups.union(kept[ia], row[jb], reason, similarity(kept[ia], row[jb]))
                merged_row.append(kept[ia])
            elif (
                ia is not None
                and jb is not None
                and same_shelf
                and _slot_score(kept[ia], row[jb], True) > _pair_score(kept[ia], row[jb]) - 1e-9
                and not (kept[ia].identified and row[jb].identified)
                and not distinct(kept[ia], row[jb])
                and _between_agreeing(pairs, n, kept, row)
            ):
                # Same guided shelf, same place, agreeing books around it: one book whose text came out garbled.
                reason = f"same place on guided shelf {shelf} as a book seen on another pass"
                groups.union(kept[ia], row[jb], reason, 0.0)
                merged_row.append(kept[ia])
            else:
                if ia is not None:
                    merged_row.append(kept[ia])
                if jb is not None:
                    merged_row.append(row[jb])  # seen only in this pass: added at its place on the shelf
        log.info("DEDUPE row %s aligned onto %s with %s agreeing spines", label, kept_label, anchors)
        kept_rows[k] = (kept_label, merged_row)

    # Final check across the sweep: no work twice, no near-identical readings twice.
    violations = enforce_unique(books, groups)
    demoted = _one_title_per_book(books, groups)
    not_counted = _glimpses(books, groups, kept_rows, guide_shelf)

    report_extra = {"titles_removed": demoted}
    read_out, res_out, report = _rebuild(read, resolved, books, groups, kept_rows, violations, guide_shelf, not_counted)
    report.update(report_extra)
    return read_out, res_out, report


def enforce_unique(books: dict[int, _Book], groups: _Groups) -> int:
    """Merges any two remaining entries for the same work or with near-identical readings; returns how many."""
    n = 0
    changed = True
    while changed:
        changed = False
        reps = sorted({groups.best(groups.find(t), books).track for t in books})
        for i, a_id in enumerate(reps):
            for b_id in reps[i + 1 :]:
                a, b = books[a_id], books[b_id]
                s = similarity(a, b)
                glimpse = (
                    a.identified != b.identified
                    and int((b if a.identified else a).read.get("observations", 0)) <= 2
                    and s >= SAME_READING
                )
                if s >= SAME_ANYWHERE or (a.identified and b.identified and s == 1.0) or glimpse:
                    reason = "same book named twice" if a.identified and b.identified else "same reading twice"
                    if glimpse and s < SAME_ANYWHERE:
                        reason = "brief glimpse of a book identified elsewhere"
                    keep, drop = (a, b) if _rank(a) >= _rank(b) else (b, a)
                    groups.union(keep, drop, f"final check: {reason}", s)
                    n += 1
                    changed = True
                    break
            if changed:
                break
    return n


def _rank(b: _Book) -> tuple[int, float, int, float]:
    return (
        int(b.identified),
        float(b.res.get("confidence", 0.0)),
        int(b.read.get("observations", 0)),
        float(b.read.get("text_score", 0.0)),
    )


def _one_title_per_book(books: dict[int, _Book], groups: _Groups) -> list[dict[str, Any]]:
    """Two different books (seen side by side, so not merged) that resolved to the same title: one of them was
    read from its neighbour's text (a spine picks up the next book's author line). The better match keeps the
    title; the other becomes unidentified, with the reason, so the packet never lists one title twice."""
    reps = {groups.best(groups.find(t), books).track for t in books}
    ided = sorted((books[t] for t in reps if books[t].identified), key=_rank, reverse=True)
    titles = {b.title.split(":")[0].lower(): tokens(b.title.split(":")[0]) for b in ided}

    def other_titles_read(b: _Book) -> int:
        """How many different identified titles this spine's reading contains (a leak reads two or more)."""
        return sum(
            1
            for tt in titles.values()
            if tt and any(len(_matched(tt, r)) / len(tt) >= SPANNED_TITLE for r in b.readings)
        )

    out: list[dict[str, Any]] = []
    for i, keep in enumerate(ided):
        for other in ided[i + 1 :]:
            if not (
                other.identified and keep.identified and same_work(keep.title, other.title) and distinct(keep, other)
            ):
                continue
            # Volumes of a set and second copies keep their title; only a reading that also holds another
            # book's title (picked up from a neighbour) loses it.
            if keep.volumes and other.volumes and not (keep.volumes & other.volumes):
                continue
            if other_titles_read(other) >= 2:
                reason = f"same title as its neighbour {keep.label()}: its reading was likely the neighbour's"
                other.res = {
                    **other.res,
                    "status": "unidentified",
                    "title": "",
                    "author": "",
                    "isbn": "",
                    "confidence": 0.0,
                    "demoted": reason,
                }
                other.identified, other.title = False, ""
                out.append({"track": other.track, "reason": reason})
                log.info("DEDUPE title removed from %s: %s", other.track, reason)
    return out


def _glimpses(
    books: dict[int, _Book], groups: _Groups, kept_rows: list[tuple[str, list[_Book]]], guide_shelf: dict[str, int]
) -> dict[int, str]:
    """Guided sweeps only: unread fragments that are not books of their own. The policyholder moved along each
    shelf once, so every real book was seen whole at some point and sits in that shelf's row. Not counted:
    an unread spine never seen whole (only ever cut by the picture's edge: the side of a box, a wall), and a
    brief unread glimpse (1-2 frames) that fits nowhere in a shelf whose row is already known."""
    members: dict[int, list[_Book]] = {}
    for t, b in books.items():
        members.setdefault(groups.find(t), []).append(b)
    out: dict[int, str] = {}
    # Titles of the identified books (and those books), for the multi-spine check below.
    works: dict[str, list[str]] = {}
    work_books: dict[str, list[_Book]] = {}
    for b in books.values():
        if b.identified:
            key = b.title.split(":")[0].lower()
            works.setdefault(key, tokens(b.title.split(":")[0]))
            work_books.setdefault(key, []).append(b)
    first_row: dict[int, str] = {}
    for label, row in kept_rows:
        shelf = guide_shelf.get(label)
        if shelf is None:
            continue
        extra_row = shelf in first_row
        first_row.setdefault(shelf, label)
        for b in row:
            root = groups.find(b.track)
            ms = members[root]
            if any(m.identified for m in ms):
                continue
            readings = [r for m in ms for r in m.readings]
            spanned = [
                w
                for w, wt in works.items()
                if wt and any(len(_matched(wt, r)) / len(wt) >= SPANNED_TITLE for r in readings)
            ]
            # Dropped only when it lies over one of those books (a box spanning several spines). A spine of
            # its own beside them, whose reading picked up the neighbours' lines, is a book (unidentified).
            covers = any(on_top(m, kb) for w in spanned for kb in work_books.get(w, []) for m in ms)
            if len(spanned) >= 2 and covers:
                out[root] = f"one detection reading the titles of {len(spanned)} different books: {spanned[:3]}"
            elif all(m.read.get("only_partial") for m in ms) and not any(len(r) >= 2 for m in ms for r in m.readings):
                out[root] = "unread and never seen whole (only cut by the picture's edge)"
            elif extra_row and sum(int(m.read.get("observations", 0)) for m in ms) <= GLIMPSE_FRAMES:
                out[root] = f"unread glimpse that fits nowhere on guided shelf {shelf}"
    for root, why in out.items():
        log.info("DEDUPE not counted: %s: %s", books[root].label(), why)
    return out


def _between_agreeing(pairs: list[tuple[int | None, int | None]], n: int, kept: list[_Book], row: list[_Book]) -> bool:
    """The pair at n has spines that agree right next to it: on both sides, or on one side with the row ending
    on the other (no other spine can be the match)."""

    def agrees(step: int) -> bool | None:
        """The pair right next to it agrees; None when the rows end there. A gap next to it means other spines
        lie between, and then the place is ambiguous (seen on a real sweep: a book joined to the wrong one)."""
        k = n + step
        if not 0 <= k < len(pairs):
            return None
        ia, jb = pairs[k]
        return ia is not None and jb is not None and _pair_score(kept[ia], row[jb]) > 0

    left, right = agrees(-1), agrees(1)
    return (left is True and right is not False) or (right is True and left is not False)


def _rebuild(
    read: dict[str, Any],
    resolved: dict[str, Any],
    books: dict[int, _Book],
    groups: _Groups,
    kept_rows: list[tuple[str, list[_Book]]],
    violations: int,
    guide_shelf: dict[str, int],
    not_counted: dict[int, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    members: dict[int, list[_Book]] = {}
    for t, b in books.items():
        members.setdefault(groups.find(t), []).append(b)
    rep_of = {root: max(ms, key=_rank) for root, ms in members.items()}
    out_books, out_res = [], []
    seen: set[int] = set()  # across rows: a book merged from another row is listed once, on the first row
    for label, row in kept_rows:
        position = 0
        for b in row:
            root = groups.find(b.track)
            if root in seen or root in not_counted:
                continue
            seen.add(root)
            rep, ms = rep_of[root], members[root]
            position += 1
            entry = dict(rep.read)
            shelf_no = guide_shelf.get(label)
            entry["shelf"] = f"shelf {shelf_no}" if shelf_no is not None else label
            entry["position"] = position
            entry["observations"] = sum(int(m.read.get("observations", 0)) for m in ms)
            variants = [v for m in sorted(ms, key=_rank, reverse=True) for v in m.read.get("text_variants", [])]
            entry["text_variants"] = list(dict.fromkeys(variants))[:5]
            entry["merged_tracks"] = sorted(m.track for m in ms if m.track != rep.track)
            out_books.append(entry)
            if rep.res:
                out_res.append(rep.res)
    report = {
        "tracks_in": len(books),
        "books_out": len(out_books),
        "merges": groups.log,
        "final_check_merges": violations,
        "not_counted": [{"track": t, "reason": why, "label": books[t].label()} for t, why in not_counted.items()],
    }
    log.info("DEDUPE %s spine tracks -> %s books (%s merges)", len(books), len(out_books), len(groups.log))
    new_read = {**read, "books": out_books}
    new_res = {**resolved, "books": out_res}
    return new_read, new_res, report


def duplicate_pairs(
    read: dict[str, Any], resolved: dict[str, Any], boxes: dict[int, dict[int, Box]] | None = None
) -> list[tuple[int, int, float]]:
    """The check on its own: pairs of entries that are still the same book (empty = no duplicates)."""
    res_by = {int(r["track"]): r for r in resolved["books"]}
    boxes = boxes or {}
    bs = []
    for b in read["books"]:
        t = int(b["track"])
        tb = dict(boxes.get(t, {}))
        for m in b.get("merged_tracks", []):  # a merged entry was seen wherever any of its tracks was
            tb.update(boxes.get(int(m), {}))
        bs.append(_Book(b, res_by.get(t, {}), tb, b.get("orientation", "upright") != "flat"))
    out = []
    for i, a in enumerate(bs):
        for b in bs[i + 1 :]:
            s = similarity(a, b)
            if s >= SAME_ANYWHERE or (a.identified and b.identified and s == 1.0):
                out.append((a.track, b.track, s))
    return out
