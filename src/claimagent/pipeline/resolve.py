"""Resolve spine text to a specific work (and an edition only when the spine supports it).

No language model. Open Library and Google Books are searched with the text read from the spine, and
each candidate is scored by how much of its title and author is actually on the spine. A title is only
accepted when the spine text explains it; otherwise the book stays unidentified.
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from claimagent.pipeline.http import CachedHttp

STOP = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "in",
    "on",
    "to",
    "for",
    "with",
    "by",
    "at",
    "from",
    "der",
    "die",
    "das",
    "und",
    "des",
    "le",
    "la",
    "les",
    "de",
    "du",
    "et",
    "el",
    "los",
    "y",
    "edition",
    "ed",
    "vol",
    "volume",
    "book",
    "books",
    "hrsg",
    "eds",
    "editor",
    "press",
    "verlag",
    "series",
    "new",
    "revised",
    "im",
    "am",
    "zum",
    "zur",
    "vom",
    "mit",
    "von",
    "ein",
    "eine",
    "einer",
    "dem",
    "den",
    "il",
    "lo",
    "di",
}
ROMAN_TOKENS = {"ii", "iii", "iv", "vi", "vii", "viii", "ix", "xi", "xii"}


def norm_tokens(text: str) -> list[str]:
    text = text.replace("ß", "ss").replace("ẞ", "SS")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    toks = re.findall(r"[a-z][a-z'\-]+|\d{4}", text)
    return [t.strip("'-") for t in toks if len(t) >= 2]


def content_tokens(text: str) -> list[str]:
    """Words that carry meaning; articles, volume numerals and numbers are left out."""
    return [t for t in norm_tokens(text) if t not in STOP and t not in ROMAN_TOKENS and not t.isdigit()]


ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}


def volume_numbers(text: str) -> set[int]:
    """Volume markers printed as a standalone roman numeral or 1-2 digit number (I, II, 2, Band 3)."""
    out: set[int] = set()
    for tok in re.findall(r"(?<![\w.])([IVX]{1,4}|\d{1,2})(?![\w.])", text):
        if tok.isdigit():
            out.add(int(tok))
        elif tok.lower() in ROMAN:
            out.add(ROMAN[tok.lower()])
    return out


WORDLIST = Path("/usr/share/dict/words")  # present on macOS and most Linux systems; no list = no cleanup


@lru_cache(maxsize=1)
def _dictionary() -> frozenset[str]:
    try:
        return frozenset(w.strip().lower() for w in WORDLIST.read_text(errors="ignore").split() if len(w.strip()) >= 2)
    except OSError:
        return frozenset()


GOOGLE_QUERIES_PER_TEXT = 6

GLUE_WORDS = {"of", "the", "and", "to", "in", "on", "for", "with", "a", "an", "der", "die", "das", "und"}


MAX_VARIANTS = 12
FALLBACK_SLOTS = 3


def same_work(a: str, b: str) -> bool:
    """Two catalogue titles name the same work: their main titles (before any subtitle) read alike as a whole.
    A one-way word overlap is not enough: "Dune Messiah" contains "Dune" but is another book."""
    ma, mb = (" ".join(content_tokens(t.split(":")[0])) for t in (a, b))
    return bool(ma and mb) and fuzz.ratio(ma, mb) >= 88


def ocr_clean(text: str) -> str:
    """Two common OCR slips on spines, repaired against a word list: one stray letter at the end of a word
    ("GARDENX" -> garden, "HISTORYA" -> history) and a small word glued to the next ("SONGOF" -> song of).
    Anything else, names included, is kept as read; the result is only used as an extra catalogue query."""
    words = _dictionary()
    if not words:
        return text
    out = []
    for tok in re.findall(r"[A-Za-zÀ-ÿ]+|[^A-Za-zÀ-ÿ]+", text):
        low = tok.lower()
        if not tok.isalpha() or len(low) < 5 or low in words:
            out.append(tok)
            continue
        fixed = low[:-1] if low[:-1] in words else None
        if fixed is None:
            for glue in GLUE_WORDS:
                if low.endswith(glue) and low[: -len(glue)] in words and len(low) - len(glue) >= 3:
                    fixed = f"{low[: -len(glue)]} {glue}"
                elif low.startswith(glue) and low[len(glue) :] in words and len(low) - len(glue) >= 3:
                    fixed = f"{glue} {low[len(glue) :]}"
        out.append(fixed or tok)
    return "".join(out)


def _covered_share(needed: list[str], pool: list[str]) -> float:
    """Share of the title explained by the spine, weighted by word length: OCR misses short words most often
    and they say least ("Go Set a Watchman" read as "WATCHMAN" is mostly covered; "Tales from a Mom" read as
    "a tale" is not)."""
    free = list(pool)
    got = 0
    for tok in sorted(needed, key=len, reverse=True):
        for i, p in enumerate(free):
            if _hit(tok, [p]):
                del free[i]
                got += len(tok)
                break
    return got / max(1, sum(len(t) for t in needed))


def _author_match(authors: list[str], spine_text: str, title_toks: list[str]) -> float:
    """1.0 when an author's surname is on the spine, 0.5 when another first name is printed right before it
    ("HARPER LEE" is not "Witness Lee"), 0 otherwise. Surnames of one or two letters are too short to prove
    anything."""
    lines = [content_tokens(ln) for ln in spine_text.split("|")]
    best = 0.0
    for author in authors:
        toks = content_tokens(author)
        if not toks or len(toks[-1]) < 3:
            continue
        surname, own = toks[-1], set(toks) | set(title_toks)
        for line in lines:
            for i, tok in enumerate(line):
                if not _hit(surname, [tok]):
                    continue
                before = line[i - 1] if i > 0 else ""
                other_first_name = len(before) >= 3 and before.isalpha() and not _hit(before, list(own))
                best = max(best, 0.5 if other_first_name else 1.0)
    return best


def _covered(needed: list[str], pool: list[str]) -> int:
    """How many of the needed words the pool explains, each pool word used once: one spine word
    ("GARDEN") cannot stand for two title words ("The Garden of Gardens")."""
    free = list(pool)
    n = 0
    for tok in sorted(needed, key=len, reverse=True):
        for i, p in enumerate(free):
            if _hit(tok, [p]):
                del free[i]
                n += 1
                break
    return n


def _hit(token: str, pool: list[str]) -> bool:
    if len(token) <= 3:
        return token in pool
    return any(fuzz.ratio(token, p) >= 84 or _same_stem(token, p) or _one_slip(token, p) for p in pool)


def _one_slip(a: str, b: str) -> bool:
    """One wrong, missing or extra letter in a word of five or more ("RIVFRS" for "rivers"); two in a word
    of nine or more."""
    n = min(len(a), len(b))
    if n < 5:
        return False
    return Levenshtein.distance(a, b) <= (2 if n >= 9 else 1)


def _same_stem(a: str, b: str) -> bool:
    """OCR drops word endings ("Wandere" for "Wandered"): the shorter word is the longer one's start, give or
    take its last letter."""
    if min(len(a), len(b)) < 5:
        return False
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n >= 4 and min(len(a), len(b)) - n <= 1


@dataclass
class Candidate:
    title: str
    authors: list[str]
    publisher: str = ""
    year: int | None = None
    isbn: str = ""
    source: str = ""
    url: str = ""
    volume_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    status: str  # identified | unidentified
    confidence: float
    spine_text: str
    title: str = ""
    author: str = ""
    publisher: str = ""
    edition: str = ""
    isbn: str = ""
    year: int | None = None
    source: str = ""
    url: str = ""
    volume_id: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


def score(spine_text: str, cand: Candidate) -> tuple[float, dict[str, Any]]:
    """Best of the spine as read and the spine with common OCR slips repaired (ocr_clean)."""
    raw = _score(spine_text, cand)
    cleaned = ocr_clean(spine_text)
    return raw if cleaned == spine_text else max(raw, _score(cleaned, cand), key=lambda r: r[0])


def _score(spine_text: str, cand: Candidate) -> tuple[float, dict[str, Any]]:
    spine = content_tokens(spine_text)
    if not spine:
        return 0.0, {}
    author_toks = {t for a in cand.authors for t in content_tokens(a)}
    # Words of the author's name inside a title ("Sven Hedin as artist") prove nothing about the title.
    title_toks = [t for t in content_tokens(cand.title) if t not in author_toks]
    if not title_toks:
        return 0.0, {}
    title_cov = _covered_share(title_toks, spine)
    author_hit = _author_match(cand.authors, spine_text, title_toks)
    explained_pool = title_toks + [t for a in cand.authors for t in content_tokens(a)] + content_tokens(cand.publisher)
    spine_cov = sum(_hit(t, explained_pool) for t in spine) / len(spine)
    s = 0.55 * title_cov + 0.25 * author_hit + 0.20 * spine_cov
    vs, vt = volume_numbers(spine_text), volume_numbers(cand.title)
    if vs and vt and not (vs & vt):
        s -= 0.3  # "Philosophen der Antike II" is not volume I
    if title_cov < 0.5:
        s = min(s, 0.5)
    unexplained = [t for t in spine if len(t) >= 5 and not _hit(t, explained_pool)]
    if spine_cov < 0.6 or unexplained:
        # A real word on the spine that the record does not explain ("im Steinhaus" under "Die Höhlenkinder")
        # can mean another volume or edition: keep the title, but not as a confident match.
        s = min(s, 0.84)
    if len(title_toks) == 1 and not author_hit:
        s = min(s, 0.6)  # a single common word is not enough to name a book
    return round(s, 3), {
        "title_cov": round(title_cov, 2),
        "author_hit": author_hit,
        "spine_cov": round(spine_cov, 2),
        "unexplained": unexplained[:5],
    }


class Resolver:
    def __init__(self, http: CachedHttp, google_key: str = "") -> None:
        self.http = http
        self.google_key = google_key
        self.deadline: float | None = None  # time.monotonic() after which no new catalogue query starts

    def query_text(self, spine_text: str) -> str:
        toks = content_tokens(spine_text)
        return " ".join(toks[:12])

    def query_variants(self, spine_text: str) -> list[str]:
        """Catalogue search ANDs terms, so one stray token (a publisher logo, a series number) returns
        nothing. Try the whole spine, then single printed lines and pairs of lines."""
        lines = [ln for ln in (s.strip() for s in spine_text.split("|")) if len(content_tokens(ln)) >= 1]
        variants = [self.query_text(spine_text)]
        if 3 <= len(lines) <= 7:
            # Leave one printed line out: drops a stray publisher or series line.
            variants += [self.query_text(" ".join(lines[:i] + lines[i + 1 :])) for i in range(len(lines))]
        rich = sorted(lines, key=lambda ln: -len(content_tokens(ln)))
        for i, a in enumerate(rich[:3]):
            for b in rich[i + 1 : 3]:
                variants.append(self.query_text(f"{a} {b}"))
        variants += [self.query_text(ln) for ln in rich[:2] if len(content_tokens(ln)) >= 2]
        cleaned = ocr_clean(spine_text)
        if cleaned != spine_text:
            variants.append(self.query_text(cleaned))
            clean_lines = [ln for ln in (s.strip() for s in cleaned.split("|")) if len(content_tokens(ln)) >= 2]
            variants += [self.query_text(ln) for ln in clean_lines[:3]]
        # Last resorts below keep slots of their own: many printed lines must not crowd them out.
        primary, variants = variants, []
        # A wrong last letter ("GARDENX"): Open Library accepts a word-start query ("garden*").
        long_words = [w for w in content_tokens(spine_text) if len(w) >= 6][:3]
        if long_words:
            variants.append(" ".join(f"{w[:-1]}*" for w in long_words))
        # OCR misreads one letter in a word ("nacb" for "nach") and every query above contains it:
        # last resort, pairs of the longest words.
        longest = sorted(set(content_tokens(spine_text)), key=len, reverse=True)[:4]
        variants += [f"{a} {b}" for i, a in enumerate(longest) for b in longest[i + 1 :]]
        seen: list[str] = []
        for v in primary:
            if len(v) >= 4 and v not in seen and len(seen) < MAX_VARIANTS - FALLBACK_SLOTS:
                seen.append(v)
        for v in variants:
            if len(v) >= 4 and v not in seen and len(seen) < MAX_VARIANTS:
                seen.append(v)
        return seen

    def candidates(self, spine_text: str, query: str | None = None, google: bool = True) -> list[Candidate]:
        q = query or self.query_text(spine_text)
        if len(q) < 4:
            return []
        out: list[Candidate] = []
        try:
            r = self.http.get(
                "https://openlibrary.org/search.json",
                {
                    "q": q,
                    "limit": 8,
                    "fields": "key,title,subtitle,author_name,first_publish_year,isbn,publisher,edition_count",
                },
                source="openlibrary",
            )
            for d in (r.data or {}).get("docs", [])[:8]:
                out.append(
                    Candidate(
                        title=d.get("title", ""),
                        authors=d.get("author_name", [])[:3],
                        publisher=", ".join(d.get("publisher", [])[:5]),
                        year=d.get("first_publish_year"),
                        source="Open Library",
                        url=f"https://openlibrary.org{d.get('key', '')}",
                        raw=d,
                    )
                )
        except (RuntimeError, LookupError):
            pass
        if not google:
            return out
        params: dict[str, Any] = {"q": q, "maxResults": 8, "printType": "books"}
        if self.google_key:
            params["key"] = self.google_key
        try:
            r = self.http.get("https://www.googleapis.com/books/v1/volumes", params, source="googlebooks")
            for it in (r.data or {}).get("items", [])[:8] if r.status < 400 else []:
                vi = it.get("volumeInfo", {})
                isbn = next(
                    (i["identifier"] for i in vi.get("industryIdentifiers", []) if i.get("type") == "ISBN_13"), ""
                )
                year = _year(vi.get("publishedDate", ""))
                out.append(
                    Candidate(
                        title=vi.get("title", "") + (f": {vi['subtitle']}" if vi.get("subtitle") else ""),
                        authors=vi.get("authors", [])[:3],
                        publisher=vi.get("publisher", ""),
                        year=year,
                        isbn=isbn,
                        source="Google Books",
                        url=vi.get("infoLink", ""),
                        volume_id=it.get("id", ""),
                        raw={"volumeInfo": vi},
                    )
                )
        except (RuntimeError, LookupError):
            pass
        return out

    def resolve(self, spine_text: str, accept: float) -> Resolution:
        cands: list[Candidate] = []
        queries: list[str] = []
        for k, q in enumerate(self.query_variants(spine_text)):
            if self.deadline is not None and time.monotonic() > self.deadline:
                break  # the packet must not wait: what is not resolved by now stays unidentified
            queries.append(q)
            # Google Books is rate-limited (100 per minute): it gets the first, broadest queries only.
            cands += self.candidates(spine_text, q, google=k < GOOGLE_QUERIES_PER_TEXT)
            if cands and max(score(spine_text, c)[0] for c in cands) >= accept:
                break
        # Ties (the same work in several languages) go to the title closest to the printed words,
        # articles included: "Der Orientalist" on the spine is the German edition, not "The Orientalist".
        # Then, the fuller title wins: "Collected Essays of Elia" over a record titled "Collected Essays".
        spine_toks = content_tokens(spine_text)
        scored = sorted(
            ((score(spine_text, c), c) for c in cands),
            key=lambda x: (
                -x[0][0],
                -_covered(content_tokens(x[1].title), spine_toks),
                -fuzz.token_set_ratio(x[1].title.lower(), spine_text.lower()),
            ),
        )
        if not scored or scored[0][0][0] <= 0:
            return Resolution("unidentified", 0.0, spine_text, evidence={"candidates": len(cands), "queries": queries})
        (best_score, parts), best = scored[0]
        # Ambiguity: a different work scoring almost the same lowers confidence.
        rivals = [s for (s, _), c in scored[1:] if fuzz.token_set_ratio(c.title, best.title) < 80]
        if rivals and rivals[0] > best_score - 0.05:
            best_score = max(0.0, round(best_score - 0.1, 3))
        res = Resolution(
            status="identified" if best_score >= accept else "unidentified",
            confidence=best_score,
            spine_text=spine_text,
            evidence={
                "score_parts": parts,
                "candidates": len(cands),
                "queries": queries,
                "runner_up": scored[1][1].title if len(scored) > 1 else "",
            },
        )
        if res.status == "identified":
            title = _fullest_title(best, [c for _, c in scored], spine_toks)
            surnames = [content_tokens(a)[-1] for a in best.authors if content_tokens(a)]
            # Some records carry the author in the title ("Go Set a Watchman by Harper Lee").
            m = re.search(r"\s+by\s+(.+)$", title, re.I)
            if m and any(s in content_tokens(m.group(1)) for s in surnames):
                title = title[: m.start()].strip()
            res.title, res.author = title, ", ".join(best.authors)
            res.source, res.url, res.volume_id = best.source, best.url, best.volume_id
            # Edition only when the spine names the publisher of a specific catalogued edition.
            ed = self._edition(spine_text, best, [c for _, c in scored])
            if ed is not None:
                res.publisher, res.isbn, res.year = ed.publisher, ed.isbn, ed.year
                res.edition = f"{ed.publisher} {ed.year or ''}".strip()
                res.volume_id = ed.volume_id or res.volume_id
                res.evidence["edition_from"] = "publisher printed on spine"
        return res

    def _edition(self, spine_text: str, work: Candidate, cands: list[Candidate]) -> Candidate | None:
        spine = content_tokens(spine_text)
        for c in cands:
            if not c.isbn or not c.publisher:
                continue
            if not same_work(c.title, work.title):
                continue
            pub = content_tokens(c.publisher)
            if pub and all(_hit(t, spine) for t in pub[:2]):
                return c
        return None

    def catalogue_height_cm(self, volume_id: str, title: str = "", author: str = "") -> float | None:
        """Printed height from Google Books: the matched volume, else the first edition of the same title and
        author that lists dimensions (Open Library matches carry no Google volume)."""
        if self.deadline is not None and time.monotonic() > self.deadline:
            return None
        params: dict[str, Any] = {"key": self.google_key} if self.google_key else {}
        if volume_id:
            h = self._height(f"https://www.googleapis.com/books/v1/volumes/{volume_id}", params)
            if h:
                return h
        if not title:
            return None
        surname = content_tokens(author.split(",")[0])[-1:] if author else []
        q = f"intitle:{' '.join(content_tokens(title.split(':')[0])[:5])}"
        if surname:
            q += f" inauthor:{surname[0]}"
        try:
            r = self.http.get(
                "https://www.googleapis.com/books/v1/volumes",
                {**params, "q": q, "maxResults": 10, "printType": "books"},
                source="googlebooks",
            )
        except (RuntimeError, LookupError):
            return None
        for it in (r.data or {}).get("items", []) if r.status < 400 else []:
            vid = it.get("id", "")
            if not same_work(it.get("volumeInfo", {}).get("title", ""), title) or not vid:
                continue
            h = self._height(f"https://www.googleapis.com/books/v1/volumes/{vid}", params)
            if h:
                return h
        return None

    def _height(self, url: str, params: dict[str, Any]) -> float | None:
        try:
            r = self.http.get(url, params, source="googlebooks")
        except (RuntimeError, LookupError):
            return None
        dims = (r.data or {}).get("volumeInfo", {}).get("dimensions", {})
        m = re.match(r"([\d.]+)\s*cm", str(dims.get("height", "")))
        return float(m.group(1)) if m else None


def _fullest_title(best: Candidate, cands: list[Candidate], spine: list[str]) -> str:
    """Catalogue records of one work differ in how much of the title they carry ("Collected Essays" vs
    "Collected Essays of Elia"). Show the longest title that contains the best one and whose extra words
    are all printed on the spine."""
    base = content_tokens(best.title)
    title = best.title
    for c in cands:
        toks = content_tokens(c.title)
        extra = [t for t in toks if t not in base]
        if (
            len(toks) > len(content_tokens(title))
            and all(t in toks for t in base)
            and extra
            and all(_hit(t, spine) for t in extra)
        ):
            title = c.title
    return title


def _year(date: str) -> int | None:
    m = re.match(r"(\d{4})", date or "")
    return int(m.group(1)) if m else None
