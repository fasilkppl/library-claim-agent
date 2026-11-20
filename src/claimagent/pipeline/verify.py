"""Independent check of each identified book's authors against a second catalogue.

A book is identified from one catalogue record (Google Books or Open Library). This step looks the same book up
in the other catalogue (by ISBN where the record has one, else by title) and compares the authors, using the
names printed on the spine as the tie-breaker. Nothing comes from a language model: every author name is from a
catalogue record or the spine.

Outcomes: the second catalogue agrees (recorded as evidence); it has no record (noted); or the authors differ.
A difference is corrected only when the spine supports the second catalogue and not the first; otherwise the
book goes to the review queue with both versions, so an adjuster decides. Authors are also listed in the order
the spine prints them.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

from claimagent.pipeline.dedupe import same_word, tokens
from claimagent.pipeline.resolve import Candidate, Resolver, content_tokens, derivative, same_work

log = logging.getLogger(__name__)

WORKERS = 6


def surnames(authors: list[str]) -> list[str]:
    return [t[-1] for t in (content_tokens(a) for a in authors) if t]


def _split(author_field: str) -> list[str]:
    return [a.strip() for a in author_field.replace("&", ",").replace(" and ", ",").split(",") if a.strip()]


def authors_agree(ours: list[str], theirs: list[str]) -> bool:
    """At least one name word in common. Every word counts, not only the last: catalogues write "Khaled
    Hosseini" and "Hosseini Khaled"; spelling may vary slightly."""
    a_words = [w for a in ours for w in content_tokens(a) if len(w) >= 3 and not w.isdigit()]
    b_words = [w for b in theirs for w in content_tokens(b) if len(w) >= 3 and not w.isdigit()]
    return any(same_word(x, y) for x in a_words for y in b_words)


def on_spine(authors: list[str], spine_text: str) -> int:
    """How many of these authors' surnames are printed on the spine."""
    spine = tokens(spine_text)
    return sum(1 for s in surnames(authors) if any(same_word(s, w) for w in spine))


def spine_order(authors: list[str], spine_text: str) -> list[str]:
    """Authors in the order the spine prints them; unchanged when the spine shows fewer than two of them."""
    spine = tokens(spine_text)

    def first_at(author: str) -> int:
        name = content_tokens(author)
        if not name:
            return 10**6
        hits = [i for i, w in enumerate(spine) if same_word(name[-1], w)]
        return hits[0] if hits else 10**6

    if sum(1 for a in authors if first_at(a) < 10**6) < 2:
        return authors
    return sorted(authors, key=first_at)


def decide(ours: list[str], theirs: list[str] | None, spine_text: str, other: str, source: str) -> dict[str, Any]:
    """The check's outcome for one book: {"status", "note", "authors" (to use), "review" (reason or "")}."""
    if theirs is None:
        return {"status": "unconfirmed", "note": f"{other} has no record of it", "authors": ours, "review": ""}
    if not theirs or authors_agree(ours, theirs):
        return {"status": "confirmed", "note": f"authors confirmed by {other}", "authors": ours, "review": ""}
    ours_seen, theirs_seen = on_spine(ours, spine_text), on_spine(theirs, spine_text)
    if theirs_seen > ours_seen:
        return {
            "status": "corrected",
            "note": f"authors taken from {other}: the spine prints them, not those from {source}",
            "authors": theirs,
            "review": f"authors corrected from {', '.join(ours)} ({source}) to {', '.join(theirs)} ({other}, "
            "matches the spine); please confirm",
        }
    return {
        "status": "disagree",
        "note": f"{source} and {other} name different authors",
        "authors": ours,
        "review": f"authors differ: {source} says {', '.join(ours)}, {other} says {', '.join(theirs)}; check the spine",
    }


class Checker:
    def __init__(self, resolver: Resolver) -> None:
        self.r = resolver

    def other_record(self, res: dict[str, Any]) -> tuple[str, list[str] | None]:
        """(the other catalogue's name, its authors for this book, or None when it has no matching record).
        Records of books about the book (summaries, trivia) are skipped; among the records of the same work, one
        whose authors agree with ours is preferred, since catalogues hold many editions of one work."""
        title = str(res.get("title", ""))
        main = title.split(":")[0]
        ours = _split(str(res.get("author", "")))
        if res.get("source") == "Open Library":
            name, found = "Google Books", self._google(main, title)
        else:
            name, found = "Open Library", self._openlibrary(main, title, str(res.get("isbn", "")))
        found = [a for a in found if a and not derivative(Candidate(title=title, authors=a))]
        if not found:
            return name, None
        return name, next((a for a in found if authors_agree(ours, a)), found[0])

    def _openlibrary(self, main: str, title: str, isbn: str) -> list[list[str]]:
        params: dict[str, Any] = {"fields": "title,author_name", "limit": 8}
        params.update({"isbn": isbn} if isbn else {"title": main})
        try:
            r = self.r.http.get("https://openlibrary.org/search.json", params, source="openlibrary")
        except (RuntimeError, LookupError):
            return []
        docs = (r.data or {}).get("docs", []) if r.status < 400 else []
        return [
            [str(a) for a in d.get("author_name", [])][:4]
            for d in docs
            if isbn or same_work(str(d.get("title", "")), title)
        ]

    def _google(self, main: str, title: str) -> list[list[str]]:
        params: dict[str, Any] = {"q": f"intitle:{main}", "maxResults": 10, "printType": "books"}
        if self.r.google_key:
            params["key"] = self.r.google_key
        try:
            r = self.r.http.get("https://www.googleapis.com/books/v1/volumes", params, source="googlebooks")
        except (RuntimeError, LookupError):
            return []
        items = (r.data or {}).get("items", []) if r.status < 400 else []
        return [
            [str(a) for a in it.get("volumeInfo", {}).get("authors", [])][:4]
            for it in items
            if same_work(str(it.get("volumeInfo", {}).get("title", "")), title)
        ]


def verify_books(
    read: dict[str, Any], resolved: dict[str, Any], resolver: Resolver, budget_s: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (resolved with checked authors, report). Books not checked within the budget are noted."""
    spine_of = {
        int(b["track"]): " | ".join([b.get("spine_text", ""), *b.get("text_variants", [])[:2]]) for b in read["books"]
    }
    checker = Checker(resolver)
    todo = [r for r in resolved["books"] if r.get("status") == "identified" and r.get("title")]
    results: dict[int, dict[str, Any]] = {}

    def check(res: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        other, theirs = checker.other_record(res)
        ours = _split(str(res.get("author", "")))
        return int(res["track"]), decide(ours, theirs, spine_of.get(int(res["track"]), ""), other, str(res["source"]))

    t0 = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=WORKERS)
    futures = [pool.submit(check, r) for r in todo]
    finished, _ = wait(futures, timeout=budget_s)
    pool.shutdown(wait=False, cancel_futures=True)
    for f in finished:
        if f.exception() is None:
            track, outcome = f.result()
            results[track] = outcome
    out_books = []
    for res in resolved["books"]:
        r = dict(res)
        track = int(r["track"])
        identified = r.get("status") == "identified" and bool(r.get("title"))
        if track in results:
            o = results[track]
            # The catalogue's order stays for searches (its first author is the main one); the spine's order is shown.
            r["author_catalogue"] = ", ".join(o["authors"])
            r["author"] = ", ".join(spine_order(o["authors"], spine_of.get(track, "")))
            r["check"] = {"status": o["status"], "note": o["note"]}
            if o["review"]:
                r["check"]["review"] = o["review"]
            log.info("VERIFY %r: %s (%s)", r.get("title"), o["status"], o["note"])
        elif identified:
            r["author_catalogue"] = str(r.get("author", ""))
            r["author"] = ", ".join(spine_order(_split(str(r.get("author", ""))), spine_of.get(track, "")))
            r["check"] = {"status": "not checked", "note": "second catalogue did not answer in time"}
        out_books.append(r)
    counts: dict[str, int] = {}
    for o in results.values():
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    report = {"checked": len(results), "of": len(todo), "outcomes": counts, "seconds": round(time.monotonic() - t0, 1)}
    log.info("VERIFY %s of %s identified books checked: %s", len(results), len(todo), counts)
    return {**resolved, "books": out_books}, report
