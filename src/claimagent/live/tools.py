"""Tools the voice agent can call. Each one only records what the policyholder said, with a timestamp,
as evidence for the pipeline. None of them produces a number, title or price."""

from __future__ import annotations

import re
from typing import Any

from claimagent.capture.store import SweepStore
from claimagent.pipeline.price.sources import COUNTRY_CURRENCY

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "set_locale",
            "description": "Record the country and currency once the policyholder confirms or corrects them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country_code": {"type": "string", "description": "ISO 3166-1 alpha-2, e.g. IN, US, GB"},
                    "currency": {"type": "string", "description": "ISO 4217, e.g. INR, USD, GBP"},
                },
                "required": ["country_code", "currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skip_region",
            "description": "The policyholder says the shelf or area in view is not theirs or must be skipped.",
            "parameters": {
                "type": "object",
                "properties": {"statement": {"type": "string", "description": "the policyholder's words"}},
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "note_book",
            "description": "The policyholder says something about the book in view: first edition, signed, rare.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "first_edition": {"type": "boolean"},
                    "signed": {"type": "boolean"},
                },
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "art_answer",
            "description": "The policyholder says whether the picture or portrait in view is an original or a print.",
            "parameters": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "is_print": {"type": "boolean"}},
                "required": ["statement", "is_print"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shelf_width",
            "description": "The policyholder explicitly states how wide the bookshelf in view is (for example "
            "'this shelf is 90 centimetres wide'). Never for the size of the room, a wall or anything else.",
            "parameters": {
                "type": "object",
                "properties": {"statement": {"type": "string"}, "width_cm": {"type": "number"}},
                "required": ["statement", "width_cm"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_sweep",
            "description": "The policyholder says the whole sweep is over and they want the claim packet now.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def run_tool(store: SweepStore, name: str, args: dict[str, Any], t_ms: int) -> dict[str, Any]:
    if name == "set_locale":
        cc = str(args.get("country_code", "")).upper()[:2]
        cur = str(args.get("currency", "")).upper()[:3] or COUNTRY_CURRENCY.get(cc, "")
        if not cc:
            return {"ok": False, "error": "country missing"}
        store.update_session(country=cc, currency=cur)
        store.log_event("locale", t_ms, country=cc, currency=cur)
        return {"ok": True, "country": cc, "currency": cur}
    if name == "skip_region":
        store.log_event("skip_region", t_ms, statement=str(args.get("statement", "")))
        return {"ok": True, "result": "the shelf in view will be excluded"}
    if name == "note_book":
        store.log_event(
            "book_note",
            t_ms,
            statement=str(args.get("statement", "")),
            first_edition=bool(args.get("first_edition")),
            signed=bool(args.get("signed")),
        )
        return {"ok": True, "result": "noted; that book goes to human appraisal"}
    if name == "art_answer":
        is_print = bool(args.get("is_print"))
        store.log_event("art_answer", t_ms, statement=str(args.get("statement", "")), is_print=is_print)
        return {
            "ok": True,
            "result": "print: will be priced as a print" if is_print else "original: goes to human appraisal",
        }
    if name == "shelf_width":
        try:
            w = float(args.get("width_cm", 0))
        except (TypeError, ValueError):
            w = 0.0
        if not 20 <= w <= 400:
            return {"ok": False, "error": "width out of range"}
        store.log_event("shelf_dimension", t_ms, statement=str(args.get("statement", "")), width_cm=w)
        return {"ok": True, "result": f"shelf width {w:.0f} cm recorded"}
    if name == "end_sweep":
        return {"ok": True, "result": "ending the sweep"}
    return {"ok": False, "error": f"unknown tool {name}"}


# Facts about the claim: recorded only while sweeping (after it, the packet is already being built from them).
FACT_TOOLS = {"skip_region", "note_book", "art_answer", "shelf_width"}
_SKIP = re.compile(r"\b(skip|not mine|isn'?t mine|aren'?t mine|don'?t count|exclude|not ours|belongs? to)\b")
_NOTE = re.compile(r"\b(first edition|1st edition|signed|autographed)\b")
_ART = re.compile(r"\b(print|original|reproduction|poster)\b")
_ORIGINAL = re.compile(r"\boriginal\b|\bnot (a |just a )?print\b")
SAID = {
    "skip_region": "Okay, I'll leave that shelf out.",
    "note_book": "Noted. That book goes to an expert for appraisal.",
    "art_print": "Got it, a print.",
    "art_original": "Noted, an original. It goes to an expert for appraisal.",
}


def rule_tool(text: str) -> tuple[str, dict[str, Any], str] | None:
    """Clear statements are recorded by rule, not left to the language model, which mixes these tools up
    ("that painting is just a print" recorded as a skipped shelf). Returns (tool, arguments, reply)."""
    t = text.lower()
    if _NOTE.search(t):
        args = {"statement": text, "first_edition": "edition" in t, "signed": bool(re.search(r"signed|autograph", t))}
        return "note_book", args, SAID["note_book"]
    if _SKIP.search(t):
        return "skip_region", {"statement": text}, SAID["skip_region"]
    if _ART.search(t):
        is_print = not _ORIGINAL.search(t)
        return (
            "art_answer",
            {"statement": text, "is_print": is_print},
            SAID["art_print" if is_print else "art_original"],
        )
    return None


# The language model may only record the country and currency (before the walk) and end the sweep; facts about
# the contents are recorded by rule (rule_tool), because the model recorded things nobody said.
LLM_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("set_locale", "end_sweep")]
