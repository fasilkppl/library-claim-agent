"""The sweep, as the brief describes it: one continuous pass, with one voice directing it.

The policyholder walks the room once: along every shelf, then across the walls and the floor. The agent directs
the pass (which part to cover next, slower, closer) but never asks for items to be photographed one by one, and
there is no separate measuring pass: items and the room come from the same walk.

Two optional markers help the packet and change nothing for the policyholder if unused: "next shelf" numbers
the shelves, and "done with books" marks where the shelf part of the walk ends. "I'm done" ends the sweep.

Every frame is tagged with the part of the walk it was taken in (and the shelf number). The pipeline uses the
tags: books come only from the shelf part, and books on different shelves are different books, so a glance back
at a shelf while panning the room does not count its books twice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

STEP_BOOKS, STEP_ROOM, STEP_DONE = "books", "room", "done"

# Commands are short and explicit. Anything about skipping, ownership or a book's condition is a statement for
# the dialog (it records an exclusion or a note), even when it mentions a shelf.
STATEMENT = re.compile(r"\b(skip|not mine|isn'?t mine|aren'?t mine|exclude|don'?t count|first edition|signed|print)\b")
NEXT_SHELF = re.compile(r"^(ok |okay |now )?(next|another|new)( one| shelf)?( please)?$|\b(next|another|new) shelf\b")
# Speech recognition mishears "done" while walking ("I'm run with books"): near-words count too.
_DONE = r"(done|dun|run|gone|finished|finish)"
BOOKS_DONE = re.compile(
    rf"\b{_DONE}\b.*\b(books?|shelf|shelves)\b"
    r"|\bbooks? (are |is )?(done|finished|covered)\b|\bno more (books|shelves)\b"
)
MOVE_ON = re.compile(
    r"\b(move on|moving on|next (item|items|part|step|thing|area|section)|the room now|go to the room)\b"
)
# The end of the walk: "I'm done", "I guess I'm done", "I'm done already", "yep, that's it", "finish the sweep".
SWEEP_DONE = re.compile(
    r"\b(i'?m|i am|we'?re|we are|all|i guess i'?m)\s+(done|finished)\b|\bthat'?s (it|all|everything)\b"
    r"|^(done|finished|all done)\b|\b(end|stop|finish) (the )?(sweep|scan|recording)\b"
    r"|\bdone with (the )?(room|sweep|everything)\b"
)
# A question about being done ("are we done?") is answered, not obeyed.
ASKS = re.compile(r"^(are|am|is|was|can|could|should|do|did|have|shall)\b")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z' ]", " ", text.lower())).strip()


@dataclass
class GuideState:
    step: str = STEP_BOOKS
    shelf: int = 1


class Guide:
    def __init__(self, ar_room: bool = False) -> None:
        self.st = GuideState()
        self.ar_room = ar_room  # camera sweep on a phone with AR: the room step switches to AR for metric depth

    # ------------------------------------------------------------------ what the phone and the pipeline see
    def tags(self) -> dict[str, Any]:
        t: dict[str, Any] = {"step": self.st.step}
        if self.st.step == STEP_BOOKS:
            t["shelf"] = self.st.shelf
        return t

    def ui(self) -> dict[str, Any]:
        s = self.st
        if s.step == STEP_BOOKS:
            return {
                "type": "guide",
                "step": s.step,
                "title": f"Shelves · shelf {s.shelf}",
                "text": "Move slowly along the shelf, left to right, about half a metre away.",
                "actions": [{"id": "next_shelf", "label": "Next shelf"}, {"id": "books_done", "label": "Books done"}],
            }
        if s.step == STEP_ROOM:
            return {
                "type": "guide",
                "step": s.step,
                "title": "Room",
                "text": (
                    "Tap Measure room (AR), move the phone slowly until AR is ready, then pan across every wall, "
                    "the floor and the ceiling. Say I'm done at the end."
                    if self.ar_room
                    else "Pan slowly across the walls, the floor and everything in the room. Say I'm done at the end."
                ),
                "actions": [{"id": "ar_room", "label": "Measure room (AR)"}] if self.ar_room else [],
            }
        return {"type": "guide", "step": s.step, "title": "Done", "text": "Building the claim packet.", "actions": []}

    def opening(self) -> str:
        return (
            "Started. Begin with the books: move slowly along each shelf from left to right, then pan across the "
            "walls and the floor. Say I'm done when you've covered the room."
        )

    def reminder(self) -> str:
        """The current direction, short: said when the policyholder asks what to do."""
        return {
            STEP_BOOKS: "Keep moving slowly along the shelves. Say done with books when you've covered them all.",
            STEP_ROOM: "Pan slowly across the walls and the floor, then say I'm done.",
        }.get(self.st.step, "")

    # ------------------------------------------------------------------ commands (voice or button)
    def command(self, text: str) -> str | None:
        """The action a spoken sentence asks for, or None when it is not a guide command (the dialog answers)."""
        t = _norm(text)
        if not t or STATEMENT.search(t):
            return None
        words = len(t.split())
        if words > 20:
            return None
        if self.st.step == STEP_BOOKS and (BOOKS_DONE.search(t) or MOVE_ON.search(t)):
            return "books_done"  # checked first: "done with the books" is not the end of the sweep
        if words <= 12 and SWEEP_DONE.search(t) and not ASKS.search(t):
            return "finish"
        if self.st.step == STEP_BOOKS and words <= 8 and NEXT_SHELF.search(t):
            return "next_shelf"
        return None

    def act(self, action: str) -> tuple[str, bool]:
        """Applies an action; returns (what to say, whether the sweep is now over)."""
        s = self.st
        log.info("GUIDE action %s in step %s (shelf %s)", action, s.step, s.shelf)
        if action == "finish":
            s.step = STEP_DONE
            return "", True
        if action == "next_shelf" and s.step == STEP_BOOKS:
            s.shelf += 1
            return f"Shelf {s.shelf}. Same again, slowly from left to right.", False
        if action == "books_done" and s.step == STEP_BOOKS:
            s.step = STEP_ROOM
            if self.ar_room:
                return (
                    "Got the shelves. Now tap Measure room on the screen, move the phone slowly until AR is ready, "
                    "then pan across every wall, the floor and the ceiling, and say I'm done.",
                    False,
                )
            return (
                "Got the shelves. Now pan slowly across the walls, the floor and everything in the room, "
                "then say I'm done.",
                False,
            )
        return "", False
