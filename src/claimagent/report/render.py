"""Readable HTML report, built only from claim_packet.json (no new numbers are created here)."""

from __future__ import annotations

import json
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from claimagent.contract import ClaimPacket

_env = Environment(loader=PackageLoader("claimagent", "report"), autoescape=select_autoescape(["html", "j2"]))


def money(v: float | None, cur: str) -> str:
    return "—" if v is None else f"{v:,.0f} {cur}"


def num(v: float | None, unit: str = "", nd: int = 1) -> str:
    return "—" if v is None else f"{v:,.{nd}f}{unit}"


_env.filters["money"] = money
_env.filters["num"] = num


def plain_json(v: Any) -> str:
    """Metrics may hold numpy numbers; written as plain JSON numbers."""
    return json.dumps(v, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))


_env.filters["plain_json"] = plain_json


def render_report(packet: ClaimPacket) -> str:
    review_ids = {r.ref_id for r in packet.review_queue}
    return _env.get_template("report.html.j2").render(p=packet, review_ids=review_ids)
