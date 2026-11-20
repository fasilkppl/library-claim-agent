"""claim_packet.json schema.

Every field the brief requires is present. Numbers the system does not know are None, never a guess.
Fields beyond the brief carry evidence (spine text, scale source, exclusions, metrics).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Sweep(_Model):
    id: str
    captured_at: str
    device: str = ""
    duration_s: float = 0.0
    country: str
    currency: str
    capture_mode: Literal["webxr", "camera"] = "camera"


class Room(_Model):
    length_m: float | None = None
    width_m: float | None = None
    height_m: float | None = None
    floor_area_m2: float | None = None
    floor_area_ft2: float | None = None
    wall_area_m2: float | None = None
    wall_area_ft2: float | None = None
    shelved_wall_area_m2: float | None = None
    shelved_wall_area_ft2: float | None = None
    scale_method: str = ""
    shape_method: str = ""
    confidence: float = 0.0
    frame_refs: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class BookPrice(_Model):
    amount: float | None = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    converted_from: str = ""
    fx_rate: float | None = None
    fx_date: str = ""
    condition_assumed: str = ""
    offers_used: int = 0
    note: str = ""


BookStatus = Literal["identified", "unidentified", "needs_appraisal"]


class Book(_Model):
    id: str
    shelf: str
    position: int
    frame_ref: str
    status: BookStatus
    title: str = ""
    author: str = ""
    publisher: str = ""
    edition: str = ""
    isbn: str = ""
    year: int | None = None
    spine_height_cm: float | None = None
    spine_thickness_cm: float | None = None
    orientation: Literal["upright", "flat"] = "upright"
    id_confidence: float = 0.0
    spine_text: str = ""
    match_source: str = ""
    match_url: str = ""
    scale_source: str = ""
    bbox: list[float] = Field(default_factory=list)
    observations: int = 1
    replacement_cost: BookPrice = Field(default_factory=BookPrice)
    used_value: BookPrice = Field(default_factory=BookPrice)


ItemStatus = Literal["priced", "range", "needs_appraisal", "unpriced"]


class Dimensions(_Model):
    w: float | None = None
    h: float | None = None
    d: float | None = None


class ItemPrice(_Model):
    low: float | None = None
    high: float | None = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    offers_used: int = 0
    query: str = ""
    note: str = ""


class Item(_Model):
    id: str
    category: str
    description: str = ""
    material: str = ""
    brand_model: str = ""
    frame_ref: str
    bbox: list[float] = Field(default_factory=list)
    dimensions_cm: Dimensions = Field(default_factory=Dimensions)
    status: ItemStatus
    replacement_cost: ItemPrice = Field(default_factory=ItemPrice)
    confidence: float = 0.0
    user_statement: str = ""
    observations: int = 1


class Totals(_Model):
    book_count: int = 0
    books_identified: int = 0
    books_unidentified: int = 0
    books_needs_appraisal: int = 0
    shelf_run_m: float | None = None
    books_replacement_cost: float = 0.0
    books_used_value: float = 0.0
    items_replacement_cost_low: float = 0.0
    items_replacement_cost_high: float = 0.0
    excluded_from_totals: int = 0
    currency: str = ""


class ReviewEntry(_Model):
    ref_id: str
    reason: str


class Exclusion(_Model):
    t_s: float
    statement: str
    frames: list[str] = Field(default_factory=list)


class ClaimPacket(_Model):
    sweep: Sweep
    room: Room
    books: list[Book]
    items: list[Item]
    totals: Totals
    review_queue: list[ReviewEntry]
    exclusions: list[Exclusion] = Field(default_factory=list)
    metrics: dict[str, object] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
