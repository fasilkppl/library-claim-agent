"""Replay end-to-end: build the test sweep (real shelf photos, simulated pan, reference marker), run every
pipeline stage with the real local models, offline against recorded catalogue responses, then score the
packet against hand-made ground truth with the same evaluator used for real sweeps.

Run: uv run pytest -m slow tests/test_replay_pipeline.py -s
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from claimagent.config import get_settings
from claimagent.contract import ClaimPacket
from claimagent.eval.evaluate import evaluate
from claimagent.pipeline.runner import run_pipeline
from make_sweep import build, build_xr

FIX = Path(__file__).parent / "fixtures"


def _bars(res: dict[str, object]) -> dict[str, dict[str, object]]:
    return {b["measure"]: b for b in res["bars"]}  # type: ignore[index, union-attr]


@pytest.mark.slow
def test_camera_sweep_replay(tmp_path: Path) -> None:
    store = build(tmp_path, "replay-pan")
    for f in (FIX / "http_cache_testpan").glob("*.json"):
        shutil.copy(f, store.cache_dir / f.name)
    packet = run_pipeline(store, get_settings(), offline=True)

    # Contract rules from the brief.
    data = json.loads((store.dir / "claim_packet.json").read_text())
    ClaimPacket.model_validate(data)
    assert data["totals"]["book_count"] == len(data["books"])
    for b in packet.books:
        assert (store.dir / b.frame_ref).exists(), "every frame_ref must open"
        for price in (b.replacement_cost, b.used_value):
            if price.amount is not None:
                assert price.url and price.retrieved_at, "a price without a source is a fabricated price"
        if b.status == "unidentified":
            assert not b.title, "no title may be guessed for an unidentified spine"
    replacement = sum(b.replacement_cost.amount or 0 for b in packet.books)
    assert abs(replacement - packet.totals.books_replacement_cost) < 0.01, "totals must add up from the lines"
    assert (store.dir / "report.html").read_text().count("<tr") > len(packet.books)

    res = evaluate(store, FIX / "truth_testpan")
    bars = _bars(res)
    print(json.dumps([{k: b[k] for k in ("measure", "result", "pass")} for b in res["bars"]], indent=1))
    assert bars["Book count"]["pass"] is True
    assert bars["Spine dimensions"]["pass"] is True
    assert bars["Time to packet"]["pass"] is True
    wrong = bars["Title identification"]["detail"]["confidently_wrong"]  # type: ignore[index]
    assert len(wrong) <= 1, wrong


@pytest.mark.slow
def test_webxr_sweep_replay(tmp_path: Path) -> None:
    store = build_xr(tmp_path, "replay-xr")
    for f in (FIX / "http_cache_testpan").glob("*.json"):
        shutil.copy(f, store.cache_dir / f.name)
    packet = run_pipeline(store, get_settings(), offline=True)
    room = packet.room
    assert room.floor_area_m2 is not None and abs(room.floor_area_m2 / (4.2 * 3.5) - 1) < 0.10
    assert room.wall_area_m2 is not None and abs(room.wall_area_m2 / (2 * (4.2 + 3.5) * 2.6) - 1) < 0.15
    assert room.height_m is not None and abs(room.height_m - 2.6) < 0.1
    assert room.floor_area_ft2 is not None and abs(room.floor_area_ft2 - room.floor_area_m2 * 10.7639) < 0.2
    depth_measured = [b for b in packet.books if b.scale_source == "webxr-depth"]
    assert len(depth_measured) >= 0.8 * len(packet.books)
