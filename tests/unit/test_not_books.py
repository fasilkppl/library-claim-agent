from types import SimpleNamespace

import numpy as np

from claimagent.pipeline.register import Observation, Track
from claimagent.pipeline.stages.read import _not_books


def _shelf(tall_px: float) -> tuple[SimpleNamespace, dict[int, dict]]:
    """Nine standing books seen in five frames, one of them taller and covered by a "cardboard box" detection in
    three frames (the detector's usual name for a book spine), plus three one-glance slivers at the frame edge."""
    det: dict[int, dict] = {seq: {"spines": [], "items": []} for seq in range(1, 6)}
    tracks: list[Track] = []

    def add(tid: int, seqs: list[int], x: float, length: float, orientation: str, edge: bool, text: bool) -> None:
        t = Track(id=tid, segment=0)
        for seq in seqs:
            bbox = [x, 100.0, x + 150.0, 100.0 + length] if orientation == "upright" else [x, 100.0, x + length, 250.0]
            spine = {
                "orientation": orientation,
                "length": length,
                "bbox": bbox,
                "touches_edge": edge,
                "text": [{"text": "TITLE"}] if text else [],
            }
            det[seq]["spines"].append(spine)
            corners = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]])
            t.observations.append(Observation(seq=seq, index=len(det[seq]["spines"]) - 1, corners=corners, spine=spine))
        tracks.append(t)

    # The spine lengths (px) of that sweep's tracks.
    for k, length in enumerate([1338, 1270, 1358, 1034, 967, 1384, 1060, 1200, 908]):
        add(k, [1, 2, 3, 4, 5], 200.0 * k, float(length), "upright", False, True)
    add(9, [1, 2, 3, 4, 5], 1900.0, tall_px, "upright", False, True)
    for seq in (1, 2, 3):
        tall = det[seq]["spines"][-1]["bbox"]
        det[seq]["items"].append({"label": "cardboard box", "bbox": list(tall), "conf": 0.3})
    add(10, [1, 2, 3], 2200.0, 435.0, "flat", False, False)
    add(11, [2], 2800.0, 390.0, "flat", True, False)
    add(12, [3], 3400.0, 466.0, "upright", True, False)
    links = {seq: SimpleNamespace(segment=0) for seq in range(1, 6)}
    return SimpleNamespace(tracks=tracks, links=links, segments=1), det


def test_the_tallest_book_is_not_taken_for_a_box_because_of_edge_slivers():
    reg, det = _shelf(tall_px=1486.0)  # sweep 20261121-171735: Wings of Fire, rejected as a cardboard box
    assert 9 not in _not_books(reg, det)


def test_a_box_far_out_of_place_in_the_row_is_still_not_a_book():
    reg, det = _shelf(tall_px=2400.0)
    assert "cardboard box" in _not_books(reg, det).get(9, "")
