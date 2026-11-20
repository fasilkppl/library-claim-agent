import cv2
import numpy as np

from claimagent.capture.quality import measure, thumb, view_shift
from claimagent.pipeline.analyze import assign_text
from claimagent.pipeline.spines import Spine, _merge_collinear
from claimagent.pipeline.vision import Box, TextLine, clean_spines


def test_clean_spines_drops_box_spanning_neighbours_and_merges_fragments():
    a, b = Box(0, 0, 40, 400, 0.9, "book"), Box(42, 0, 80, 400, 0.9, "book")
    span = Box(0, 0, 80, 400, 0.5, "book")
    top, bottom = Box(100, 0, 140, 210, 0.8, "book"), Box(101, 190, 141, 400, 0.7, "book")
    out = clean_spines([a, b, span, top, bottom])
    assert len(out) == 3


def _sp(x0, y0, x1, y1):
    corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return Spine(
        cx=(x0 + x1) / 2,
        cy=(y0 + y1) / 2,
        length=y1 - y0,
        width=x1 - x0,
        angle=0,
        orientation="upright",
        conf=0.9,
        source="sam+yolo",
        corners=corners,
    )


def test_collinear_fragments_join_but_neighbours_do_not():
    out = _merge_collinear([_sp(100, 0, 140, 200), _sp(101, 210, 141, 400), _sp(145, 0, 185, 400)], typical=400)
    assert len(out) == 2 and max(s.length for s in out) > 390


def test_cross_spine_line_split_by_words():
    spines = [
        {"corners": [[0, 0], [100, 0], [100, 400], [0, 400]], "orientation": "upright"},
        {"corners": [[100, 0], [200, 0], [200, 400], [100, 400]], "orientation": "upright"},
    ]
    words = [
        ("Sven", Box(10, 50, 50, 70)),
        ("Hedin,", Box(52, 50, 95, 70)),
        ("Sven", Box(110, 50, 150, 70)),
        ("Hedin,", Box(152, 50, 195, 70)),
    ]
    out = assign_text(spines, [TextLine("Sven Hedin, Sven Hedin,", 1.0, Box(10, 50, 195, 70), False, words)])
    assert [t["text"] for t in out[0]] == ["Sven Hedin,"] and [t["text"] for t in out[1]] == ["Sven Hedin,"]


def test_blur_and_glare_measures():
    rng = np.random.default_rng(0)
    sharp = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
    assert measure(cv2.GaussianBlur(sharp, (31, 31), 10)).sharpness < measure(sharp).sharpness / 10
    white = np.full((480, 640, 3), 255, np.uint8)
    assert measure(white).glare == 1.0


def test_view_shift_phase_correlation():
    rng = np.random.default_rng(1)
    big = cv2.GaussianBlur((rng.random((480, 900)) * 255).astype(np.uint8), (5, 5), 1)
    a, b = thumb(big[:, :640]), thumb(big[:, 64:704])
    assert 0.08 < view_shift(a, b) < 0.12
