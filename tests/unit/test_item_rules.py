import numpy as np

from claimagent.pipeline import vlm


def _item(iid, label, category, seq, bbox, boxes=None):
    return {"id": iid, "label": label, "category": category, "frame_seq": seq, "bbox": bbox, "boxes": boxes or {}}


def test_a_print_inside_a_gift_box_is_a_part_of_it_but_a_padlock_on_a_table_is_not():
    box = _item("B", "gift box", "household", 7, [0, 0, 400, 300], {"5": [0, 0, 400, 300]})
    print_on_box = _item("A", "kite", "other", 5, [100, 100, 150, 150])
    table = _item("T", "desk", "table", 9, [0, 0, 800, 600], {"5": [0, 0, 800, 600]})
    assert vlm.parts_of_other_items([box, print_on_box]) == {"A": "gift box"}
    assert vlm.parts_of_other_items([table, _item("P", "padlock", "household", 5, [100, 100, 150, 150])]) == {}


def test_spine_cover():
    assert vlm.spine_cover([0, 0, 100, 100], [[0, 0, 50, 100], [50, 0, 100, 100]]) == 1.0
    assert vlm.spine_cover([0, 0, 100, 100], []) == 0.0


def test_book_answers_need_spines_and_tables_under_books_stay(monkeypatch):
    rng = np.random.default_rng(0)
    frames = {s: rng.integers(0, 255, (400, 400, 3), dtype=np.uint8) for s in (1, 2, 3)}  # detailed pictures
    items = [
        _item("D", "desk", "table", 1, [10, 10, 300, 300]),
        _item("C", "cardboard box", "household", 2, [10, 10, 300, 300]),
        _item("W", "bottle", "kitchen", 3, [10, 10, 300, 300]),
    ]
    monkeypatch.setattr(vlm, "_ask", lambda crops, timeout: ["books", "books", "book"])
    spines = {1: [], 2: [[0, 0, 160, 320], [160, 0, 320, 320]], 3: []}
    removed = vlm.name_items(items, frames, spines=spines)
    kept = {it["id"]: it for it in items}
    assert "kept" in kept["D"]["vision_check"]  # the table the books stand on
    assert [r["id"] for r in removed] == ["C"]  # spines under it: books, counted in the book list
    assert kept["W"]["vision_check"] == "unknown"  # "book" with no spine there: listed, not priced


def test_a_mat_in_front_of_a_wall_item_is_not_a_part_of_it():
    from claimagent.pipeline.vlm import parts_of_other_items

    mirror = {
        "id": "M",
        "label": "mirror",
        "category": "decor",
        "frame_seq": 1,
        "bbox": [0, 0, 400, 600],
        "boxes": {"1": [0, 0, 400, 600]},
    }
    mat = {
        "id": "D",
        "label": "doormat",
        "category": "rug",
        "frame_seq": 1,
        "bbox": [100, 450, 250, 560],
        "boxes": {"1": [100, 450, 250, 560]},
    }
    print_ = {
        "id": "P",
        "label": "kite",
        "category": "other",
        "frame_seq": 1,
        "bbox": [50, 50, 120, 120],
        "boxes": {"1": [50, 50, 120, 120]},
    }
    parts = parts_of_other_items([mirror, mat, print_])
    assert "D" not in parts and parts.get("P") == "mirror"
