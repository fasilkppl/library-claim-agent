from claimagent.pipeline.vlm import canonical


def test_answers_map_to_household_labels_and_categories():
    assert canonical("pressure cooker") == ("pressure cooker", "kitchen", False)
    assert canonical("iron") is not None and canonical("clothes iron")[1] == "appliance"
    assert canonical("light switch")[2] is True  # a building fixture, not contents
    assert canonical("unknown") is None
    assert canonical("a very long rambling answer here") is None


def test_books_seen_by_the_item_detector_leave_the_item_list():
    assert canonical("books")[2] is True and canonical("book")[1] == "book"


def test_contents_words_are_not_fixtures():
    assert canonical("floor lamp") is not None and canonical("floor lamp")[2] is False
    assert canonical("wall clock") is not None and canonical("wall clock")[2] is False
    assert canonical("doormat")[2] is False  # "door" only as a whole word
    assert canonical("kitchen sink")[2] is True
    assert canonical("light switch")[2] is True
    assert canonical("not sure") is None and canonical("unknown object") is None


def test_a_building_word_in_front_of_a_contents_word_keeps_the_item():
    assert canonical("door mat") == ("doormat", "rug", False)
    assert canonical("floor mat") == ("mat", "rug", False)
    assert canonical("wall mirror") == ("mirror", "decor", False)
    assert canonical("window curtain") == ("curtain", "decor", False)
    assert canonical("wall art") == ("wall art", "art", False)  # art goes to appraisal, it is never dropped
    fixtures = ("light switch", "wall socket", "kitchen sink", "plug point", "floor", "door", "wall", "door handle")
    for fixture in fixtures:
        assert canonical(fixture)[2] is True, fixture
