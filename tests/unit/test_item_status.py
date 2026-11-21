from claimagent.pipeline.assemble import item_price_status


def test_an_item_the_picture_check_never_reached_is_not_priced():
    # Sweep 20261121-171735: the check timed out and a gift box was priced as a "laptop".
    assert item_price_status({"label": "laptop", "confidence": 0.9}, vision_check_on=True) == "unpriced"


def test_items_named_from_their_picture_are_priced():
    assert item_price_status({"vision_check": "bucket", "confidence": 0.3}, vision_check_on=True) == "range"


def test_an_item_the_vision_model_cannot_name_is_not_priced():
    # Same sweep: a dark corner the detector called an "electric kettle" (confidence 0.6+) was priced.
    assert item_price_status({"vision_check": "unknown", "confidence": 0.3}, vision_check_on=True) == "unpriced"
    assert item_price_status({"vision_check": "unknown", "confidence": 0.8}, vision_check_on=True) == "unpriced"


def test_with_the_picture_check_switched_off_detector_names_are_priced():
    assert item_price_status({"label": "bucket", "confidence": 0.9}, vision_check_on=False) == "range"
