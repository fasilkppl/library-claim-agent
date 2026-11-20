from claimagent.pipeline.verify import authors_agree, decide, spine_order


def test_authors_listed_in_the_order_the_spine_prints_them():
    catalogue = ["David Heinemeier Hansson", "Jason Fried"]
    spine = "REWORK | JASON FRIED | DAVID HEINEMEIER HANSSON"
    assert spine_order(catalogue, spine) == ["Jason Fried", "David Heinemeier Hansson"]
    assert spine_order(catalogue, "REWORK") == catalogue  # the spine shows no names: order kept


def test_same_authors_in_another_order_agree():
    assert authors_agree(["David Heinemeier Hansson", "Jason Fried"], ["Jason Fried", "David Heinemeier Hansson"])
    out = decide(["Jason Fried"], ["Jason Fried", "David Heinemeier Hansson"], "", "Open Library", "Google Books")
    assert out["status"] == "confirmed" and out["review"] == ""


def test_disagreement_goes_to_review_unless_the_spine_settles_it():
    ours, theirs = ["Trivia-on-Books Staff"], ["Khaled Hosseini"]
    out = decide(ours, theirs, "KHALED HOSSEINI | And the Mountains Echoed", "Open Library", "Google Books")
    assert out["status"] == "corrected" and out["authors"] == theirs and "please confirm" in out["review"]
    out = decide(ours, theirs, "AND THE MOUNTAINS ECHOED", "Open Library", "Google Books")
    assert out["status"] == "disagree" and out["authors"] == ours and "authors differ" in out["review"]


def test_second_catalogue_without_a_record_is_noted_not_flagged():
    out = decide(["Sam Bhagwat"], None, "", "Open Library", "Google Books")
    assert out["status"] == "unconfirmed" and out["review"] == ""


class _Resp:
    def __init__(self, data):
        self.data, self.status = data, 200


class _Http:
    def __init__(self, data):
        self.data = data

    def get(self, url, params, source=""):
        return _Resp(self.data)


class _Resolver:
    def __init__(self, data):
        self.http, self.google_key = _Http(data), ""


def test_an_agreeing_edition_is_preferred_over_a_summary_publisher():
    from claimagent.pipeline.verify import Checker

    google = {
        "items": [
            {"volumeInfo": {"title": "Go Set a Watchman", "authors": ["Superread Books"]}},
            {"volumeInfo": {"title": "Go Set a Watchman", "authors": ["Harper Lee"]}},
        ]
    }
    res = {"title": "Go Set A Watchman", "author": "Harper Lee", "source": "Open Library", "isbn": ""}
    other, authors = Checker(_Resolver(google)).other_record(res)
    assert other == "Google Books" and authors == ["Harper Lee"]


def test_surname_first_spelling_agrees():
    assert authors_agree(["Hosseini Khaled"], ["Khaled Hosseini"])
    assert authors_agree(["Charles 1812-1870 Dickens"], ["Charles Dickens"])
    assert not authors_agree(["Superread Books"], ["Harper Lee"])
