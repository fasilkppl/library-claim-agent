from claimagent.pipeline.resolve import Candidate, Resolver, content_tokens, norm_tokens, score, volume_numbers


def test_norm_folds_accents_and_sharp_s():
    assert norm_tokens("Auf großer Fahrt") == ["auf", "grosser", "fahrt"]
    assert content_tokens("Gödel, Escher, Bach") == ["godel", "escher", "bach"]


def test_volume_numbers():
    assert volume_numbers("RICKEN (Hrsg.) • Philosophen der Antike II") == {2}
    assert volume_numbers("UTB 1136") == set()
    assert volume_numbers("Erzählungen 3 Jorge Luis Borges") == {3}


def test_full_match_scores_high():
    c = Candidate("Einführung in die Philosophie des Mythos", ["Luc Brisson"])
    s, _ = score("Brisson • Einführung in die Philosophie des Mythos", c)
    assert s >= 0.95


def test_author_name_is_not_title_evidence():
    c = Candidate("Sven Hedin as artist", ["Sven Hedin"])
    s, _ = score("Sven Hedin,", c)
    assert s < 0.65


def test_wrong_volume_penalised():
    v1 = Candidate("Philosophen der Antike I", ["Friedo Ricken"])
    v2 = Candidate("Philosophen der Antike II", ["Friedo Ricken"])
    spine = "RICKEN (Hrsg.) • Philosophen der Antike II"
    assert score(spine, v2)[0] > score(spine, v1)[0] + 0.2


def test_single_common_word_capped():
    c = Candidate("Magazin", ["Somebody"])
    assert score("Magazin | Obk | Cose", c)[0] <= 0.6


def test_query_variants_drop_stray_lines():
    r = Resolver.__new__(Resolver)
    vs = Resolver.query_variants(r, "THE CATHEDRAL & THE BAZAAR | ERIC S. RAYMOND | O'REILLY")
    assert vs[0].startswith("cathedral bazaar")
    assert "cathedral bazaar eric raymond" in vs


def test_eval_title_match_is_strict():
    from claimagent.eval.evaluate import title_match

    assert title_match("Auf großer Fahrt", "Auf grosser fahrt")
    assert title_match("Gödel, Escher, Bach", "Gödel, Escher, Bach: An Eternal Golden Braid")
    assert title_match("Antike Denker über den Staat", "Rechts- und Staatsphilosophie I. Antike Denker über den Staat")
    assert not title_match("Die Höhlenkinder im Steinhaus", "Die Höhlenkinder")
    assert not title_match("Die Natur und die Griechen", "Griechen")
    assert title_match("The Cathedral & the Bazaar", "The Cathedral and the Bazaar")


def test_volume_aware_eval_match_and_unexplained_spine_words_cap():
    from claimagent.eval.evaluate import title_match

    assert title_match("Gesammelte Aufsätze II", "Gesammelte Aufsätze")
    assert not title_match("Philosophen der Antike II", "Philosophen der Antike I")
    c = Candidate("Die Höhlenkinder", ["Alois Theodor Sonnleitner"])
    assert score("SONNLEITNER | Höhlenkinder | im Steinhaus | Die", c)[0] < 0.85


def test_another_first_name_before_the_surname_is_another_author():
    from claimagent.pipeline.resolve import Candidate, score

    spine = "MARA HOLT | LANTERNS"
    right = score(spine, Candidate(title="Go Set the Lanterns", authors=["Mara Holt"]))[0]
    namesake = score(spine, Candidate(title="Lanterns Ree", authors=["Peter Holt"]))[0]
    assert right > namesake + 0.05


def test_one_or_two_letter_surnames_prove_nothing():
    from claimagent.pipeline.resolve import Candidate, score

    assert score("A RIVER OK", Candidate(title="River Stories From Afar", authors=["Ian Ok"]))[1]["author_hit"] == 0.0


def test_long_title_words_weigh_more_than_short_ones():
    from claimagent.pipeline.resolve import Candidate, score

    # only the long word is readable on the spine, the short ones are not
    s, parts = score("MARA HOLT | LANTERNS", Candidate(title="Go Set the Lanterns", authors=["Mara Holt"]))
    assert parts["title_cov"] > 0.5 and s >= 0.65


def test_sequel_is_not_the_same_work():
    from claimagent.pipeline.resolve import same_work

    assert same_work("The Almanack of Naval Ravikant", "The Almanack of Naval Ravikant: A Guide to Wealth")
    assert same_work("ReWork", "Rework")
    assert not same_work("Dune Messiah", "Dune")


def test_books_about_a_book_are_not_the_book():
    from claimagent.pipeline.resolve import derivative, same_work

    trivia = Candidate(
        title="And the Mountains Echoed by Khaled Hosseini",
        authors=["Trivia-on-Books Staff", "And the Mountains Echoed"],
    )
    novel = Candidate(title="And the Mountains Echoed", authors=["Khaled Hosseini"])
    own_by = Candidate(title="Go Set a Watchman by Harper Lee", authors=["Harper Lee"])
    assert derivative(trivia) and not derivative(novel) and not derivative(own_by)
    assert same_work("Go Set a Watchman by Harper Lee", "Go Set a Watchman")


def test_study_notes_are_not_the_book():
    from claimagent.pipeline.resolve import derivative

    assert derivative(Candidate(title="Charles Dickens: A Tale of Two Cities: Notes", authors=["George Taylor"]))
    assert not derivative(Candidate(title="Notes from Underground", authors=["Fyodor Dostoyevsky"]))


def test_author_possessive_before_the_title_is_removed() -> None:
    from claimagent.pipeline.resolve import without_author_prefix

    # Seen on sweep 20261120-124155: one Lost Symbol came back under this title and was counted twice.
    assert without_author_prefix("Dan Brown's the Lost Symbol", ["Dan Brown"]) == "The Lost Symbol"
    assert without_author_prefix("Brown's The Lost Symbol", ["Dan Brown"]) == "The Lost Symbol"
    # A possessive that is part of the real title, or another person's name, stays.
    assert without_author_prefix("Ender's Game", ["Orson Scott Card"]) == "Ender's Game"
    assert without_author_prefix("The Handmaid's Tale", ["Margaret Atwood"]) == "The Handmaid's Tale"
    assert (
        without_author_prefix("Agatha Christie's Secret Notebooks", ["John Curran"])
        == "Agatha Christie's Secret Notebooks"
    )


def test_a_full_name_possessive_by_another_author_is_a_book_about_the_book() -> None:
    from claimagent.pipeline.resolve import Candidate, derivative

    # Seen on sweep 20261120-124155: a parody record was taken for the novel and the book counted twice.
    assert derivative(Candidate(title="Dan Brown's the Lost Symbol", authors=["Nigel Tomm"]))
    assert not derivative(Candidate(title="Dan Brown's The Lost Symbol", authors=["Dan Brown"]))
    assert not derivative(Candidate(title="Ender's Game", authors=["Orson Scott Card"]))
    assert not derivative(Candidate(title="Charlotte's Web", authors=["E. B. White"]))
    assert not derivative(Candidate(title="The Handmaid's Tale", authors=["Margaret Atwood"]))


def test_catalogue_thickness_from_listed_size_or_page_count():
    from claimagent.pipeline.resolve import Resolver

    class _Resp:
        def __init__(self, data):
            self.data, self.status = data, 200

    class _Http:
        def __init__(self, data):
            self.data = data

        def get(self, url, params, source=""):
            return _Resp(self.data)

    listed = Resolver(_Http({"volumeInfo": {"dimensions": {"thickness": "2.40 cm"}}}))
    assert listed.catalogue_thickness_cm("v1") == (2.4, "catalogue thickness")
    pages = Resolver(_Http({"volumeInfo": {"pageCount": 244}}))  # The Almanack: 1.5 cm on the hand-measured shelf
    assert pages.catalogue_thickness_cm("v2") == (1.6, "244 pages in the catalogue")
    assert Resolver(_Http({"volumeInfo": {}})).catalogue_thickness_cm("v3") is None
    assert pages.catalogue_thickness_cm("") is None
    # The page count wins over a listed thickness (often another edition's).
    both = Resolver(_Http({"volumeInfo": {"dimensions": {"thickness": "3.00 cm"}, "pageCount": 180}}))
    assert both.catalogue_thickness_cm("v4") == (1.2, "180 pages in the catalogue")


def test_thickness_from_open_library_editions_when_there_is_no_google_volume():
    from claimagent.pipeline.resolve import Resolver

    class _Resp:
        def __init__(self, data):
            self.data, self.status = data, 200

    class _Http:
        def __init__(self, data):
            self.data = data

        def get(self, url, params, source=""):
            return _Resp(self.data)

    editions = {
        "entries": [
            {"number_of_pages": 640},
            {"number_of_pages": 656, "physical_format": "Hardcover"},
            {"number_of_pages": 672, "physical_format": "Hardcover"},
            {"number_of_pages": None},
        ]
    }
    got = Resolver(_Http(editions)).catalogue_thickness_cm("", "https://openlibrary.org/works/OL1W")
    assert got == (4.1, "656 pages (median of 3 Open Library editions)")  # mostly hardcover: 656 x 0.0055 + 0.45


def test_a_first_name_glued_to_the_surname_still_names_the_author() -> None:
    spine = "THE ALMANACK OF NAVAL RAVIKANT ERICJORGENSON"  # read on a real sweep
    english, parts = score(spine, Candidate("The Almanack of Naval Ravikant", ["Eric Jorgenson"]))
    spanish, _ = score(spine, Candidate("El Almanaque de Naval Ravikant", []))
    assert parts["author_hit"] == 1.0 and not parts["unexplained"]
    assert english > spanish  # the two used to tie, and the translation was listed first


def test_thickness_from_the_work_found_by_title_when_the_matched_record_has_no_page_count():
    from claimagent.pipeline.resolve import Resolver

    class _Resp:
        def __init__(self, data, status=200):
            self.data, self.status = data, status

    class _Http:
        def get(self, url, params, source=""):
            if "googleapis" in url:
                return _Resp({"volumeInfo": {"title": "The Lost Symbol"}})  # no page count, no size
            if url.endswith("search.json"):
                assert params["title"] == "The Lost Symbol" and params["author"] == "brown"
                return _Resp(
                    {"docs": [{"key": "/works/OL5W", "title": "The Lost Symbol", "number_of_pages_median": 604}]}
                )
            return _Resp({"entries": [{"number_of_pages": 509}, {"number_of_pages": 639}, {"number_of_pages": 670}]})

    got = Resolver(_Http()).catalogue_thickness_cm("VOL1", title="The Lost Symbol", author="Dan Brown")
    assert got == (3.8, "639 pages (median of 3 Open Library editions)")


def test_a_record_far_from_the_works_median_page_count_is_an_unusual_edition():
    from claimagent.pipeline.resolve import Resolver, without_author_prefix

    class _Resp:
        def __init__(self, data):
            self.data, self.status = data, 200

    class _Http:
        def get(self, url, params, source=""):
            if "googleapis" in url:
                return _Resp({"volumeInfo": {"pageCount": 87}})  # an abridged edition was matched
            if url.endswith("search.json"):
                return _Resp({"docs": [{"title": "A Tale of Two Cities", "number_of_pages_median": 388}]})
            return _Resp(
                {},
            )

    t, why = Resolver(_Http()).catalogue_thickness_cm("V", title="A Tale of Two Cities", author="Charles Dickens")
    assert t == 2.4 and "unusual edition" in why
    assert without_author_prefix("A Tale of Two Cities -", ["Charles Dickens"]) == "A Tale of Two Cities"


def test_query_variants_do_not_depend_on_set_order():
    import os
    import subprocess
    import sys

    spine = "Phystcull | Sambursky | of the | Greeks | World | und ihre moderne Verwandlung"
    code = f"from claimagent.pipeline.resolve import Resolver; print(Resolver(None).query_variants({spine!r}))"
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed}
        ).stdout
        for seed in ("1", "2", "3", "4")
    }
    assert len(runs) == 1, runs  # the same queries whatever the string hashing
