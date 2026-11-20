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
