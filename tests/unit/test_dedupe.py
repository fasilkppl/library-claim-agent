from claimagent.pipeline.dedupe import dedupe_books, duplicate_pairs, reading_similarity, tokens


def book(track, shelf, pos, text, obs=5):
    return {
        "track": track,
        "shelf": shelf,
        "position": pos,
        "spine_text": text,
        "text_variants": [text] if text else [],
        "text_score": 50.0,
        "observations": obs,
        "frame_ref": f"frames/{track:06d}.jpg",
    }


def res(track, title="", author="", conf=0.9):
    if not title:
        return {"track": track, "readable": True, "status": "unidentified"}
    return {
        "track": track,
        "readable": True,
        "status": "identified",
        "title": title,
        "author": author,
        "confidence": conf,
    }


def run(books, resolved):
    read, out, report = dedupe_books({"books": books}, {"books": resolved})
    return read["books"], out["books"], report


def test_revisit_of_a_shelf_is_merged_in_order():
    first = [
        book(1, "A", 1, "WINGS OF FIRE"),
        book(2, "A", 2, "WATCHMAN HARPER LEE"),
        book(3, "A", 3, ""),
        book(4, "A", 4, "REWORK"),
    ]
    again = [
        book(11, "B", 1, "WINGS OF FIRR"),
        book(12, "B", 2, "WATCHMAN"),
        book(13, "B", 3, ""),
        book(14, "B", 4, "REWORK"),
    ]
    books, resolved, report = run(first + again, [res(1, "Wings of Fire", "A P J Abdul Kalam")])
    assert len(books) == 4  # the unreadable spine between the same neighbours is the same book too
    assert {b["shelf"] for b in books} == {"A"}
    assert not duplicate_pairs({"books": books}, {"books": resolved})
    assert report["tracks_in"] == 8 and report["books_out"] == 4


def test_spine_split_in_two_neighbours_is_one_book():
    books, _, _ = run(
        [book(1, "A", 1, "A TALLOR PWO OTTES"), book(2, "A", 2, "ATALLOR"), book(3, "A", 3, "REWORK")], []
    )
    assert len(books) == 2


def test_different_books_are_never_merged():
    books, _, _ = run(
        [
            book(1, "A", 1, "THE LOST SYMBOL DAN BROWN"),
            book(2, "A", 2, "MY EXPERIMENTS WITH TRUTH"),
            book(3, "A", 3, "PRINCIPLES RAY DALIO"),
            book(4, "A", 4, "PATTERNS OF SOFTWARE"),
        ],
        [res(1, "The Lost Symbol", "Dan Brown")],
    )
    assert len(books) == 4


def test_leaked_author_on_a_long_reading_is_not_a_match():
    # A neighbour's "DAN BROWN" read onto this spine must not make it The Lost Symbol.
    a, b = tokens("MITH TRUTW HY EXPERIMENITS DAN BROWN"), tokens("DAN BROWNI")
    assert reading_similarity(a, b) == 0.0


def test_same_book_named_twice_anywhere_is_one_entry():
    books, resolved, _ = run(
        [book(1, "A", 1, "ALMANACK NAVAL RAVIKANT"), book(2, "C", 1, "TAR ALMANACK")],
        [res(1, "The Almanack of Naval Ravikant", "Eric Jorgenson"), res(2, "Almanack of Naval Ravikant", "")],
    )
    assert len(books) == 1
    assert books[0]["merged_tracks"] == [2]
    assert not duplicate_pairs({"books": books}, {"books": resolved})


def test_glimpse_of_an_author_merges_into_the_identified_book():
    books, _, _ = run(
        [book(1, "A", 1, "THE LOST SYMBOL DAN BROWN"), book(2, "Z", 1, "DAN BROWNI", obs=1)],
        [res(1, "The Lost Symbol", "Dan Brown")],
    )
    assert len(books) == 1


def test_lone_unreadable_spines_are_kept():
    books, _, _ = run([book(1, "A", 1, ""), book(2, "B", 1, "")], [])
    assert len(books) == 2  # no evidence they are one book


def test_volumes_of_a_set_are_different_books():
    books, _, _ = run(
        [
            book(1, "A", 1, "Hanser Erzählungen1 Jorge Luis Borges"),
            book(2, "A", 2, "Hanser Erzählungen 2 Jorge Luis Borges"),
            book(3, "A", 3, "Philosophen der Antike II"),
        ],
        [res(3, "Philosophen der Antike I")],
    )
    assert len(books) == 3


def rect(x0, x1):
    return [[x0, 0.0], [x1, 0.0], [x1, 300.0], [x0, 300.0]]


def test_two_spines_side_by_side_in_one_frame_are_two_books():
    shelf = [book(1, "A", 1, "TRANSHIMALAJA SVEN HEDIN"), book(2, "A", 2, "TRANSHIMALAJA SVEN HEDIN")]
    boxes = {1: {7: rect(100, 140)}, 2: {7: rect(142, 180)}}
    read, _, _ = dedupe_books({"books": shelf}, {"books": []}, boxes)
    assert len(read["books"]) == 2
    # the same spine detected twice on top of itself is one book
    boxes = {1: {7: rect(100, 140)}, 2: {7: rect(104, 142)}}
    read, _, _ = dedupe_books({"books": shelf}, {"books": []}, boxes)
    assert len(read["books"]) == 1


def test_leaked_neighbour_lines_do_not_merge_two_books():
    # Both spines carry the neighbour's "WATCHMAN HARPER LEE" (seen on a real sweep): still two books.
    shelf = [
        book(4, "A", 1, "WINGS OF FIRE An Autobiography | with Arun Tiwari | WATCHMAN | HARPER LEE"),
        book(11, "A", 2, "SYMBOL | DAN BROWNI | The Lost | WATCHMAN | HARPER LEE | WINGS OF FIRE"),
    ]
    boxes = {4: {7: rect(100, 150)}, 11: {7: rect(110, 160)}}  # overlapping boxes
    read, _, _ = dedupe_books({"books": shelf}, {"books": [res(4, "Wings of Fire: An Autobiography", "Kalam")]}, boxes)
    assert len(read["books"]) == 2


def guided(track, pos, text, obs=5, partial=False, shelf_label="A"):
    b = book(track, shelf_label, pos, text, obs)
    b["guide_shelf"], b["only_partial"] = 1, partial
    return b


def test_guided_shelf_leaves_out_fragments_that_are_not_books():
    shelf = [
        guided(1, 1, "", partial=True),  # the side of a box at the shelf's end, never seen whole
        guided(2, 2, "THE LOST SYMBOL DAN BROWN"),
        guided(3, 3, "GO SET A WATCHMAN HARPER LEE"),
        guided(4, 4, "SYMBOL | DAN BROWNI | The Lost | WATCHMAN | HARPER LEE"),  # one box over two spines
        guided(5, 5, "PRINCIPLES OF BUILDING AI AGENTS"),
        guided(6, 1, "unfalis atal", obs=1, shelf_label="B"),  # a one-frame glimpse placed nowhere
    ]
    resolved = [res(2, "The Lost Symbol", "Dan Brown"), res(3, "Go Set a Watchman", "Harper Lee")]
    read, _, report = dedupe_books({"books": shelf}, {"books": resolved})
    # the multi-spine box is merged into a book it covers or left out: never a book of its own
    assert sorted(b["track"] for b in read["books"]) == [2, 3, 5]
    assert {n["track"] for n in report["not_counted"]} >= {1, 6}


def test_unguided_sweeps_keep_every_spine():
    shelf = [book(1, "A", 1, ""), book(2, "A", 2, "SYMBOL DAN BROWN WATCHMAN HARPER LEE")]
    shelf[0]["only_partial"] = True
    read, _, _ = dedupe_books({"books": shelf}, {"books": []})
    assert len(read["books"]) == 2


def test_evaluator_title_matching_on_real_pairs():
    from claimagent.eval.evaluate import title_match

    assert not title_match("Principles of Building AI Agents", "Patterns for Building AI Agents")
    assert title_match("Wings of Fire", "Wings of Fire: An Autobiography")
    assert title_match("My Experiments with Truth", "Mahatma Gandhi: The Story of My Experiments with Truth")
    assert title_match("A Tale of Two Cities", "Charles Dickens' A Tale of Two Cities")
    assert not title_match("Dune", "Dune Messiah")
    assert not title_match("Philosophen der Antike I", "Philosophen der Antike II")


def test_two_books_side_by_side_never_share_a_title():
    # Real case: a spine's reading picked up two neighbours' lines (Wings of Fire, Go Set a Watchman).
    shelf = [
        book(1, "A", 1, "WATCHMAN HARPER LEE GO SET A"),
        book(2, "A", 2, "WINGS OF FIRE AN AUTOBIOGRAPHY | WATCHMAN | HARPER LEE | GO SET A"),
        book(3, "A", 3, "WINGS OF FIRE AN AUTOBIOGRAPHY KALAM"),
    ]
    resolved = [
        res(1, "Go Set a Watchman", "Harper Lee", 0.95),
        res(2, "Go Set A Watchman", "Harper Lee", 0.7),
        res(3, "Wings of Fire: An Autobiography", "Kalam", 0.9),
    ]
    boxes = {1: {7: rect(100, 140)}, 2: {7: rect(142, 200)}, 3: {7: rect(202, 260)}}
    read, out, report = dedupe_books({"books": shelf}, {"books": resolved}, boxes)
    assert len(read["books"]) == 3  # three physical books
    titles = [r.get("title") for r in out["books"]]
    assert titles.count("Go Set a Watchman") + titles.count("Go Set A Watchman") == 1
    assert report["titles_removed"] and report["titles_removed"][0]["track"] == 2


def test_volumes_of_a_set_side_by_side_keep_their_titles():
    shelf = [book(1, "A", 1, "GRIECHISCHES RECHTSDENKEN I"), book(2, "A", 2, "GRIECHISCHES RECHTSDENKEN II")]
    resolved = [res(1, "Griechisches Rechtsdenken", "Wolf", 0.9), res(2, "Griechisches Rechtsdenken", "Wolf", 0.8)]
    boxes = {1: {7: rect(100, 140)}, 2: {7: rect(142, 200)}}
    _read, out, report = dedupe_books({"books": shelf}, {"books": resolved}, boxes)
    assert [r.get("title") for r in out["books"]].count("Griechisches Rechtsdenken") == 2
    assert not report["titles_removed"]


def test_one_book_under_two_catalogue_titles_is_counted_once():
    # Sweep 20261121-185148: My Experiments came back as three catalogue records and was counted three times.
    shelf = [
        book(1, "A", 1, "MY EXPERIMEIITS | M.K.GANDHI | WITH TRUTH"),
        book(2, "B", 4, "WITH TRUTH | MY EXPERIMENTS"),
    ]
    resolved = [
        res(1, "Mahatma Gandhi: The Story of My Experiments with Truth", "Mahatma Gandhi"),
        res(2, "My Experiments with Truth", "M.K. Gandhi"),
    ]
    books, _, _ = run(shelf, resolved)
    assert len(books) == 1
    # Shared words are not enough between two different works.
    shelf = [book(3, "A", 1, "PRINCIPLES OF BUILDING AI AGENTS"), book(4, "A", 2, "PATTERNS FOR BUILDING AI AGENTS")]
    resolved = [
        res(3, "Principles of Building AI Agents", "Sam Bhagwat"),
        res(4, "Patterns for Building AI Agents", "Sam Bhagwat"),
    ]
    books, _, _ = run(shelf, resolved)
    assert len(books) == 2


def test_a_quick_glimpse_read_with_glued_words_joins_its_book():
    # Same sweep: a first look at the shelf read "WATCHMAN | GO SETA | ND HARPERLEE O" (2 frames).
    shelf = [
        book(5, "A", 3, "WATCHMAN | GO SETA | ND HARPERLEE O", obs=2),
        book(6, "B", 3, "WATCHMÄNT HARPER LEE | GO SET A", obs=16),
    ]
    books, _, _ = run(shelf, [res(6, "Go Set a Watchman", "Harper Lee")])
    assert len(books) == 1
    # A glimpse that spells another book's title is kept apart.
    shelf = [
        book(7, "A", 1, "IWO CTMPS | A TADEON | A TAIL OR", obs=2),
        book(8, "B", 2, "THE ALMANACK OF NAVAL RAVIKANT", obs=12),
    ]
    books, _, _ = run(shelf, [res(8, "The Almanack of Naval Ravikant", "Eric Jorgenson")])
    assert len(books) == 2


def test_a_glimpse_named_from_one_frame_is_matched_by_what_it_reads():
    # Real sweep: a 2-frame glimpse of this book was named "Principle Agent" from one frame and counted twice.
    glimpse = book(3, "A", 1, "Principles of Builctno Al Apente", obs=2)
    seen_well = book(27, "B", 1, "Sam Bhagwat | Principles of Building Al Agents", obs=10)
    weak = {
        **res(3, "Principle Agent", "Sarah Suzor", conf=0.65),
        "evidence": {"matched_on_single_frame_reading": "Principles of bulateg Al Agente"},
    }
    good = res(27, "Principles of Building AI Agents", "Sam Bhagwat")
    books, resolved, _ = run([glimpse, seen_well], [weak, good])
    assert len(books) == 1 and resolved[0]["title"] == "Principles of Building AI Agents"
    # Named by its own cross-frame reading, the same glimpse keeps its title and stays a book of its own.
    books, _, _ = run([glimpse, seen_well], [res(3, "Principle Agent", "Sarah Suzor", conf=0.65), good])
    assert len(books) == 2
