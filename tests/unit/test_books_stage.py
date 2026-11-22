from claimagent.pipeline.stages.books import holds_spine


def test_a_frame_that_also_read_a_neighbours_whole_spine():
    almanack = "THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSON"
    assert holds_spine(f"{almanack} | ATADKOR | TVO CINPS | HARPER", [almanack])
    assert holds_spine("TIO CTNPS | A TALON | THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSOM", [almanack])
    assert not holds_spine("IWO CTMPS | A TADEON | A TAIL OR", [almanack])
    assert not holds_spine("ALMANACK", [almanack])  # one word of it is not the spine
    assert not holds_spine("REWORK JASON FRIED", ["REWORK"])  # a one-word spine proves nothing


def test_photo_and_page_count_thickness_are_combined_unless_the_outline_is_wrong():
    from claimagent.pipeline.stages.books import fuse_thickness

    t, why = fuse_thickness(1.87, (1.2, "173 pages"))  # photo too wide, page count too thin: 1.5 on the shelf
    assert round(t, 2) == 1.5 and "combined" in why
    t, why = fuse_thickness(3.12, (0.8, "100 pages"))  # four times the page count: an outline over two spines
    assert t == 0.8 and "photo gave 3.1 cm" in why
    assert fuse_thickness(None, (1.8, "279 pages"))[0] == 1.8
    # A photo measured against a printed marker stands on its own (test shelf: 3.18 cm measured, 3.25 true, 287
    # pages of another edition said 1.8).
    assert fuse_thickness(3.18, (1.8, "287 pages"), metric_scale=True)[0] == 3.18


def test_a_match_from_one_frame_needs_the_author_or_most_of_the_reading():
    from claimagent.config import Settings
    from claimagent.pipeline.resolve import Resolution
    from claimagent.pipeline.stages.books import resolve_books

    def res(title, conf, author_hit, spine_cov):
        parts = {"title_cov": 1.0, "author_hit": author_hit, "spine_cov": spine_cov, "unexplained": []}
        return Resolution("identified", conf, "", title=title, evidence={"score_parts": parts})

    answers = {
        "IWO CTMPS | A TADEON | A TAIL OR": Resolution("unidentified", 0.59, ""),
        "TWO SYPTES | A TAIL OR": res("Two Tails on the Trail", 0.65, 0.0, 0.5),  # real sweep: another book
        "WINGS OF FIRE An Autobiography | APJ Abdul Kalam": res("Wings of Fire", 0.95, 1.0, 1.0),
    }

    class FakeResolver:
        deadline = None

        def resolve(self, text, threshold):
            return answers.get(text, Resolution("unidentified", 0.0, text))

    class FakeStore:
        def write_stage(self, name, data):
            return None

    def book(track, text, variants):
        return {"track": track, "spine_text": text, "text_variants": variants}

    read = {
        "books": [
            book(1, "IWO CTMPS | A TADEON | A TAIL OR", ["TWO SYPTES | A TAIL OR"]),
            book(2, "W1NGS", ["WINGS OF FIRE An Autobiography | APJ Abdul Kalam"]),
        ]
    }
    out = {b["track"]: b for b in resolve_books(FakeStore(), read, FakeResolver(), Settings())["books"]}
    assert not out[1].get("title")  # left unidentified, not named after another book
    assert out[2]["title"] == "Wings of Fire" and out[2]["confidence"] < Settings().id_accept
