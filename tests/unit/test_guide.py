from claimagent.capture.store import FrameMeta
from claimagent.live.guide import Guide
from claimagent.pipeline.runner import books_only


def test_one_continuous_sweep_shelves_then_room():
    g = Guide()
    assert g.tags() == {"step": "books", "shelf": 1}
    assert g.command("next shelf") == "next_shelf"
    g.act("next_shelf")
    assert g.tags() == {"step": "books", "shelf": 2}
    assert g.command("Done with the books.") == "books_done"
    said, over = g.act("books_done")
    assert not over and "walls" in said and g.tags() == {"step": "room"}
    assert g.command("I'm done") == "finish"
    _, over = g.act("finish")
    assert over and g.st.step == "done"


def test_im_done_ends_the_sweep_from_any_part_of_the_walk():
    g = Guide()
    assert g.command("I'm done.") == "finish"
    assert g.command("that's all") == "finish"
    assert g.command("finish the sweep") == "finish"


def test_statements_go_to_the_dialog_not_the_guide():
    g = Guide()
    for said in (
        "this one on the left is a signed first edition from my grandfather",
        "not mine",
        "Skip that shelf. Those books are not mine.",
        "I'm in India. Rupees is right.",
        "How many books have you found so far?",
        "The painting is just a print.",
    ):
        assert g.command(said) is None, said
    assert g.command("Okay, next.") == "next_shelf"


def test_spines_count_only_in_shelf_frames_of_a_guided_sweep():
    def meta(seq, step):
        return FrameMeta(seq=seq, t_ms=seq * 600, width=10, height=10, extra={"step": step})

    det = {1: {"spines": [{"x": 1}]}, 2: {"spines": [{"x": 2}]}}
    out = books_only(det, {1: meta(1, "books"), 2: meta(2, "room")})
    assert out[1]["spines"] and out[2]["spines"] == []


def test_clear_statements_are_recorded_by_rule():
    from claimagent.live.tools import rule_tool

    assert rule_tool("That painting is just a print")[:2] == (
        "art_answer",
        {"statement": "That painting is just a print", "is_print": True},
    )
    assert rule_tool("No, it's an original oil painting")[1]["is_print"] is False
    assert rule_tool("Skip that shelf. Those books are not mine.")[0] == "skip_region"
    assert rule_tool("this one is a signed first edition")[0] == "note_book"
    assert rule_tool("How many books have you found so far?") is None


def test_dialog_directions_are_replaced_by_the_guide():
    from claimagent.live.session import strip_directions

    out = strip_directions("Yes, I can see it. Just hold the camera steady and say next.", "Keep moving slowly.")
    assert out == "Yes, I can see it. Keep moving slowly."
    assert strip_directions("Alright, India and rupees set.", "x") == "Alright, India and rupees set."


def test_spoken_titles_are_short():
    from claimagent.live.session import spoken_title

    assert (
        spoken_title("Die Höhlenkinder im Steinhaus von Dr. A. Th. Sonnleitner [pseud.].")
        == "Die Höhlenkinder im Steinhaus"
    )
    assert spoken_title("Wings of Fire: An Autobiography") == "Wings of Fire"


def test_yes_to_the_locale_question():
    from claimagent.live.locale import confirms

    for said in ("Yes.", "Yeah, that's right", "I'm in India. Rupees is right.", "Correct"):
        assert confirms(said), said
    for said in ("No, I'm in the US", "Not India, Germany", "What do you mean?"):
        assert not confirms(said), said


def test_natural_sentence_moves_on_from_the_books():
    g = Guide()
    assert g.command("hey im done with the books. are we good to move on to the next item.") == "books_done"
    assert g.command("Can we move on to the room now?") == "books_done"


def test_answers_during_the_walk_invent_nothing():
    from claimagent.live.session import sweep_reply

    assert "summary" in sweep_reply("How many books have you found so far?", "Keep going.")
    assert sweep_reply("Scanning the books, okay?", "Keep going.") == "Yes. Keep going."
    assert sweep_reply("It can be worn.", "Keep going.") == "Okay."


def test_language_model_cannot_record_contents_facts():
    from claimagent.live.tools import LLM_TOOLS

    assert {t["function"]["name"] for t in LLM_TOOLS} == {"set_locale", "end_sweep"}


def test_country_named_in_an_answer():
    from claimagent.live.locale import country_named

    assert country_named("I'm in India. Rupees is right.") == "IN"
    assert country_named("no, we are in the UK") == "GB"
    assert country_named("Scanning the books, okay?") == ""


def test_real_phrases_from_a_recorded_sweep():
    g = Guide()
    assert g.command("Yes, I'm run with books already, I told you.") == "books_done"  # "done" misheard as "run"
    assert g.command("Okay, moving on to the forest.") == "books_done"
    assert g.command("Are we good?") is None
    assert g.command("This is the height.") is None
    assert g.command("Wait.") is None
    g.act("books_done")
    for said in ("I guess I'm done.", "I'm done already. I'm done.", "Yep, that's it.", "Yeah, I'm done buddy."):
        assert g.command(said) == "finish", said
    assert g.command("Are we done?") is None


def test_post_sweep_analysis_fills_the_biggest_gaps_first_within_budget():
    from claimagent.pipeline.stages.frames import _gap

    assert _gap(5000, []) == float("inf")
    assert _gap(5000, [1000, 4000, 9000]) == 1000.0


def test_a_camera_sweep_on_an_ar_phone_switches_to_ar_for_the_room():
    g = Guide(ar_room=True)
    said, over = g.act("books_done")
    assert not over and "Measure room" in said
    ui = g.ui()
    assert ui["step"] == "room" and [a["id"] for a in ui["actions"]] == ["ar_room"]
    assert Guide().act("books_done")[0].startswith("Got the shelves. Now pan")  # no AR: no button, no mention


def test_im_done_on_the_shelves_asks_for_the_room_once():
    g = Guide()
    said, over = g.act("finish")
    assert not over and g.ui()["step"] == "room" and "room" in said.lower()
    assert g.act("finish") == ("", True)  # the second time it ends
