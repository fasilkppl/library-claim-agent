from types import SimpleNamespace

from claimagent.config import get_settings
from claimagent.live.coach import Coach


def _frame(seq: int, art_conf: float | None) -> dict:
    items = [{"label": "framed picture", "conf": art_conf}] if art_conf is not None else []
    return {"seq": seq, "spines": [], "items": items, "quality": {"book_glare": 0.0, "book_sharpness": 100.0}}


def _coach() -> tuple[Coach, list[str]]:
    said: list[str] = []
    store = SimpleNamespace(log_event=lambda *a, **k: None, load_depth=lambda m: None)
    coach = Coach(get_settings(), store, lambda: True, lambda kind, text: said.append(text), lambda: 0)
    coach.reset()
    coach.step_of = lambda: "room"
    return coach, said


def test_one_frame_framed_picture_is_not_asked_about():
    # Sweep 20261121-185148: a book cover seen once as a "framed picture" (0.40) made the agent ask.
    coach, said = _coach()
    coach.on_frame(_frame(13, 0.40), 0, lambda: 0)
    for seq in range(14, 30):
        coach.on_frame(_frame(seq, None), 0, lambda: 0)
    assert not any("original or a print" in s for s in said)


def test_a_picture_seen_in_two_nearby_frames_is_asked_about_once():
    coach, said = _coach()
    for seq in (40, 42, 44):
        coach.on_frame(_frame(seq, 0.6), 0, lambda: 0)
    assert sum("original or a print" in s for s in said) == 1
