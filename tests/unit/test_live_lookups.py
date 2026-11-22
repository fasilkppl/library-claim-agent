import numpy as np

from claimagent.capture.store import FrameMeta
from claimagent.live.session import LOOKUPS_PER_TRACK, SETTLED_FRAMES, LiveSession
from claimagent.pipeline.register import Observation, Track
from claimagent.pipeline.resolve import Candidate, pool_from_json, pool_to_json


def _session(n_frames: int) -> LiveSession:
    s = LiveSession.__new__(LiveSession)  # only the lookup bookkeeping is exercised
    s.metas = [FrameMeta(seq=k, t_ms=k * 500, width=10, height=10) for k in range(n_frames)]
    s.titles, s.queued = {}, {}
    return s


def _track(last_seq: int) -> Track:
    obs = Observation(seq=last_seq, index=0, corners=np.zeros((4, 2)), spine={})
    return Track(id=1, segment=0, observations=[obs])


def test_a_spine_is_looked_up_once_the_camera_has_moved_on():
    s = _session(10)
    order = {m.seq: k for k, m in enumerate(s.metas)}
    assert not s._settled(_track(9), order)  # still in view: its text may still grow
    assert s._settled(_track(9 - SETTLED_FRAMES), order)


def test_a_track_is_looked_up_again_only_when_its_text_grew():
    s = _session(10)
    assert s._wants_lookup(1, "WINGS OF FIRE")
    s.queued[1] = ["WINGS OF FIRE"]
    assert not s._wants_lookup(1, "WINGS OF FIRE")
    assert not s._wants_lookup(1, "WINGS OF FIRX")  # same length: a misread of the same words
    assert s._wants_lookup(1, "WINGS OF FIRE | KALAM")
    s.queued[1] = ["a"] * LOOKUPS_PER_TRACK
    assert not s._wants_lookup(1, "WINGS OF FIRE | APJ ABDUL KALAM")
    assert not _session(1)._wants_lookup(2, "AB | 7")  # too few letters to look up


def test_the_walks_catalogue_records_are_kept_for_the_packet():
    pool = [
        Candidate("Rework", ["Jason Fried"], source="Open Library", url="https://openlibrary.org/works/OL1W"),
        Candidate("Wings of Fire", ["A. P. J. Abdul Kalam"], isbn="9788173711466", volume_id="v1", raw={"a": 1}),
    ]
    assert pool_from_json(pool_to_json(pool)) == pool
