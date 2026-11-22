import cv2
import numpy as np

from claimagent.capture.store import FrameMeta
from claimagent.capture.tracking import assess
from claimagent.pipeline import itemtrack


def meta(seq, t_ms, x, emulated=False):
    pose = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, 1.4, 0, 1]
    return FrameMeta(seq=seq, t_ms=t_ms, width=10, height=10, pose=pose, extra={"client_t": t_ms, "emulated": emulated})


def no_depth(m):
    return None


def test_slow_walk_is_trusted():
    frames = [meta(i, i * 600, 0.2 * i) for i in range(1, 20)]  # 0.33 m/s
    r = assess(frames, no_depth)
    assert len(r.trusted) == 19 and not r.rejected and r.note == ""


def test_runaway_scale_rejects_the_rest_of_the_sweep():
    xs = [0, 0.05, 0.1, 0.8, 2.0, 3.5, 3.6, 3.7, 3.8]  # the jump seen on a real phone, then "settled"
    r = assess([meta(i + 1, i * 600, x) for i, x in enumerate(xs)], no_depth)
    assert r.lost_for_good and max(r.trusted) < 4 and all(s in r.rejected for s in range(4, 10))
    assert "implausible camera speed" in r.note


def test_one_glitch_is_dropped_and_tracking_continues():
    frames = [meta(i, i * 600, 0.1 * i, emulated=(i == 5)) for i in range(1, 12)]
    r = assess(frames, no_depth)
    assert set(r.rejected) == {5} and not r.lost_for_good


def test_indoor_depth_limit():
    r = assess([meta(1, 0, 0), meta(2, 600, 0.1), meta(3, 1200, 0.2)], lambda m: np.full((4, 4), 30.0))
    assert not r.trusted


def _scene(seed=0, w=900, h=700):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w)) * 255).astype(np.uint8)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_tracker_follows_an_object_across_a_pan():
    big = _scene()
    frames, sightings = [], []
    for k in range(6):
        dx = 40 * k
        frames.append((k, big[:, dx : dx + 600].copy()))
        x0 = 300 - dx  # the object is fixed in the scene, so it moves left in the frame
        sightings.append(itemtrack.Sighting(k, 0, "laptop" if k % 2 else "television", (x0, 200, x0 + 150, 320), 0.8))
    ids = itemtrack.link(frames, sightings)
    assert len(set(ids.values())) == 1  # one object, even with the label switching between synonyms


def test_two_objects_seen_together_are_never_merged():
    img = np.zeros((400, 400, 3), np.uint8)
    img[50:150, 50:120] = (200, 200, 200)
    img[50:150, 250:320] = (200, 200, 200)
    tracks = {
        0: [itemtrack.Sighting(1, 0, "cup", (50, 50, 120, 150), 0.9)],
        1: [itemtrack.Sighting(1, 1, "cup", (250, 50, 320, 150), 0.9)],
        2: [itemtrack.Sighting(2, 0, "cup", (250, 50, 320, 150), 0.9)],
    }
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    groups = itemtrack.same_object_groups(tracks, {1: gray, 2: gray}, {1: img, 2: img})
    joined = next(g for g in groups if 0 in g)
    assert 1 not in joined  # 0 and 1 share frame 1


def test_keyboard_inside_laptop_is_part_of_it():
    s = [
        itemtrack.Sighting(1, 0, "laptop", (0, 0, 100, 100), 0.9),
        itemtrack.Sighting(1, 1, "keyboard", (10, 60, 90, 95), 0.7),
        itemtrack.Sighting(1, 2, "keyboard", (200, 60, 280, 95), 0.7),
    ]
    assert [x.index for x in itemtrack.drop_parts(s)] == [0, 2]


def test_a_new_tracking_segment_after_an_origin_reset_is_checked_afresh():
    def seg_meta(seq, t_ms, x, seg):
        m = meta(seq, t_ms, x)
        m.extra["ar_segment"] = seg
        return m

    runaway = [0, 0.05, 0.8, 2.0, 3.5, 3.6]  # scale lost in segment 0
    frames = [seg_meta(i + 1, i * 600, x, 0) for i, x in enumerate(runaway)]
    # After the browser's reset event the phone walks slowly in new coordinates.
    frames += [seg_meta(10 + i, 6000 + i * 600, 0.1 * i, 1) for i in range(12)]
    r = assess(frames, no_depth)
    assert r.trusted == set(range(10, 22))  # the longer, clean segment
    assert all(s in r.rejected for s in range(1, 7))
