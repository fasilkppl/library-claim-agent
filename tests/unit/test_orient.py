import numpy as np

from claimagent.capture.orient import rotate_frame, turns_from_gravity, turns_from_pose
from claimagent.pipeline.geometry import Camera


def pose_from_axes(right, up, back, pos=(0.3, 1.4, 0.2)):
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, back, pos
    return m.T.reshape(-1).tolist()  # column-major, as WebXR sends it


UPRIGHT = pose_from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1))
# phone turned so the screen's right edge points up (rolled 90 degrees counter-clockwise)
SIDEWAYS = pose_from_axes((0, 1, 0), (-1, 0, 0), (0, 0, 1))
INTR = {"fx": 1300.0, "fy": 1310.0, "cx": 430.0, "cy": 975.0, "width": 881, "height": 1920}


def test_up_is_found_from_the_pose_and_from_gravity():
    assert turns_from_pose(UPRIGHT) == 0
    assert turns_from_pose(SIDEWAYS) == 1
    assert turns_from_gravity([0.2, 9.7, 0.5]) == 0
    assert turns_from_gravity([9.6, 0.3, 1.0]) == 1  # right edge up
    assert turns_from_gravity([-9.6, 0.3, 1.0]) == 3
    assert turns_from_gravity([0.1, -9.7, 0.0]) == 2
    looking_down = pose_from_axes((1, 0, 0), (0, 0, -1), (0, 1, 0))
    assert turns_from_pose(looking_down) is None  # undefined: the caller keeps the last turn


def test_turned_pixels_see_the_same_3d_point():
    rng = np.random.default_rng(0)
    img = np.zeros((1920, 881, 3), np.uint8)
    for k in (1, 2, 3):
        _, _, pose2, intr2 = rotate_frame(img, k, pose=SIDEWAYS, intrinsics=INTR)
        cam1, cam2 = Camera.from_meta(SIDEWAYS, INTR), Camera.from_meta(pose2, intr2)
        u = rng.uniform(0, 881, 20)
        v = rng.uniform(0, 1920, 20)
        z = rng.uniform(0.5, 3.0, 20)
        uu, vv, w, h = u.copy(), v.copy(), 881.0, 1920.0
        for _ in range(k):  # numpy.rot90, counter-clockwise: (u, v) -> (v, W - u)
            uu, vv, w, h = vv, w - uu, h, w
        assert np.allclose(cam1.backproject(u, v, z), cam2.backproject(uu, vv, z), atol=1e-6)


def test_sideways_frame_becomes_upright_and_its_depth_turns_with_it():
    img = np.zeros((1920, 881, 3), np.uint8)
    depth = np.arange(72 * 128, dtype=np.float32).reshape(128, 72)
    k = turns_from_pose(SIDEWAYS)
    img2, depth2, pose2, _ = rotate_frame(img, k, depth=depth, pose=SIDEWAYS, intrinsics=INTR)
    assert img2.shape[:2] == (881, 1920) and depth2.shape == (72, 128)
    assert turns_from_pose(pose2) == 0  # after turning, up is up
