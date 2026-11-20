"""Browser end-to-end: real server, real local models, Chrome with a fake camera (a pan over shelf photos)
and a fake microphone (recorded policyholder speech).

Checks the whole live path: greeting by voice, speech recognised and turned into tool calls, the inventory
filling during the sweep, the packet built after Stop within the 5-minute budget.

Run:  CLAIM_E2E_URL=http://127.0.0.1:8000 uv run pytest -m slow tests/test_e2e_browser.py -s
(the server must be running: `uv run claim-agent`).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from make_sweep import load_photo, pan_frames, paste_marker

URL = os.environ.get("CLAIM_E2E_URL", "http://127.0.0.1:8000")
MEDIA = Path(__file__).parent / "fixtures" / "media"
REPEAT = 6


def build_video(path: Path) -> None:
    frames = []
    for i, name in enumerate(["philosophy.jpg", "german.jpg"]):
        img = load_photo(name)
        if i == 0:
            img = paste_marker(img, 420, 55)
        frames += pan_frames(img, 90, out_w=1280, seed=i)
    h = max(f.shape[0] for f in frames)
    with path.open("wb") as fh:
        for f in frames:
            canvas = np.full((h, 1280, 3), 128, np.uint8)  # letterbox, never squash
            canvas[: f.shape[0]] = f
            jpg = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()
            # Chrome plays MJPEG at ~30 fps; repeating frames gives a ~36 s pan over both shelves,
            # a realistic walking speed instead of a 6 s blur.
            fh.write(jpg * REPEAT)


def server_up() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=3) as r:
            return bool(json.loads(r.read())["ok"])
    except OSError:
        return False


@pytest.mark.slow
def test_live_sweep_end_to_end(tmp_path: Path) -> None:
    if not server_up():
        pytest.skip(f"server not running at {URL}")
    from playwright.sync_api import sync_playwright

    video = tmp_path / "shelf_pan.mjpeg"
    build_video(video)
    events: list[dict[str, object]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={video}",
                f"--use-file-for-fake-audio-capture={MEDIA / 'user_voice.wav'}",
                "--autoplay-policy=no-user-gesture-required",
            ]
        )
        page = browser.new_context(permissions=["microphone", "camera"]).new_page()
        page.on(
            "websocket",
            lambda ws: ws.on("framereceived", lambda f: events.append(json.loads(f)) if isinstance(f, str) else None),
        )
        page.goto(URL)
        page.click("#btnConnect")
        page.wait_for_selector("#btnCam:not([disabled])", timeout=30_000)
        page.wait_for_function("document.querySelector('#log').innerText.includes('Agent:')", timeout=60_000)
        page.click("#btnCam")
        t_start = time.time()
        # Sweep for ~75 s so the recorded speech (locale, skip, print, question) is heard during capture.
        page.wait_for_timeout(75_000)
        n_live = int(page.inner_text("#nBooks"))
        # The sweep as the brief describes it: shelves, then the room in the same pass, then "I'm done" (Finish).
        page.click("#guideActions button:has-text('Books done')")
        page.wait_for_timeout(5_000)
        page.click("#btnStop")
        t_stop = time.time()
        page.wait_for_selector("#packet:not([hidden])", timeout=300_000)
        t_packet = time.time() - t_stop
        browser.close()

    kinds = [e.get("type") for e in events]
    user = [e["text"] for e in events if e.get("type") == "user_text"]
    tools = [e["name"] for e in events if e.get("type") == "tool"]
    packet = next(e for e in events if e.get("type") == "packet")
    print(
        json.dumps(
            {
                "live_spines": n_live,
                "user_heard": user,
                "tools": tools,
                "packet_s": round(t_packet, 1),
                "sweep_s": round(t_stop - t_start, 1),
                "totals": packet["totals"],
            },
            indent=2,
        )
    )
    guide_steps = [e.get("step") for e in events if e.get("type") == "guide"]
    assert guide_steps[:1] == ["books"] and "room" in guide_steps, f"sweep flow not followed: {guide_steps}"
    assert "agent_text" in kinds, "agent never spoke"
    assert n_live > 0, "inventory did not fill during the sweep"
    assert user, "no speech recognised from the microphone"
    assert "set_locale" in tools
    assert t_packet < 300, "packet took longer than 5 minutes"
    sweep = packet["sweep_id"]
    with urllib.request.urlopen(f"{URL}/sweeps/{sweep}/claim_packet.json") as r:
        data = json.loads(r.read())
    assert data["totals"]["book_count"] == len(data["books"])
    with urllib.request.urlopen(f"{URL}/sweeps/{sweep}/report.html") as r:
        assert b"Contents claim" in r.read()
