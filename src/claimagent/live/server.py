"""FastAPI app: phone web client, WebSocket for the live sweep, and the saved sweeps (packet, report, frames).

WebSocket protocol
  client -> server
    text   {"type":"hello", locale, timezone, device, capture_mode, marker_side_cm?}
           {"type":"ar_status", state: starting | ready | lost}
           {"type":"start"} | {"type":"stop"} | {"type":"text","text":...}
    binary 0x01 + PCM16 mono 16 kHz audio
           0x02 + uint32 header length (LE) + header JSON + JPEG [+ depth uint16 millimetres]
  server -> client
    text   session, agent_text, user_text, hint, inventory, tool, locale, state, pipeline, packet, audio_stop, error
    binary 0x11 + PCM16 mono 24 kHz agent speech
Binds to 127.0.0.1; the phone reaches it through `adb reverse tcp:8000 tcp:8000` (a secure context for
WebXR and the microphone without certificates).
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from collections.abc import AsyncIterator, Coroutine
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from claimagent.config import get_settings
from claimagent.live.session import LiveSession, Shared

log = logging.getLogger(__name__)
MAX_FRAME_BYTES = 8_000_000
_shared: Shared | None = None


async def _load_shared() -> Shared:
    from claimagent.live.voice import SpeechToText, TextToSpeech
    from claimagent.pipeline.analyze import load_models

    stt, tts, models = await asyncio.gather(
        asyncio.to_thread(SpeechToText), asyncio.to_thread(TextToSpeech), asyncio.to_thread(load_models)
    )
    return Shared(stt=stt, tts=tts, analyzer_models=models)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _shared
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its request lines carry API keys
    _shared = await _load_shared()
    await asyncio.to_thread(_warm_llm)
    await asyncio.to_thread(_warm_stt, _shared)
    log.info("models loaded; open http://localhost:8000")
    yield


def _warm_stt(shared: Shared) -> None:
    """The first transcriptions on the Apple GPU are several times slower (kernels are prepared on first use):
    run a few utterances of different lengths now, so the policyholder's first sentences are answered fast."""
    import time

    import numpy as np

    from claimagent.live.voice import IN_RATE, OUT_RATE

    t0 = time.perf_counter()
    for text in ("Hello.", "Skip that shelf, those books are not mine.", "This one is a signed first edition, please."):
        pcm = np.frombuffer(shared.tts.synth(text), np.int16).astype(np.float32) / 32768.0
        audio = np.interp(np.arange(0, len(pcm), OUT_RATE / IN_RATE), np.arange(len(pcm)), pcm).astype(np.float32)
        shared.stt.transcribe(audio)
    log.info("speech recognition warmed up in %.1f s", time.perf_counter() - t0)


def _warm_llm() -> None:
    import httpx

    s = get_settings()
    try:
        httpx.post(
            f"{s.ollama_url}/api/chat",
            timeout=120,
            json={
                "model": s.llm_model,
                "stream": False,
                "think": False,
                "keep_alive": "60m",
                "messages": [{"role": "user", "content": "Say ok."}],
                "options": {"num_predict": 3},
            },
        )
    except httpx.HTTPError as exc:
        log.warning("LLM warm-up failed (is Ollama running? `ollama serve`): %s", exc)


class RevalidatedFiles(StaticFiles):
    """Static files the browser must revalidate on every load (ETag), so an updated UI shows at once."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app = FastAPI(title="Library contents claim agent", lifespan=lifespan)
settings = get_settings()
settings.sweeps_dir.mkdir(parents=True, exist_ok=True)
app.mount("/sweeps", StaticFiles(directory=settings.sweeps_dir, html=True), name="sweeps")
app.mount("/web", RevalidatedFiles(directory=settings.web_dir), name="web")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/manifest.webmanifest")
def manifest() -> FileResponse:
    return FileResponse(settings.web_dir / "manifest.webmanifest", media_type="application/manifest+json")


@app.get("/sw.js")
def service_worker() -> FileResponse:
    # Served from the root so its scope covers the whole app; never cached, so updates apply at once.
    return FileResponse(settings.web_dir / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health() -> JSONResponse:
    s = get_settings()
    return JSONResponse(
        {
            "ok": _shared is not None,
            "llm": s.llm_model,
            "price_sources": {
                "serpapi": bool(s.serpapi_key),
                "serper": bool(s.serper_key),
                "ebay": bool(s.ebay_client_id),
                "google_books_key": bool(s.google_books_key),
            },
        }
    )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    assert _shared is not None
    session = LiveSession(_shared, ws.send_json, ws.send_bytes)
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            try:
                if msg.get("bytes") is not None:
                    await _binary(session, msg["bytes"])
                elif msg.get("text") is not None:
                    await _text(session, msg["text"])
            except WebSocketDisconnect:
                raise
            except Exception:  # one bad frame or utterance is logged and skipped; the sweep goes on
                log.exception("message handling failed; continuing")
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()


async def _binary(session: LiveSession, data: bytes) -> None:
    if not data:
        return
    tag, body = data[0], data[1:]
    if tag == 0x01:
        if len(body) % 2 == 0:
            await session.on_audio(body)
    elif tag == 0x02 and 4 < len(body) < MAX_FRAME_BYTES:
        (hlen,) = struct.unpack("<I", body[:4])
        header: dict[str, Any] = json.loads(body[4 : 4 + hlen])
        rest = body[4 + hlen :]
        jlen = int(header.get("jpeg_len", len(rest)))
        jpeg, depth = rest[:jlen], rest[jlen:]
        if jpeg[:2] == b"\xff\xd8":
            await asyncio.to_thread(session.on_frame, header, jpeg, depth)


_tasks: set[asyncio.Task[None]] = set()


def _spawn(coro: Coroutine[Any, Any, None]) -> None:
    """Run a session action without blocking the receive loop; keep a reference so it is not collected."""
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


async def _text(session: LiveSession, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except ValueError:
        return
    kind = msg.get("type")
    if kind == "hello":
        _spawn(session.hello(msg))
    elif kind == "start":
        _spawn(session.start_sweep())
    elif kind == "stop":
        _spawn(session.end_sweep())
    elif kind == "ar_status" and msg.get("state") in ("starting", "ready", "lost"):
        _spawn(session.on_ar_status(str(msg["state"])))
    elif kind == "text" and isinstance(msg.get("text"), str) and msg["text"].strip():
        await session.on_user_text(msg["text"].strip()[:500])


def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser(prog="claim-agent", description="Live library contents claim agent")
    ap.add_argument("--host", default="127.0.0.1", help="bind address (keep 127.0.0.1; use adb reverse for the phone)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    uvicorn.run("claimagent.live.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
