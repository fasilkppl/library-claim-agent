"""One live sweep: voice conversation, frame capture, live inventory, capture coaching, then the packet.

Threads/tasks:
- the WebSocket receive loop (audio + frames + control messages);
- a frame worker thread (per-frame analysis, live registration, coaching rules);
- the speaker task (Kokoro sentences, cancelled on barge-in);
- the pipeline thread after the sweep ends.
Coaching hints are produced by code from measurements; the language model only handles conversation.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import queue
import re
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import cv2
import httpx
import numpy as np

from claimagent.capture import quality
from claimagent.capture.orient import rotate_frame, turns_from_gravity, turns_from_pose
from claimagent.capture.store import FrameMeta, SweepStore, new_sweep_id, utc_now
from claimagent.config import Settings, get_settings
from claimagent.live.coach import AR_READY, AR_STARTING, Coach
from claimagent.live.locale import COUNTRY_NAMES, CURRENCY_WORDS, guess_country
from claimagent.live.tools import TOOLS, run_tool
from claimagent.live.voice import IN_RATE, OUT_RATE, Segmenter, SpeechToText, StreamingVad, TextToSpeech, sentences
from claimagent.pipeline.analyze import FrameAnalyzer
from claimagent.pipeline.http import CachedHttp
from claimagent.pipeline.price.sources import COUNTRY_CURRENCY
from claimagent.pipeline.register import Registrar
from claimagent.pipeline.resolve import Resolver
from claimagent.pipeline.stages import CATEGORY, consensus_text, is_keyframe

log = logging.getLogger(__name__)
PLAYBACK_TAIL_S = 0.5  # network and playback delay on the phone after the last audio chunk is sent
ECHO_WINDOW_S = 6.0  # an utterance this soon after the agent spoke is checked against its words

SYSTEM_PROMPT = """You are the voice of a home-insurance contents-claim app. The policyholder is walking through \
their home library with the phone camera on while you inventory books and other items.
Rules:
- Reply in one or two short spoken sentences. No lists, no markdown, no emojis.
- Never state a price, a book title, a count or a measurement unless it appears in the STATE line below.
- When the policyholder states a fact, call the matching tool: country/currency -> set_locale; "skip that shelf / \
not mine" -> skip_region; "first edition / signed" -> note_book; original or print -> art_answer; a shelf width -> \
shelf_width; "I'm done / finished" -> end_sweep.
- Do not ask them to pull books out, scan barcodes, type ISBNs or photograph items one by one.
- If they ask what happens next: after the walk you identify, price and measure everything and read back a summary."""


@dataclass
class Shared:
    """Models loaded once per server process."""

    stt: SpeechToText
    tts: TextToSpeech
    analyzer_models: Any = None


class LiveSession:
    def __init__(
        self,
        shared: Shared,
        send_json: Callable[[dict[str, Any]], Awaitable[None]],
        send_bytes: Callable[[bytes], Awaitable[None]],
        settings: Settings | None = None,
    ) -> None:
        self.s = settings or get_settings()
        self.shared = shared
        self._send_json, self._send_bytes = send_json, send_bytes
        self.loop = asyncio.get_running_loop()
        self.store = SweepStore(self.s.sweeps_dir, new_sweep_id())
        self.store.update_session(
            country=self.s.country,
            currency=self.s.currency,
            captured_at=utc_now(),
            capture_mode="camera",
            state="intro",
        )
        self.segmenter = Segmenter(vad=StreamingVad())
        self.history: list[dict[str, Any]] = []
        self.speaking_until = 0.0
        self.speak_task: asyncio.Task[None] | None = None
        self.reply_task: asyncio.Task[None] | None = None
        self.end_task: asyncio.Task[None] | None = None
        self.state = "intro"  # intro -> sweeping -> processing -> done
        self.frames_q: queue.Queue[tuple[FrameMeta, np.ndarray] | None] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.analyzer: FrameAnalyzer | None = None
        self.registrar = Registrar()
        self.det: dict[int, dict[str, Any]] = {}
        self.metas: list[FrameMeta] = []
        self.coach = Coach(self.s, self.store, self._may_hint, self._hint_out, self.elapsed_ms)
        self.turns = 0  # quarter turns applied to make frames upright (kept while "up" is undefined)
        self.prev_thumb: np.ndarray | None = None
        self.prev_key_t = 0
        self.titles: dict[str, tuple[str, float]] = {}
        self.resolving: set[str] = set()
        self.http = CachedHttp(self.store.cache_dir)
        self.resolver = Resolver(self.http, self.s.google_books_key)
        self.voice_metrics: dict[str, list[float]] = {
            "stt_s": [],
            "llm_first_token_s": [],
            "tts_first_audio_s": [],
            "turn_latency_s": [],
        }
        self.utterance_end = 0.0
        self.voice_busy = threading.Event()  # a voice turn is in progress: frame analysis yields to it
        self.closed = False

    # ------------------------------------------------------------------ output helpers
    async def send(self, msg: dict[str, Any]) -> None:
        if not self.closed:
            try:
                await self._send_json(msg)
            except Exception:  # client went away
                self.closed = True

    async def send_audio(self, pcm: bytes) -> None:
        if not self.closed:
            try:
                await self._send_bytes(b"\x11" + pcm)
            except Exception:
                self.closed = True

    def post(self, msg: dict[str, Any]) -> None:
        """Thread-safe send from worker threads."""
        asyncio.run_coroutine_threadsafe(self.send(msg), self.loop)

    @property
    def speaking(self) -> bool:
        # The phone is still playing for a moment after the server has sent the last audio.
        return time.monotonic() < self.speaking_until + PLAYBACK_TAIL_S

    def elapsed_ms(self) -> int:
        return self.store.elapsed_ms()

    # ------------------------------------------------------------------ speech out
    async def say(self, text: str, kind: str = "agent", interrupt: bool = False) -> None:
        if interrupt:
            await self.stop_speaking()
        await self.send({"type": "agent_text", "text": text, "kind": kind})
        self.store.log_event("agent", self.elapsed_ms(), text=text, agent_kind=kind)
        self.history.append({"role": "assistant", "content": text})
        await self._speak_stream(self._one(text))

    @staticmethod
    async def _one(text: str) -> AsyncIterator[str]:
        for sent in sentences(iter([text])):
            yield sent

    async def _speak_stream(self, chunks: AsyncIterator[str]) -> None:
        if self.speak_task and not self.speak_task.done():
            await asyncio.wait({self.speak_task})

        async def run() -> None:
            first = True
            async for sentence in chunks:
                if not re.search(r"\w", sentence):
                    continue
                t = time.perf_counter()
                pcm = await asyncio.to_thread(self.shared.tts.synth, sentence)
                if first and self.utterance_end:
                    self.voice_metrics["tts_first_audio_s"].append(round(time.perf_counter() - t, 2))
                    self.voice_metrics["turn_latency_s"].append(round(time.monotonic() - self.utterance_end, 2))
                    self.utterance_end = 0.0
                if first:
                    self.voice_busy.clear()
                first = False
                dur = len(pcm) / 2 / OUT_RATE
                self.speaking_until = max(self.speaking_until, time.monotonic()) + dur
                for i in range(0, len(pcm), OUT_RATE):  # ~0.5 s chunks
                    await self.send_audio(pcm[i : i + OUT_RATE])

        self.speak_task = asyncio.create_task(run())
        with contextlib.suppress(asyncio.CancelledError):
            await self.speak_task

    async def stop_speaking(self) -> None:
        if self.speak_task and not self.speak_task.done():
            self.speak_task.cancel()
        self.speaking_until = 0.0
        await self.send({"type": "audio_stop"})

    # ------------------------------------------------------------------ start
    async def hello(self, msg: dict[str, Any]) -> None:
        device = str(msg.get("device", ""))[:200]
        mode = "webxr" if msg.get("capture_mode") == "webxr" else "camera"
        cc = guess_country(str(msg.get("locale", "")), str(msg.get("timezone", ""))) or self.s.country
        cur = COUNTRY_CURRENCY.get(cc, self.s.currency)
        marker = msg.get("marker_side_cm")
        self.store.update_session(
            device=device,
            capture_mode=mode,
            country=cc,
            currency=cur,
            marker_side_cm=float(marker) if marker else None,
            client_locale=msg.get("locale"),
            client_timezone=msg.get("timezone"),
        )
        await self.send(
            {"type": "session", "sweep_id": self.store.id, "country": cc, "currency": cur, "capture_mode": mode}
        )
        self.analyzer = FrameAnalyzer(self.store, self.shared.analyzer_models)
        self.worker = threading.Thread(target=self._frame_worker, daemon=True)
        self.worker.start()
        greet = (
            f"Hi, I'm your contents claim assistant. It looks like you're in {COUNTRY_NAMES.get(cc, cc)}, "
            f"so I'll value everything in {CURRENCY_WORDS.get(cur, cur)}. Is that right? "
            "When you're ready, press start and walk the room once, slowly panning across every shelf, "
            "each wall, the floor and up to the ceiling. I'll tell you if I need a closer look."
        )
        await self.say(greet, "greeting")

    async def start_sweep(self) -> None:
        if self.state == "sweeping":
            return
        self.state = "sweeping"
        self.store.update_session(state="sweeping", started_monotonic=time.monotonic(), started_at=utc_now())
        self.coach.reset()
        await self.send({"type": "state", "state": "sweeping"})
        if self.store.read_session().get("capture_mode") == "webxr":
            return  # the phone reports when AR tracking is ready (on_ar_status)
        await self.say("Started. Begin with the first bookshelf, top to bottom, at about an arm's length.", "status")

    async def on_ar_status(self, state: str) -> None:
        """The phone holds back frames until ARCore tracks reliably; it reports each change here."""
        if self.state != "sweeping":
            return
        self.store.log_event("ar_status", self.elapsed_ms(), state=state)
        if state == "starting":
            await self.say(AR_STARTING, "status")
        elif state == "ready":
            await self.say(AR_READY, "status")
        elif state == "lost":
            self.coach.on_ar_lost()

    # ------------------------------------------------------------------ audio in
    async def on_audio(self, pcm: bytes) -> None:
        for ev, audio in self.segmenter.feed(pcm, self.speaking):
            if ev == "start" and self.speaking:
                await self.stop_speaking()  # barge-in
            elif ev == "end" and audio is not None:
                self.utterance_end = time.monotonic()
                self.voice_busy.set()
                t = time.perf_counter()
                ratio = self.segmenter.last_speech_ratio
                result = await asyncio.to_thread(self.shared.stt.transcribe, audio, self._agent_recent(), ratio)
                stt_s = round(time.perf_counter() - t, 2)
                self.voice_metrics["stt_s"].append(stt_s)
                # Every utterance is kept as evidence, with what was understood or why it was dropped.
                clip = await asyncio.to_thread(self.store.save_audio, audio, IN_RATE)
                self.store.log_event(
                    "utterance",
                    self.elapsed_ms(),
                    audio=clip,
                    seconds=round(len(audio) / IN_RATE, 2),
                    text=result.text,
                    dropped=result.dropped,
                    speech_ratio=round(ratio, 2),
                    avg_logprob=result.avg_logprob,
                    stt_s=stt_s,
                )
                if result.text:
                    await self.on_user_text(result.text)
                else:
                    self.voice_busy.clear()

    def _agent_recent(self) -> str:
        """What the agent said in the last few seconds: the microphone can hear it back (echo)."""
        recent = [e for e in self.history[-4:] if e["role"] == "assistant"]
        if not recent or time.monotonic() - self.speaking_until > ECHO_WINDOW_S:
            return ""
        return " ".join(str(e["content"]) for e in recent)

    async def on_user_text(self, text: str) -> None:
        await self.send({"type": "user_text", "text": text})
        self.store.log_event("user", self.elapsed_ms(), text=text)
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        self.reply_task = asyncio.create_task(self.reply(text, self.elapsed_ms()))

    # ------------------------------------------------------------------ dialog
    def state_line(self) -> str:
        sess = self.store.read_session()
        books = len(self.registrar.tracks)
        readable = sum(1 for t in self.registrar.tracks if self._track_text(t))
        items = sorted({it["label"] for d in self.det.values() for it in d.get("items", []) if it["conf"] > 0.35})
        return (
            f"STATE: phase={self.state}; country={sess.get('country')}; currency={sess.get('currency')}; "
            f"spines registered so far={books}; spines with readable text={readable}; "
            f"items seen={', '.join(items[:12]) or 'none yet'}."
        )

    async def reply(self, text: str, t_ms: int) -> None:
        try:
            await self._reply(text, t_ms)
        except asyncio.CancelledError:
            raise  # a newer utterance took over
        except Exception:
            log.exception("reply failed")
            self.voice_busy.clear()
            await self.say("Sorry, I didn't catch that. Could you say it again?", "error")

    async def _reply(self, text: str, t_ms: int) -> None:
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + self.state_line()}, *self.history[-12:]]
        try:
            content, calls = await self._chat(messages, tools=True)
        except (httpx.HTTPError, RuntimeError) as exc:
            log.warning("llm failed: %s", exc)
            self.voice_busy.clear()
            await self.say("Sorry, I didn't catch that. Could you say it again?", "error")
            return
        ended = False
        if calls:
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                result = run_tool(self.store, name, args, t_ms)
                self.store.log_event("tool", t_ms, name=name, args=args, result=result)
                await self.send({"type": "tool", "name": name, "args": args, "result": result})
                if name == "set_locale" and result.get("ok"):
                    await self.send({"type": "locale", "country": result["country"], "currency": result["currency"]})
                ended = ended or name == "end_sweep"
                messages.append({"role": "tool", "content": json.dumps(result), "tool_name": name})
            if ended:
                # Its own task: a later utterance cancels the reply task, and must not cancel the packet build.
                self.end_task = asyncio.create_task(self.end_sweep())
                self.voice_busy.clear()
                return
            if not content.strip():
                messages[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n" + self.state_line()}
                try:
                    content, _ = await self._chat(messages, tools=False)
                except (httpx.HTTPError, RuntimeError):
                    content = "Noted."
        if not content.strip():
            self.voice_busy.clear()
        if content.strip():
            self.history.append({"role": "assistant", "content": content})
            await self.send({"type": "agent_text", "text": content, "kind": "reply"})
            self.store.log_event("agent", self.elapsed_ms(), text=content, agent_kind="reply")
            await self._speak_stream(self._one(content))

    async def _chat(self, messages: list[dict[str, Any]], tools: bool) -> tuple[str, list[dict[str, Any]]]:
        body: dict[str, Any] = {
            "model": self.s.llm_model,
            "messages": messages,
            "stream": True,
            "think": False,
            "keep_alive": "60m",
            "options": {"temperature": 0.3, "num_ctx": 4096, "num_predict": 120},
        }
        if tools:
            body["tools"] = TOOLS
        content = ""
        calls: list[dict[str, Any]] = []
        t = time.perf_counter()
        first = True
        async with (
            httpx.AsyncClient(timeout=60) as client,
            client.stream("POST", f"{self.s.ollama_url}/api/chat", json=body) as r,
        ):
            if r.status_code >= 400:
                raise RuntimeError(f"ollama {r.status_code}")
            async for line in r.aiter_lines():
                if not line:
                    continue
                d = json.loads(line)
                m = d.get("message", {})
                if m.get("content"):
                    if first:
                        self.voice_metrics["llm_first_token_s"].append(round(time.perf_counter() - t, 2))
                        first = False
                    content += m["content"]
                calls += m.get("tool_calls") or []
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
        return content, calls

    # ------------------------------------------------------------------ frames
    def on_frame(self, header: dict[str, Any], jpeg: bytes, depth_raw: bytes) -> None:
        if self.state != "sweeping":
            return
        depth = None
        if depth_raw and header.get("depth_w") and header.get("depth_h"):
            mm = np.frombuffer(depth_raw, np.uint16).reshape(int(header["depth_h"]), int(header["depth_w"]))
            depth = mm.astype(np.float32) / 1000.0
            depth[mm == 0] = np.nan
        img = quality.decode_jpeg(jpeg)
        pose, intrinsics = header.get("pose"), header.get("intrinsics")
        # Frames are stored upright, however the phone was held (gravity from the AR pose or the motion sensor).
        turns = (
            turns_from_pose(pose) if pose else (turns_from_gravity(header["gravity"]) if header.get("gravity") else 0)
        )
        self.turns = self.turns if turns is None else turns
        if self.turns:
            img, depth, pose, intrinsics = rotate_frame(img, self.turns, depth, pose, intrinsics)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        meta = self.store.add_frame(
            jpeg,
            self.elapsed_ms(),
            img.shape[1],
            img.shape[0],
            pose=pose,
            intrinsics=intrinsics,
            depth=depth,
            extra={"client_t": header.get("t"), "emulated": bool(header.get("emulated")), "turns": self.turns},
        )
        self.frames_q.put((meta, img))

    def _frame_worker(self) -> None:
        while True:
            item = self.frames_q.get()
            if item is None:
                return
            meta, img = item
            # The conversation has priority: wait (bounded) while a voice turn is being answered.
            waited = 0.0
            while self.voice_busy.is_set() and waited < 8.0:
                time.sleep(0.05)
                waited += 0.05
            # Stay live: if frames pile up, analyse the newest; skipped ones are done after the sweep.
            while self.frames_q.qsize() > 3:
                nxt = self.frames_q.get()
                if nxt is None:
                    return
                meta, img = nxt
            try:
                self._process(meta, img)
            except Exception:
                log.exception("frame %s failed", meta.seq)

    def _process(self, meta: FrameMeta, img: np.ndarray) -> None:
        q = quality.measure(img)
        self.coach.on_motion(meta)
        if q.is_blurry(self.s.blur_min):
            self.coach.on_blur(True)
            return
        self.coach.on_blur(False)
        th = quality.thumb(img)
        keep, _ = is_keyframe(self.prev_thumb, th, meta.t_ms - self.prev_key_t)
        if not keep:
            return  # the post-sweep selection applies the same rule, so nothing is lost
        self.prev_thumb, self.prev_key_t = th, meta.t_ms
        if q.is_dark():
            self.coach.on_dark()
        assert self.analyzer is not None
        d = self.analyzer.analyze(meta, img)
        self.det[meta.seq] = d
        self.metas.append(meta)
        self.registrar.add_frame(meta.seq, img, d["spines"])
        self.coach.on_frame(d, len(self.registrar.tracks), self._readable_count)
        self._push_inventory()

    # ------------------------------------------------------------------ coaching output
    def _may_hint(self) -> bool:
        # Capture coaching only while the policyholder is walking, and never over speech.
        return self.state == "sweeping" and not self.speaking and not self.segmenter.in_speech

    def _hint_out(self, kind: str, text: str) -> None:
        self.post({"type": "hint", "kind": kind, "text": text})
        asyncio.run_coroutine_threadsafe(self.say(text, "hint"), self.loop)

    def _readable_count(self) -> int:
        return sum(1 for t in self.registrar.tracks if self._track_text(t))

    # ------------------------------------------------------------------ live inventory
    def _track_text(self, t: Any) -> str:
        obs = [o for o in t.observations if o.seq in self.det]
        return consensus_text(
            [self.det[o.seq]["spines"][o.index].get("text", []) for o in obs],
            [float(self.det[o.seq]["spines"][o.index]["width"]) for o in obs],
        )

    def _push_inventory(self) -> None:
        books = []
        for t in self.registrar.tracks:
            text = self._track_text(t)
            title = ""
            if text:
                if text in self.titles:
                    title = self.titles[text][0]
                elif text not in self.resolving and len(re.sub(r"[^A-Za-z]", "", text)) >= 5:
                    self.resolving.add(text)
                    threading.Thread(target=self._resolve_live, args=(text,), daemon=True).start()
            books.append(
                {"n": t.id + 1, "unit": t.segment + 1, "text": text[:80], "title": title, "seen": len(t.observations)}
            )
        items: dict[str, int] = {}
        for d in self.det.values():
            for it in d.get("items", []):
                if it["conf"] > 0.35:
                    items[it["label"]] = max(items.get(it["label"], 0), 1)
        self.post(
            {
                "type": "inventory",
                "book_count": len(books),
                "readable": sum(1 for b in books if b["text"]),
                "identified": sum(1 for b in books if b["title"]),
                "books": books[-60:],
                "items": [{"label": k, "category": CATEGORY.get(k, "other")} for k in sorted(items)],
                "frames": len(self.metas),
                "elapsed_s": round(self.elapsed_ms() / 1000),
            }
        )

    def _resolve_live(self, text: str) -> None:
        try:
            r = self.resolver.resolve(text, self.s.id_accept)
            self.titles[text] = (r.title if r.status == "identified" else "", r.confidence)
        except Exception:
            self.titles[text] = ("", 0.0)

    # ------------------------------------------------------------------ end
    async def end_sweep(self) -> None:
        if self.state != "sweeping":
            return
        self.state = "processing"
        start = self.store.read_session().get("started_monotonic") or time.monotonic()
        self.store.update_session(
            state="processing",
            duration_s=round(time.monotonic() - start, 1),
            ended_at=utc_now(),
            voice_metrics=self.voice_metrics,
        )
        await self.send({"type": "state", "state": "processing"})
        await self.say(
            "Thanks, that's the sweep. I'm finishing identification, pricing and measurements now. "
            "This takes a few minutes.",
            "status",
        )
        self.frames_q.put(None)
        self.voice_busy.clear()
        if self.worker:
            # The pipeline reads what the live worker builds: start only once it has stopped.
            waited = 0
            while self.worker.is_alive():
                await asyncio.to_thread(self.worker.join, 10)
                waited += 10
                if waited % 60 == 0 and self.worker.is_alive():
                    log.info("live frame worker still finishing after %s s", waited)
        t0 = time.monotonic()

        def progress(kind: str, data: dict[str, Any]) -> None:
            self.post({"type": "pipeline", "kind": kind, **data})

        from claimagent.pipeline.runner import run_pipeline

        try:
            packet = await asyncio.to_thread(run_pipeline, self.store, self.s, False, self.analyzer, progress)
        except Exception as exc:
            log.exception("pipeline failed")
            self.state = "failed"
            self.store.update_session(state="failed", error=str(exc)[:300])
            await self.send({"type": "error", "message": f"pipeline failed: {exc}"})
            await self.send({"type": "state", "state": "failed"})
            await self.say(
                "Something went wrong while building the packet. The frames are saved, so it can be rebuilt.", "error"
            )
            return
        self.state = "done"
        await self.send(
            {
                "type": "packet",
                "sweep_id": self.store.id,
                "seconds": round(time.monotonic() - t0, 1),
                "json": f"/sweeps/{self.store.id}/claim_packet.json",
                "report": f"/sweeps/{self.store.id}/report.html",
                "pdf": f"/sweeps/{self.store.id}/report.pdf" if (self.store.dir / "report.pdf").exists() else None,
                "totals": packet.totals.model_dump(),
            }
        )
        await self.say(summary_speech(packet), "summary")

    async def close(self) -> None:
        self.closed = True
        self.frames_q.put(None)
        if self.state == "sweeping":
            self.store.update_session(state="abandoned")


def summary_speech(p: Any) -> str:
    cur = CURRENCY_WORDS.get(p.sweep.currency, p.sweep.currency)
    t = p.totals
    parts = [
        f"Your claim packet is ready. I counted {t.book_count} books: {t.books_identified} identified, "
        f"{t.books_unidentified} I couldn't identify, and {t.books_needs_appraisal} for an expert to appraise."
    ]
    if t.books_replacement_cost:
        parts.append(
            f"The priced books come to about {t.books_replacement_cost:,.0f} {cur} to replace new, "
            f"or {t.books_used_value:,.0f} {cur} used."
        )
    else:
        parts.append("I couldn't find sourced prices for the books, so none are in the total.")
    parts.append(
        f"I also logged {len(p.items)} other items"
        + (
            f", between {t.items_replacement_cost_low:,.0f} and {t.items_replacement_cost_high:,.0f} {cur}."
            if t.items_replacement_cost_high
            else "."
        )
    )
    if p.room.floor_area_m2:
        parts.append(
            f"The floor is {p.room.floor_area_m2:.1f} square metres and the walls "
            f"{p.room.wall_area_m2:.0f} square metres."
        )
    parts.append(f"{len(p.review_queue)} lines need a human check; they're listed in the review queue.")
    return " ".join(parts)
