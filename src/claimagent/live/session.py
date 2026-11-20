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
from claimagent.live.coach import AR_STARTING, Coach
from claimagent.live.guide import Guide
from claimagent.live.locale import COUNTRY_NAMES, CURRENCY_WORDS, confirms, country_named, guess_country
from claimagent.live.tools import FACT_TOOLS, LLM_TOOLS, rule_tool, run_tool
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

DIRECTION_WORDS = re.compile(
    r"\b(say|tap|press|point|hold|move|scan|scanning|walk|pan|shelf|shelves|next|got it|done|camera|"
    r"start with|begin with|go ahead|keep going)\b",
    re.I,
)


def strip_directions(content: str, reminder: str) -> str:
    """One voice gives directions: the guide. The language model, told not to, still invents its own ("say next
    or move on", commands that do not exist), so its sentences that direct are dropped and the guide's current
    instruction is said instead."""
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", content.strip()) if x]
    kept = [x for x in sentences if not DIRECTION_WORDS.search(x)]
    if len(kept) == len(sentences):
        return content
    log.info("LLM directions dropped: %s", [x for x in sentences if DIRECTION_WORDS.search(x)])
    return " ".join([*kept, reminder]).strip()


def spoken_title(title: str) -> str:
    """A catalogue title as said aloud: no subtitle, no bracketed notes, no "by <author>" tail."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", title.split(":")[0])
    t = re.split(r"\s+(?:by|von|de|par)\s+", t, maxsplit=1, flags=re.I)[0]
    return t.strip(" .,;-")


SWEEP_EXPLAINED = (
    "When you're ready, press start and walk the room once: slowly along every shelf, then across the walls and "
    "the floor. I'll tell you if I need a closer look."
)
_ASKS_NUMBERS = re.compile(r"\b(how many|count|counted|total|price|prices|cost|worth|value|how much)\b", re.I)
_QUESTION = re.compile(r"\?\s*$|^(can|could|are|is|do|did|does|will|would|have|has|what|how|why|where|when)\b", re.I)


def sweep_reply(text: str, reminder: str) -> str:
    """What the agent answers during the walk when the words are neither a guide command nor a fact to
    record: nothing invented, only what is true now."""
    if _ASKS_NUMBERS.search(text):
        return "I'll read you the counts and prices in the summary as soon as the sweep ends."
    if _QUESTION.search(text.strip()):
        return f"Yes. {reminder}".strip()
    return "Okay."


LIVE_LOOKUPS_MAX = 30  # live title lookups per sweep (2 catalogue queries each); the packet looks up the rest
MIC_LOG_S = 5.0  # microphone diagnostics in debug.log this often
# Longest the frame worker waits for a voice turn: longer waits leave frames to analyse after the sweep.
VOICE_YIELD_S = 2.0
SYSTEM_PROMPT = """You are the voice of a home-insurance contents-claim app. The policyholder is walking through \
their home library with the phone camera on while you inventory books and other items.
Rules:
- Reply in one or two short spoken sentences. No lists, no markdown, no emojis.
- Never state a price, a book title, a count or a measurement unless it appears in the STATE line below.
- When the policyholder states a fact, call the matching tool: country/currency -> set_locale; "skip that shelf / \
not mine" -> skip_region; "first edition / signed" -> note_book; original or print -> art_answer; a shelf width -> \
shelf_width; "I'm done / finished" -> end_sweep.
- Do not ask them to pull books out, scan barcodes, type ISBNs or photograph items one by one.
- If they ask what happens next: after the walk you identify, price and measure everything and read back a summary.
- The app's guide gives every direction (which shelf, which item, when to move on). Never tell them where to point \
the camera or what to do next; answer only what they asked."""


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
        self.guide = Guide()
        self.guide_lock = threading.Lock()  # the guide is used from the frame worker and the event loop
        self.coach.step_of = lambda: self.guide.st.step
        self.guide_opened = False
        self.ar_start_said = False
        self.processing_reply_at = 0.0
        self.spoken_titles: set[str] = set()
        self.awaiting_locale = False  # the greeting asked "is that right?" and waits for the answer
        self.mic_log_at = 0.0
        self.locale_confirmed = False
        self.explain_after_reply = False
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
        # Everything the session decides goes to sweeps/<id>/debug.log, for diagnosing a recorded sweep.
        self.debug_handler = logging.FileHandler(self.store.dir / "debug.log", encoding="utf-8")
        self.debug_handler.setLevel(logging.DEBUG)
        self.debug_handler.setFormatter(
            logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logging.getLogger("claimagent").addHandler(self.debug_handler)
        logging.getLogger("claimagent").setLevel(logging.DEBUG)
        self.frame_stats: dict[str, int] = {
            "received": 0,
            "blurry": 0,
            "duplicate": 0,
            "analysed": 0,
            "skipped_backlog": 0,
        }
        log.info("session %s created", self.store.id)

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
        log.info("SAY [%s] t=%.1fs state=%s: %s", kind, self.elapsed_ms() / 1000, self.state, text)
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
                log.debug("TTS %.2fs audio in %.2fs: %s", dur, time.perf_counter() - t, sentence[:80])
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
        # The question stands alone, so the policyholder can answer it; the sweep is explained after the answer.
        greet = (
            f"Hi, I'm your contents claim assistant. It looks like you're in {COUNTRY_NAMES.get(cc, cc)}, "
            f"so I'll value everything in {CURRENCY_WORDS.get(cur, cur)}. Is that right?"
        )
        self.awaiting_locale = True
        await self.say(greet, "greeting")

    async def start_sweep(self) -> None:
        if self.state == "sweeping":
            return
        self.state = "sweeping"
        self.awaiting_locale = False
        self.store.update_session(state="sweeping", started_monotonic=time.monotonic(), started_at=utc_now())
        self.coach.reset()
        ar = self.store.read_session().get("capture_mode") == "webxr"
        with self.guide_lock:
            self.guide = Guide()
        self.store.log_event("guide_step", self.elapsed_ms(), action="start", **self.guide.tags())
        await self.send({"type": "state", "state": "sweeping"})
        await self.send(self.guide.ui())
        if ar:
            return  # the phone reports when AR tracking is ready (on_ar_status); the guide opens then
        await self._open_guide()

    async def _open_guide(self) -> None:
        if self.guide_opened:
            return
        self.guide_opened = True
        await self.say(self.guide.opening(), "guide")

    async def guide_action(self, action: str) -> None:
        """A guide step from a button or a spoken command: one voice, one target at a time."""
        if self.state != "sweeping":
            return
        with self.guide_lock:
            speech, over = self.guide.act(action)
        self.store.log_event("guide_step", self.elapsed_ms(), action=action, **self.guide.tags())
        await self.send(self.guide.ui())
        if speech:
            await self.say(speech, "guide", interrupt=True)
        if over:
            self.end_task = asyncio.create_task(self.end_sweep())

    async def on_ar_status(self, state: str) -> None:
        """The phone holds back frames until ARCore tracks reliably; it reports each change here."""
        if self.state != "sweeping":
            return
        self.store.log_event("ar_status", self.elapsed_ms(), state=state)
        if state == "starting" and not self.ar_start_said:
            self.ar_start_said = True  # said once; later restarts show on screen only
            await self.say(AR_STARTING, "status")
        elif state == "ready":
            await self._open_guide()
        elif state == "lost":
            self.coach.on_ar_lost()

    # ------------------------------------------------------------------ audio in
    async def on_audio(self, pcm: bytes) -> None:
        now = time.monotonic()
        if now - self.mic_log_at >= MIC_LOG_S:
            self.mic_log_at = now
            chunks, peak = self.segmenter.window()
            log.info(
                "MIC t=%.1fs chunks=%s (~%.0f%% of real time) vad_max=%.2f in_speech=%s agent_speaking=%s",
                self.elapsed_ms() / 1000,
                chunks,
                100 * chunks * 0.032 / MIC_LOG_S,
                peak,
                self.segmenter.in_speech,
                self.speaking,
            )
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
        log.info("USER t=%.1fs state=%s: %s", self.elapsed_ms() / 1000, self.state, text)
        await self.send({"type": "user_text", "text": text})
        self.store.log_event("user", self.elapsed_ms(), text=text)
        if self.state == "processing":
            # Nothing is known for sure until the packet is built: a fixed answer, never an invented count.
            self.voice_busy.clear()
            if time.monotonic() - self.processing_reply_at > 20:
                self.processing_reply_at = time.monotonic()
                await self.say("I'm still building your claim packet. I'll read you the totals as soon as it's ready.")
            return
        if self.state == "intro" and self.awaiting_locale:
            self.awaiting_locale = False
            self.voice_busy.clear()
            if confirms(text):
                sess = self.store.read_session()
                args = {"country_code": sess.get("country", ""), "currency": sess.get("currency", "")}
                result = run_tool(self.store, "set_locale", args, self.elapsed_ms())
                self.locale_confirmed = True
                self.store.log_event("tool", self.elapsed_ms(), name="set_locale", args=args, result=result, by="rule")
                await self.send({"type": "tool", "name": "set_locale", "args": args, "result": result})
                await self.say(f"Great. {SWEEP_EXPLAINED}", "reply")
                return
            # A correction ("no, I'm in the US"): the dialog records it, then the sweep is explained.
            self.explain_after_reply = True
        if self.state == "sweeping":
            action = self.guide.command(text)
            if action:
                log.info("USER command -> guide %s", action)
                self.voice_busy.clear()
                await self.guide_action(action)
                return
        if self.reply_task and not self.reply_task.done():
            self.reply_task.cancel()
        self.reply_task = asyncio.create_task(self.reply(text, self.elapsed_ms()))

    def on_client_log(self, msg: dict[str, Any]) -> None:
        """Capture counters and errors from the phone: what it held back or failed to send."""
        data = {k: v for k, v in msg.items() if k != "type"}
        self.store.log_event("client", self.elapsed_ms(), **data)
        level = logging.WARNING if msg.get("type") == "client_error" else logging.INFO
        log.log(level, "PHONE %s %s", msg.get("type"), json.dumps(data)[:600])

    # ------------------------------------------------------------------ dialog
    def state_line(self) -> str:
        sess = self.store.read_session()
        items = sorted({it["label"] for d in self.det.values() for it in d.get("items", []) if it["conf"] > 0.35})
        # No running counts: live counts still include books seen twice; totals come in the final summary.
        return (
            f"STATE: phase={self.state}; guide step={self.guide.st.step}; "
            f"country={sess.get('country')}; currency={sess.get('currency')}; "
            f"items seen={', '.join(items[:12]) or 'none yet'}; counts: only in the final summary."
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
        if self.state == "sweeping":
            ruled = rule_tool(text)
            if ruled:
                name, args, said = ruled
                if name in FACT_TOOLS and self.state != "sweeping":
                    result = {"ok": False, "error": "the sweep has ended; nothing more is recorded"}
                else:
                    result = run_tool(self.store, name, args, t_ms)
                log.info("RULE tool %s %s -> %s", name, args, result)
                self.store.log_event("tool", t_ms, name=name, args=args, result=result, by="rule")
                await self.send({"type": "tool", "name": name, "args": args, "result": result})
                self.history += [{"role": "user", "content": text}]
                self.voice_busy.clear()
                await self.say(said, "reply")
                return
        if self.state == "sweeping" and not self.locale_confirmed and (confirms(text) or country_named(text)):
            # The answer to "is that right?" came after start was pressed: still recorded.
            sess = self.store.read_session()
            cc = country_named(text) or str(sess.get("country", ""))
            args = {"country_code": cc, "currency": COUNTRY_CURRENCY.get(cc, str(sess.get("currency", "")))}
            result = run_tool(self.store, "set_locale", args, t_ms)
            self.locale_confirmed = True
            self.store.log_event("tool", t_ms, name="set_locale", args=args, result=result, by="rule")
            await self.send({"type": "tool", "name": "set_locale", "args": args, "result": result})
            await self.send({"type": "locale", "country": result.get("country"), "currency": result.get("currency")})
            self.voice_busy.clear()
            await self.say(f"Noted: {COUNTRY_NAMES.get(cc, cc)}, {CURRENCY_WORDS.get(args['currency'], '')}.", "reply")
            return
        if self.state == "sweeping":
            # During the walk the language model neither speaks nor records: it invented facts here (a "first
            # edition" note for "scanning the books, okay?", a book "The Art of War" that is not in the room).
            # Facts are recorded by rule above; everything else gets a fixed, true answer.
            self.history.append({"role": "user", "content": text})
            self.voice_busy.clear()
            await self.say(sweep_reply(text, self.guide.reminder()), "reply")
            return
        self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n" + self.state_line()}, *self.history[-12:]]
        try:
            content, calls = await self._chat(messages, tools=True)
            log.info(
                "LLM reply to %r: content=%r tools=%s",
                text,
                content,
                [(c.get("function", {}).get("name"), c.get("function", {}).get("arguments")) for c in calls],
            )
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
                    self.locale_confirmed = True
                    await self.send({"type": "locale", "country": result["country"], "currency": result["currency"]})
                ended = ended or name == "end_sweep"
                messages.append({"role": "tool", "content": json.dumps(result), "tool_name": name})
            if ended:
                self.voice_busy.clear()
                if self.state == "sweeping":
                    await self.guide_action("finish")
                return
            if not content.strip():
                messages[0] = {"role": "system", "content": SYSTEM_PROMPT + "\n" + self.state_line()}
                try:
                    content, _ = await self._chat(messages, tools=False)
                except (httpx.HTTPError, RuntimeError):
                    content = "Noted."
        if self.state == "sweeping":
            content = self._without_directions(content)
        if not content.strip():
            self.voice_busy.clear()
        if content.strip():
            self.history.append({"role": "assistant", "content": content})
            await self.send({"type": "agent_text", "text": content, "kind": "reply"})
            self.store.log_event("agent", self.elapsed_ms(), text=content, agent_kind="reply")
            await self._speak_stream(self._one(content))
        if self.explain_after_reply and self.state == "intro":
            self.explain_after_reply = False
            await self.say(SWEEP_EXPLAINED, "reply")

    def _without_directions(self, content: str) -> str:
        return strip_directions(content, self.guide.reminder())

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
            body["tools"] = LLM_TOOLS
        content = ""
        calls: list[dict[str, Any]] = []
        t = time.perf_counter()
        first = True
        log.debug("LLM request tools=%s system=%r last=%r", tools, messages[0]["content"][-300:], messages[-1])
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
        log.debug("LLM done in %.2fs", time.perf_counter() - t)
        return content, calls

    # ------------------------------------------------------------------ frames
    def on_frame(self, header: dict[str, Any], jpeg: bytes, depth_raw: bytes) -> None:
        if self.state != "sweeping":
            log.debug("frame dropped: state=%s client_t=%s", self.state, header.get("t"))
            return
        self.frame_stats["received"] += 1
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
        if pose is not None and header.get("ar_trusted") is False:
            # Sent while AR was not tracking: the image is kept, its pose and depth are not (they would mislead).
            log.debug("frame client_t=%s: AR not tracking, kept as a camera frame", header.get("t"))
            pose, depth = None, None
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
            extra={
                "client_t": header.get("t"),
                "emulated": bool(header.get("emulated")),
                "turns": self.turns,
                **self.guide.tags(),
            },
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
            while self.voice_busy.is_set() and waited < VOICE_YIELD_S:
                time.sleep(0.05)
                waited += 0.05
            # Stay live: if frames pile up, analyse the newest; skipped ones are done after the sweep.
            while self.frames_q.qsize() > 3:
                nxt = self.frames_q.get()
                if nxt is None:
                    return
                log.debug("live analysis behind: frame %s left for after the sweep", meta.seq)
                self.frame_stats["skipped_backlog"] += 1
                meta, img = nxt
            try:
                self._process(meta, img)
            except Exception:
                log.exception("frame %s failed", meta.seq)

    def _process(self, meta: FrameMeta, img: np.ndarray) -> None:
        t0 = time.perf_counter()
        q = quality.measure(img)
        self.coach.on_motion(meta)
        client_t = (meta.extra or {}).get("client_t")
        tag = f"frame {meta.seq} client_t={client_t} turns={self.turns} {img.shape[1]}x{img.shape[0]}"
        if q.is_blurry(self.s.blur_min):
            self.frame_stats["blurry"] += 1
            log.debug("%s BLURRY sharpness=%.1f (min %s)", tag, q.sharpness, self.s.blur_min)
            self.coach.on_blur(True)
            return
        self.coach.on_blur(False)
        th = quality.thumb(img)
        keep, why = is_keyframe(self.prev_thumb, th, meta.t_ms - self.prev_key_t)
        if not keep:
            self.frame_stats["duplicate"] += 1
            log.debug("%s DUPLICATE view shift=%.3f sharpness=%.1f", tag, why, q.sharpness)
            return  # the post-sweep selection applies the same rule, so nothing is lost
        self.prev_thumb, self.prev_key_t = th, meta.t_ms
        if q.is_dark():
            self.coach.on_dark()
        assert self.analyzer is not None
        d = self.analyzer.analyze(meta, img)
        self.det[meta.seq] = d
        self.metas.append(meta)
        self.registrar.add_frame(meta.seq, img, d["spines"])
        self.frame_stats["analysed"] += 1
        log.info(
            "%s ANALYSED sharpness=%.1f spines=%s readable=%s texts=%s items=%s tracks=%s in %.2fs queue=%s",
            tag,
            q.sharpness,
            len(d["spines"]),
            sum(1 for sp in d["spines"] if sp.get("text")),
            [" ".join(ln.get("text", "") for ln in sp.get("text", []))[:40] for sp in d["spines"] if sp.get("text")][
                :6
            ],
            [(it["label"], round(it["conf"], 2)) for it in d.get("items", [])][:10],
            len(self.registrar.tracks),
            time.perf_counter() - t0,
            self.frames_q.qsize(),
        )
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
                elif (
                    text not in self.resolving
                    and len(re.sub(r"[^A-Za-z]", "", text)) >= 5
                    and len(self.resolving) < LIVE_LOOKUPS_MAX  # the catalogue's daily quota is shared by all sweeps
                ):
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
            r = self.resolver.resolve(text, self.s.id_accept, live=True)
            self.titles[text] = (r.title if r.status == "identified" else "", r.confidence)
        except Exception:
            self.titles[text] = ("", 0.0)
            return
        title = self.titles[text][0]
        # The agent says what it is logging: a book it has identified (confidently) during the walk.
        if title and r.confidence >= self.s.id_accept and title.lower() not in self.spoken_titles:
            self.spoken_titles.add(title.lower())
            self.coach.hint("logged", f"Logged {spoken_title(title)}.", gap=6.0, per_kind=6.0)

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
        log.info("session %s done; live frame stats %s", self.store.id, self.frame_stats)
        self._detach_debug_log()

    def _detach_debug_log(self) -> None:
        logging.getLogger("claimagent").removeHandler(self.debug_handler)
        self.debug_handler.close()

    async def close(self) -> None:
        log.info("socket closed in state %s; live frame stats %s", self.state, self.frame_stats)
        if self.state not in ("processing",):
            self._detach_debug_log()
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
