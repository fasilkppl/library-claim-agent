"""Local speech stack: streaming VAD (Silero), speech-to-text (Whisper), text-to-speech (Kokoro).

Everything runs on the Mac. Audio in: 16 kHz mono PCM16 from the phone. Audio out: 24 kHz mono PCM16.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from rapidfuzz import fuzz

from claimagent.config import get_settings

log = logging.getLogger(__name__)

IN_RATE = 16000
CPU_FALLBACK = "small"
OUT_RATE = 24000
CHUNK = 512  # Silero works on 32 ms chunks at 16 kHz


class StreamingVad:
    """Silero VAD with state carried across chunks (the bundled helper resets state on every call)."""

    def __init__(self) -> None:
        import onnxruntime
        from faster_whisper.utils import get_assets_path

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.session = onnxruntime.InferenceSession(
            os.path.join(get_assets_path(), "silero_vad_v6.onnx"), providers=["CPUExecutionProvider"], sess_options=opts
        )
        self.reset()

    def reset(self) -> None:
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.context = np.zeros((1, 64), np.float32)

    def prob(self, chunk: np.ndarray) -> float:
        x = np.concatenate([self.context, chunk.reshape(1, -1).astype(np.float32)], axis=1)
        out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        self.context = x[:, -64:]
        return float(np.asarray(out).reshape(-1)[0])


@dataclass
class Segmenter:
    """Turns a stream of VAD probabilities into utterances, with a stricter start while the agent talks."""

    start_prob: float = 0.6
    end_prob: float = 0.35
    start_chunks: int = 4  # ~130 ms of speech starts an utterance
    end_chunks: int = 31  # ~1 s of silence ends it (people pause mid-sentence)
    barge_prob: float = 0.85
    barge_chunks: int = 8  # ~250 ms of loud speech interrupts the agent
    preroll: int = 10
    max_chunks: int = 31 * 20  # 20 s cap
    vad: StreamingVad = field(default_factory=StreamingVad)

    def __post_init__(self) -> None:
        self.buf: list[np.ndarray] = []
        self.history: list[np.ndarray] = []
        self.in_speech = False
        self.run = 0
        self.silence = 0
        self.speech_chunks = 0
        self.last_speech_ratio = 0.0  # share of the last utterance's chunks that the VAD called speech
        self.window_chunks = 0  # diagnostics: audio chunks and the highest speech probability since last read
        self.window_max_prob = 0.0
        self.pending = np.zeros(0, np.float32)

    def window(self) -> tuple[int, float]:
        """(audio chunks, highest speech probability) since the last call: is audio arriving, does it sound
        like speech at all."""
        out = (self.window_chunks, round(self.window_max_prob, 2))
        self.window_chunks, self.window_max_prob = 0, 0.0
        return out

    def feed(self, pcm16: bytes, agent_speaking: bool) -> list[tuple[str, np.ndarray | None]]:
        """Returns events: ('start', None) when speech begins, ('end', audio) when an utterance ends."""
        events: list[tuple[str, np.ndarray | None]] = []
        samples = np.frombuffer(pcm16, np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        while len(self.pending) >= CHUNK:
            chunk, self.pending = self.pending[:CHUNK], self.pending[CHUNK:]
            p = self.vad.prob(chunk)
            self.window_chunks += 1
            self.window_max_prob = max(self.window_max_prob, p)
            if not self.in_speech:
                self.history = [*self.history[-self.preroll :], chunk]
                thr, need = (
                    (self.barge_prob, self.barge_chunks) if agent_speaking else (self.start_prob, self.start_chunks)
                )
                self.run = self.run + 1 if p >= thr else 0
                if self.run >= need:
                    self.in_speech, self.silence, self.speech_chunks = True, 0, self.run
                    self.buf = list(self.history)
                    events.append(("start", None))
            else:
                self.buf.append(chunk)
                self.speech_chunks += p >= self.start_prob
                self.silence = self.silence + 1 if p < self.end_prob else 0
                if self.silence >= self.end_chunks or len(self.buf) >= self.max_chunks:
                    audio = np.concatenate(self.buf)
                    voiced = len(self.buf) - self.silence  # the trailing silence is not part of the utterance
                    self.last_speech_ratio = self.speech_chunks / max(1, voiced)
                    self.in_speech, self.run, self.buf, self.history = False, 0, [], []
                    events.append(("end", audio))
        return events


# Phrases Whisper invents on noise or silence (learned from captioned web videos).
INVENTED = re.compile(
    r"thanks? (you )?for watching|see you (in the )?next (one|video|time)|link in the description|"
    r"(like and )?subscribe|thanks for listening|subtitles? by|captions? by|amara\.org|www\.|\.com\b",
    re.I,
)
MIN_SPEECH_RATIO = 0.35  # an utterance that is mostly not speech (a fan, a clatter) is not transcribed
MIN_AVG_LOGPROB = -1.0  # Whisper's own token confidence; below it the text is mostly guessed
MAX_COMPRESSION = 2.4  # repetitive output ("Bye. Bye. Bye.") compresses far better than speech
MAX_NO_SPEECH = 0.6
MAX_NON_ENGLISH = 0.15  # share of non-English letters (þ, æ, accents) above which an English transcript is noise


@dataclass
class Transcript:
    text: str
    dropped: str = ""  # why the text was not used, empty when it was
    avg_logprob: float | None = None
    no_speech_prob: float | None = None


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def invented_reason(
    text: str, segments: list[Any], hint_words: set[str], agent_recent: str, speech_ratio: float
) -> str:
    """Why a transcript is not what the person said (empty if it looks real)."""
    words = _words(text)
    if not words:
        return "no words"
    if speech_ratio < MIN_SPEECH_RATIO:
        return f"mostly not speech (VAD {speech_ratio:.0%})"
    if INVENTED.search(text):
        return "a phrase Whisper invents on noise"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(1 for c in letters if not c.isascii()) / len(letters) > MAX_NON_ENGLISH:
        return "not English letters (Whisper inventing on noise)"
    if len(words) >= 4 and len(set(words)) / len(words) < 0.45:
        return "repeated words"
    content = [w for w in words if len(w) > 3]
    if len(content) >= 3 and sum(w in hint_words for w in content) / len(content) >= 0.7:
        return "repeats the recognition hint"
    if agent_recent and len(text) >= 12 and fuzz.partial_ratio(text.lower(), agent_recent.lower()) >= 85:
        return "the agent's own voice (echo)"
    if segments:
        n = sum(len(s.text) for s in segments) or 1
        logprob = sum(s.avg_logprob * len(s.text) for s in segments) / n
        no_speech = max(s.no_speech_prob for s in segments)
        if logprob < MIN_AVG_LOGPROB:
            return f"low confidence (avg logprob {logprob:.2f})"
        if no_speech > MAX_NO_SPEECH and logprob < -0.6:
            return f"likely silence (no-speech {no_speech:.2f})"
        if max(s.compression_ratio for s in segments) > MAX_COMPRESSION:
            return "repetitive output"
    return ""


class SpeechToText:
    _lock = threading.Lock()

    def __init__(
        self, model: str | None = None, beam_size: int | None = None, hint: str | None = None, device: str | None = None
    ) -> None:
        from faster_whisper import WhisperModel

        s = get_settings()
        self.model_name = model or s.whisper_model
        self.device = device or s.whisper_device
        # Apple GPU (CTranslate2 MPS backend) runs float16; the CPU runs int8. Without the GPU backend the large
        # model is too slow for a conversation (about 5 s per sentence), so the CPU fallback uses the small one.
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="float16" if self.device == "mps" else "int8",
                cpu_threads=s.whisper_threads,
            )
        except (RuntimeError, ValueError) as exc:
            if self.device == "cpu":
                raise
            log.warning("Whisper on %s failed (%s); falling back to %s on the CPU", self.device, exc, CPU_FALLBACK)
            self.model_name, self.device = CPU_FALLBACK, "cpu"
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8", cpu_threads=s.whisper_threads)
        self.beam_size = beam_size or s.whisper_beam_size
        # A short hint biases spelling of the few domain words; a long word list gets read back on noise.
        self.hint = s.whisper_hint if hint is None else hint
        self.hint_words = {w for w in _words(self.hint) if len(w) > 3}

    def transcribe(self, audio: np.ndarray, agent_recent: str = "", speech_ratio: float = 1.0) -> Transcript:
        if len(audio) < IN_RATE * 0.4:
            return Transcript("", "too short")
        with self._lock:
            segs, _ = self.model.transcribe(
                audio,
                language="en",
                beam_size=self.beam_size,
                vad_filter=False,
                initial_prompt=self.hint or None,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            segments = list(segs)
        text = " ".join(s.text.strip() for s in segments).strip()
        why = invented_reason(text, segments, self.hint_words, agent_recent, speech_ratio)
        n = sum(len(s.text) for s in segments) or 1
        return Transcript(
            "" if why else text,
            why,
            round(sum(s.avg_logprob * len(s.text) for s in segments) / n, 3) if segments else None,
            round(max(s.no_speech_prob for s in segments), 3) if segments else None,
        )


class TextToSpeech:
    _lock = threading.Lock()

    def __init__(self) -> None:
        from kokoro_onnx import Kokoro

        s = get_settings()
        self.voice = s.tts_voice
        self.kokoro = Kokoro(str(s.models_dir / "kokoro-v1.0.onnx"), str(s.models_dir / "voices-v1.0.bin"))

    def synth(self, text: str) -> bytes:
        with self._lock:
            audio, sr = self.kokoro.create(text, voice=self.voice, speed=1.05, lang="en-us")
        if sr != OUT_RATE:
            idx = np.linspace(0, len(audio) - 1, int(len(audio) * OUT_RATE / sr))
            audio = np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)
        pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
        return pcm.tobytes()


def sentences(stream: Iterator[str]) -> Iterator[str]:
    """Group streamed tokens into speakable sentences so speech starts before the reply is complete."""
    buf = ""
    for tok in stream:
        buf += tok
        while True:
            m = re.search(r"(.+?[.!?])(\s+|$)", buf)
            if not m or (m.end() == len(buf) and not buf.endswith((" ", "\n"))):
                break
            yield m.group(1).strip()
            buf = buf[m.end() :]
    if buf.strip():
        yield buf.strip()
