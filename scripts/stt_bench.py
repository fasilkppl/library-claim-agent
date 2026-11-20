"""Speech-to-text benchmark: word error rate on spoken commands, and text invented on noise.

    uv run python scripts/stt_bench.py --synthetic            # macOS Indian-English voices + room noise
    uv run python scripts/stt_bench.py --clips sweeps/<id>    # real utterances saved by the app, scored
                                                              # against sweeps/<id>/audio/expected.tsv

Synthetic voices (macOS `say`: Aman, Rishi, Tara) are a stand-in for real recordings: they show relative
differences between configurations, not the absolute error rate on a real phone in a real room.
Each configuration runs the full chain: 48 kHz capture -> conversion to 16 kHz (the phone's job) -> VAD
speech share -> Whisper -> invented-text filter.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from claimagent.live.voice import IN_RATE, SpeechToText, StreamingVad, _words

PHRASES = [
    "Yes, India and rupees is correct.",
    "No, I live in the United States, please use dollars.",
    "Skip that shelf, those books are not mine.",
    "This one is a signed first edition.",
    "The painting on the wall is just a print.",
    "That portrait is an original oil painting.",
    "The top shelf belongs to my roommate.",
    "I'm done with the sweep.",
    "Can you hear me clearly?",
    "What happens after I finish walking?",
    "This book was my grandfather's, please be careful with it.",
    "The lamp in the corner is brand new.",
    "Should I move closer to the shelf?",
    "Everything on this table is mine.",
    "Please read the summary again.",
]
VOICES = ["Aman", "Rishi", "Tara"]
AGENT_LINES = [
    "Slow down a little, the picture is blurry.",
    "I can see a framed picture. Is it an original or a print?",
]
OLD_HINT = (
    "Home library contents claim. Bookshelf, first edition, signed copy, print, original, "
    "painting, rupees, dollars, India, skip that shelf."
)


@dataclass
class Clip:
    audio48: np.ndarray
    expected: str  # "" = nothing was said: any text is invented
    agent_recent: str = ""


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        rate, n = w.getframerate(), w.getnframes()
        x = np.frombuffer(w.readframes(n), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(axis=1)
    return x, rate


def say(text: str, voice: str, folder: Path) -> np.ndarray:
    out = folder / "s.wav"
    subprocess.run(
        ["say", "-v", voice, "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@48000", text], check=True
    )
    x, _ = read_wav(out)
    return x


def to16_nearest(x48: np.ndarray) -> np.ndarray:
    """The old phone conversion: every third sample, no filter (aliasing)."""
    return x48[::3].copy()


def lowpass_taps(cutoff: float, n: int = 63) -> np.ndarray:
    """Windowed-sinc low-pass FIR; cutoff as a fraction of the sample rate."""
    k = np.arange(n) - (n - 1) / 2
    h = np.sinc(2 * cutoff * k) * np.blackman(n)
    return h / h.sum()


def to16_filtered(x48: np.ndarray) -> np.ndarray:
    """A proper conversion: low-pass below the new Nyquist frequency, then every third sample."""
    return np.convolve(x48, lowpass_taps(7200 / 48000), mode="same")[::3].astype(np.float32)


def noise(kind: str, n: int, rng: np.random.Generator) -> np.ndarray:
    white = rng.standard_normal(n).astype(np.float32)
    if kind == "white":
        return white
    if kind == "brown":  # fan / air-conditioner rumble
        b = np.cumsum(white)
        b -= np.convolve(b, np.ones(4801) / 4801, mode="same")  # remove drift
        return (b / (np.abs(b).max() + 1e-9)).astype(np.float32)
    if kind == "clatter":  # kitchen clicks
        x = np.zeros(n, np.float32)
        for at in rng.integers(0, n - 2000, 6):
            x[at : at + 2000] += rng.standard_normal(2000).astype(np.float32) * np.exp(-np.arange(2000) / 200)
        return x
    raise ValueError(kind)


def mix(speech: np.ndarray, kind: str, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    pad = np.zeros(int(0.4 * 48000), np.float32)
    s = np.concatenate([pad, speech, pad])
    nz = noise(kind, len(s), rng)
    ps, pn = np.mean(s**2) + 1e-12, np.mean(nz**2) + 1e-12
    return (s + nz * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def speech_ratio(audio16: np.ndarray, vad: StreamingVad) -> float:
    vad.reset()
    chunks = [audio16[i : i + 512] for i in range(0, len(audio16) - 511, 512)]
    probs = [vad.prob(c) for c in chunks]
    return sum(p >= 0.6 for p in probs) / max(1, len(probs))


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def old_transcribe(stt: SpeechToText, audio16: np.ndarray) -> str:
    """The transcription as it was before the speech fixes: greedy, word-list prompt, one regex filter."""
    if len(audio16) < IN_RATE * 0.3:
        return ""
    segs, _ = stt.model.transcribe(
        audio16,
        language="en",
        beam_size=1,
        vad_filter=False,
        initial_prompt=OLD_HINT,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
    if re.fullmatch(r"(thank you\.?|thanks for watching\.?|you|\.|bye\.?)", text.lower()):
        return ""
    return text


def synthetic_clips(seed: int = 7) -> list[Clip]:
    rng = np.random.default_rng(seed)
    clips: list[Clip] = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        for v in VOICES:
            for k, text in enumerate(PHRASES):
                kind = ["brown", "white", "clatter"][k % 3]
                clips.append(Clip(mix(say(text, v, folder), kind, 10.0, rng), text))
        for kind, level in [("white", 0.003), ("brown", 0.02), ("clatter", 0.05), ("brown", 0.06)]:
            for _ in range(3):
                clips.append(Clip(noise(kind, 3 * 48000, rng) * level, ""))
        for line in AGENT_LINES:  # the agent's own voice heard back by the microphone
            clips.append(Clip(mix(say(line, "Samantha", folder), "brown", 15.0, rng) * 0.5, "", agent_recent=line))
    return clips


def real_clips(sweep: Path) -> list[Clip]:
    rows = list(csv.reader((sweep / "audio" / "expected.tsv").open(), delimiter="\t"))
    out = []
    for name, expected in rows:
        x, rate = read_wav(sweep / "audio" / name)
        x48 = np.interp(np.arange(0, len(x), rate / 48000), np.arange(len(x)), x).astype(np.float32)
        out.append(Clip(x48, expected))
    return out


def run(configs: list[tuple[str, str, int, str, bool, str]], clips: list[Clip], device: str = "cpu") -> None:
    vad = StreamingVad()
    loaded: dict[str, SpeechToText] = {}
    print(
        f"{len([c for c in clips if c.expected])} spoken clips, {len([c for c in clips if not c.expected])} noise/echo"
    )
    print("| configuration | audio path | word error rate | invented text on noise/echo | s per clip |")
    print("|---|---|---|---|---|")
    for label, model, beam, hint, new_filter, path in configs:
        if model not in loaded:
            loaded[model] = SpeechToText(model=model, beam_size=beam, hint=hint, device=device)
        stt = loaded[model]
        stt.beam_size, stt.hint = beam, hint
        stt.hint_words = {w for w in _words(hint) if len(w) > 3}
        errs = refs = invented = 0
        t0 = time.perf_counter()
        for c in clips:
            a16 = to16_filtered(c.audio48) if path == "filtered" else to16_nearest(c.audio48)
            if new_filter:
                text = stt.transcribe(a16, c.agent_recent, speech_ratio(a16, vad)).text
            else:
                text = old_transcribe(stt, a16)
            if c.expected:
                e, n = wer(c.expected, text)
                errs, refs = errs + e, refs + n
            elif text:
                invented += 1
        per = (time.perf_counter() - t0) / len(clips)
        n_noise = len([c for c in clips if not c.expected])
        print(f"| {label} | {path} | {errs / max(1, refs):.1%} | {invented}/{n_noise} | {per:.2f} |", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--clips", type=Path, help="sweep folder with audio/expected.tsv")
    ap.add_argument("--models", default="small,large-v3-turbo")
    ap.add_argument("--device", default="cpu", help="cpu or mps (Apple GPU)")
    ap.add_argument("--no-baseline", action="store_true")
    args = ap.parse_args()
    clips = real_clips(args.clips) if args.clips else synthetic_clips()
    hint = "Home library contents claim."
    configs = (
        []
        if args.no_baseline
        else [("before: small, greedy, word-list hint, old filter", "small", 1, OLD_HINT, False, "nearest")]
    )
    for m in args.models.split(","):
        configs.append((f"after: {m}, beam 5, short hint, new filter", m, 5, hint, True, "nearest"))
        configs.append((f"after: {m}, beam 5, short hint, new filter", m, 5, hint, True, "filtered"))
    run(configs, clips, args.device)


if __name__ == "__main__":
    main()
