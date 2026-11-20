from dataclasses import dataclass

import numpy as np

from claimagent.live.voice import CHUNK, Segmenter, invented_reason


@dataclass
class Seg:
    text: str
    avg_logprob: float = -0.3
    no_speech_prob: float = 0.05
    compression_ratio: float = 1.3


HINT = {"home", "library", "contents", "claim"}


def check(text, segs=None, agent="", ratio=0.9):
    return invented_reason(text, segs if segs is not None else [Seg(text)], HINT, agent, ratio)


def test_real_speech_passes():
    assert check("Skip that shelf, those books are not mine.") == ""


def test_text_whisper_invents_on_noise_is_dropped():
    assert check("Thank you for watching. We'll see you in the next one.")
    assert check("You can find the link in the description.")
    assert check("Bye. Bye. Bye. Bye. Bye. Bye.")
    assert check("Home library contents claim.")  # the recognition hint read back
    assert check("Okay, sure.", [Seg("Okay, sure.", avg_logprob=-1.4)])  # Whisper itself unsure
    assert check("Skip that shelf.", ratio=0.1)  # mostly not speech


def test_agents_own_voice_heard_back_is_dropped():
    agent = "Slow down a little, the picture is blurry."
    assert check("slow down a little the picture is", agent=agent) == "the agent's own voice (echo)"
    assert check("The painting is a print.", agent=agent) == ""


class FakeVad:
    def __init__(self, probs):
        self.probs = iter(probs)

    def prob(self, chunk):
        return next(self.probs)


def feed(probs):
    seg = Segmenter(vad=FakeVad(probs))
    events = []
    silence = np.zeros(CHUNK, np.int16).tobytes()
    for _ in probs:
        events += seg.feed(silence, agent_speaking=False)
    return seg, [e for e, _ in events]


def test_a_short_pause_does_not_cut_the_sentence():
    # 0.8 s of speech, a 0.8 s pause (25 chunks), more speech, then 1.2 s of silence
    seg, events = feed([0.9] * 25 + [0.1] * 25 + [0.9] * 15 + [0.1] * 38)
    assert events == ["start", "end"]
    assert seg.last_speech_ratio > 0.5


def test_noise_that_barely_triggers_the_vad_has_a_low_speech_ratio():
    seg, events = feed([0.7] * 4 + [0.4, 0.2] * 20 + [0.1] * 35)
    assert events == ["start", "end"] and seg.last_speech_ratio < 0.35


def test_non_english_letters_are_noise():
    from claimagent.live.voice import invented_reason

    class Seg:
        def __init__(self, t):
            self.text, self.avg_logprob, self.compression_ratio, self.no_speech_prob = t, -0.3, 1.0, 0.1

    noise = "Hél hér, Gúl, Þjón, Vatnsæl, Æsjó, Þjó, Þjó, Þjó."
    assert invented_reason(noise, [Seg(noise)], set(), "", 0.7).startswith("not English")
    ok = "Books are done. Now I'm moving on to other items."
    assert invented_reason(ok, [Seg(ok)], set(), "", 0.7) == ""
