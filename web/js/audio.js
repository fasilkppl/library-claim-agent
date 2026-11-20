// Microphone in (downsampled to 16 kHz PCM16 in an AudioWorklet) and agent voice out (24 kHz PCM16).
import { sendTagged } from "./net.js";

// Converts the microphone to 16 kHz PCM16. If the context already runs at 16 kHz (the browser resampled
// properly), samples pass through; otherwise a low-pass filter runs before samples are dropped, so sounds
// above 8 kHz do not fold back into the speech band (aliasing makes speech recognition worse).
const WORKLET = `
function lowpass(cut, n) {
  const h = new Float32Array(n); let sum = 0;
  for (let i = 0; i < n; i++) {
    const k = i - (n - 1) / 2;
    const sinc = k === 0 ? 2 * cut : Math.sin(2 * Math.PI * cut * k) / (Math.PI * k);
    const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (n - 1));
    h[i] = sinc * w; sum += h[i];
  }
  for (let i = 0; i < n; i++) h[i] /= sum;
  return h;
}
class Down extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000; this.phase = 0; this.acc = [];
    this.taps = this.ratio > 1.01 ? lowpass(0.45 / this.ratio, 63) : null;
    this.ring = new Float32Array(63); this.pos = 0;
  }
  process(inputs) {
    const ch = inputs[0][0]; if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      let v = ch[i];
      if (this.taps) {
        const n = this.taps.length;
        this.ring[this.pos] = v; this.pos = (this.pos + 1) % n;
        let s = 0; for (let k = 0; k < n; k++) s += this.taps[k] * this.ring[(this.pos + k) % n];
        v = s;
      }
      this.phase += 1;
      if (this.phase >= this.ratio) { this.phase -= this.ratio; this.acc.push(v); }
    }
    if (this.acc.length >= 1600) {
      const out = new Int16Array(this.acc.length);
      for (let i = 0; i < out.length; i++) out[i] = Math.max(-1, Math.min(1, this.acc[i])) * 32767;
      this.port.postMessage(out.buffer, [out.buffer]); this.acc = [];
    }
    return true;
  }
}
registerProcessor('down', Down);`;

let playCtx = null, playHead = 0, playing = [];

let micStarted = false;
export async function startMic() {
  if (micStarted) return; // a retry or reconnect keeps the microphone already running
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  // Prefer a 16 kHz context: the browser then resamples the microphone with a proper filter.
  let micCtx, source;
  try {
    micCtx = new AudioContext({ sampleRate: 16000 });
    source = micCtx.createMediaStreamSource(stream);
  } catch {
    if (micCtx) micCtx.close();
    micCtx = new AudioContext(); // e.g. browsers that cannot mix sample rates: the worklet filters instead
    source = micCtx.createMediaStreamSource(stream);
  }
  await micCtx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET], { type: "application/javascript" })));
  const node = new AudioWorkletNode(micCtx, "down");
  node.port.onmessage = (e) => sendTagged(0x01, new Uint8Array(e.data));
  source.connect(node);
  playCtx = new AudioContext({ sampleRate: 24000 });
  micStarted = true;
}

export function playAgentAudio(buf) {
  if (new Uint8Array(buf, 0, 1)[0] !== 0x11 || !playCtx) return;
  const pcm = new Int16Array(buf.slice(1));
  const f = new Float32Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) f[i] = pcm[i] / 32768;
  const b = playCtx.createBuffer(1, f.length, 24000); b.copyToChannel(f, 0);
  const src = playCtx.createBufferSource(); src.buffer = b; src.connect(playCtx.destination);
  playHead = Math.max(playHead, playCtx.currentTime + 0.05);
  src.start(playHead); playHead += b.duration; playing.push(src);
  src.onended = () => { playing = playing.filter((s) => s !== src); };
}

export function stopPlayback() {
  for (const s of playing) { try { s.stop(); } catch { /* already stopped */ } }
  playing = []; playHead = 0;
}
