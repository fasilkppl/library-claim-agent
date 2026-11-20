// One WebSocket per session. Text = JSON control messages; binary = tagged media:
//   0x01 mic audio (PCM16, 16 kHz) · 0x02 frame (uint32 header length, header JSON, JPEG, uint16 depth mm)
//   0x11 agent audio from the server (PCM16, 24 kHz)

let ws = null;

export function connect({ onJson, onBinary, onClose }) {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => resolve();
    ws.onerror = (e) => reject(e);
    ws.onclose = onClose;
    ws.onmessage = (ev) => (typeof ev.data === "string" ? onJson(JSON.parse(ev.data)) : onBinary(ev.data));
  });
}

// Bytes still waiting to go out: frames are skipped while the link is slow instead of piling up in memory.
export const backlog = () => (ws ? ws.bufferedAmount : 0);

export const sendJson = (o) => ws && ws.readyState === 1 && ws.send(JSON.stringify(o));

export function sendTagged(tag, ...parts) { // parts: Uint8Array
  if (!ws || ws.readyState !== 1) return;
  const out = new Uint8Array(1 + parts.reduce((a, p) => a + p.byteLength, 0));
  out[0] = tag; let o = 1;
  for (const p of parts) { out.set(p, o); o += p.byteLength; }
  ws.send(out.buffer);
}

export function sendFrame(header, jpeg, depth) {
  header.jpeg_len = jpeg.byteLength;
  const h = new TextEncoder().encode(JSON.stringify(header));
  const len = new Uint8Array(4); new DataView(len.buffer).setUint32(0, h.byteLength, true);
  const parts = [len, h, new Uint8Array(jpeg)];
  if (depth) parts.push(new Uint8Array(depth.buffer, depth.byteOffset, depth.byteLength));
  sendTagged(0x02, ...parts);
}
