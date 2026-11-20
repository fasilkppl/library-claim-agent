// Frame capture. WebXR (ARCore): camera image + metric pose + depth in one immersive-ar session.
// Camera-only fallback: getUserMedia frames, no pose or depth.
import { micStats } from "./audio.js";
import { backlog, sendFrame, sendJson } from "./net.js";
import { $, log, setAr, status } from "./ui.js";

const KEYFRAME_MS = 900; // a frame every 0.9 s: live analysis keeps up, fewer frames are left for after the sweep
// ARCore needs a moment of slow movement to fix its metric scale; frames are held back until then.
const AR_READY_FRAMES = 45;    // ~1.5 s of continuous (not emulated) tracking
const AR_READY_MOVE_M = 0.10;  // ... while the phone moved at least this far
const DEPTH_LONG = 128, DEPTH_SHORT = 72; // depth grid, oriented like the camera image
const MAX_BACKLOG = 4_000_000; // bytes queued on the socket before new frames are skipped

let xrSession = null, camStream = null, camTimer = null, sweeping = false, tStart = 0;

// Diagnostics: what the phone captured, held back and why. Sent to the Mac every few seconds (debug.log).
const stats = { mode: "", sent: 0, not_ready: 0, busy: 0, backlog: 0, no_camera: 0, encode_failed: 0, ar_state: "", ar_changes: [] };
let statsTimer = null;
function count(k) { stats[k] = (stats[k] || 0) + 1; }
function flushStats() {
  if (!sweeping) return;
  sendJson({ type: "client_log", t: Math.round(performance.now() - tStart), backlog_bytes: backlog(), ...stats, ...micStats(), ar_changes: stats.ar_changes.splice(0) });
}

export function startSweepClock() {
  sweeping = true; tStart = performance.now();
  if (!statsTimer) statsTimer = setInterval(flushStats, 3000);
}

export function stopCapture() {
  flushStats();
  if (statsTimer) { clearInterval(statsTimer); statsTimer = null; }
  sweeping = false;
  if (xrSession) xrSession.end();
  if (camTimer) clearInterval(camTimer);
  if (camStream) camStream.getTracks().forEach((t) => t.stop());
  window.removeEventListener("devicemotion", onMotion);
  setAr("");
}

function sampleDepth(frame, view, W, H) {
  const [dw, dh] = W >= H ? [DEPTH_LONG, DEPTH_SHORT] : [DEPTH_SHORT, DEPTH_LONG];
  try {
    const di = frame.getDepthInformation(view);
    if (!di) return null;
    const depth = new Uint16Array(dw * dh);
    for (let i = 0; i < dh; i++) for (let j = 0; j < dw; j++) {
      const d = di.getDepthInMeters((j + 0.5) / dw, (i + 0.5) / dh); // normalized view coords, top-left origin
      depth[i * dw + j] = d > 0 && d < 65 ? Math.round(d * 1000) : 0;
    }
    return { depth, dw, dh };
  } catch { return null; }
}

export async function startAR() {
  try { await setupAR(); }
  catch (e) { if (xrSession) xrSession.end().catch(() => {}); xrSession = null; document.body.classList.remove("ar"); throw e; }
}

async function setupAR() {
  xrSession = await navigator.xr.requestSession("immersive-ar", {
    requiredFeatures: ["local-floor"],
    optionalFeatures: ["camera-access", "depth-sensing", "dom-overlay", "hit-test"],
    depthSensing: { usagePreference: ["cpu-optimized"], dataFormatPreference: ["luminance-alpha", "float32"] },
    domOverlay: { root: $("overlay") },
  });
  document.body.classList.add("ar");
  const gl = document.createElement("canvas").getContext("webgl2", { xrCompatible: true, alpha: true });
  await xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
  const ref = await xrSession.requestReferenceSpace("local-floor");
  const binding = ("XRWebGLBinding" in window) ? new XRWebGLBinding(xrSession, gl) : null;
  const fb = gl.createFramebuffer();
  const c2d = $("canvas"), ctx = c2d.getContext("2d");
  let last = 0, busy = false, warned = false, arState = "", stable = 0, startPos = null;

  stats.mode = "ar";
  const setState = (s, why = "") => {
    if (s === arState) return;
    stats.ar_state = s; stats.ar_changes.push({ t: Math.round(performance.now() - tStart), state: s, why });
    arState = s; setAr(s); sendJson({ type: "ar_status", state: s });
    status({ starting: "Move the phone slowly left and right to start AR…", ready: "AR tracking. Sweep the room slowly.",
             lost: "AR lost tracking: hold still, then move slowly." }[s]);
  };

  const onFrame = (t, frame) => {
    xrSession.requestAnimationFrame(onFrame);
    const layer = xrSession.renderState.baseLayer;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
    if (!sweeping) return;
    const pose = frame.getViewerPose(ref);
    // emulatedPosition: ARCore lost the position and reports rotation only.
    // Frames are never held back: while AR is not ready they are still sent, marked untrusted (no pose or depth
    // is used from them), so a sweep never loses what the camera saw.
    let trusted = true;
    if (!pose || pose.emulatedPosition) {
      stable = 0; startPos = null; trusted = false;
      count("not_ready");
      setState(arState === "ready" ? "lost" : "starting", !pose ? "no pose" : "position emulated (tracking lost)");
      if (!pose) return; // no view to read the camera from
    } else {
      const at = pose.transform.position;
      stable++; startPos = startPos || [at.x, at.y, at.z];
      if (arState !== "ready") {
        const moved = Math.hypot(at.x - startPos[0], at.y - startPos[1], at.z - startPos[2]);
        if (stable >= AR_READY_FRAMES && moved >= AR_READY_MOVE_M) setState("ready", `stable ${stable} frames, moved ${moved.toFixed(2)} m`);
        else { count("not_ready"); trusted = false; setState(arState === "lost" ? "lost" : "starting", "waiting for steady tracking"); }
      }
    }
    if (t - last < KEYFRAME_MS) return;
    if (busy) { count("busy"); return; }
    if (backlog() > MAX_BACKLOG) { count("backlog"); return; }
    const view = pose.views[0];
    if (!view.camera || !binding) {
      count("no_camera");
      if (!warned) { warned = true; log("This browser gives no camera image in AR (camera-access). Use the camera-only sweep.", "h"); }
      return;
    }
    last = t; busy = true;
    const W = view.camera.width, H = view.camera.height;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, binding.getCameraImage(view.camera), 0);
    const px = new Uint8ClampedArray(W * H * 4);
    gl.readPixels(0, 0, W, H, gl.RGBA, gl.UNSIGNED_BYTE, px);
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    const img = new ImageData(W, H), row = W * 4; // GL rows are bottom-up; images are top-down
    for (let y = 0; y < H; y++) img.data.set(px.subarray((H - 1 - y) * row, (H - y) * row), y * row);
    const p = view.projectionMatrix; // column-major; intrinsics in top-left image pixel coordinates
    const intrinsics = { fx: p[0] * W / 2, fy: p[5] * H / 2, cx: (1 - p[8]) * W / 2, cy: (1 + p[9]) * H / 2, width: W, height: H };
    const d = sampleDepth(frame, view, W, H);
    const header = { t: Math.round(t - tStart), pose: Array.from(view.transform.matrix), intrinsics,
                     emulated: !!pose.emulatedPosition, ar_trusted: trusted, depth_w: d ? d.dw : 0, depth_h: d ? d.dh : 0 };
    c2d.width = W; c2d.height = H; ctx.putImageData(img, 0, 0);
    c2d.toBlob(async (blob) => {
      try { if (blob) { sendFrame(header, await blob.arrayBuffer(), d && d.depth); count("sent"); } else count("encode_failed"); }
      catch (e) { count("encode_failed"); console.warn("frame not sent", e); }
      finally { busy = false; } // one failed frame must not stop capture for the rest of the sweep
    }, "image/jpeg", 0.92);
  };
  xrSession.requestAnimationFrame(onFrame);
  xrSession.addEventListener("end", () => { document.body.classList.remove("ar"); xrSession = null; });
}

// World-up from the motion sensor, in screen coordinates (the server turns sideways frames upright with it).
let gravity = null;
function onMotion(e) {
  const g = e.accelerationIncludingGravity;
  if (!g || g.x == null || g.y == null || g.z == null) return;
  const a = ((screen.orientation && screen.orientation.angle) || 0) * Math.PI / 180;
  gravity = [g.x * Math.cos(a) - g.y * Math.sin(a), g.x * Math.sin(a) + g.y * Math.cos(a), g.z];
}

export async function startCamera() {
  // Full resolution with continuous autofocus where the browser supports it (the AR camera stream has neither).
  // The full 4:3 sensor at 30 fps: a 16:9 or 60 fps stream is a crop of the sensor and looks zoomed in.
  camStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment", aspectRatio: { ideal: 4 / 3 }, width: { ideal: 1920 }, height: { ideal: 1440 },
      frameRate: { ideal: 30, max: 30 }, advanced: [{ focusMode: "continuous" }],
    },
  });
  const camTrack = camStream.getVideoTracks()[0];
  try {
    const caps = camTrack.getCapabilities ? camTrack.getCapabilities() : {};
    // 1x: the main lens uncropped. Below 1x some phones switch to the ultra-wide lens, which blurs spine text.
    if (caps.zoom) await camTrack.applyConstraints({ advanced: [{ zoom: Math.max(caps.zoom.min, Math.min(1, caps.zoom.max)) }] });
  } catch (e) { console.warn("zoom not set", e); }
  window.addEventListener("devicemotion", onMotion);
  stats.mode = "camera";
  const v = $("video"); v.srcObject = camStream; v.hidden = false; v.style.width = "100%";
  const set = camTrack && camTrack.getSettings ? camTrack.getSettings() : {};
  const caps = camTrack && camTrack.getCapabilities ? camTrack.getCapabilities() : {};
  sendJson({ type: "client_log", event: "camera_started", width: set.width, height: set.height, focusMode: set.focusMode,
             frameRate: set.frameRate, zoom: set.zoom, zoom_range: caps.zoom ? [caps.zoom.min, caps.zoom.max] : null,
             aspect: set.aspectRatio, label: camTrack ? camTrack.label : "" });
  try { await v.play(); }
  catch (e) { camStream.getTracks().forEach((t) => t.stop()); camStream = null; throw e; }
  const c = $("canvas"), ctx = c.getContext("2d");
  let encoding = false;
  camTimer = setInterval(() => {
    if (!sweeping) return;
    if (!v.videoWidth) { count("no_camera"); return; }
    if (encoding) { count("busy"); return; }
    if (backlog() > MAX_BACKLOG) { count("backlog"); return; }
    encoding = true;
    c.width = v.videoWidth; c.height = v.videoHeight; ctx.drawImage(v, 0, 0);
    const header = { t: Math.round(performance.now() - tStart), gravity };
    c.toBlob(async (b) => {
      try { if (b) { sendFrame(header, await b.arrayBuffer(), null); count("sent"); } else count("encode_failed"); }
      catch (e) { count("encode_failed"); console.warn("frame not sent", e); }
      finally { encoding = false; }
    }, "image/jpeg", 0.92);
  }, KEYFRAME_MS);
}
