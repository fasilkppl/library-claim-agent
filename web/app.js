// Phone client entry point: wires the page, the socket, audio and capture together.
import { playAgentAudio, startMic, stopPlayback } from "./js/audio.js";
import { startAR, startCamera, startSweepClock, stopCapture } from "./js/capture.js";
import { connect, sendJson } from "./js/net.js";
import {
  $, hint, initTabs, log, renderInventory, renderPacket, setConnected, setLocale, setPhase, status,
} from "./js/ui.js";

const MARKER_CM = 10;
let arSupported = false;

function onJson(m) {
  switch (m.type) {
    case "session":
      $("sweepId").textContent = m.sweep_id; setLocale(m.country, m.currency);
      status("Connected. Answer the agent, then start the sweep."); break;
    case "agent_text": log(`Agent: ${m.text}`, m.kind === "hint" ? "h" : ""); if (m.kind === "hint") hint(m.text); break;
    case "user_text": log(`You: ${m.text}`, "u"); break;
    case "hint": hint(m.text); break;
    case "locale": setLocale(m.country, m.currency); break;
    case "tool": log(`[recorded ${m.name}]`, "s"); break;
    case "audio_stop": stopPlayback(); break;
    case "inventory": renderInventory(m); break;
    case "state":
      if (m.state === "processing") { stopCapture(); setPhase("processing"); status("Sweep ended. Building the claim packet…"); }
      if (m.state === "failed") { setPhase("failed"); status("The packet could not be built. The frames are saved; reload the page to start again."); }
      break;
    case "pipeline":
      if (m.kind === "stage") status(`Building the packet: ${m.stage}…`);
      if (m.kind === "analyze") status(`Analysing frames ${m.done}/${m.total}…`);
      break;
    case "packet": finished = true; setPhase("done"); renderPacket(m); status("Done. The agent is reading the summary."); break;
    case "error": log(`Error: ${m.message}`, "h"); status(m.message); break;
  }
}

let finished = false; // a packet arrived: a later disconnect keeps it on screen
function onClose() {
  setConnected(false);
  stopCapture(); // nothing more can reach the Mac: stop the camera instead of capturing into nothing
  if (finished) { status("Disconnected. The packet above is saved on the Mac."); return; }
  setPhase("idle");
  status("Disconnected from the Mac. Check the USB cable and that the server runs, then press Connect.");
}

$("btnConnect").onclick = async () => {
  $("btnConnect").disabled = true;
  try {
    await startMic();
    await connect({ onJson, onBinary: playAgentAudio, onClose });
    setConnected(true);
    arSupported = !!(navigator.xr && await navigator.xr.isSessionSupported("immersive-ar").catch(() => false));
    sendJson({ type: "hello", locale: navigator.language, timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
               device: navigator.userAgent, capture_mode: arSupported ? "webxr" : "camera",
               marker_side_cm: $("chkMarker").checked ? MARKER_CM : null });
    setPhase("connected", { ar: arSupported });
    if (!arSupported) log("AR is not available on this device: camera-only sweep (no room measurement).", "s");
  } catch (e) { status(`Could not start: ${e}`); $("btnConnect").disabled = false; }
};

async function begin(kind) {
  setPhase("sweeping");
  try { kind === "ar" ? await startAR() : await startCamera(); }
  catch (e) { log(`Could not start the ${kind} sweep: ${e}`, "h"); setPhase("connected", { ar: arSupported }); return; }
  startSweepClock(); sendJson({ type: "start" });
}
$("btnAR").onclick = () => begin("ar");
$("btnCam").onclick = () => begin("camera");
$("btnStop").onclick = () => { stopCapture(); sendJson({ type: "stop" }); };
$("typed").onsubmit = (e) => {
  e.preventDefault();
  const t = $("typedText").value.trim();
  if (t) { sendJson({ type: "text", text: t }); $("typedText").value = ""; }
};

// Installable app: Chrome offers installation once the manifest and service worker are in place.
let installPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => { e.preventDefault(); installPrompt = e; $("btnInstall").hidden = false; });
$("btnInstall").onclick = async () => { if (!installPrompt) return; installPrompt.prompt(); await installPrompt.userChoice; installPrompt = null; $("btnInstall").hidden = true; };
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});

initTabs();
setPhase("idle");
