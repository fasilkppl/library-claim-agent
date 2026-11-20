# Library Contents Claim Agent

A live voice-and-vision agent that inventories and values a home library from **one continuous camera
sweep**. The policyholder walks the room with an Android phone while talking to the agent. The agent coaches
the capture as it happens ("slow down", "step closer", "is that portrait an original or a print?"). When the
walk ends, the system writes a claim packet: every book read from its spine, measured and priced locally;
every other item classified and priced; and the room's floor and wall area.

Everything runs **locally on a Mac with Apple Silicon**: speech, dialog, vision and OCR. The only network
calls are catalogue lookups and price retrieval, and every one is saved as evidence.

- Architecture (one page): [docs/architecture.md](docs/architecture.md)
- Failure log (one page): [docs/failure_log.md](docs/failure_log.md)
- Ground-truth templates: [ground_truth/template/](ground_truth/template/); results come from `make eval`

## Quick start (about 15 minutes, most of it downloads)

Requirements: macOS 14+ on Apple Silicon with 16 GB RAM or more, [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com/download), Android Platform Tools (`adb`), and an ARCore-capable Android phone
with Chrome.

```bash
cd claim-agent
make setup                 # Python 3.12 environment and Playwright Chromium
make models                # ~10 GB: Qwen3-VL 8B (Ollama), Kokoro TTS, YOLO11x, YOLO-World, FastSAM, Whisper turbo
cp .env.example .env       # add price-source keys (optional, see below)
make run                   # http://localhost:8000; loads models, about 30 s
```

Phone: enable USB debugging, connect the cable, then run `make phone` (`adb reverse tcp:8000 tcp:8000`).
On the phone, open **http://localhost:8000** in Chrome. Through adb, localhost counts as a secure context,
so the camera, microphone and WebXR work without certificates. A laptop browser works too, using the
camera-only mode.

**Install as an app (optional):** in Chrome, menu (⋮) → *Add to home screen* → *Install* (or *Options →
Install as app* in the page). It opens full screen from its own icon. It is the same web page: the Mac
server (`make run`) and the USB link (`make phone`) must be running; otherwise the app shows how to start them.

1. **Connect & talk**. The agent greets you and confirms the country and currency (from the phone's time
   zone; say so if it is wrong).
2. **Start AR sweep**. Walk the room once: shelves top to bottom at arm's length, then each wall, the floor
   and the ceiling. Talk normally: "skip that shelf, those aren't mine", "that one is a signed first
   edition", "it's a print".
3. **Finish sweep**, or say "I'm done". The packet appears on screen (report link and `claim_packet.json`)
   and the agent reads out the summary.

Output for each sweep is in `sweeps/<id>/`: `claim_packet.json`, `report.html`, `report.pdf` (the same
report printed by headless Chromium; the page has *Download PDF* and *Download JSON* buttons), `frames/` (every
`frame_ref`), `depth/`, `crops/`, `stages/*.json` (the output of every stage), `cache/` (raw catalogue and
price responses) and `metrics.json` (latency per stage, API calls, cost).

### Price sources (`.env`)

| Key | Used for | Without it |
|---|---|---|
| `SERPER_API_KEY` | Google Shopping new prices (country-local `gl=`), item ranges, via Serper.dev (2500 free credits) | falls back to `SERPAPI_API_KEY` |
| `SERPAPI_API_KEY` | the same Google Shopping results via SerpAPI | no new prices; lines blank and in review |
| `EBAY_CLIENT_ID/SECRET` | used prices (eBay Browse API; for India the US marketplace, converted and labelled) | optional: used prices come from Google Shopping second-hand marketplace offers first |
| `GOOGLE_BOOKS_API_KEY` | catalogue matches, catalogue book heights, ebook fallback | Open Library only; the anonymous Google quota is usually exhausted |

No key means **no price**, never an estimated one. FX rates come from the ECB via frankfurter.app (no key).

## Commands

```bash
make check                                   # ruff, mypy --strict, unit tests (no models, no network)
make test-slow                               # replay pipeline with real models + browser E2E (needs `make run`)
uv run python -m claimagent.pipeline run --sweep <id> [--offline] [--fresh]   # re-run the post-sweep stages
make eval SWEEP=<id> TRUTH=ground_truth/mine # pass-bar table vs your ground truth -> sweeps/<id>/eval.md
make reprice SWEEP=<id> COUNTRY=US           # same 10 books priced for a second country
make marker                                  # docs/marker_10cm.pdf, reference for camera-only sweeps
```

## How it works (short)

`select → analyze → register → read → resolve → measure → items → room → price → assemble`. Each stage is
a function that writes `stages/<name>.json` and can be re-run alone. Details in
[docs/architecture.md](docs/architecture.md).

- **Count**: FastSAM segments every spine and YOLO11 says where the books are. Spines are matched across
  frames through homographies of the shelf plane, so each book is counted once.
- **Read**: Apple Vision OCR on every frame; text is split per spine and voted across frames. No language
  model reads titles.
- **Identify**: Open Library and Google Books are searched with the spine text. A match is scored only by
  words actually on the spine. Below threshold, the book is `unidentified` with its dimensions.
- **Measure**: ARCore metric depth and pose (WebXR). Without AR: an ArUco reference marker, a shelf width
  stated by voice, or catalogue book heights. The room comes from the ARCore depth point cloud.
- **Price**: retrieved offers only, with source, URL, date and condition. Rare, signed, first-edition and
  above-threshold lines and art are never auto-priced. Totals are summed in code.

## Reference app: what was kept, changed and dropped

The reference ([Insurance Claim Live Agent Team](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/voice_ai_agents/insurance_claim_live_agent_team))
was installed and its own test suites run: 55 Python tests, 18 client regressions and 7 avatar tests all
passed. A live session was not possible without a Gemini API key.

| | Kept | Changed | Dropped |
|---|---|---|---|
| Transport | FastAPI + one WebSocket per session; browser mic/camera capture; localhost-only binding | binary frames carrying pose and depth; frames saved to disk as evidence | base64 JSON media |
| Live model | voice session with non-blocking tools; barge-in | local Whisper → Qwen3-VL 8B (tools) → Kokoro instead of Gemini Live; capture coaching from code measurements, not the model | avatar |
| Background work | "graph runs after each turn" idea | a staged, re-runnable pipeline with per-stage outputs and metrics | policy rules, routing, mock policy DB |
| UI | notebook that fills live | live inventory (spines, readable, identified, items) | narrative notebook |
| Packet | downloadable packet | contract `claim_packet.json` + HTML report + frames + crops | markdown ZIP |

## Tools and models used

Pretrained only, nothing trained or fine-tuned: Ultralytics YOLO11x (COCO), YOLO-World v2, FastSAM-x;
Apple Vision text recognition (macOS); faster-whisper `large-v3-turbo` (Apple GPU; `small` on the CPU as fallback); Silero VAD; Qwen3-VL 8B Instruct (Ollama);
Kokoro-82M (kokoro-onnx); OpenCV (SIFT, homographies, ArUco); rapidfuzz. Services: Open Library, Google
Books, Serper.dev or SerpAPI (Google Shopping), eBay Browse API, frankfurter.app. Built with Claude Code as the AI coding
assistant.

## Assumptions (where the brief is ambiguous)

- Capture is WebXR on Android Chrome (ARCore pose, depth and camera image in one session). Other devices
  fall back to camera-only: books are measured from the reference marker, a voice-stated shelf width or
  catalogue heights, and the room is reported as not measured.
- Book count includes books seen only cut off at a frame edge, flagged in review. Books standing behind
  others are invisible and not counted.
- Edition and ISBN are filled only when the spine shows the publisher of a catalogued edition.
- Replacement = median of the main price band of new offers in the policyholder's country, from established
  booksellers when at least two list the book (general marketplaces also carry cheap unlicensed prints);
  box sets, study notes, translations sold under the same title and collectors' copies are excluded. Used = median of at least two
  second-hand marketplace offers, condition "good", never above the new price; local offers first, else US
  offers converted and labelled. Blank (and in review) when neither exists.
- Appraisal threshold: 150 USD equivalent in local currency, configurable.
- Wall area = perimeter × ceiling height; doors and windows are not subtracted. Shelved wall area = front
  area of detected shelving units.
- Item status `unpriced` was added to the contract's statuses for items with too few comparable offers.
- Plan deviations: an ArUco marker replaces the credit card (detection on a card was unreliable next to book
  covers of the same aspect ratio); FastSAM + YOLO replaces Hough-line spine splitting (measured better on
  real photos); OCR-first reading replaces VLM reading (a VLM grounding pass took 203 s per frame).

## Limitations (honest)

- On the test phone (OnePlus CPH2745) ARCore depth is unusable (37 m read at 1 m), so the room is reported
  "not measured" and books are measured from catalogue heights. A phone whose AR depth passes a 1 m wall check
  is needed for room areas. See the failure log.
- Titles come from public catalogues; books missing from Open Library and Google Books stay unidentified
  (6 of 35 in the test set).
- Thin spines (under 2 cm) carry about ±0.4 cm thickness error; italic script titles are often not read.
