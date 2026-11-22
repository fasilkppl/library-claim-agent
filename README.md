# Library Contents Claim Agent

A live voice-and-vision agent that inventories and values a home library from **one continuous camera
sweep**. The policyholder walks the room with an Android phone while talking to the agent. The agent coaches
the capture as it happens ("slow down", "step closer", "is that portrait an original or a print?"). When the
walk ends, the system writes a claim packet: every book read from its spine, measured and priced locally;
every other item classified and priced; and, when the phone's AR tracking works, the room's floor and wall area.

Everything runs **locally on a Mac with Apple Silicon**: speech, dialog, vision and OCR. The only network
calls are catalogue lookups and price retrieval, and every one is saved as evidence.

- Architecture (one page): [docs/architecture.md](docs/architecture.md)
- Failure log (one page): [docs/failure_log.md](docs/failure_log.md)
- Ground-truth templates: [ground_truth/template/](ground_truth/template/); results come from `make eval`

## Quick start (about 15 minutes on a fast connection; most of it is the ~10 GB model download)

Requirements: macOS 14+ on Apple Silicon with 16 GB RAM or more, [uv](https://docs.astral.sh/uv/),
[Ollama](https://ollama.com/download), Android Platform Tools (`adb`), and an Android phone with Chrome
(ARCore is needed only for the optional AR mode).

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
2. **Camera-only sweep** (the default; open `/?ar=1` to get the AR sweep instead). Walk the room once: along
   each shelf from left to right, about half a metre away (tap **Next shelf** between shelves and **Books
   done** at the end, or say it), then pan slowly across the walls, the floor and everything in the room. At
   the room step an AR phone shows **Measure room (AR)**: tap it, move until AR is ready, then pan the walls,
   floor and ceiling (floor and wall area come only from AR depth).
   Talk normally: "skip that shelf, those aren't mine", "that one is a signed first edition", "it's a print".
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
make price-sheet SWEEP=<id>                  # blank hand-check sheet (search links only) for the sweep's book prices
make ar-check LABEL=native SECONDS=45        # ARCore's own log while an AR app runs (phone on USB)
make ar-sweep SWEEP=<id>                     # AR tracking segments, trust and depth of a sweep
make submission SWEEP=<id> TRUTH=ground_truth/mine  # packet + frames zip, results sheet, the two notes as PDF
```

## How it works (short)

`select → analyze → register → read → resolve → dedupe → verify → measure → items → room → price → assemble`.
Each stage writes `stages/<name>.json` and can be re-run alone; `sweeps/<id>/debug.log` records every
decision. Details in [docs/architecture.md](docs/architecture.md).

- **Sweep**: the agent confirms country and currency, then directs one continuous walk: the shelves, then the
  walls, floor and items in the same pass, ended with "I'm done". It says what it logs, asks to slow down or
  step closer, and asks whether art is a print. During the walk no language model speaks or records facts;
  corrections ("skip that shelf", "it's a print") are recorded by rule.
- **Count**: FastSAM + YOLO11 find spines; spines are tracked across frames, and `dedupe` merges repeat
  sightings (revisits, split detections) with guards for set volumes and books side by side.
- **Read and identify**: Apple Vision OCR; Open Library and Google Books searched with the spine text, scored
  only by words on the spine; books about a book are rejected; `verify` confirms the authors in the second
  catalogue. Below threshold, the book is `unidentified` with its dimensions.
- **Measure**: catalogue heights of identified books are rulers in the same picture and shelf row. AR depth is
  used when the phone's AR tracks reliably (check with `web/artest.html`). The room needs AR depth.
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

- **Test space**: 10 books on one shelf and 18 non-book items in one room (the brief asks for at least 60 books
  on 2 shelving units; only 10 books were available). Ground-truth samples are therefore 10 measured spines
  and 10 reference prices instead of 20 and 15. Numbers are reported on this space only. The reference prices
  were read from Google Shopping India by the AI assistant (`ground_truth/mine/prices.csv` says so per row),
  not checked by a person; `make price-sheet` gives the links for a hand check.
- Capture is a camera sweep on Android Chrome (full 4:3 sensor, 1 frame per 0.9 s). WebXR AR is opt-in
  (`/?ar=1`); without trusted AR the room is reported as not measured.
- Book count includes books seen whole at least once. Fragments only ever cut by the picture's edge, and
  one-frame glimpses outside the shelf row, are not counted (listed in `stages/dedupe.json`).
- Edition and ISBN are filled only when the spine shows the publisher of a catalogued edition; otherwise the
  common edition is priced (a reason prices can differ from a specific edition's).
- Replacement = median of the cheapest price range holding a third of the new offers in the policyholder's
  country (a jump of more than 2.5 times separates editions), from established booksellers; searched by title
  and author, not ISBN; box sets, study notes, translations and collectors' copies are excluded. Used = median of at least two second-hand offers, never above the new price; else US offers
  converted and labelled. Blank (and in review) when neither exists.
- Appraisal threshold: 150 USD equivalent in local currency, configurable.
- Wall area = perimeter × ceiling height; doors and windows are not subtracted.
- Item status `unpriced` was added for items with too few comparable offers or that could not be recognised.

## Limitations (honest)

- The submitted packet (sweep `20261121-185148-d1ac89`) was rebuilt with the final code from the sweep's saved
  frames and recorded responses; the live packet shown in the video is kept beside it. This sweep was used
  during development; see [docs/failure_log.md](docs/failure_log.md).
- Room areas need ARCore depth; the submitted sweep predates the AR room step. On the test phone
  (OnePlus 15, Android 17) ARCore failed in earlier AR sweeps: a standalone
  ARCore app (AR Ruler), Google's model-viewer demo and our test page (`web/artest.html`) all lose tracking,
  and ARCore logs camera-to-motion-sensor clock offsets of -54 to +44 ms against its 5 ms limit. Our AR
  geometry passes the synthetic replay (floor within 10 %, walls within 15 %). Camera sweeps therefore
  report the room as not measured; evidence in [docs/failure_log.md](docs/failure_log.md).
- Very thin neighbouring spines can be detected as one (two 1 cm books counted once); spine thickness is
  overestimated where masks bleed into neighbours.
- Items seen only in passing are named from partial crops and can be misnamed (an iron called a hand
  mixer); unrecognised objects are left unpriced.
- Prices are for the common edition unless the spine names the edition.
- Titles come from public catalogues; books missing from Open Library and Google Books stay unidentified.
