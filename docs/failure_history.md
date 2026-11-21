# Failure history (full engineering diary)

Every defect found while building, with root causes and before/after numbers. The one-page failure log for
the submitted run is [failure_log.md](failure_log.md).

**Status:** measured on a real room (10 books, 18 items, tape-measured 4.45 × 2.69 × 3.15 m, OnePlus CPH2745
phone, ground truth in `ground_truth/mine/`) and on the earlier test sweeps (45 books from two shelf photos).
The final one-take sweep will be re-evaluated with `make eval`; numbers below are from the practice sweep.

Real room, practice sweep (`sweeps/room-1-fix/eval.md`): book count 10 vs 10 (PASS) · titles 70% correct,
0% confidently wrong (PASS) · spine height MAPE 4.3%, thickness 9.0%, 10/10 matched (PASS) · items 61% with
correct category (FAIL) · book prices 2/10 within 25% (FAIL, see 6) · floor and wall area not measured (FAIL,
AR depth unusable on this phone, see 5) · packet 41 s after the sweep (PASS).

Test sweeps (`sweeps/test-pan-keys/eval.md`): count 43 vs 45 (PASS) · titles 71%, 0% confidently wrong
(PASS) · height MAPE 10.1%, thickness 5.7% (PASS).

## 1. Titles missed (9 of 35 legible spines with a Google Books key, 11 without)

- **What:** 74% correct with the key (69% without). No wrong title is reported with confidence.
- **Root causes, counted:**
  - 8 are missing from the catalogue: 6 Borges/Hanser volumes and 2 Höhlenkinder volumes. Open Library has
    no record, and the Google Books anonymous quota was exhausted. These correctly stay unidentified or
    unconfident.
  - 1 is a segmentation merge: the thin Schrödinger spine was absorbed into its neighbour.
  - 1 is an italic script title ("Shame and Necessity") that OCR does not read.
  - 1 has OCR letter swaps ("nacb" for "nach") in every reading.
- **Fixes measured on this set** (lenient matcher at the time):
  - 57% → 63%: text assigned to spines by area overlap instead of centre point, plus cross-frame line voting.
  - → 66%: leave-one-line-out and word-pair catalogue queries.
  - → 69%: lines running across two spines split at word boxes.
  - → 74%: observations with an outlier spine width excluded from voting.
  - The evaluator was then made strict, because the lenient one counted truncated titles as correct, and
    the same packet scores **69%**. Confidently-wrong went 8.6% → 0% after one rule: an unexplained spine
    word of 5+ letters caps confidence (Höhlenkinder vs "… im Steinhaus").
- **Next:** a Google Books key (more German catalogue coverage); OCR at 180° on crops that read as
  garbage; a second segmentation pass on spine widths more than 1.8× the shelf median.

## 2. Spine sizes truncated

- **What:** height MAPE was 18.0% and thickness 18.5%. Grassi measured 11.4 cm against 21.6 cm true;
  Brisson's thickness was 1.0 cm against 1.75 cm.
- **Root cause (proved per observation):** in frames where FastSAM missed a spine, the fallback YOLO box
  covered only half of it (520 px vs 990 px). The stage also used the observation cut off at the frame edge.
  An earlier 7.6% error was in my test generator: anisotropic crops. Diagnosed and fixed in the generator,
  not hidden in the pipeline.
- **Fix:** prefer segmentation observations; length = 80th percentile (errors are truncations); width =
  median over near-whole observations; collinear fragments joined.
- **Measured:** height 18.0% → **10.1%**, thickness 18.5% → **14.9%**; Brisson 1.0 → 1.8 cm (true 1.75).
- **Still weak:** spines under 2 cm carry about ±0.4 cm error, close to my ±10 px pixel ground truth.

## 3. Room area biased low

- **What:** synthetic rooms came out 3.3% small, and 7.7% small at 2% depth noise (13.57 vs 14.70 m²).
- **Root cause:** depth noise spreads each wall into a band of cells, and the flood fill stopped at the
  inner edge of the band.
- **Fix:** walls = ridge cells (≥ 60% of the local maximum density). Fill grown onto the wall cells.
- **Measured:** rectangle 0.0% at 1% noise, −1.9% at 2%; L-shape +0.2% to −1.9% up to 3% noise.
- **Unverified:** real ARCore depth in a furnished room, where furniture against walls hides the walls.

## 4. First real-phone sweep (OnePlus CPH2745, Chrome 157, ARCore 1.57)

- **What:** a 60 s desk sweep gave a 7.4 m "ceiling", no room, and 53 items for about 12 real objects
  (the same laptop counted 21 times). The item total was ₹13–31 lakh.
- **Root causes (proved from the saved frames):**
  - ARCore lost its scale in the first 10 s: positions ran 12 m in 10 s and depth grew to 37 m. Even before
    that, depth read about 3 m for objects about 0.4 m away. Every metric output inherits this.
  - Item detection ran on every second keyframe only, and items were merged only by 3D position or shelf
    registration, so with no trusted pose no sighting was ever merged.
  - One network timeout in a price search crashed the whole packet.
- **Fixes:** AR tracking guard (above, in the architecture note), AR start gate on the phone, item tracker
  with appearance re-identification, detection on every keyframe, price-search failures become blank lines.
- **Measured on the same sweep:** items 53 → 26; room "not measured" with the tracking reason instead of a
  wrong number; packet no longer crashes. The office background really has many monitors, so 26 is not
  a clean count: the open question is whether ARCore depth on this phone is reliable when tracking is
  started properly. A measured test (phone at 0.75 m from a wall, then moved 50 cm) is the next step.

## 5. AR depth on the test phone is unusable (room not measured)

- **What:** on the OnePlus CPH2745 (Chrome 157, ARCore 1.57) the depth grid read a median of 37 m with the
  phone 1 m from a wall, and the reported phone height rose from 1.3 m to 2.6 m while it was hand-held.
- **Root cause:** ARCore's metric scale on this device; it was wrong from the first frame, before any fast
  motion. Not fixable in the app.
- **What the app does:** the tracking guard (median depth over 8 m, phone above 2.4 m, speed over 1 m/s)
  rejects those poses, the agent says so live, and the room is reported "not measured" with the reason
  instead of a wrong number. Books are measured from catalogue heights instead (below).
- **Next:** sweep with an ARCore phone whose depth passes a 1 m wall check (Pixel or Samsung), or measure the
  room from a known object in view.

## 6. Book prices vs hand-checked prices

- **What:** 2 of 10 within 25%. The app is consistently lower than the reference prices.
- **Root causes (from the offers):** India's Google Shopping lists many cheap prints on general marketplaces
  (₹139–195 for Wings of Fire), study notes, box sets, other-language editions and collectors' copies
  (₹14,000+). Fixed: those listings are filtered, the median is taken over the main price band, and
  established booksellers are preferred. The remaining gap is mostly the reference itself: the hand-checked
  prices came from different shops (Crossword ₹509 vs Amazon ₹280–300 for the same paperback).
- **Next:** check every reference price on one retailer (Amazon.in) before the final evaluation.

## 7. Practice sweep defects (fixed, measured on the same frames)

| Defect | Root cause | Fix | Before → after |
|---|---|---|---|
| 13 books counted for 10 | a gift box lying by the books, 2 edge strips | spine must not be both out of place in the row and seen as another object; strips seen ≤2× at the edge, across the row, dropped | 13 → 10 |
| titles 30% | only the cross-frame consensus was searched; OCR slips ("BROWNI", "TALEOF", "CIFIES") | close re-reading of each spine's 3 clearest views; single-frame readings (capped below confident); one-letter tolerance; word-list repair | 30% → 70%, still 0 confidently wrong |
| no spine sizes | AR rejected, catalogue rulers needed a confident match and Google volume | review-level matches as rulers (≥3 must agree within 15%), height looked up by title | none → height 4.3%, thickness 9.0% |
| thickness +0.5 cm | width taken from the mask's bounding rectangle | mask mean width (area / length), low percentile over views | 28% → 9% |
| Google Books switched off mid-run | per-minute quota treated as exhausted | pacing at 1.5 requests/s; a per-minute 429 waits instead of disabling | — |
| API key in logs | httpx logs full request URLs | httpx logging off | — |

## 8. Speech misheard (fixed, measured)

- **What:** on the practice sweeps the agent "heard" sentences nobody said: "Thank you for watching. We'll
  see you in the next one. Bye. Bye. Bye…", "You can find the link in the description", and the recognition
  hint read back word for word ("first edition, signed copy, original, painting, rupees…").
- **Root causes:** Whisper invents captions on noise; the long word-list hint is echoed when audio is
  unclear; Whisper `small` with greedy decoding is weak on Indian-accented English; the phone converted
  48 kHz to 16 kHz by dropping samples (aliasing); a 0.7 s pause cut sentences in half; and nothing was saved,
  so nothing could be measured.
- **Fixes:** every utterance saved as WAV with its transcript (or the reason it was dropped); the browser
  resamples the microphone (filtered fallback in the worklet); Whisper `large-v3-turbo` on the Apple GPU,
  beam 5, short hint, warmed up at start; an invented-text filter (Whisper's own confidence, repeats, known
  caption phrases, hint and agent-echo copies, VAD speech share); 1 s end-of-sentence pause.
- **Measured** (`scripts/stt_bench.py`, 45 sentences in three macOS Indian-English voices with room noise,
  14 noise/echo clips): word errors 13.2% → 1.3%; invented text on noise 2/14 → 0/14; 1.5 s per sentence
  (2.6–3.4 s while frames are analysed). Synthetic voices are a stand-in: the saved clips of the final sweep
  are the real check.

## 9. Trial sweep: "blurry" complaints, sideways phone, AR camera quality (fixed or worked around)

- **What:** the agent kept saying "the picture is blurry"; 60 of 106 frames were discarded as blurry; the phone
  was held sideways for part of the sweep, so standing books looked like lying ones and no sizes came out.
- **Root causes:** the sharpness check was a whole-frame Laplacian variance, low for any frame with a lot of
  plain wall however well focused (a readable frame scored 24 against a limit of 60); frames were analysed in
  whatever orientation the phone was held; Chrome's AR camera stream is softer than the phone camera (no
  autofocus or sharpening), and on this phone AR depth is unusable anyway.
- **Fixes:** sharpness measured where the picture has detail (90th percentile of tile variances; in-focus
  frames 60–230, the same frames blurred 8–17; limit 20); every frame turned upright from gravity (AR pose or
  motion sensor) with the camera geometry turned with it (tested to be exact); camera-only mode requests
  continuous autofocus and full resolution. Not-a-book rules refined on what the extra frames exposed (a water
  bottle, edge slivers, glance noise), each checked on the 45-book set so it did not drop real books.
- **Measured:** frames kept 31 → 61 of 106; items 78% → 89% on the trial. Counts still vary by ±1 between
  runs (a thin side face next to a book cannot be told from a thin book; it is kept, so real books are not lost).

## 10. Prices wrong on the trial packet (fixed; the remaining gap is market spread)

- **What:** Go Set a Watchman ₹13,927, And the Mountains Echoed ₹112, Rework ₹234; items ₹55–93k with a "bed",
  a "dining table" and a "helmet" that were a desk and a pressure cooker.
- **Root causes (from the saved offers):** the catalogue title "Go Set a Watchman by Harper Lee" made every
  normal listing fail the word match, leaving antiquarian copies (Biblio, ₹8,000–4,56,000); a critical study
  ("Ecocritical Approach to … And the Mountains Echoed"), bundles and audio CDs passed the match; the real
  listing "ReWork: Change the Way You Work Forever" failed it (subtitle words); rare-book marketplaces counted
  as new copies; item prices followed wrong detector names.
- **Fixes:** listings must match the book's main title (author and subtitle removed on both sides, at most one
  extra word); bundles, audio, studies and summaries rejected; rare-book marketplaces excluded from the new
  price; items renamed by the local vision-language model from their crops (nine per call), building fixtures
  (light switches) and books removed from the item list.
- **Measured (trial, reference = Flipkart, owned edition):** no price off by more than 46%; within 25%: 3 of 8.
  The rest is market spread, not a bug: Amazon ₹326 vs Flipkart ₹471 for the same Rework; Flipkart itself lists
  The Almanack at ₹193 and ₹276; the owned Lost Symbol is a hardcover the photo cannot reveal. Items now
  ₹25–38k, with a pressure cooker at ₹2,160–3,136 and a clothes iron at ₹667–1,204.

## What was designed after seeing the practice room (disclosure)

The answer key (`ground_truth/mine/`) is read only by `make eval`; no title, count, size, price or room
measurement from it is in the code. But several rules were written after looking at the practice sweep's
failures: the not-a-book check, close spine re-reading, the one-letter tolerance, the width percentile, the
price band and seller rules, the item tracker's cross-label matching, and the evaluator's pairing of
unidentified books by shelf position. Each was checked on the second test set (45 books) so it did not make
that set worse. The item detector's vocabulary is a general 148-word household list written without the
room in mind; an earlier version had borrowed 13 words from the room's item list and was replaced (items
found went 61% → 72% with the general list). On 21 Nov, after the practice sweeps, the item name check stopped
deleting objects named with a building word in front ("door mat", "wall mirror", "wall art": the word list
gained "wall art" and "mat"); it had deleted a wall decoration and a floor mat, and no score changed on the
saved sweeps. Also on 21 Nov, from a practice sweep: the not-a-book check takes the typical spine length from
spines seen properly (one-frame slivers at the frame edge had made the tallest book, Wings of Fire, look out of
place and be dropped as a "cardboard box"), and records titled "<First Last>'s <title>" by someone else (a
parody of The Lost Symbol) are rejected as books about a book. Replayed on that sweep's frames: count 8 → 10,
titles 60% → 80%. The final sweep, and the reviewers' room, are the real test.

## Live-path defects found by the end-to-end test (fixed)

- A keyword collision in event logging raised inside `say()`. It silenced all agent audio and stopped the
  packet. Found from the server log, fixed, and the E2E test now passes.
- Speech-to-text took 4.6 s under load (0.93 s alone) because of CPU contention with frame analysis. Fix:
  torch threads capped, Whisper given 6 threads, frame analysis pauses during a voice turn. Now 1.0–3.8 s.
- Registration took 114 s after a looping video caused constant relocalisation. Fix: candidates bounded to
  16 keyframes. Now 2–11 s.
- "That painting is just a print" got no `art_answer` tool call when no art had been detected. Open.
- The packet stalled (identification ran 20 minutes) when Open Library slowed to 9–30 s per request. Fix:
  8 s timeouts, a site is skipped after 3 timeouts in a row, a hard 100 s identification budget, Google Books
  paced to its per-minute quota. The packet is now 115–142 s on the end-to-end test.

## Cost per sweep and latency per stage

Models are local, so they cost nothing per sweep. The paid part is SerpAPI at about 0.015 USD per search
(Developer plan): about 2 searches per identified book (local + US fallback) and 1 per item. 60 books and
10 items is roughly 130 searches, **about 2 USD per sweep**. eBay, Open Library, Google Books and
frankfurter.app are free.

Post-sweep latency, live browser E2E (75 s sweep, 124 frames): select 0.8 s · analyze 28.5 s (frames not
analysed live) · register 11.4 s · read 0.03 s · resolve 3.3 s · price + assemble 1.6 s · **total 45.6 s**.
Per frame: YOLO 0.3 s, FastSAM 0.4 s, segmentation 0.2 s, OCR 0.25 s, items 0.2 s.
Voice turn: STT 1.0–3.8 s, LLM first token 0.2–4.8 s, TTS first audio 0.8–3.6 s.

## With two more weeks

Validate WebXR camera-access and depth on several phones. Collect three real rooms of ground truth and tune
nothing until it exists. Add a 180° OCR retry and a spine re-split pass. Run a checker agent that re-reads
every confident title from its crop. Add per-book condition grading from the crop. Move STT to the GPU
when frame analysis is idle.
