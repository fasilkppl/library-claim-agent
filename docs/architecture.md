# Architecture

```
 Phone (Chrome, camera sweep; WebXR AR optional)            Mac (FastAPI, every model local)
 ┌──────────────────────────────┐  one WebSocket  ┌────────────────────────────────────────────────────┐
 │ mic 16 kHz ─────────────────────┼──────────────▶│ VAD → Whisper turbo → guide commands / fixed rules   │
 │ speaker ◀───────────────────────┼───────────────│ → Kokoro TTS. Qwen3-VL talks only before the walk.   │
 │ camera 4:3, 1 frame / 0.9 s ────┼──────────────▶│ frame worker: sharpness → keyframe → spines, OCR,    │
 │ guide card + live inventory ◀───┼───────────────│ items → coaching (blur, closer, glare, cut-off shelf) │
 └──────────────────────────────┘                 └───────────────────────────┬────────────────────────┘
                                                                               │ "I'm done"
 select → analyze (time budget) → register → read → resolve → dedupe → verify → measure → items → price → assemble
 every stage writes sweeps/<id>/stages/<name>.json; debug.log records each decision
```

## The sweep (the brief's scenario)

The agent confirms country and currency and waits for the answer, then explains the walk in two sentences.
The policyholder walks the room once: along the shelves ("next shelf", "done with books" are optional
markers), then across walls, floor and items in the same pass, and says "I'm done". The agent says what it
logs ("Logged Rework"), asks to slow down or step closer, and asks whether art is an original or a print.
During the walk the language model neither speaks nor records: it invented facts in tests. Corrections
("skip that shelf", "it's a print", "first edition") are recorded by rule; other questions get fixed, true
answers. Each frame is tagged with the part of the walk it belongs to (shelves or room, shelf number).

## Which model does what

| Step | Model / method |
|---|---|
| Speech in | Silero VAD + faster-whisper large-v3-turbo (Apple GPU); filter for invented text (confidence, repeats, caption phrases, echo, non-English letters) |
| Speech out | Kokoro-82M |
| Spines | FastSAM masks checked by YOLO11x "book"; Apple Vision OCR per frame and close re-reads of each spine |
| Titles | no model: Open Library and Google Books searches scored against the spine words; books *about* a book (summaries, trivia, study notes) rejected |
| Author check | second catalogue confirms the authors (by ISBN or title); disagreements go to the review queue |
| Items | YOLO-World + YOLO11x find candidates (recall first); Qwen3-VL names each item from its crop; building parts removed |
| Prices | Serper (Google Shopping, India); listings must be this product (no sets, summaries, accessories); main price band |

## Counting each book once (`pipeline/dedupe.py`)

Spines are tracked frame to frame; then: neighbours that read the same and lie on each other are one spine;
a second pass over a shelf is aligned in order with the first (sequence alignment); the same catalogue work
or near-identical reading twice is merged. Guards: different volume numbers or two spines side by side in
one frame are never merged; a spine whose reading picked up a neighbour's title loses that title instead of
duplicating it. On a guided shelf, fragments never seen whole or seen once outside the shelf row are not
counted. Every merge and exclusion is logged with its reason.

## Where metric scale comes from

Catalogue heights (Google Books dimensions, Open Library editions) of identified books are rulers: in each
frame, a ruler standing in the same shelf row gives that picture's pixels per centimetre for its neighbours
(height error 2-10% on real sweeps). AR depth is used when the phone's AR tracks reliably (`web/artest.html`
tests it). Never estimated by a model. The room needs AR depth; without it the room is reported as not
measured.

## Where prices come from

Every price is a dated shopping listing with its URL. Replacement = median of new offers from established
sellers in the main price band; used = second-hand sellers only, capped at the new price; no local offer:
US converted and labelled; none: blank and excluded from totals. Rare, signed, art and above-threshold lines
go to appraisal unpriced. Totals are computed in code from the lines.

## Time budget

The packet is due within 5 minutes. Frame analysis after the walk fills the largest gaps first within its
budget; slow lookups (titles, rulers, items) get what remains after the later stages' reserve. Measured:
160-210 s after "I'm done".
