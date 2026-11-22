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
 select → analyze (time budget) → register → read → resolve → second look → dedupe → verify → measure → room → assemble
 beside them: items (needs only the analysis), book prices (start once the titles are final)
 every stage writes sweeps/<id>/stages/<name>.json; debug.log records each decision
```

## The sweep (the brief's scenario)

The agent confirms country and currency, then the policyholder walks the room once: along the shelves,
then walls, floor and items, and says "I'm done". The agent says what it logs ("Logged Rework"), coaches
(slow down, step closer) and asks whether art is a print. During the walk the language model neither speaks
nor records (it invented facts in tests): corrections ("skip that shelf", "it's a print") are rules.

## Which model does what

| Step | Model / method |
|---|---|
| Speech in | Silero VAD + faster-whisper large-v3-turbo (Apple GPU); filter for invented text (confidence, repeats, caption phrases, echo, non-English letters) |
| Speech out | Kokoro-82M |
| Spines | FastSAM masks checked by YOLO11x "book"; Apple Vision OCR per frame and close re-reads of each spine; Qwen3-VL copies the text of a spine OCR could not read (never confident, always reviewed) |
| Titles | no model: Open Library and Google Books searches scored against the spine words; books *about* a book (summaries, trivia, study notes) rejected |
| Author check | second catalogue confirms the authors (by ISBN or title); disagreements go to the review queue |
| Items | YOLO-World + YOLO11x find candidates (recall first); Qwen3-VL names each item from its crop; building parts removed |
| Prices | Serper (Google Shopping, India) searched by title and author; listings must be this book from an established bookseller (no sets, translations, summaries) |

## Counting each book once (`pipeline/dedupe.py`)

Spines are tracked frame to frame; a second pass over a shelf is aligned in order with the first. Different
volume numbers, or two spines side by side in one frame, are never merged; fragments never seen whole are not
counted. Every merge and exclusion is logged with its reason.

## Where metric scale comes from

Catalogue heights (Google Books dimensions, Open Library editions) of identified books are rulers: in each
frame, a ruler standing in the same shelf row gives that picture's pixels per centimetre for its neighbours
(height error 2-10% on real sweeps). Never estimated by a model. ARCore depth and pose are used when the
phone's AR tracks reliably. The room needs ARCore depth: at the room step a camera sweep offers *Measure room
(AR)*, which switches the same sweep to WebXR; without it the room is reported as not measured.

## Where prices come from

Every price is a dated shopping listing with its URL. Replacement = median of the cheapest price range that
holds a third of the new listings from established booksellers (a jump of more than 2.5 times separates
editions: imported copies cost several times the local one); used = second-hand sellers only, capped at the
new price; no local offer:
US converted and labelled; none: blank and excluded from totals. Rare, signed, art and above-threshold lines
go to appraisal unpriced. Totals are computed in code from the lines.

## Time budget

The packet is due within 5 minutes. Frame analysis after the walk fills the largest gaps first within its
budget; slow lookups (titles, rulers) get what remains. Final run: 222 s after "I'm done"; items and book
prices now run beside the book stages (same sweeps replay in 39-53 s instead of 71-114 s, same results).
