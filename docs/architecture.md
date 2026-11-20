# Architecture

```
 Android Chrome (WebXR immersive-ar)                     Mac (FastAPI, all models local)
 ┌───────────────────────────────┐  WebSocket  ┌──────────────────────────────────────────────────────┐
 │ mic 16 kHz PCM ───────────────┼────────────▶│ Silero VAD → Whisper turbo → Qwen3-VL 8B (tools) →    │
 │ speaker ◀─────────────────────┼─────────────│ Kokoro TTS (sentence streaming, barge-in stops it)    │
 │ camera image (camera-access)  │             │                                                      │
 │ + metric pose + depth grid    ├────────────▶│ frames/ depth/ events.jsonl (evidence on disk)        │
 │   (ARCore, ~1.7 fps)          │             │ frame worker: quality → keyframe? → analyze → register│
 │ live inventory ◀──────────────┼─────────────│ coaching rules (blur, glare, too fast, cut-off shelf, │
 └───────────────────────────────┘             │ unreadable spines, missing ceiling/floor/walls, art?) │
                                               └──────────────────────────┬───────────────────────────┘
                                                                          │ stop / "I'm done"
 select ─▶ tracking ─▶ analyze ─▶ register ─▶ read ─▶ resolve ─▶ measure ─▶ items ─▶ room ─▶ price ─▶ assemble
 (each stage writes stages/<name>.json; frames analysed live are cached and not recomputed)
```

## Which model does what

| Step | Model / method | Notes |
|---|---|---|
| Speech in | Silero VAD (streaming) + faster-whisper `large-v3-turbo` on the Apple GPU (float16), beam 5; invented-text filter (Whisper confidence, repeats, known caption phrases, hint and echo copies); every utterance saved as WAV evidence | ~1.5 s per utterance; synthetic Indian-English test: 13.2% → 1.3% word errors |
| Dialog and tool calls | Qwen3-VL 8B Instruct via Ollama | only conversation; it never produces a title, number or price (they come from a STATE line computed in code) |
| Speech out | Kokoro-82M | first audio 0.8–3.6 s |
| Spine segmentation | FastSAM-x masks gated by YOLO11x "book" boxes; fragment merge and neighbour-span removal in code | 0.6–1 s per frame |
| Spine text | Apple Vision OCR on each frame; word boxes split lines that run across spines; then a close re-read of each spine's 3 clearest views (straightened, enlarged); per-line vote across frames | ~0.25 s per frame + ~0.2 s per spine |
| Identification | no model: Open Library / Google Books search + rapidfuzz scoring of spine words | confident only if every long spine word is explained |
| Cross-frame dedupe | SIFT + RANSAC homographies of the shelf plane; tracks by overlap; relocalisation on re-capture | |
| Items | YOLO-World v2 (open vocabulary) + YOLO11x COCO classes; 3D (depth) or plane clustering | brand only from OCR text printed on the object |
| Room | ARCore depth point cloud → floor/ceiling peaks → wall ridge occupancy grid → flood fill from the camera path → polygon | no model |

## When AR tracking fails

ARCore can lose its metric scale (seen on a real phone: the reported position ran 12 m in 10 s and depth
grew with it). The phone holds back frames until tracking has been stable for 1.5 s while moving at least
10 cm, and sends Chrome's `emulatedPosition` flag with each frame. The server rejects poses that move faster
than 1 m/s or see a median depth beyond 12 m; after a sustained failure every later pose is rejected
(ARCore does not repair the scale). The agent says so live. In the packet, rejected frames lose their pose,
so books fall back to marker / voice / catalogue scale, items to image tracking, and the room is reported as
not measured with the reason. Code: `capture/tracking.py`, stage `tracking`.

## Counting each object once

Items are followed from keyframe to keyframe by image motion (ORB + RANSAC homography) and box overlap, with
synonym labels treated as one kind (laptop / tv / monitor). Revisits are joined by 3D position when AR is
trusted, otherwise by appearance (feature match, or colour histogram and shape) between tracks that never
appear in the same frame. Code: `pipeline/itemtrack.py`.

## Where metric scale comes from

1. **WebXR / ARCore depth (primary).** Each keyframe carries the camera pose (metres, `local-floor`), the
   intrinsics (from the projection matrix) and a 96×72 depth grid. Spine size = pixel extent × depth / focal
   length, from the 80th percentile across the frames that show the whole spine.
2. **ArUco reference marker (camera-only mode)**, 10 cm (`docs/marker_10cm.pdf`). Its side in pixels, carried
   into shelf-plane coordinates by the homography, gives pixels per centimetre for the whole shelf segment.
3. **Policyholder states a shelf width** by voice (`shelf_width` tool).
4. **Catalogue heights** (Google Books `dimensions`, looked up by volume or by title and author) of identified
   books, used as rulers when at least 3 agree within 15% (median absolute deviation). On the test phone, whose
   AR depth is unusable, this gave spine heights within 4% and thicknesses within 9%.
Every book records its `scale_source`. Sources that disagree by more than 10% send the line to review.
Room: the depth points are already metric through the ARCore pose, so no extra scale is needed.

## Where prices come from

| Figure | Source | Rule |
|---|---|---|
| Book replacement | Google Shopping via Serper.dev (or SerpAPI), `gl=<country>`, new offers whose title matches; sets, notes, other-language and collector listings removed; established booksellers preferred; median of the main price band | median after removing multipacks; otherwise US offers converted (ECB rate); otherwise Google Play ebook price (labelled); otherwise blank |
| Book used | Google Shopping with the "used" filter, offers from second-hand book marketplaces only (Google ignores the filter when it has no used listings); local first, else US converted; else eBay Browse if keyed | median of at least 2, offers above the new price dropped (collectible copies are not a used value) |
| Item | Google Shopping via Serper.dev (or SerpAPI) | brand/model printed on the object → `priced`; otherwise 25th–75th percentile of comparable offers → `range` |
| Never auto-priced | art and portraits (unless the user says "print"); signed, first edition or rare (voice or spine text); above 150 USD equivalent | `needs_appraisal`, excluded from totals, kept as a review note |

Every figure stores its source, URL, `retrieved_at` and assumed condition; the raw response is in
`sweeps/<id>/cache/`. Totals are summed in Python from the lines (`compute_totals`), never by a model.
