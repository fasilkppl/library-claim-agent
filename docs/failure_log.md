# Failure log

Run: `20261121-185148-d1ac89` (21 Nov) · 10 books on one shelf, 18 non-book items, one room (4.45 × 2.69 ×
3.15 m) · OnePlus 15 (CPH2745, Android 17, Chrome 157, ARCore 1.57) · results in `results.md` of the submission.
Pass bars, 4 of 8: book count 10/10, titles 10/10 (0 confidently wrong), spine size (height 7.4 %, thickness
7.1 %), time to packet (222 s live) pass; prices 5/10, items 11/18, floor and wall area fail.

**How this packet was made:** rebuilt on 22 Nov with the final code (`python -m claimagent.pipeline run --sweep
20261121-185148-d1ac89`) from the sweep's saved frames and recorded catalogue and shop responses (14 new
searches). The packet at the end of the video is the live one (21 Nov code: 14 books, 11 identified), included
as `claim_packet.live.json`. This sweep was one of three used to develop and check fixes after 21 Nov, so its
numbers are optimistic: a sweep captured after the code was final (`20261122-064126-ccdcac`) passed 3 bars.

## 1. Room not measured (floor and wall area fail)

- **What:** no floor or wall area; the sweep ran in camera mode.
- **Root cause:** the room is measured only from ARCore depth. The AR room step (*Measure room (AR)*) was
  added after this sweep. Earlier AR sweeps on this phone failed inside ARCore: a standalone ARCore app (AR
  Ruler) found no plane and logged camera-to-motion-sensor clock offsets of −54 to +22 ms (limit 5 ms) and 32
  tracking resets; in Chrome the position drifted up to 40 m standing still. Our AR geometry passes the
  synthetic replay (floor within 10 %, walls within 15 %).
- **What I would change:** start the AR step by itself when ARCore passes a 1 m depth check; otherwise
  measure the room from the camera frames, with the book rulers as scale.

## 2. Non-book items: 11 of 18 found (bar 80 %)

- **What:** the pressure cooker, frying pan, kitchen shelves, curtain rod and clothes hanger were not found;
  the bucket was called a laundry basket and the gift box a plain box (other categories than the list).
- **Root cause:** one pan shows small and kitchen objects briefly and partly; the detector has no reliable class
  for a bare curtain rod, and the vision model names each item from one crop.
- **What I would change:** have the agent ask for a pause on an object seen once; merge repeats by position and
  colour (a colour-only rule found a second doormat on another sweep but listed a pressure cooker twice).

## 3. Book prices: 5 of 10 within 25 %

- **What:** A Tale of Two Cities ₹88 (reference ₹229), Rework ₹197 (₹326), Principles of Building AI Agents ₹449
  (₹851), Patterns for Building AI Agents ₹625 (₹449); Wings of Fire unpriced.
- **Root cause:** the shop results recorded on the sweep day lacked a matching print listing for two books (A
  Tale of Two Cities fell back to an ebook price, flagged for review; Wings of Fire got none) and an established
  bookseller for Rework; Principles and Patterns listings moved between the sweep and the reference reading
  (22 Nov 01:06). The reference prices were read from the same searches by the AI assistant, not by a person.
- **What I would change:** search again with the paperback edition when only an ebook is found, and re-read the
  reference by hand within an hour of the sweep.

**Measured fix:** spine heights on sweep `20261120-150808-ef87ca` went from 64.3 % to 7.9 % error when the
catalogue look-ups for the ruler books ran 6 at a time instead of one at a time within the same budget. Items and
book prices now run beside the book stages: three sweeps replay in 39, 53 and 47 s instead of 71, 114 and
103 s, with the same books and items.

## Cost per sweep and latency per stage

- **Cost:** USD 0.094 live (47 Serper Google Shopping searches, 2 credits each); the rebuild added 14 searches
  (USD 0.028). All models run locally; Open Library, Google Books and frankfurter.app are free.
- **Latency after "I'm done", live (21 Nov code, 222 s):** select 4.9 s, analyze 50.0, register 27.7, read
  13.6, resolve 26.8, verify 4.3, measure 13.3, items 64.6, prices and report 16.5. Rebuild with the final code
  (frames already analysed): 121 s. Voice (medians): speech-to-text 2.4 s, first spoken audio 0.6 s, turn 3.1 s.

Rules written after seeing the practice room are disclosed, with every earlier defect, in
[failure_history.md](failure_history.md).
