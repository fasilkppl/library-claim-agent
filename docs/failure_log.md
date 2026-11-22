# Failure log

Run: `20261122-064126-ccdcac` · 10 books on one shelf, 18 non-book items, one room (4.45 × 2.69 × 3.15 m) ·
OnePlus 15 (CPH2745, Android 17, Chrome 157, ARCore 1.57) · full results in `results.md` of the submission.
Pass bars, 3 of 8: book count 10/10, titles 10/10 (0 confidently wrong), time to packet 226 s pass; spine size
(height 6.4 %, thickness 15.2 % against 15 %), prices 6/10, items 10/18, floor and wall area fail.

## 1. Room not measured (floor and wall area fail)

- **What:** no floor or wall area; the whole sweep ran in camera mode.
- **Root cause:** the room is measured only from ARCore depth, and the AR room step (*Measure room (AR)*,
  added on the last day) was not started in this sweep. Earlier AR sweeps on this phone failed inside ARCore:
  in a standalone ARCore app (AR Ruler) no plane was found and ARCore logged camera-to-motion-sensor clock
  offsets of −54 to +22 ms (its limit is 5 ms) and 32 tracking resets; in Chrome the phone's position drifted
  up to 40 m while standing still. Our AR geometry passes the synthetic replay (floor within 10 %, walls 15 %).
- **What I would change:** start the AR step by itself at the room step when ARCore passes a 1 m depth check;
  otherwise measure the room from the camera frames, with the book rulers as scale.

## 2. Non-book items: 10 of 18 found (bar 80 %)

- **What:** missed the padlock and curtain rod (never detected), chair and bucket (seen once or twice at low
  confidence, dropped as background noise), the mirror (called a light switch by the vision model and removed
  as a wall fixture) and the second doormat; the gift box was named "box" (household, not decor) and the
  kitchen shelves "cabinet".
- **Root cause:** a pan shows small objects briefly, and repeats are merged by name when two objects never
  share a picture: the two doormats differ in colour (match 0.11 and 0.15) yet were listed as one.
- **What I would change:** merge repeats by their position in the room and colour, not by name. A colour-only
  rule found the doormat on this sweep but listed one pressure cooker twice, so it was not shipped.

## 3. Book prices: 6 of 10 within 25 %

- **What:** Principles of Building AI Agents ₹1,969 (reference ₹851), Patterns for Building AI Agents ₹625
  (₹449), The Lost Symbol ₹294 (₹547), My Experiments with Truth ₹360 (₹236).
- **Root cause:** listings move: Patterns and The Lost Symbol changed price between the reference reading
  (01:06) and the sweep (12:15). Two listings 7.8 times apart (₹449, imported ₹3,489) were averaged. A
  neighbour's author leaked into the Gandhi spine text, so a "vol 1" record matched and that edition was
  priced. The reference prices were read from the same searches by the AI assistant, not by a person.
- **What I would change:** done after the run: two listings far apart are two editions (₹449). Next: keep
  looking past a review-level match, and re-read the reference by hand within an hour of the sweep.

**Near miss, spine thickness 15.2 %:** photo widths run wide (angled views; one mask covered two thin books,
5.1 cm) and catalogue page counts miss both ways (−33 % to +33 %); the two are combined. No constant was tuned
to this shelf.

**Measured fix:** spine heights on sweep `20261120-150808-ef87ca` went from 64.3 % to 7.9 % error when the
catalogue look-ups for the ruler books ran 6 at a time instead of one at a time within the same budget. Since
this run, items and book prices run beside the book stages: three sweeps replay in 39, 53 and 47 s instead of
71, 114 and 103 s, with the same books and items.

## Cost per sweep and latency per stage

- **Cost:** USD 0.096: 48 Serper Google Shopping searches (2 credits each). All models run locally; Open
  Library, Google Books and frankfurter.app are free.
- **Latency after "I'm done" (226 s):** select 5.3 s, analyze 50.3 (frames left over from the walk),
  register 23.9, read 10.0, resolve 33.9, verify 2.2, measure 15.6, items 69.0, prices and report 15.5.
  Voice (medians): speech-to-text 2.1 s, first spoken audio 0.9 s, whole turn 2.6 s.

Rules written after seeing the practice room are disclosed, with every earlier defect, in
[failure_history.md](failure_history.md).
