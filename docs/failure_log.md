# Failure log

Run: `TODO(final run): sweep id` · 10 books, 18 non-book items, one room (4.45 × 2.69 × 3.15 m) · OnePlus 15
(CPH2745, Android 17, Chrome 157, ARCore 1.57) · full results in [results](TODO(final run): link to eval.md).
Pass bars: `TODO(final run): one line from make eval`.

## 1. Room not measured (floor and wall area fail)

- **What:** the packet has no floor or wall area; the sweep runs in camera-only mode.
- **Root cause (proved): ARCore itself fails on this phone.** In a standalone ARCore app (AR Ruler: no browser,
  none of our code) aimed at a patterned bedspread, no plane was ever found, and ARCore's own log reported the
  camera and motion-sensor clocks out of step 74 times (−54 to +22 ms; ARCore's limit is 5 ms) and 32 tracking
  resets. Chrome shows the same errors (Google's model-viewer and our test page `web/artest.html`): the phone's
  position drifted up to 40 m while standing still, and depth read 0.5–50 m at 1 m. Our AR geometry passes the
  synthetic replay (floor within 10 %, walls within 15 %, height within 0.1 m).
- **What I would change:** measure the room without ARCore: scale from the catalogue book rulers already used
  for spines, direction from the phone's motion sensors, room shape from the frames. Or sweep with a phone
  whose ARCore passes the 1 m check on `web/artest.html`.

## 2. TODO(final run): second-worst error (likely non-book items, TODO% found against 80 %)

- **What:** TODO(final run).
- **Root cause:** a room pan shows most objects briefly and partly. The detector misses household objects it
  has no name for, and the vision model names items from partial crops (a clothes iron called a hand mixer).
- **What I would change:** have the agent ask for a short pause on a corner where it saw an unnamed object
  (the brief allows re-capturing a corner during the pass); name each item from its three best crops.

## 3. TODO(final run): third-worst error (likely book prices, TODO of 8 within 25 %)

- **What:** TODO(final run). The app's prices are mostly lower than the hand-checked ones.
- **Root cause:** the reference is Flipkart's listing for the owned edition; the app takes the median of the
  main price band across Indian sellers, where cheaper reprints dominate (A Tale of Two Cities: ₹208 against
  ₹569 for the owned deluxe hardcover). The spine rarely shows which edition it is.
- **What I would change:** when the spine names the publisher, price that edition's ISBN at a named retailer
  first; otherwise report the market range and mark the edition as unconfirmed.

## Measured fix

Spine heights on the same sweep (`20261120-150808-ef87ca`, re-run on the same frames): height error
**64.3 % → 7.9 %** (values within 15 %: 19 % → 62 %). Root cause: the catalogue look-ups that give each
in-frame ruler book its true height ran one at a time and most did not finish within the stage's time
budget, so most books had no ruler. Fix: look-ups run in parallel (6 at a time) inside the same budget.

Tried and not shipped: on the same sweep one outline covered a leaning book and its neighbour, so the
neighbour's title went to the wrong spine (Go Set a Watchman measured 5.6 cm thick; true 1.9 cm). Ignoring
text near an outline's long edge fixed that book, but on another sweep it kept junk text from an outline that
spanned two books and listed one book twice. Re-run on three sweeps, no pass bar improved, so it was dropped.

## Cost per sweep and latency per stage

- **Cost:** TODO(final run) from `metrics.json`. All models run locally (no per-sweep cost). Paid calls are
  Serper Google Shopping searches (2 credits each; 2,500 free credits at sign-up); Open Library, Google Books
  and frankfurter.app are free.
- **Latency after "I'm done":** TODO(final run) per stage from `metrics.json` (select, analyze, register, read,
  resolve, dedupe, verify, measure, items, room, price_assemble; total). Voice turn: speech-to-text TODO s,
  speech out TODO s.

Rules written after seeing the practice room are disclosed, with every earlier defect, in
[failure_history.md](failure_history.md).
