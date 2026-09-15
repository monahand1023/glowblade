# lightsaber_fx: vision-assisted multi-object auto-detection

Status: draft (pending Gemini gap-review + Dan's review)
Date: 2026-09-15

## Problem

Today's automatic detection (`detect.py`'s `detect_blade`) finds exactly one
object per clip, via optical-flow motion seeding + SAM2 scoring. This was a
deliberate scope cut in the multi-saber design
(`2026-09-15-multi-saber-tracking-design.md`, "Explicitly out of scope for
this pass": *"Automatic detection of multiple objects... The user places all
initial points manually, per saber"*) -- at the time, multi-object tracking
robustness itself was the open question, and automatic multi-object detection
would have been a second unknown stacked on the first.

That pipeline now exists and works (the multi-slot picker built earlier
tonight lets a user add up to 4 saber slots and manually draw a line on each
object). But manual selection on a busy multi-person clip is exactly the
tedious, error-prone case that motivated tonight's other work: the
mask-preview safety check exists *because* a hand-drawn line can miss a thin
blade and grab a background flag or a torso instead. A vision-language model
looking at one frame of a 3-person swordfight and returning "here are the 3
swords, roughly here" removes that manual step for the common case, the same
way `detect_blade` already removes it for one object.

## Scope

- Detect up to **4** objects per clip (matches `MAX_SABERS`, the multi-slot
  picker's own cap).
- **Gemini 3.8 Flash** as the vision backend, via the official `google-genai`
  Python SDK (`pip install google-genai`) -- not the `gemini` CLI shelled
  out to; a real dependency, not a subprocess wrapper.
  - Cost is not a design constraint here: ~1,548 image tokens for a 1280x720
    frame (Gemini's tiling formula: `768x768` tiles at 258 tokens/tile) plus
    a few hundred prompt/response tokens comes to roughly **$0.003 per
    frame analyzed** at Flash's $0.75/$3.75 per-million-token rates -- call
    it a few cents for a whole day of testing, effectively free at this
    tool's actual usage volume.
  - Auth via `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in the environment,
    read by the SDK itself. No key stored or handled by this codebase.
- Runs **automatically on upload**, same trigger point as today's
  `POST /api/jobs/{id}/detect` (which currently auto-fires from
  `uploadFile()` in `app.js`).
- **Falls back to today's behavior** -- unchanged `detect_blade` -- whenever
  the vision path can't be used: no API key configured, the network call
  errors or times out, or Gemini's response, once validated (see below),
  yields zero usable candidates. The no-key/offline case is not a degraded
  mode to apologize for; it's the tool working exactly as it does tonight.
- **Web app only.** The CLI's `run --auto` keeps calling `detect_blade`
  directly. Same reasoning the multi-saber design already used for its own
  web-only cut: this is inherently a picker-UI-facing feature (multiple
  proposals shown for the user to accept/override), and the CLI's
  interactive-popup model would need its own design for that. Not requested.

### Explicitly out of scope

- **Fixing tracking drift.** This feature changes where SAM2 gets seeded on
  frame 0 (or wherever the chosen frame is); it does not touch
  `propagate_in_video` or anything about how a track behaves once it starts.
  Tonight's investigation (independent-session test, exclude points, the
  `hiera_large` checkpoint, box+point prompts, periodic cross-repulsion
  re-prompting) ruled out every cheap fix for sustained drift on hard footage
  -- a better initial box is not that fix, and this spec does not claim it
  is. It happens to dodge one specific failure mode (the box+point spike
  earlier tonight avoided the flag-lock case) without fixing the underlying
  problem (the same object then drifted onto a different wrong target
  instead) -- an incidental, not a reason to scope this feature around drift.
- **Multi-frame analysis / cross-frame object identity.** V1 sends exactly
  one representative frame in one API call. Sending several frames and
  matching "sword A in frame 10" to "sword A in frame 80" is a real, useful
  idea (worth a future pass if single-frame proves unreliable in practice)
  but it's a second hard problem -- an easier version of exactly the
  identity-confusion problem SAM2 itself struggles with -- not a small
  addition to this one.
- **Mid-render interactive correction.** Still not built (the multi-saber
  design's Phase C). This feature is a complementary, much smaller piece:
  better *initial* seeding, not a replacement for correcting a track that's
  already gone wrong mid-clip.
- CLI support.
- Any object class beyond "a held, swung prop" -- the prompt is written for
  swords/bats/staffs specifically, not general object detection.

## Design

### New module: `vision_detect.py`

A new file alongside `detect.py`, not inside it. Different failure modes
(network errors, API keys, per-call cost) belong in their own file, matching
how this codebase already splits concerns one-file-per-module
(`blade.py`/`detect.py`/`track.py`/`glow.py`). `detect.py` itself is
untouched by this feature -- it remains the fallback, exactly as it is today.

```python
def detect_blades_vlm(video_path, checkpoint_path, config_name, device,
                       client=None) -> list[BladeProposal]:
    """Ask Gemini to find every held, swung prop in one representative
    frame, then validate each candidate through the same SAM2 + fit_blade
    gate detect_blade already uses. Returns 0-4 BladeProposal (the same
    type detect_blade returns), sorted by elongation descending.

    `client` is injectable (a genai.Client, or a test double) so tests never
    make a real network call -- same pattern as detect.py's own
    `_build_image_predictor` docstring: "Isolated so tests can replace the
    whole [external] dependency in one place."
    """
```

Pipeline, step by step:

1. **Pick a frame.** The middle frame of the clip (`n_frames // 2`) --
   simplest possible choice, and it matches `detect.py`'s own bias toward
   mid-swing frames (its own docstring: motion-based detection finds the
   frame "where the object was easiest to find, which is usually
   mid-swing"). No optical flow involved here; this is just "give the VLM
   a frame where people are mid-action, not standing still at frame 0."
2. **One Gemini call.** Send the frame plus a prompt asking for a bounding
   box (`[x_min, y_min, x_max, y_max]` in the frame's own pixel coordinates)
   around every person-held sword/bat/staff-like object, using
   `response_json_schema` for a typed, parseable response (a JSON array of
   `{box: [x0,y0,x1,y1], label: str}`) rather than free text -- avoids
   writing a JSON-extraction regex against prose.
3. **Validate the response shape.** Malformed JSON, an empty array, boxes
   with non-numeric or out-of-frame coordinates, or more than 4 entries
   (truncate to 4, keeping the first 4 -- the model isn't asked to rank
   them, so "first 4" is as good a cut as any) are all handled here, not
   left to blow up downstream.
4. **Per box, run the existing SAM2 + `fit_blade` gate.** For each
   validated box: `_build_image_predictor(...).predict(box=box,
   multimask_output=False)` (the same box-prompt API path tonight's
   drift-investigation spike already exercised and confirmed works), then
   `fit_blade` on the resulting mask, then the same `MIN_ELONGATION` check
   `detect.py`'s own `_candidate_masks` already applies. A box Gemini
   proposed that doesn't actually segment into a blade-shaped mask (e.g. it
   pointed at a shield, or at a person rather than tightly at their weapon)
   is dropped here -- defense in depth, not a decision to trust the VLM's
   labeling blindly.
5. **Return proposals.** Each surviving candidate becomes a `BladeProposal`
   (reusing the existing type from `detect.py`) with `frame_index` (the one
   chosen frame, shared by all of them -- satisfies `track_objects`' "all
   objects in one session must share a prompt_frame" constraint for free,
   with no extra handling needed), `points` (the same
   `_points_on_axis`-style sampled points `detect.py` already derives from
   a fitted mask, for consistency with how a manual line-draw becomes
   points), and `elongation`.

### Web API change

`POST /api/jobs/{id}/detect`'s response shape changes from one proposal to
a list:

```jsonc
// today:
{"found": true, "frame_index": 42, "points": [...], "elongation": 11.2, ...}
// new:
{"found": true, "proposals": [
  {"frame_index": 42, "points": [...], "elongation": 11.2, "mask_url": "..."},
  {"frame_index": 42, "points": [...], "elongation": 8.7, "mask_url": "..."}
], "source": "vlm"}  // or "source": "motion" when the fallback ran
```

The endpoint itself tries `detect_blades_vlm` first (when a Gemini client
can be constructed -- i.e. an API key is present); on any failure at any
step above (import error building the client, network error, zero validated
proposals), it falls back to today's `detect_blade` and wraps its single
result as a one-entry `proposals` list. The frontend never needs to know
which path ran except via the informational `source` field (useful for a
"found via AI" vs. "found via motion" hint in the UI, not load-bearing).

**Partial success is still success, not a fallback trigger.** If Gemini
finds 2 of the 3 swords actually in frame, the endpoint returns those 2
proposals as-is -- it does not additionally run `detect_blade` to try to
fill the third slot, and it does not treat "fewer than expected" as a
failure. The user sees 2 pre-filled slots and adds the third manually via
"+ Add saber," same interaction as adding any slot today. Mixing a VLM
proposal and a motion-detected proposal in the same response would mean
reasoning about two different `frame_index` values (the VLM's chosen middle
frame vs. whatever frame motion detection independently picks), which
breaks the "all objects share one prompt_frame" constraint `track_objects`
depends on -- not worth the complexity for a case the manual add-a-slot
flow already handles.

`source` is also useful for debugging -- worth threading into
`job_meta.json`'s already-existing `prompts` bookkeeping (added earlier
tonight specifically so a bad auto-detect could be diagnosed after the
fact) so `lightsaber-fx inspect` can show which path proposed a job's
points.

### Frontend change

`app.js`'s `detectOverlay` is currently a single value hard-coded to slot 0
(`activeSaberIndex === 0 && detectOverlay` in `redrawPoints()`), because
auto-detect only ever found one object. This needs to generalize to a
per-slot detected overlay:

- `detect()`'s success handler, instead of always writing into
  `sabers[0]`, creates one saber slot per returned proposal (up to today's
  `MAX_SABERS`), each carrying its own `points`/`prompt_frame` and its own
  `detectedMask` (renamed from the singular `detectOverlay`, now a per-saber
  field rather than a module-level variable).
- `redrawPoints()`'s overlay branch draws the *active* slot's own
  `detectedMask` instead of a single global -- the existing "first manual
  gesture on this slot discards its detected proposal" logic
  (`mouseup`'s `if (activeSaberIndex === 0 && detectOverlay)` block)
  generalizes the same way, keyed off the active slot instead of hardcoded
  to slot 0.
- Slot colors default per the existing `DEFAULT_SABER_COLORS` cycle
  (red/blue/green/red), same as manually-added slots -- vision detection
  doesn't propose colors, just locations.
- The one-object fallback case is visually identical to today: one slot,
  one overlay, nothing about the existing single-saber experience changes.

## Testing

- **`vision_detect.py` unit tests**: injected fake `genai.Client` (matching
  `detect.py`'s own test convention of stubbing `_build_image_predictor`)
  covering: a clean multi-box response producing N proposals; malformed
  JSON; an empty array; boxes with out-of-frame or non-numeric coordinates;
  more than 4 boxes (truncation); a box that fails the elongation gate once
  segmented (dropped, not returned). No real network call, no real SAM2
  inference in this layer's own tests -- `_build_image_predictor` is itself
  stubbed here the same way `test_detect.py` already stubs it.
- **Web endpoint tests**: `/detect`'s new `proposals` list shape; the
  fallback path (client construction raises, or a stubbed VLM call returns
  zero proposals) reaching `detect_blade` and wrapping its result correctly;
  `source` field correctness in both cases.
- **Frontend**: no automated test coverage (this codebase's established
  pattern for `app.js` -- manual browser verification only, per the
  session's own precedent for the picker/preview UI work). Manual pass:
  upload a multi-person clip with a configured API key, confirm multiple
  slots populate with distinguishable per-slot overlays; upload the same
  clip with `GEMINI_API_KEY` unset, confirm the exact today's-behavior
  single-slot fallback.
- **One real integration smoke test, gated behind an env var** (matching
  how `test_track.py` already skips its real-SAM2 tests when the checkpoint
  isn't installed): if `GEMINI_API_KEY` is set, run `detect_blades_vlm`
  against a real Mixkit knights-battling clip and assert it returns 2+
  proposals -- the only place this spec touches a real API call in test.

## Open questions for the plan phase

- Exact Gemini prompt wording -- needs iteration against real frames, not
  something to lock down before implementation starts.
- Whether `response_json_schema` support requires a specific `google-genai`
  SDK version floor -- check at implementation time, pin in `pyproject.toml`.
- Timeout value for the Gemini call (the fallback path depends on this not
  hanging the upload flow indefinitely).
