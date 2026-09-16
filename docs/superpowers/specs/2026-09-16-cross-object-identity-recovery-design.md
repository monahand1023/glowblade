# lightsaber_fx: recovering tracked-object identity after a crossing

Status: approved
Date: 2026-09-16

## Problem

On real multi-object footage (a two-person fencing bout, job `58a8f662`),
one tracked blade's glow rendered at a wildly wrong length and angle for a
stretch of the clip, and separately, one blade's glow vanished entirely for
the rest of the clip after the two blades crossed.

Root-caused to two distinct bugs:

1. **Fixed already** (`fit_blade` in `blade.py`): a noisy per-frame SAM2 mask
   occasionally included a small, disconnected speck of foreground far from
   the real blade (in this footage, a fencer's body cord). PCA gives that
   speck outsized leverage, badly skewing the fitted axis and length.
   `fit_blade` now restricts itself to the mask's largest connected
   component before fitting.
2. **This design**: when the two tracked blades visually cross/touch,
   SAM2's shared multi-object tracking session (`track_objects` in
   `track.py`) can permanently lose the distinction between the two
   `obj_id`s -- both converge onto tracking the *same* physical blade for
   the rest of the clip. Confirmed directly on the real footage: from frame
   ~296 onward, both objects' mask centroids sit within 1-8px of each other
   (fully merged) all the way to the end of the clip (frame 505), even
   though the two blades are clearly visually separate again by frame 480.
   The tracker never self-corrects once this happens.

Multiple tracked objects crossing is not a rare edge case for this
feature -- it's close to guaranteed on any real multi-person clip -- so this
needs an actual fix, not just a documented limitation (the current state,
per `README.md`).

### Relationship to the existing multi-saber tracking design

`docs/superpowers/specs/2026-09-15-multi-saber-tracking-design.md` already
proposed a fix for tracking failures during multi-object rendering: a
mid-render *interactive* correction flow (pause the render, show the user
the flagged frame, let them click new points). That design was written and
approved but **never implemented** -- today's `track_objects()` is the
plain synchronous version with no pause/resume/correction machinery.

Two reasons this design doesn't build on top of that one:

- That design's detection heuristic (mask area empty, or >6x a reference
  size) would **not have caught this failure**. During the crossing here,
  both objects' masks stayed normal-sized (~1300-2700px, in line with
  their usual size) -- it's a positional/identity mixup, not a size
  anomaly.
- This fix is fully automatic (matching how initial detection already works
  via Gemini), needing no new SSE event shapes, pause/resume state, or
  picker UI.

## Scope

- **Pairwise only**: this reconciliation stage runs only for jobs tracking
  **exactly 2 objects**. Jobs with 3 or 4 tracked objects skip it entirely
  and keep today's behavior (undetected, documented limitation) even if
  only 2 of the 3-4 objects actually cross. Matches the only failure mode
  actually observed, and avoids the added complexity of checking multiple
  object pairs and reasoning about overlapping/simultaneous merges within
  one job. A future pass could extend detection to run per-pair within a
  larger job if that turns out to matter in practice.
- **During the crossing itself**: the "lost" object's blade holds at its
  last known-good position (frozen) rather than disappearing, per Dan's
  call. This needs new data (frozen mask/geometry written into the gap),
  not new rendering logic -- `render_glow_multi` already renders whatever
  mask each frame has.
- **Failure is silent, not fatal**: if any step of recovery fails (no
  `GEMINI_API_KEY`, a Gemini error, no clean separation found within the
  search window), that pair's masks are left exactly as `track_objects`
  produced them -- today's behavior. The render always completes; it never
  blocks or errors out because recovery didn't work.

## Approach

A new post-hoc **reconciliation** stage, run after `track_objects()`
finishes and before `compute_motion()` runs. `track_objects()` itself is
untouched -- it keeps producing (possibly identity-corrupted) masks exactly
as it does today. The new stage reads those masks directly off disk, and
where it finds a crossing, patches the affected object's mask files in
place before anything downstream (`compute_motion`, `render_glow_multi`)
ever sees them.

This was chosen over detecting/handling the crossing *during* SAM2's
`propagate_in_video` loop because it keeps the existing, already-working
tracking code completely unmodified -- all new complexity lives in one new,
independently testable module that only ever reads and writes mask files.
The cost is some redundant compute (re-tracking a stretch that already ran
once, for the object being recovered) but that's negligible next to a
render's total runtime, and it's only ever paid when a crossing is actually
detected.

New module: `src/lightsaber_fx/pipeline/reacquire.py`.

## Design

All frame counts and thresholds below (IoU 0.8, 15-frame sustain, 90-frame
lookback, 10-frame search step, 150-frame search cap) are starting
defaults validated only loosely against the one real clip this was
diagnosed on -- tune during implementation, same as the area-health
heuristic's `K = 5` in the older multi-saber design.

### 1. Detecting a crossing

For every pair of tracked objects in the job, compute per-frame mask IoU
(both masks are already on disk). Flag a crossing when IoU stays above a
high threshold (0.8) for a sustained run of at least 15 consecutive frames
(~0.5s at typical frame rates).

The *sustained* requirement is the important part: blades briefly touching
and correctly separating again (normal fencing contact, tracked correctly)
must not trigger this -- only a merge that doesn't recover on its own does.
Confirmed on the real footage: once merged at frame 296, IoU-equivalent
separation never recovered on its own through frame 505 (200+ frames), a
clearly different signature from a momentary touch.

`merge_start_frame` is defined as the *first* frame IoU crossed the
threshold, not the frame the sustained check confirms it at -- that's the
frame the frozen-blade gap (design section 4) starts from.

### 2. Finding a clean reference frame

Walking backward from `merge_start_frame` using the two objects' already-
computed centroids (from their existing, pre-corruption mask data), find
the frame of maximum separation within a lookback window (90 frames,
~3.6s). This is deliberately *not* "the last frame before the merge" --
the spike that validated this approach found that frame is often already
contaminated (on the test clip, frame 294 -- one frame before the
confirmed merge -- was already down to 7px separation, only 23px more than
the fully-merged state, and using it as the identity reference produced a
**swapped** match). Walking back to the true local separation maximum
avoids this.

### 3. Re-acquiring the lost object

Walking forward from `merge_start_frame` in steps (every 10 frames), ask
Gemini (same call shape as `detect_blades_vlm` in `vision_detect.py`, but
pointed at a specific frame instead of its own motion-scored frame choice)
to find blade-like boxes in that frame. Stop at the first frame where it
returns >=2 boxes whose SAM2-predicted masks don't overlap each other --
this is the `reacquire_frame`.

Match each of the 2 new detections back to the original `obj_id`s by
nearest centroid to the clean reference frame from step 2 (validated
directly on the real footage: 43px and 93px, unambiguous). Whichever
original object's post-merge track landed far from its own match is the
"lost" one that needs recovery; the other is "kept" (its `obj_id` never
actually left the right target, despite the duplicate).

Start a fresh, independent `track_object()` session for the lost object
only, prompted with its matched detection's points at `reacquire_frame`,
tracking across the full clip (reusing the existing `frames_dir` --  no
re-extraction needed).

### 4. Patching the masks

For the lost object only:

- Frames `[merge_start_frame, reacquire_frame)`: overwritten with a frozen
  copy of that object's own last-good mask (the reference frame from step
  2).
- Frames `[reacquire_frame, end)`: overwritten with the freshly tracked
  masks from step 3.

The kept object's masks are never touched.

After patching, `compute_motion()` runs fresh on the corrected masks for
the recovered object (unchanged function, unchanged call site in
`runner.py` -- it has no idea reconciliation happened).

### 5. Integration point

`runner.py`, immediately after `track_objects()` returns and before the
per-object `compute_motion()` loop, for jobs with exactly 2 tracked
objects. Jobs with 1, 3, or 4 objects skip this stage entirely (1 has
nothing to cross with; 3-4 are out of scope per the pairwise-only
decision above).

### 6. Failure handling

Any of the following causes that pair's reconciliation to be skipped
(masks left exactly as `track_objects` produced them, matching today's
behavior) rather than raising:

- No `GEMINI_API_KEY` / `GOOGLE_API_KEY` configured.
- A Gemini API error (network, auth, bad response).
- No frame within the forward search window (capped, e.g. 150 frames from
  `merge_start_frame`) produces 2 non-overlapping detections.
- The re-acquisition track itself looks implausible (e.g. `fit_blade`
  returns `None` immediately, or the recovered object's length is a wild
  outlier vs. its own pre-merge sizes) -- a basic sanity check, not a hard
  requirement, since a real recovered blade's size can legitimately vary
  with pose.

This matches the project's existing "decline rather than guess wrong"
philosophy (same one behind `_detect_proposals`'s VLM-then-motion fallback
in `server.py`, and the elongation-based rejection in `detect.py`).

## Testing

- **`detect_merges`**: synthetic mask sequences (no SAM2/Gemini) --
  asserts a sustained high-IoU run triggers, a brief touch-then-separate
  run does not, and `merge_start_frame` lands on the first threshold
  crossing, not the confirmation frame.
- **`find_clean_reference`**: synthetic centroid sequences with a gradual
  approach into a merge -- asserts it returns the true local-maximum-
  separation frame, not simply the last frame before the merge (the exact
  case the spike caught).
- **`reacquire_pair`**: Gemini client injected as a test double (same
  pattern as `detect_blades_vlm`'s `client` parameter), a stubbed SAM2
  image predictor -- asserts correct identity matching given known
  reference/detection centroids, and that it returns nothing (triggering
  the failure-handling path) when the double reports <2 non-overlapping
  boxes within the search window.
- **`patch_masks`**: asserts the frozen-gap and freshly-tracked frame
  ranges land exactly where expected, and that the kept object's mask
  files are byte-identical to their pre-reconciliation state.
- **Integration-style**: the real job's recorded masks
  (`58a8f662`) as a fixture, asserting reconciliation detects the known
  merge at the known frame and produces two spatially-separated objects
  from `reacquire_frame` onward -- mirrors the spike's manual verification,
  automated.
