# lightsaber_fx: recovering hilt position via optical flow during sustained blade contact

Status: approved (in chat; awaiting written-spec review)
Date: 2026-09-17

## Problem

On the real fencing footage (job `58a8f662`), a sustained close-range
engagement between the two tracked blades (frames 293-454, 162 frames,
6.6s at 25fps) makes both objects' independent SAM2 masks -- and the
`fit_blade` PCA geometry fitted from them -- converge onto essentially the
same shape for large stretches of the run. `blade.suppress_overlap_bleed`'s
confidence-weighted smoother (`cadb19f`, committed earlier in this session)
already handles this by falling back to a low-curvature path between the
run's two known-good boundary frames wherever the two objects' raw fits
can't be trusted apart. That fix is real and correct as far as it goes, but
measured directly on this run: **~65% of its frames have essentially zero
confidence** (the two objects' raw centroids land within a few px of each
other), so for most of the run's 6.6 seconds the rendered glow blade is
just following a smooth curve between two distant anchors with no real
per-frame position data at all. Visually, this looks like the blade
floating disconnected from the fencer's hand -- confirmed by rendering the
real job and inspecting frames 360-430, where one or both glow blades sit
in mid-air, not attached to either hand.

The prior smoother can't improve on this on its own: it only has the
contaminated mask-derived geometry to work with, and that geometry has no
signal left in the dead zone. Fixing the dead zone needs a genuinely
different source of information.

### Key insight (external second opinion, via Gemini CLI)

The two blades are thin, low-texture, and visually merge easily -- exactly
what makes SAM2 mask segmentation confusable here. The fencers' hands/hilts
(gripped, gloved) are comparatively high-texture and, on this footage,
rarely occupy the same pixels even when the blades themselves cross. A
classical point tracker (optical flow) seeded on the hand/glove region,
run directly on the raw video frames, is a different technique with a
different failure mode -- it doesn't care what the blade masks are doing at
all, so it can succeed exactly where the mask-based approaches get
confused.

## Scope

- **Position only, not angle.** This recovers each tracked object's
  **hilt position** through the dead zone. It does **not** attempt to
  recover the blade's pointing angle/length there -- that stays exactly
  as today, a smoothed approximation anchored at the run's boundaries
  (`blade._smooth_interpolate_run`). Chosen over also deriving angle via a
  local line/edge search: meaningfully more complex (a new image-processing
  step with its own failure modes on a thin, motion-blurred blade), and the
  actual complained-about visual bug -- the blade floating disconnected
  from the hand -- is fixed by position alone.
- **Runs only for cross-object overlap runs `retrack_overlap_runs` didn't
  already resolve.** A run in `resolved_ranges` already has a real,
  validated independent SAM2 re-track for both objects and needs nothing
  further.
- **Fails silently, same philosophy as every other recovery stage in this
  pipeline** (`reconcile_pair`, `retrack_overlap_runs`): if tracking can't
  be validated for a given run/object, that run/object gets no override
  and keeps today's behavior (the confidence-weighted smoother's existing
  output). The render always completes.
- **Two-object jobs only**, matching `suppress_overlap_bleed`'s own
  existing scope (`runner.run_pipeline_multi` only calls any of this
  cross-object-recovery machinery when `len(object_ids) == 2`).

## Approach

A new module, `pipeline/hilt_track.py`, with no dependency on SAM2, masks,
or `blade.py`'s internals -- it operates purely on raw video frames and a
handful of known-good (x, y) points, making it independently testable with
synthetic images.

### Core primitive: `track_hilt_through_run`

```python
def track_hilt_through_run(
    frames_dir, frame_indices, run_start_frame, run_end_frame,
    before_frame, before_hilt, before_length,
    after_frame, after_hilt, after_length,
):
    """Recover per-frame hilt (x, y) positions for one tracked object
    through [run_start_frame, run_end_frame] (inclusive) by tracking
    forward from before_hilt (known-good, at before_frame) and backward
    from after_hilt (known-good, at after_frame) via classical optical
    flow on the raw video frames.

    Returns {frame_number: (x, y)} for whichever frames in the run a
    validated estimate exists for -- a subset of
    [run_start_frame, run_end_frame], possibly empty.
    """
```

For each direction (forward / backward):

1. **Seed**: `cv2.goodFeaturesToTrack` within a `HILT_SEED_WINDOW_RADIUS_PX`
   square window around the starting known-good hilt point, on the
   grayscale starting frame. Fewer than `MIN_SEED_FEATURES` good corners
   found -> decline this direction entirely (no trackable texture there,
   matching this codebase's established "decline rather than guess"
   philosophy -- see `reacquire._two_separate_detections`,
   `detect.MIN_ELONGATION`).
2. **Track**: `cv2.calcOpticalFlowPyrLK`, frame-to-frame, sequentially
   across the run in that direction's order. Points whose tracking status
   flag comes back failed are dropped; if *all* points are lost before
   reaching the far end, the direction stops there (matches
   `reacquire._retrack_one_object`'s "lost the blade" early-decline). Each
   frame's position estimate is the **median** (not mean) of whichever
   points are still tracked -- resistant to any single point drifting
   off the real feature.
3. **Validate**: the direction's landing position at the *opposite* known-
   good anchor is compared against that anchor's true position (already
   known -- it's a good frame). Must land within
   `HILT_TRACK_MAX_DRIFT_FRAC * (that anchor's own fitted blade length)`
   to be trusted at all -- directly mirrors
   `reacquire.RETRACK_MAX_DRIFT_FRAC`'s already-proven validate-against-
   the-far-known-good-anchor approach, applied to a tracked point instead
   of a whole re-tracked mask. A direction that fails validation, or never
   reached the far anchor, is discarded entirely (none of its frames are
   used, even the ones before the point it lost tracking or drifted).

**Blend**: where both directions validate and both cover a frame, blend by
each direction's own distance from its start anchor -- the same
`t = (frame - before_frame) / (after_frame - before_frame)` fraction
`blade._interpolate_row`/`_smooth_interpolate_run` already use elsewhere in
this codebase for the identical before/after-anchored-run shape:
`blended = (1 - t) * forward_estimate + t * backward_estimate`. At `t=0`
(the `before_frame` end) this is 100% the forward track, which started
there and has had zero distance to drift; at `t=1` (the `after_frame` end)
it's 100% the backward track, for the same reason at its own end. Where
only one direction validates, use it alone (weight 1.0) for whichever
frames it covers. Where neither validates, the run/object gets no
override.

### Wiring into the existing pipeline

A thin wrapper, `compute_hilt_overrides`, loops `blade._find_overlap_runs`
(the same shared run-detection step `suppress_overlap_bleed` and
`retrack_overlap_runs` already use) for both objects, skipping any run
inside `exclude_frame_ranges` (`retrack_overlap_runs`'s `resolved_ranges`)
and any run missing either anchor (nothing to validate against, same as
every other stage here), calling `track_hilt_through_run` per object per
remaining run, and merging the results into two flat
`{frame_number: (x, y)}` dicts (one per object).

```python
def compute_hilt_overrides(
    frames_dir, masks_dir_a, masks_dir_b, motion_path_a, motion_path_b,
    exclude_frame_ranges=(),
):
    """Returns (hilt_overrides_a, hilt_overrides_b)."""
```

`blade.suppress_overlap_bleed` gains two new optional parameters,
`hilt_overrides_a=None` / `hilt_overrides_b=None` (each a
`{frame_number: (x, y)}` dict, matching `compute_hilt_overrides`'s output).
After its existing per-run smoothing (`_smooth_interpolate_run`, unchanged)
produces that run's `tip`/`hilt`/`centroid`/`width`, a new step
(`_apply_hilt_overrides`) replaces `hilt` with the tracked position for any
frame present in the override dict, then re-derives `axis`/`length`/`angle`
from the (already-smoothed) `tip` and the new `hilt` -- the same
re-derive-from-tip-and-hilt pattern `_smooth_interpolate_run` and
`_interpolate_geometry` already use elsewhere in this file. `centroid` and
`width` are left as the smoother produced them -- hilt-tracking only has
evidence about the hand's position, not the blade's overall shape or the
mask's centroid.

`runner.run_pipeline_multi` calls `compute_hilt_overrides` once, right
after `retrack_overlap_runs` (and the `compute_motion` re-run for whatever
it patched) and right before `suppress_overlap_bleed`, passing
`resolved_ranges` through as `exclude_frame_ranges` and threading the two
resulting override dicts into the `suppress_overlap_bleed` call.

## Constants (starting values -- calibrate against the real job during implementation)

Every constant in this codebase that governs a fuzzy real-footage decision
(`RUN_SMOOTHING_STRENGTH`, `POSITION_GLITCH_JUMP_PX`,
`CROSS_OBJECT_OVERLAP_IOU_THRESHOLD`, `RETRACK_MAX_DRIFT_FRAC`, ...) was
picked by measuring against the real job, not guessed a priori and left
alone. These follow the same rule -- the values below are reasoned starting
points for the implementer to begin from, not final:

- `HILT_SEED_WINDOW_RADIUS_PX = 45` -- half-width of the square seed
  window. Large enough to cover a gloved hand at this footage's framing,
  small enough to avoid pulling in the *other* tracked object's hand or
  blade during close contact.
- `MIN_SEED_FEATURES = 4` -- fewer good corners than this makes the
  per-frame median position too noisy to trust.
- `HILT_TRACK_MAX_DRIFT_FRAC = 0.3` -- tighter than
  `RETRACK_MAX_DRIFT_FRAC`'s 0.5, since a hilt point should track more
  precisely than a whole re-tracked blade mask.
- `cv2.calcOpticalFlowPyrLK`'s own `winSize`/`maxLevel` and
  `cv2.goodFeaturesToTrack`'s `qualityLevel`/`minDistance` -- standard
  OpenCV defaults are a reasonable starting point (`winSize=(21, 21)`,
  `maxLevel=3`, `qualityLevel=0.3`, `minDistance=7`); tune only if real-data
  validation shows a specific failure these don't explain.

## Testing

**Synthetic (no real video needed):** small JPEG frame sequences written
to a temp `frames_dir` (matching the `{idx:05d}.jpg` convention
`reacquire.py`'s `_frame_path` already establishes) with a distinctively
textured patch (e.g. a small checkerboard) translated by a known amount
frame to frame. Covers: the tracker recovers a known translation; a
direction that lands far from its validating anchor gets discarded; a
window with insufficient texture (e.g. a flat gray patch) declines rather
than guessing; forward+backward blending when both validate; graceful
"no override" when neither direction validates.

**Real-footage validation**, same rigor as every fix this session: re-run
the real job's masks/motion through the full corrected pipeline, read the
WARNING logs, and -- critically, since the previous fix's regression was
only caught by looking at rendered frames -- render the full clip and
visually inspect frames throughout the 293-454 run (not just a few
samples that happen to land in the "already fine" edges) against the raw
source video, confirming the hand-holding-the-blade attachment holds
through the previously-dead middle. If real footage shows the fencers'
hands *also* get confused/occluded during part of this run, that's the
actual limit of this technique on this footage -- to be reported honestly,
not forced.

## Out of scope / explicit non-goals

- Blade angle/length recovery in the dead zone (see Scope).
- Jobs with 3-4 tracked objects.
- Any change to `retrack_overlap_runs`, `reconcile_pair`, or the
  confidence-weighted smoother's existing per-run logic for `tip`/`width`.
- A general-purpose point tracker reusable outside this specific
  before/after-anchored-run scenario.
