# lightsaber_fx: curved blade geometry during blade-on-blade contact

Status: approved (in chat; awaiting written-spec review)
Date: 2026-09-17

## Problem

Every fix made earlier in this session (confidence-weighted smoothing
`cadb19f`, hilt optical-flow tracking `bbc7414`+, tip confidence weighting
`83e94f6`/`3d75b05`, cable-component continuity rejection `bfbd39e`,
extended-range anchor-gap correction `a195214`) improved **where the
straight line's two endpoints sit**. None of them can fix what showed up
next: at frame 292 of the real job (fencing video, job
`58a8f662`/`120e85ed-...`), the raw video shows both épées visibly
**bowing** where they cross during a bind, while the rendered glow draws a
perfectly straight capsule from hilt to tip. The gap between "real,
non-straight mask shape" and "rendered straight line" is a genuinely
different kind of bug from anything fixed so far.

Root-caused with real-data numbers, not a guess:

- At frame 292, object 1 (blue)'s raw SAM2 mask bows up to **38.7px off a
  straight hilt-to-tip line (~17% of its 230px length)**; its pixel count
  spikes to 2084px vs. ~1400-1600px on clean neighboring frames -- a real
  contact-induced mask distortion, not noise.
- Object 0 (red) shows a smaller ~7.7px (~3%) deviation at the same frame.
- On frames far from any contact (50, 100, 150, 200, 480, 500), both
  objects deviate from straight by only **3-7px (1.3-2.9% of length)** --
  ordinary PCA-fit noise. This confirms the effect is contact-specific, not
  a general fitting-noise problem the existing smoothing constants could
  absorb.
- Frame 292 already falls *inside* the range corrected by `a195214`
  (`before_idx=290`, `after_idx=455`, corrected span 291-454) -- this is
  not a gap the extended-range fix missed. The hilt/tip endpoints there are
  already coming from validated, corrected data.

**Root cause:** `glow.py::_capsule_mask` -- the actual renderer -- always
draws one straight `cv2.line`-based capsule from hilt to tip, every frame,
unconditionally. `blade.py::fit_blade`'s PCA fit, the confidence-weighted
smoother, and `hilt_track.py`'s optical-flow recovery all only ever produce
*two* points per frame. There is no representation anywhere in the
pipeline for a bent blade, so no amount of endpoint-placement accuracy can
close this gap.

## Scope

- **Contact-only, not universal.** Curvature is only fit and rendered when
  a frame's own mask shows real, measured bow. Baseline noise elsewhere
  (1.3-2.9% of length) is well below render-visible thresholds at typical
  blade width -- spending new fitting/smoothing/rendering complexity there
  buys nothing and risks a visible wobble on otherwise-clean footage.
- **No fabricated curvature in the dead zone.** Inside a sustained overlap
  run's fully-merged-mask "dead zone" (where `hilt_track.py`'s
  optical-flow endpoints are the only signal at all), the per-object mask
  still technically exists on disk but is heavily contaminated --
  confirmed on the real job at frames 330/340/350: object 0's mask nearly
  fully *contains* object 1's mask (intersection ~100% of the smaller
  mask). A naive per-frame significance check can't tell that apart from
  genuine bow (see "Cross-object contamination gate" below); the straight
  hilt-to-tip line stays exactly as it is today for those frames --
  rendering a guessed curve from contaminated pixels would trade one kind
  of visible inaccuracy for another, less honest one.
- **Quadratic bend, not an N-point spline.** One new mid-blade control
  point per frame (hilt + bend + tip = a quadratic Bezier). Every bow
  measured so far, in every sampled frame, is a single-direction sag --
  never an S-curve or multiple inflections. A 3-5 point spline (evaluated
  and rejected below) would solve a generality nothing in this pipeline's
  real footage has ever exhibited.
- **Trigger is a per-frame significance check, gated by cross-object mask
  IoU.** A significance threshold on the frame's own mask shape alone is
  *not* sufficient on its own -- see "Cross-object contamination gate"
  below, a correction made after spiking this against the real job and
  finding it would otherwise fire across ~50-frame stretches deep in the
  known dead zone. The IoU gate reuses signal `_find_overlap_runs` already
  computes, rather than adding a new dependency on run/anchor detection
  logic itself.
- **Two-object jobs benefit most** (this is where blade-on-blade contact
  happens), but the mechanism itself is single-object -- it fits from one
  object's own mask, with no cross-object comparison. It's not restricted
  to the two-object code path.

## Approach considered and rejected: N-point spline

Consulted Gemini CLI for a second opinion on curve-fitting technique.
Two families of approaches came back beyond the one chosen:

1. **Distance-transform ridge spline:** `cv2.distanceTransform` on the
   mask, treat high-distance ridge points as a confidence-weighted spline
   fit (contamination pixels sit at the mask boundary, low distance value,
   naturally down-weighted). Robust, but a new dependency pattern (EDT) not
   used elsewhere in `blade.py`, and more compute per frame.
2. **Iterative principal curve:** initialize at the straight PCA line,
   iteratively re-project mask points onto the current curve and re-fit
   with robust local regression (LOWESS-style). Most general and most
   rigorous, also the most expensive, and the extra generality (handling a
   severe S-curve) has no real-footage evidence backing it.

Both were rejected in favor of the quadratic-bend approach below: it's the
smallest change that explains every measurement made this session, reuses
an existing codebase pattern almost verbatim, and if a real S-curve case
ever turns up, the bend-point field generalizes to more points later
without a rearchitecture.

## Approach: quadratic bend point

### Data model

`BladeGeometry` (in `blade.py`) gains one new field:

```python
class BladeGeometry(NamedTuple):
    centroid: tuple
    axis: tuple
    tip: tuple
    hilt: tuple
    length: float
    width: float
    angle: float
    bend: tuple | None   # NEW -- (x, y) mid-blade control point, or None
```

`motion.npz` (via `save_motion`/`load_motion`) gains a matching `bend`
array, `(N, 2)`, NaN-filled in rows where the frame's blade has no
significant bow -- mirroring exactly how `tip`/`hilt` already represent a
missing frame in this schema.

### Fitting, inside `fit_blade`

After the existing PCA axis/endpoint fit (unchanged): bin the mask's
points along the axis, reusing `_median_perpendicular_extent`'s existing
bin machinery (`width_bins`, default 20). Take the **median** perpendicular
offset in the bin nearest the blade's midpoint -- median, not mean, for
exactly the reason `_median_perpendicular_extent` already uses it: robust
to a single contaminated bin without any new robust-statistics machinery.

```python
BEND_SIGNIFICANCE_PX = 8  # see Constants for how this was calibrated
```

If the midpoint bin's median perpendicular offset exceeds
`BEND_SIGNIFICANCE_PX`, `bend` is populated as the point at the blade's
midpoint projection, offset perpendicular to the straight axis by that
median value. Otherwise `bend = None`.

This is a strictly local, single-frame, single-object computation -- no
change to `_largest_component`, no new parameters threaded in from
`compute_motion`'s caller, no coupling to the other tracked object's mask
or motion data at all. `fit_blade`'s existing single-object boundary is
preserved exactly (its own docstring: "this function only ever sees one
object's masks, so it can't tell a mask that's bled into a nearby tracked
object's blade from genuine fast motion") -- `bend` at this stage is a
**candidate**, not a final decision. The final decision needs cross-object
information `fit_blade` deliberately doesn't have, which is exactly what
the next step provides.

### Cross-object contamination gate (`suppress_overlap_bleed`)

A per-frame significance check on the mask alone is not sufficient. Spiked
directly against the real job before finalizing this spec: without a
cross-object gate, the significance check (at an 8-15px candidate
threshold) fires across long stretches deep inside the known 293-454
contact run -- e.g. frames 323-369 (47 frames), 393-425 (33 frames) for
object 0. Checked directly: at frames 330/340/350, object 0's raw mask
*nearly fully contains* object 1's mask (intersection 1015-1511px against
an object-1 mask of only 1041-1537px -- essentially all of object 1's
pixels also counted as object 0's). This is the dead zone's actual
signature -- SAM2 merging the two objects' masks -- not blade bow, and
rendering it as a curve would be worse than the straight line it would
replace.

The fix reuses signal that already exists: `_find_overlap_runs` already
computes a per-frame cross-object IoU array (`ious`, via `_mask_iou`) over
the *entire* shared frame range to detect runs in the first place. That
function's return value gains `ious` (or an equivalent `overlapping`
boolean array, `ious > CROSS_OBJECT_OVERLAP_IOU_THRESHOLD` -- the exact
comparison it already makes internally) alongside the `OverlapRun` list it
returns today. `suppress_overlap_bleed`, which already calls
`_find_overlap_runs` and already has both objects' masks/motion in hand,
uses this to clear `bend` back to `None` for both objects on any frame
where cross-object IoU exceeds `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD` --
no new IoU computation, no new constant, just an additional consumer of
data this function already produces.

Re-checked with this gate applied, across the *entire* 506-frame job: the
47/33/etc.-frame false-positive stretches disappear completely (0 frames
above threshold anywhere in the dead zone), and the only frames left with
a significant candidate `bend` are **291 and 292** -- exactly the pair at
the edge of the known run, exactly where the original screenshot showed
the divergence. This is strong confirmation the gate is correctly
separating "real, measurable bow" from "mask contamination," not just
suppressing the signal generally.

### `_stabilize_tip_hilt` (`glow.py`)

When this function's existing continuity fix flips tip/hilt on an axis
sign-flip, it must swap `bend` alongside them -- otherwise the bend point
stays associated with the wrong (now-relabeled) end and the curve bows the
wrong way for that frame.

### Temporal stability

Two problems with the *gated* per-frame `bend` estimate (i.e. after the
contamination gate above has already cleared contaminated candidates):
it can still jitter between frames from ordinary mask noise, and it can
visually "pop" into existence the instant a frame crosses
`BEND_SIGNIFICANCE_PX`.

This must run **after** the contamination gate, not before -- smoothing a
window that still contains an uncleared contaminated neighbor would blend
bad data into an adjacent legitimate frame's value before the gate ever
gets to clear it. So although the smoothing math itself needs no
cross-object data (no confidence weights, no run/anchor awareness, just
"smooth this one object's own already-gated per-frame estimate"), it runs
as a final step inside `suppress_overlap_bleed`, immediately after that
function clears contaminated candidates -- not in `compute_motion`, which
runs before the gate exists and would have nothing valid yet to smooth.
Rather than reusing the heavier cross-object curvature-penalized smoother
(`_smooth_run_field` -- built for a different problem: filling a *fully
absent* signal across a whole anchor-to-anchor gap using cross-object
confidence weights), it's a separate, simpler pass:

- **Denoise:** within each contiguous stretch of non-`None`, gated `bend`
  frames, a small local median/weighted-average window
  (`BEND_TEMPORAL_SMOOTH_WINDOW`) smooths neighboring frames' estimates.
- **Ramp in/out:** the smoothed bend offset is multiplied by a rise/fall
  fraction over the first/last few frames of a stretch
  (`BEND_RAMP_FRAMES`), mirroring the existing `ignition_fraction`
  rise/fall shape already used in `glow.py` for blade ignite/extinguish --
  same shape of problem (a signal turning on/off over a short window),
  reused rather than reinvented.

### Renderer (`glow.py`)

`_capsule_mask(shape, hilt, tip, width, extend_frac, hilt_taper_frac,
hilt_taper_min_frac)` currently builds a straight 4-point quad body between
a tapered hilt wedge and an extended, rounded tip cap. It needs a `bend`
parameter (default `None`) and, when given, must build that same
tapered/rounded capsule shape along a **curved centerline** instead of a
straight segment:

1. Sample the quadratic Bezier (hilt, bend, tip) into a short polyline
   (~12-16 points, plenty at these blade lengths/widths).
2. Apply the existing tip-extension logic along the curve's tangent at the
   tip (not a straight extrapolation past the raw tip).
3. Build the capsule outline as a sequence of quads/discs walking the
   polyline, instead of the single quad drawn today -- same constant
   width, same tapered hilt wedge, same rounded tip cap.

When `bend is None`, the polyline degenerates to the existing 2-point
case -- this is a **strict generalization** of `_capsule_mask`, not a
fork. Today's behavior (every frame outside contact) falls out for free.

`_build_blade_shape` and `_composite_blade_contribution` need to thread
`bend` through from the per-frame `motion.npz` read in `render_glow`/
`render_glow_multi`, the same way `tip`/`hilt` already are.

## Constants (calibrated against the real job before writing the implementation plan)

Spiked directly against every mask frame in the real job (506 frames,
both objects, not just the six spot-checked during brainstorming) before
committing to a value, matching the same discipline used for every other
fuzzy-decision constant in this codebase.

- `BEND_SIGNIFICANCE_PX = 8` -- threshold above which a frame's median
  midpoint-bin perpendicular offset is treated as real bow. With the
  cross-object IoU gate applied, the gated noise ceiling across the whole
  clip is 3.2px (object 0) / 4.2px (object 1, p99; max 28.0px, entirely
  attributable to frames 291-292 themselves), and the real signal sits at
  21.4-28.0px -- 8px sits with comfortable margin on both sides and
  produces zero false positives and zero false negatives against this
  job's one real contact run.
- **No new IoU constant** -- the contamination gate reuses the existing
  `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD` (already governs
  `_find_overlap_runs`'s own `overlapping` decision) rather than
  introducing a second, potentially-inconsistent threshold for the same
  underlying question ("are these two objects' masks colliding right
  now?").
- `BEND_TEMPORAL_SMOOTH_WINDOW` / `BEND_RAMP_FRAMES` -- small (2-3
  frames). With the IoU gate applied, this job's only bend-active stretch
  is frames 291-292 -- 2 frames, not the up-to-47-frame stretches seen
  before the gate was added. A single real 2-frame stretch isn't enough to
  fully pin an exact value (unlike `BEND_SIGNIFICANCE_PX`, which has a
  clean, wide separating margin) -- exact values to be finalized during
  implementation against this stretch plus any additional contact runs
  found in other real footage, but the real data available now rules out
  anything large.

## Testing

**Synthetic unit tests** (`tests/pipeline/test_blade.py`, extending the
existing `_bar_mask`/`_mask_at` conventions):

- A new bowed-mask helper (draws a mask along a quadratic curve, mirroring
  the existing straight-bar helpers) -- `fit_blade` populates `bend` when
  the bow exceeds `BEND_SIGNIFICANCE_PX`.
- **Regression guard:** every existing straight-mask test must keep
  returning `bend=None` unchanged. Since this touches `fit_blade`, a
  heavily-exercised, central function, this is the most important test in
  the set.
- Contamination robustness: inject an extra pixel cluster into one bin of
  an otherwise-straight synthetic mask (mimicking cross-object bleed),
  confirm the median-per-bin fit isn't thrown off.
- `_stabilize_tip_hilt` swaps `bend` correctly on a tip/hilt flip.
- **Cross-object contamination gate:** `_find_overlap_runs` returns `ious`
  (or equivalent) alongside its run list; `suppress_overlap_bleed` clears
  a synthetic frame's `bend` to `None` when cross-object IoU exceeds
  `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD`, and leaves it untouched below that
  threshold -- this is the single most important test in the whole
  feature given what real-data spiking found without it.

**Renderer tests** (`tests/pipeline/test_glow.py` -- read its existing
conventions before writing these):

- `bend=None` produces **byte-identical** output to today's straight
  capsule -- the critical no-regression guarantee for every non-contact
  frame in every existing job.
- A given `bend` point produces a capsule whose sampled polyline points
  are offset from the straight hilt-tip line in the expected
  direction/magnitude.

**Temporal-stability tests:** ramp in/out at stretch boundaries (mirroring
existing `ignition_fraction` tests), denoising reduces synthetic per-frame
jitter across a noisy synthetic stretch.

**Real-data validation** (same discipline used for every fix this
session -- not optional):

- Re-run `compute_motion` + render on the real job.
- Frame 292 specifically: numeric check that the rendered curve now tracks
  the raw mask's measured bow (not just the straight endpoints), plus
  direct visual inspection (the same raw-vs-render zoomed-crop technique
  used to root-cause this in the first place).
- A dense visual montage across 286-299 to confirm a smooth transition in
  and out of curvature, no popping.
- A broad sweep across the rest of the clip to confirm zero regressions on
  every non-contact frame.
- Full test suite + ruff before any commit, as established all session.

## Out of scope / explicit non-goals

- **Not universal.** Straight-line rendering stays the default and the
  only behavior outside a measured bow; this spec does not change
  `fit_blade`'s output on any frame that isn't visibly bowed.
- **Not an N-point spline.** Explicitly rejected above; revisit only if
  real footage ever shows a multi-inflection curve, which nothing sampled
  this session does.
- **Not curvature in the dead zone.** No new signal is fabricated where
  `hilt_track.py`'s optical-flow endpoints are the only trustworthy data.
  The per-object mask on disk there is contaminated, not absent -- `bend`
  is kept `None` by the cross-object IoU gate in `suppress_overlap_bleed`,
  not because `fit_blade` finds nothing to fit (see "Cross-object
  contamination gate").
- **Not a change to tip/hilt position or orientation logic.** This spec
  only adds a third control point between two endpoints whose own
  placement logic (PCA fit, confidence-weighted smoothing, hilt
  optical-flow override) is unchanged.
