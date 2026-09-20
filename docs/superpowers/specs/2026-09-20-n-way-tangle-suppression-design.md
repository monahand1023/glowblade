# N-way tangle suppression — design spec

**Status:** designed and reviewed, deliberately deferred (2026-09-20) --
left as a known, documented limitation rather than built now. Not queued
for implementation; pick this up by reading this spec fresh rather than
re-deriving it, if it becomes worth doing later.
**Follows:** `docs/findings/2026-09-19-multi-saber-tracking-investigation.md`
(background, the real-footage evidence this design is built from, and the
"known gap, not urgent" note this spec now resolves)

## Problem

`runner.run_pipeline_multi`'s cross-object correction pipeline (commit
`371933d`) loops `reconcile_pair` → `retrack_overlap_runs` →
`suppress_overlap_bleed` over every *pair* of tracked objects. Each pair is
corrected independently, against its own before/after anchors. This has no
way to represent "three or more objects are simultaneously indistinguishable
right now" — it can only ever reason about two objects at a time.

Confirmed on real footage (job `0eb4fda2`, three Korean martial artists):
frames 43-76 (34 frames) show every one of the three tracked objects with
2+ simultaneous high-IoU partners — a genuine three-way raw-pixel collision,
not three independent pairwise contacts. Root cause is upstream and
unfixable: two of the three real blades are not visible in the source
footage during this stretch (confirmed by three independent from-scratch
SAM2 re-seed attempts, all converging onto the one visible blade). No
tracking algorithm can recover a position with zero pixel evidence behind
it.

A related, distinct problem surfaced investigating this: `suppress_overlap_bleed`
already detects when a single pair's overlap run is unusually long
(`LONG_INTERPOLATION_SPAN_FRAMES = 90`) — a sign that a single object's own
signal has been lost for an extended stretch, with nothing but a straight-
line interpolation between distant anchors to show for it. Today this only
escalates a warning log; it doesn't change what gets rendered. Confirmed
on the same job: object 0's own entanglement (unrelated to the 43-76
tangle) produces a smoothly-interpolated-but-untrustworthy position for
the rest of the clip after frame ~76, visually landing near wherever the
other two objects happen to legitimately be.

## Goal

**In scope:** when an object's position for a stretch of frames has no
trustworthy signal — either because it's part of a genuine N-way raw-pixel
collision, or because its pairwise correction has been extrapolating a
single low-confidence run for an unusually long time — fade its rendered
blade out rather than show a position that doesn't correspond to anything
real. This does not attempt to recover a *more accurate* position in either
case; by construction, neither trigger fires unless the signal to recover
one doesn't exist.

**Explicitly out of scope, tracked as a distinct future item:** genuine
N-way disambiguation (jointly reacquiring/re-tracking 3+ simultaneously
merged objects to recover more accurate positions when *some* real signal
exists across them). Bigger effort, no evidence yet that it's needed —
every real tangle this session found had zero recoverable signal, not
partial.

## Architecture

Two independent detection signals, one shared suppression mechanism:

```
                    ┌─ find_n_way_tangle_runs(masks_dirs) ──────┐
                    │  (new, runs once after the pairwise loop) │
                    │                                            ▼
runner.py's pairwise │                              mark_suppressed_ranges(
correction loop      │                                motion_path, ranges)
(per pair a, b)      │                                            ▲
                    │  find_long_interpolation_runs(a, b)        │
                    └─ (new, runs once per pair, right   ────────┘
                        after that pair's suppress_overlap_bleed)
                                        │
                                        ▼
                          motion.npz gains a new
                          `suppressed` boolean array
                                        │
                                        ▼
                    glow.py's render_glow_multi: visibility_fraction()
                    (generalizes the existing ignition_fraction taper
                     to also ramp down/up around suppressed ranges)
```

Neither new detector touches `reconcile_pair`, `retrack_overlap_runs`, or
`suppress_overlap_bleed`'s own geometry-correction logic. Both are pure
post-hoc annotation: "this stretch of an object's already-computed geometry
shouldn't be trusted enough to render," decided after that geometry is
finalized.

## Components

### 1. `blade.find_n_way_tangle_runs(masks_dirs, iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD)`

```python
N_WAY_MIN_PARTNERS = 2

def find_n_way_tangle_runs(masks_dirs, iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD):
    """For 3+ tracked objects, find every frame where an object's raw mask
    simultaneously overlaps `N_WAY_MIN_PARTNERS` or more OTHER objects past
    `iou_threshold` -- a genuine multi-way pixel collision, not the ordinary
    single-partner contact `_find_overlap_runs`/`suppress_overlap_bleed`
    already handle pairwise. Confirmed on real footage (job 0eb4fda2): a
    34-frame stretch where all three tracked objects' masks mutually
    overlapped, because 2 of the 3 real blades were genuinely invisible in
    the source and every independent re-seed converged onto the one visible
    target.

    Returns `dict[object_id, list[(run_start, run_end)]]` -- inclusive
    frame-index ranges (positions in the shared frame sequence, not video
    frame numbers), one list per object that had at least one qualifying
    run. An object never appearing as a key had none.

    A job with fewer than 3 objects always returns `{}`: an object can have
    at most one "other" object to overlap with, so the >=2-partners
    condition can never trigger. Safe to call unconditionally regardless of
    object count -- callers don't need to special-case N < 3.
    """
```

Implementation: reuse `_cross_object_ious(masks_dir_a, masks_dir_b, frame_indices)`
once per pair (`itertools.combinations(sorted(masks_dirs), 2)` — at most 6
pairs for 4 objects), accumulate a per-object "how many other objects am I
overlapping right now" count array, threshold at `N_WAY_MIN_PARTNERS`, group
contiguous `True` runs per object (same grouping logic `_find_overlap_runs`
already does inline — worth factoring into a shared `_runs_of_true` helper
both can call, but not required for correctness).

Validated against real data before writing this spec: this exact rule,
run against job `0eb4fda2`'s current masks, flags frames 43-76 for all
three objects and nothing else in the 100-frame clip — zero false
positives against the rest of the clip, including frames that show
ordinary single-pair contact.

### 2. `blade.find_long_interpolation_runs(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b, iou_threshold=..., anchor_iou_threshold=None, span_threshold=LONG_INTERPOLATION_SPAN_FRAMES)`

```python
def find_long_interpolation_runs(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b,
                                  iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD,
                                  anchor_iou_threshold=None,
                                  span_threshold=LONG_INTERPOLATION_SPAN_FRAMES):
    """Same overlap-run detection `suppress_overlap_bleed` uses (see
    `_find_overlap_runs`), filtered to runs whose corrected span --
    `[before+1, after-1]` when both anchors exist, else `[run_start,
    run_end]`, matching suppress_overlap_bleed's own correct_start/
    correct_end -- is at least `span_threshold` frames long. A run this
    long already gets a louder warning from suppress_overlap_bleed (see
    LONG_INTERPOLATION_SPAN_FRAMES's own docstring) because neither the
    shared tracking session nor an independent re-track could tell the two
    objects apart for that whole stretch -- confidence is ~0 throughout, so
    the interpolated result is a plain straight line with no real signal
    behind it, for both objects together (matching suppress_overlap_bleed's
    own reasoning for why it corrects both together rather than guessing
    which one looks more wrong).

    Returns `(ranges_a, ranges_b)`, each a list of inclusive (start, end)
    frame-index tuples for object a/object b respectively -- matching this
    function's own a/b argument order, the same convention
    compute_hilt_overrides and compute_direction_overrides already use, not
    positional indices into anything else.

    Deliberately a separate function rather than an additional
    suppress_overlap_bleed return value: that function's return is
    depended on by ~15 existing tests as a bare int (n_held), and changing
    its shape would touch all of them for a concern -- which stretches to
    hide, not what corrected geometry to compute -- that's orthogonal to
    what it already does.
    """
```

Implementation: `load_motion` both paths, call `_find_overlap_runs` exactly
as `suppress_overlap_bleed` does, compute each run's corrected span the same
way, keep the ones `>= span_threshold`, return the same `(start, end)` for
both `ranges_a` and `ranges_b` (a long run means neither object's data in
that span is trustworthy, not just one).

### 3. `blade.mark_suppressed_ranges(motion_path, ranges)`

```python
def mark_suppressed_ranges(motion_path, ranges):
    """OR `ranges` (a list of inclusive (start, end) frame-index tuples)
    into `motion_path`'s `suppressed` boolean array, creating it all-False
    first if this is the first call for this object. Frame indices are
    positions in the array, matching every other frame-index convention in
    this module (OverlapRun.run_start/run_end, etc.), not real video frame
    numbers.

    Safe to call multiple times for the same motion_path (e.g. once for
    find_long_interpolation_runs' result, again for
    find_n_way_tangle_runs') -- each call only ever adds True positions,
    never clears an earlier call's.
    """
```

Implementation: `load_motion`, get-or-create a `suppressed` array
(`np.zeros(n, dtype=bool)` if absent) sized to any existing per-frame field,
set `True` over each range, write back via `np.savez(motion_path, **motion)`
-- the exact pattern `suppress_overlap_bleed` already uses for its own
in-place patches. `load_motion` already returns whatever keys exist in the
npz, so this round-trips transparently with zero changes to `save_motion`,
`load_motion`, `_FIELDS`, or `BladeGeometry`.

### 4. `glow.visibility_fraction(n, first_active, last_active, suppressed_ranges, ramp_frames)`

```python
def _suppression_local_visibility(n, start, end, ramp_frames):
    """1.0 far outside [start, end], ramping to 0.0 approaching either
    edge, 0.0 throughout the range itself."""
    if ramp_frames <= 0:
        return 0.0 if start <= n <= end else 1.0
    if n < start:
        return max(0.0, min(1.0, (start - n) / ramp_frames))
    if n > end:
        return max(0.0, min(1.0, (n - end) / ramp_frames))
    return 0.0


def visibility_fraction(n, first_active, last_active, suppressed_ranges, ramp_frames):
    """Like `ignition_fraction` (which this calls internally, unchanged),
    generalized to also taper to zero around each entry in
    `suppressed_ranges` -- same rise/fall shape as ignition, just centered
    on an interior range instead of only the clip's start/end. Overall
    visibility is the minimum of the boundary taper and every suppressed
    range's local taper: whichever reason currently demands the lowest
    visibility wins. An empty `suppressed_ranges` makes this identical to
    calling `ignition_fraction` directly.
    """
    frac = ignition_fraction(n, first_active, last_active, ramp_frames)
    for start, end in suppressed_ranges:
        frac = min(frac, _suppression_local_visibility(n, start, end, ramp_frames))
    return frac
```

`ignition_fraction` itself is unchanged. `render_glow_multi`'s existing
per-frame `frac = ignition_fraction(n, first_active, last_active,
ignition_ramp_frames)` becomes `frac = visibility_fraction(n, first_active,
last_active, suppressed_ranges, ignition_ramp_frames)`, where
`suppressed_ranges` is derived once per object (outside the per-frame loop,
alongside where `first_active`/`last_active` are already computed) from
`motion.get("suppressed")` via the same contiguous-run grouping used in
component 1.

The resulting `frac` already feeds into the existing `_apply_ignition(tip,
hilt, frac)` call, which lerps `tip` toward `hilt` (0 = collapsed to a
point at the hilt = invisible). No new visual code path: a suppressed blade
reads as powering down and back up, the same visual language the clip's
own ignition/extinguish already uses.

`render_glow` (the single-object, non-multi path) is untouched: a
single-object job can have neither trigger (both need at least a second
object to compare against), so it has nothing to suppress.

## Data flow: exact call sites in `runner.py`

Inside the existing pairwise loop, immediately after that pair's
`suppress_overlap_bleed` call:

```python
long_ranges_a, long_ranges_b = find_long_interpolation_runs(
    paths["motion_paths"][a], paths["masks_dirs"][a],
    paths["motion_paths"][b], paths["masks_dirs"][b],
)
if long_ranges_a:
    mark_suppressed_ranges(paths["motion_paths"][a], long_ranges_a)
if long_ranges_b:
    mark_suppressed_ranges(paths["motion_paths"][b], long_ranges_b)
```

After the entire pairwise loop finishes (every pair corrected):

```python
for oid, ranges in find_n_way_tangle_runs(paths["masks_dirs"]).items():
    mark_suppressed_ranges(paths["motion_paths"][oid], ranges)
```

Both blocks live inside the same `if object_pairs:` branch the existing
pairwise loop already runs in (2+ objects) -- a single-object job's
`stabilize_blade_length`-only branch is untouched.

## Testing plan

- `test_blade.py::find_n_way_tangle_runs` — synthetic 3-object masks:
  genuine 3-way tangle detected with correct run boundaries; only 2 of 3
  overlapping (ordinary pairwise contact) NOT flagged; a 2-object job
  always returns `{}`; a 4-object job where only one sub-triple tangles
  flags only the involved three.
- `test_blade.py::find_long_interpolation_runs` — a run shorter than
  `span_threshold` returns empty lists for both objects; a run at or above
  it returns the correct `(start, end)` for both; a run missing an anchor
  (falls back to `[run_start, run_end]`) is measured against that span, not
  `[before+1, after-1]`.
- `test_blade.py::mark_suppressed_ranges` — first call on a motion.npz with
  no `suppressed` key creates one; a second call on an already-marked file
  ORs in rather than clobbering. Ranges are always in-bounds by
  construction (both callers derive them from run-detection over the same
  `frame_indices` the target motion.npz was built from), so no
  out-of-bounds handling is needed -- not testing for it is deliberate, not
  an oversight.
- `test_glow.py::visibility_fraction` — empty `suppressed_ranges` matches
  `ignition_fraction` exactly; a suppressed range in the clip's interior
  ramps down/holds-zero/ramps-up on schedule; a suppressed range overlapping
  the clip's own ignition/extinguish taper takes the minimum correctly.
- `test_runner.py` — extend the 3-object end-to-end test (or add a sibling)
  so a synthetic 3-way tangle results in the affected objects' motion.npz
  gaining a `suppressed` array covering the right frames. A full
  glow-render assertion is out of scope for this test tier (the existing
  end-to-end tests stop at motion.npz, not pixel output).

## Constants

- `N_WAY_MIN_PARTNERS = 2` (new, `blade.py`) — the partner count that
  distinguishes ordinary pairwise contact from a genuine multi-way tangle.
- `LONG_INTERPOLATION_SPAN_FRAMES = 90` (existing, `blade.py`) — reused
  as-is for `find_long_interpolation_runs`' default `span_threshold`.
- `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD = 0.1` (existing) — reused as-is for
  both new detectors' default `iou_threshold`.

No new tunable constants in `glow.py`: `visibility_fraction` reuses the
same `ignition_ramp_frames` already computed from `IGNITION_RAMP_SECONDS`.

## Error handling / edge cases

- **Empty `suppressed_ranges` everywhere**: both new detectors return empty
  structures when nothing qualifies (confirmed by construction for < 3 /
  < 2 objects respectively), and `visibility_fraction` with an empty list
  degrades to exactly today's `ignition_fraction` behavior — a 2-object job
  with no long runs, or any job with no tangles, renders bit-for-bit
  identically to today.
- **A suppressed range at the very start or end of the clip** (no ramp room
  before/after): `_suppression_local_visibility` already clamps to `[0, 1]`
  the same way `ignition_fraction` does, so this degrades gracefully to an
  abrupt-ish transition rather than crashing or going out of range.
- **Overlapping ranges from the two triggers on the same object**:
  `mark_suppressed_ranges`' OR-in-place semantics make this a non-issue —
  a frame suppressed by both is just suppressed.

## Out of scope (tracked separately)

Genuine N-way disambiguation — jointly reacquiring/re-tracking 3+
simultaneously merged objects when partial real signal exists across them,
rather than suppressing. Not designed here; revisit if a future job shows
evidence that partial (not total) N-way signal loss is common enough to be
worth the much larger effort.
