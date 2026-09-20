# Multi-saber tracking investigation (2026-09-19)

Findings from running the pipeline end-to-end on three real, previously-untested
source videos, each with 2-3 tracked objects. Two of the three needed real fixes;
one exposed a genuine architectural gap. This doc is the retrospective; the
fix itself landed as commit `371933d` ("Run cross-object bleed correction for
every pair, not just 2-object jobs").

## The headline bug

`runner.run_pipeline_multi`'s entire cross-object correction pipeline —
`reconcile_pair`, `retrack_overlap_runs`, `compute_hilt_overrides`,
`compute_direction_overrides`, `suppress_overlap_bleed` — was gated behind
`if len(object_ids) == 2`. It was built and validated against a 2-saber
fencing video earlier this project, and the 3/4-object branch was written
under the assumption that "no single other object to reconcile against"
meant none of this machinery applied. That assumption was never tested
against a real 3+ object job until this session.

**Confirmed on real footage** (job `0eb4fda2`, three Korean martial artists,
100 frames): object 0's mask fully swapped identity onto object 1's blade
for 63 of 100 frames. Zero correction was applied, because the entire
correction block was skipped for this 3-object job. The rendered video
showed two glowing blades stacked on top of each other for well over half
the clip.

### The fix

Every one of the pairwise functions above already takes exactly two
`masks_dir`/`motion_path` arguments and is symmetric in them — nothing
about their internals assumes there are only two objects *total* in the
job, only that they're being asked to reconcile two *specific* objects.
The fix loops the same calls over `itertools.combinations(object_ids, 2)`
instead of a single hardcoded `(0, 1)` pair. A 2-object job still gets
exactly one pair (unchanged behavior); a 3-object job gets 3 pairs; a
4-object job gets 6.

No changes were needed inside `reacquire.py`, `hilt_track.py`, or
`blade.py` — the bug was entirely in `runner.py`'s orchestration.

## Two more bugs the fix surfaced

Both were found by actually running the new pairwise loop against real
job data rather than trusting it from the diff. Both are fixed and have
regression tests (see commit `371933d`).

**1. Positional-index vs. real object_id confusion.**
`retrack_overlap_runs` returns `(patched, resolved_ranges)` where `patched`
is a subset of the *positional* `{0, 1}` — 0 meaning "whichever masks_dir/
motion_path was passed as this call's own first argument," not "object 0."
The old 2-object-only branch always called it with `(a, b) == (0, 1)`, so
positional and real indices coincided by construction and the distinction
never mattered. The first version of the generalized loop indexed straight
into `patched` as if it were real object_ids, which is correct for the
pair `(0, 1)` and silently wrong for any other pair. Confirmed on real
data: calling the loop for pair `(0, 2)`, a validated patch to object 2
returned positional index `1`, which — read as a real object_id — would
have triggered a redundant `compute_motion` re-run for object 1 while
leaving object 2's own freshly-patched masks un-recomputed. Fixed by
translating through `pair = (a, b)` before use.

**2. Unhandled crash on an empty anchor mask.**
`_retrack_one_object` (in `reacquire.py`) loads a "before" anchor mask and
passes it straight to `_points_on_axis` without checking whether it has
any foreground pixels. `_points_on_axis` runs PCA over the mask's nonzero
pixel coordinates; given zero pixels, `np.linalg.svd` returns an empty `vt`
and the next line's `vt[0]` raises `IndexError: index 0 is out of bounds
for axis 0 with size 0`. This is a real, reachable crash — confirmed on
real footage (pair `(1, 2)` in the martial-artists job) — that only
"worked" because `retrack_overlap_runs`' outer `try/except Exception` is
deliberately unconditional ("never raises, structurally," per its own
docstring) and swallowed it, logging a warning and falling back to
interpolation. The function already had the equivalent guard for an empty
*after*-mask (`fit_blade(after_mask) is None: return False`); it just
didn't have the symmetric guard for the before-mask. Fixed by adding
`if not before_mask.any(): return False` before the `_points_on_axis`
call, so this case is declined cleanly instead of reached via a crash a
broad exception handler happens to catch.

## The three videos: what shipped, what didn't, and why

### Fencing video (job `58a8f662`) — 2 objects
Already fixed and delivered earlier this project (frame-320 cross-object
angle recovery via optical-flow hilt/direction tracking, commit `422f8ef`).
One known, investigated, unfixed limitation remains: the red saber's fitted
length visibly drifts/flickers in the final ~15% of the clip. Root cause:
sustained motion blur genuinely erases the blade's silhouette in the raw
frames for a long stretch — there is no recoverable signal, mask-based or
optical-flow-based, to track through it. Extending the optical-flow
technique built for the frame-320 fix to this longer, more degraded stretch
was attempted and failed validation (the reconstructed direction pointed
into empty space when overlaid on the real frame). Documented as a genuine
footage limitation, not a pipeline defect.

### Knights video (job `92b675aa`) — 2 of 3 apparent weapons
Two weapons track and render correctly. A third (a background figure's
stick/spear) failed 7 independent SAM2 seeding attempts — points-only,
points with negative prompts, single-frame box+point, box+point at a
different frame, dual-anchor conditioning — each either grabbing background
scenery or losing tracking within a few frames of its prompt. The one
technique that gave full-clip coverage (box+point at frame 110) still
produced a wildly unstable `width` field (oscillating 20-140px throughout,
not an isolated spike), which the existing length-stabilizer has no
equivalent for and which further investigation found no clean reference
signal to gate against. Dropped by agreement rather than shipped broken.

### Martial artists video (job `0eb4fda2`) — the 3-object case above
Beyond the headline architectural bug, two of the three tracked objects had
their own independent, per-object SAM2 segmentation failures that the
architectural fix alone doesn't touch (they're not cross-object bleed —
each is a single object's raw mask drifting onto the wrong target):

- **Object 2** locked onto its own wearer's helmet for frames 0-43 (a
  compact, high-contrast, easy-for-SAM2-to-grab feature right next to the
  real target). Fixed by re-seeding with a box+point prompt at a clean
  anchor frame and propagating; verified by checking *which real person*
  the mask belonged to at every sampled frame, not just whether it looked
  blade-shaped (see the next section — this check was missing on the first
  attempt and let a wrong fix through).
- **Object 1** locked onto a static background flag pole for nearly the
  entire clip (it visually overlapped the real target at the prompt frame,
  and SAM2 kept the larger, more stable flag rather than following the
  thinner, moving blade once they separated). Fixed the same way.

**A residual, unfixed limitation:** during roughly frames 22-84, 2-3 of the
three real blades become genuinely invisible in the source footage at the
same time (the two grapplers' weapons are tucked into a close embrace).
Confirmed via three independent from-scratch re-seed attempts — for object
1, object 2, and (to check whether a better recipe existed) object 0 —
all converging onto the *same* one visible blade once their own true
target left the frame. This is the same class of limitation as the fencing
video's motion blur: no raw pixel evidence exists to track, so no seeding
strategy can recover it. The pipeline's interpolation fallback
(`suppress_overlap_bleed`) is the best available approximation and is what
ships. Delivered as-is per explicit decision.

## A verification mistake worth naming

The first fix attempt for object 2 (re-seed at frame 47, propagate
*backward* to frame 0) looked correct under the verification done at the
time: sampled frames showed an elongated, blade-shaped mask with a
plausible, continuous trajectory. It was wrong — the backward propagation
had drifted onto a *different real person* (the lone third warrior) for
roughly the first 30 frames, and the verification never checked "is this
the same person as before," only "does this look like a blade." The
mistake was only caught by later cross-checking mask centroids/IoU
*between* objects, which revealed two supposedly-independent tracks
occupying the same screen position. The corrected approach (re-seed
forward from frame 0, where the target is unambiguous) was verified by
plotting the centroid's x-position across the whole clip and confirming it
never jumps to the other side of the frame where the other tracked people
are — a cheap, mechanical check that would have caught the first mistake
immediately.

**Lesson for future SAM2 re-seeding work in this codebase:** verifying a
fix against "is the mask on a plausible target" is not sufficiently
strong when multiple similar targets exist in the same shot. The stronger
check is either (a) a centroid/position trajectory across the *whole*
clip, watching for jumps that cross into another tracked object's territory,
or (b) pairwise mask IoU between all tracked objects, watching for
unexpected convergence.

## Hardening recommendations

Two concrete, small, high-value items came directly out of this session;
one larger item is a known gap worth tracking but not urgent.

1. **Harden `_points_on_axis` at its own definition (`detect.py`), not just
   at the one call site that crashed.** It's called from four places
   (`reacquire.py` twice, `vision_detect.py` once, `detect.py` once); only
   the call site involved in this session's crash got a guard. The other
   three have the identical latent risk — an empty mask reaching
   `_points_on_axis` crashes via `IndexError` on `np.linalg.svd`'s output,
   not a clear, documented failure. Fixing it once at the source (raise a
   clear `ValueError` for an empty mask, or return `None` and update the
   three unguarded callers to handle it) closes the whole risk class
   instead of relying on every future caller to remember to check
   `.any()` first — exactly the mistake this session found.

2. **Add a 3-object end-to-end synthetic test for the pairwise correction
   loop**, mirroring the existing
   `test_run_pipeline_multi_recovers_from_a_simulated_crossing_end_to_end`
   (2-object case). The current 3/4-object tests all mock the pairwise
   functions and assert they're *called* with the right pairs — none of
   them exercise the real `reconcile_pair` → `retrack_overlap_runs` →
   `suppress_overlap_bleed` chain together for a 3-object job the way the
   2-object end-to-end test does. This is the exact feature this session
   shipped; it currently has no test that would catch a regression in the
   real (not mocked) multi-pair interaction.

3. **Known gap, not urgent: pairwise correction has no cross-pair
   consistency check.** Each pair is corrected independently, in sequence,
   against its own anchors. A genuine *simultaneous* three-way tangle
   (all three masks overlapping at once, as happened in the martial-artists
   video) isn't fully resolved this way — there's no arbitration across
   all three objects at once, only three independent pairwise
   reconciliations. Worth a real design pass if this recurs on a future
   job; not worth building speculatively now.
