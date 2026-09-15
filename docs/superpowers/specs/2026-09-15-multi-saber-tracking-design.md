# lightsaber_fx: multi-saber tracking with mid-render correction

Status: approved
Date: 2026-09-15

## Problem

The pipeline currently tracks exactly one object per clip, end to end:
detection, SAM2 tracking, motion computation, glow compositing, and the
click-to-select UI all assume a single mask sequence. Dan wants to track
several sabers swinging at once in the same clip (e.g. a duel).

Two feasibility spikes were run against real footage
(`/Users/danm/Desktop/lightsaber/lightsaber-sword.mp4`, 3 sword props, two of
them unlit/plain) before this design, because SAM2 does support multi-object
video tracking natively (multiple `obj_id`s in one predictor session) and the
open question was whether it actually tracks cleanly enough to build on:

- **Spike 1** (single include-point per object): one object tracked cleanly.
  A second object's mask never isolated the thin blade at all -- it engulfed
  the actor's entire body (~30,000px vs. ~2,700px for a real blade mask) from
  frame 0. A third object tracked a small sliver for a few frames then went
  permanently empty from frame 10 onward.
- **Spike 2** (tight bounding-box prompts + heuristic re-prompting): the box
  prompt fixed the whole-body engulfment convincingly (525-2241px throughout,
  no re-prompting needed). It did **not** fix the permanent-loss case -- same
  object died at the same frame regardless of prompt shape; a verified
  re-prompt revived it for only 7 frames before it died again for good. Box
  prompts also **introduced a new failure**: the previously-clean object
  ballooned onto an unrelated background object (a stone lantern) during two
  windows. A re-prompt fixed one window; a second, unnecessary re-prompt
  elsewhere destroyed a segment that would have self-recovered on its own.

Conclusion: raw SAM2 multi-object tracking on anything but clean,
well-separated, well-lit footage is a whack-a-mole problem, not something a
better prompt shape alone solves. **This design does not attempt "fully
automatic" multi-object tracking.** Instead it builds tracking failure
*correction* into the render flow itself, as a first-class, expected step --
the same way today's single-object flow already lets a user click to
override a bad detection before rendering.

## Scope

- Up to **4** simultaneous tracked sabers per render (a fixed, generous-enough
  cap -- keeps the UI and the SAM2 session's per-object overhead bounded).
- Each tracked saber gets its own **color, intensity, and voice** -- the
  whole point of tracking several at once (e.g. a red saber and a blue saber
  in the same duel).
- **Mid-render interactive correction**: tracking runs as a frame-by-frame
  stream instead of one blocking call. When an object's mask looks wrong
  (empty for several frames, or implausibly large), the render pauses,
  surfaces the flagged frame and object(s) to the user, and waits for new
  points before resuming propagation from exactly that frame.
- Builds on the existing single-object pipeline and `JobManager`/SSE model --
  this is an extension of that architecture, not a rewrite of it. Every
  per-object helper already in `blade.py` (`save_mask`, `load_mask`,
  `compute_motion`, `fit_blade`, ...) is reused unchanged, just invoked once
  per object instead of once per job.
- **Web app only.** The CLI's `run` command stays single-object for now --
  mid-render correction is inherently a live-UI concept (showing a flagged
  frame, waiting for a click), and the CLI's blocking-popup model would need
  its own separate design. Not requested, not built here.

### Explicitly out of scope for this pass

- Box-prompt / drag-a-box UI for initial or correction points. Spike 2 found
  box prompts don't durably fix the failure mode that mid-render correction
  is actually designed to address (permanent loss recovers no better with a
  box than a point), and points reuse the exact picker interaction that
  already exists. Worth revisiting later if correction accuracy turns out to
  need it in practice.
- Automatic *detection* of multiple objects (today's single-object
  auto-detect via optical flow is not extended to propose several objects).
  The user places all initial points manually, per saber.
- More than 4 objects.
- CLI support.

## Data model

Every per-object pipeline artifact gets its own subdirectory/file, keyed by
a small integer `obj_id` (0-3), instead of one shared set of files:

```
<job_dir>/
  masks/
    0/   00000.npz, 00001.npz, ...   # unchanged per-frame format
    1/   ...
    2/   ...
  motion/
    0.npz
    1.npz
    2.npz
  frames/                             # unchanged -- frames are shared, not per-object
  glow_frames/                        # unchanged -- one composited sequence, N blades summed in
```

`blade.py`'s functions all already take a `masks_dir`/`motion_path` argument
rather than assuming a fixed location, so multi-object support here is pure
orchestration (call each function once per `obj_id` with its own directory),
not a change to `blade.py` itself.

`job_meta.json` grows one new field: `object_ids: [0, 1, 2]` (however many
objects this job tracked), so `require_rerenderable` and `rerender_pipeline`
know how many objects' cached masks/motion to expect and loop over. Per-object
color/intensity/voice are **not** persisted server-side, matching how the
single-object job never persists them today -- they live in the browser's
per-saber UI state and are re-sent on every render/rerender call.

**Compatibility note**: this new layout (`masks/{obj_id}/`,
`motion/{obj_id}.npz`, `object_ids` in `job_meta.json`) replaces today's flat
single-object layout (`masks/`, `motion.npz`) -- a render with 1 saber is
just the N=1 case of the same code path, not a separate one. Job directories
created before this change won't have `object_ids` and won't be
rerenderable afterward; `lightsaber-fx clean` clears them out. Acceptable for
a local single-user tool with no precious job history, but worth calling out
explicitly rather than discovering it as a surprise.

## Tracking + mid-render correction

### One shared SAM2 session for all objects

`track_object()` (single-object) is replaced by `track_objects()`
(multi-object) for this flow. It builds **one** predictor / inference_state
for the whole job -- SAM2 encodes each frame's image features once per
session, so tracking N objects in one session is cheaper than N separate
single-object sessions, not more expensive. Initial prompts (points, one
include point minimum per object like today) are added for each `obj_id` on
its own chosen prompt frame via `add_new_points_or_box`, then tracking
proceeds via `propagate_in_video`.

### Frame-by-frame streaming instead of one blocking call

`propagate_in_video` is a generator that already yields one frame's results
(all objects at once) at a time -- today's single-object `track_object`
simply drains it in a loop before returning. The multi-object version keeps
that loop open: as each frame's masks arrive, it (a) writes each object's
mask to its own `masks/{obj_id}/` directory, and (b) runs the area-health
check below.

**Correction pauses are frame-synchronous across all objects.** Because one
predictor session yields all objects' results together per frame, there is
no way for "object A tracks on into frame 200 while object B pauses at frame
80" -- if any object looks wrong at frame N, the whole session pauses at
frame N until every currently-flagged object is fixed, then resumes together
from there. This is a hard constraint from SAM2's API shape, not a design
choice, and the frontend should present it that way: one pause can carry
more than one flagged object at once.

### Area-health heuristic

Per object, per frame, using the object's own mask area (pixel count):

- **Lost**: mask empty (`area == 0`) for `K` consecutive frames (`K = 5`
  as a starting default -- tune during implementation) after having had a
  valid mask at least once.
- **Drifted**: `area` exceeds `6x` a running reference size for that object.
  The reference starts as the object's area on its own prompt frame and
  updates as a running median over recent *accepted* (non-flagged) frames --
  this is what let spike 2's ~2,700px "real blade" reference distinguish a
  correctly-sized mask from an ~11,000px+ one that had drifted onto
  background.

Either condition flags that `obj_id` at that frame.

### Pause / resume mechanics

- New `JobState.status`: `"awaiting_correction"`, carrying
  `{frame_index, flagged: [obj_id, ...]}`.
- The background thread (same `JobManager` thread model as today) reports
  this via `progress_cb` as a new SSE event shape, then blocks on a
  `threading.Event`. Concretely, this means the SAM2 predictor's
  `inference_state` (holding every frame's image embeddings for the session)
  sits resident in memory for as long as the user takes to respond -- could
  be minutes. Acceptable for a local single-user, one-job-at-a-time tool
  (same trust boundary as today's single long-running render thread), but
  worth naming as a real resource cost rather than leaving it implicit.
- New endpoint `GET /api/jobs/{id}/correction-frame` serves the flagged
  frame's image (same pattern as today's `detect-frame`), so the picker
  canvas can show it.
- New endpoint `POST /api/jobs/{id}/correct`, body
  `{corrections: [{obj_id, points: [[x,y,label],...]}, ...]}` -- one entry
  per currently-flagged object the user is fixing this round. `frame_index`
  is **not** client-supplied; the server uses its own recorded pause frame,
  so a stale/slow client can't apply a correction to the wrong frame.
- On receiving corrections, the thread applies each one via
  `add_new_points_or_box` **before** resuming `propagate_in_video` from that
  frame. This matters: spike 2 found that calling `add_new_points_or_box` on
  a frame `propagate_in_video` has already produced results for silently
  no-ops (SAM2 blends the new prompt with the existing, possibly-empty mask
  logits as a prior instead of replacing them) -- corrections must be applied
  to the *paused* frame, before propagation continues past it, never after.
- Status returns to `"running"` and the generator loop continues.

### Giving up on an object

A correction entry may be `{obj_id, give_up: true}` instead of points --
that object's mask simply stays empty for the rest of the clip from there on
(the glow stage already tolerates a per-frame missing mask; see
`test_render_glow_frame_with_missing_mask_still_renders`). Also applied
automatically once an object has been corrected **5 times** in one job,
so a truly unrecoverable object can't stall the render indefinitely --
the pause reports this as an automatic give-up rather than asking again.

If every object ends up abandoned (zero usable frames across all of them),
the render hard-fails with the same "nothing to render" error the
single-object pipeline already raises via `_require_usable_track`. If at
least one object has real coverage, rendering proceeds -- same "partial
coverage is legitimate, warn and continue" tolerance the single-object path
already has.

## Glow compositing

This turns out to be the lowest-risk part of the whole feature: `glow.py`
already composites in additive linear light (`render_glow`'s core/colour-band
/wide-glow layers are summed into `full_fx`, then tonemapped once at the very
end), so going from one blade's contribution to N blades' summed
contributions is a natural extension, not a rewrite:

- Per frame, loop over each object that has a mask this frame. Build that
  object's own capsule shape (unchanged `_build_blade_shape`/
  `_capsule_mask`, including its own ignition-ramp fraction from its own
  first/last-active window) and its own core/colour/wide-glow using **its
  own color**.
- The expensive per-frame Gaussian-blur work stays confined to a bounding
  box, but now **per object** (each gets cropped, processed in its own
  local box, and placed back into the shared `full_fx` accumulator at its
  own offset) rather than one box spanning every object -- objects on
  opposite sides of a wide frame shouldn't force one huge shared crop.
  Overlapping regions between two nearby blades sum correctly since
  placement into `full_fx` is additive.
- Knoll darken (dim the plate near the blade before adding colour) darkens
  around the **union** of every object's blade shape for that frame (OR the
  per-object `blade_u8` masks together before feeding the combined mask into
  `knoll_darken`) -- the plate darkens near any blade, the physically
  sensible generalization of "near the blade."
- The temporal trail and final tonemap/encode already operate on the
  frame-level combined buffer, not on any single object's contribution --
  **no change needed there.** This is already blade-count-agnostic.
- Directional motion blur is computed per object (each has its own velocity
  from its own motion.npz) on that object's local patch before it's placed
  into the shared accumulator.
- The single-object case (1 object) must produce **pixel-identical or
  near-identical output** to today's `render_glow` -- this is the regression
  guard the test plan below is built around.

## Audio

`synthesize_audio()` is called once per object (its own motion.npz, its own
`voice`), producing N temporary WAV files. A new small mixing step sums them
into one track with peak-normalization (scale the mixed buffer down if its
peak would clip) -- avoids the obvious failure of two sabers' hums clipping
when they move in sync. No per-object spatial panning or anything fancier;
just a straightforward mix.

## Web API changes

- `POST /api/jobs/{id}/points` body shape changes from today's flat
  `{points, prompt_frame, color, intensity, voice, blade_extend}` to
  `{sabers: [{points, prompt_frame, color, intensity, voice}, ...],
  blade_extend}` -- one entry per tracked saber (1-4), `blade_extend` stays a
  single job-level toggle (a structural rendering choice, not a per-saber
  style choice -- never asked to be per-object).
- `POST /api/jobs/{id}/rerender` body shape changes the same way: a list of
  per-object color/intensity/voice instead of scalars, `blade_extend` still
  global. Reuses `object_ids` from `job_meta.json` to know how many entries
  to expect.
- New `GET /api/jobs/{id}/correction-frame` (see above).
- New `POST /api/jobs/{id}/correct` (see above).
- `GET /api/jobs/{id}/events` SSE stream gains the `awaiting_correction`
  stage shape: `{stage: "awaiting_correction", frame_index, flagged:
  [{obj_id, color}, ...], frame_url}`.
- `GET /api/jobs/{id}/preview` (the live-render preview added earlier this
  session) is unaffected -- it already just serves the latest glow frame
  regardless of how many objects contributed to it.

## Frontend flow

**Upfront selection**: the single picker canvas gains a "+ Add another
saber" control (up to 4 slots). Each slot has its own color/intensity/voice
controls and its own "currently picking for this saber" active state;
clicking the canvas routes the point to whichever slot is active. Points from
different slots are drawn in their slot's chosen color so they stay visually
distinguishable on one shared canvas -- no need for separate canvases per
object.

**Mid-render correction**: the existing progress screen gains a new state,
entered when an SSE event's `stage` is `awaiting_correction`. It shows the
flagged frame (`correction-frame` endpoint) with the same click-to-place-
points canvas interaction as the upfront picker, one color-tagged sub-section
per currently-flagged object (there may be more than one, per the
frame-synchronous pause behavior above), plus a "give up on this saber"
option per flagged object. Submitting posts to `/correct` and returns to the
normal progress view.

## Rerender

`rerender_pipeline` already skips tracking entirely and just re-runs
glow/audio/mux from cached masks/motion -- that shape is unchanged. It now
loops over `job_meta.json`'s `object_ids`, reading each object's cached
`masks/{obj_id}/` and `motion/{obj_id}.npz`, and accepts a list of
color/intensity/voice (one per object, matching the render endpoint) instead
of scalars.

## Testing

- **Area-health heuristic**: pure-function unit tests (same style as
  `ignition_fraction` -- lost after K empty frames, drifted above the
  threshold, neither on a normal steady object, reference-area update
  behavior), no SAM2/GPU required.
- **Multi-object `render_glow`**: extend the existing `_build_blade_clip`
  synthetic-fixture pattern to place 2 independently-moving, independently-
  colored blades in one clip. Assert both render in their own colors, assert
  a region only one blade's glow should reach doesn't pick up the other
  blade's color, and -- the critical regression guard -- assert the
  single-object case's output is unchanged from before this feature (same
  fixtures the current single-object tests already use).
- **Pause/resume cycle**: a stubbed predictor (real SAM2 is far too slow for
  tests, as observed all session) that deterministically returns an empty
  mask for a chosen `obj_id` at a chosen frame, driving `JobManager` into
  `awaiting_correction`; a test posts a correction via `TestClient` and
  asserts propagation resumes and the job reaches `done` -- mirrors how
  `test_server.py` already stubs `run_pipeline`/`rerender_pipeline` for its
  job-manager-focused tests.
- **Web endpoint tests**: `/points` and `/rerender`'s new multi-saber body
  shape (validation: 1-4 sabers, at least one include point per saber,
  rejects >4), `/correction-frame` and `/correct` (including the "frame
  already visited" ordering guarantee and the give-up path), matching the
  existing traversal-rejection and validation-error test patterns in
  `test_server.py`.
- **Manual end-to-end pass**: the real `lightsaber-sword.mp4` clip through
  the actual browser UI, deliberately choosing points likely to need a
  correction (based on what the spikes already learned about this clip),
  confirming the full loop: upfront multi-slot selection -> pause -> correct
  -> resume -> multi-colored final render -> rerender.

## Out of scope (explicitly deferred)

- Box-prompt / drag-a-box correction UI.
- Automatic multi-object detection.
- More than 4 simultaneous objects.
- CLI support for multi-saber tracking.
- Per-saber `blade_extend` toggle.
- Spatial audio panning between sabers (beyond simple mixed/normalized hum
  tracks).
