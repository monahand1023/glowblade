"""Lightsaber glow compositing (Phase B1 of the fidelity upgrade).

Rewritten around Phase A's ``BladeGeometry`` contract
(``lightsaber_fx.pipeline.blade``) and the VFX practices catalogued in
docs/design-notes.md's "B1 -- the glow renderer" table. Each helper below
is labelled with the item it implements.

Old behaviour this replaces: ``colored[mask > 0] = color`` traced the prop's
exact silhouette (the "glowing bat"), core and colour blurred the same
mask so the white core was as wide as the blade, there was no temporal
state, compositing ran in sRGB gamma with a uint8 round-trip per layer, and
colour washed out on bright plates because nothing pulled the plate down
first.

Renders a lossless PNG sequence instead of an OpenCV mpeg4 video (B1.10) --
a later phase encodes it once with libx264.
"""

import os

import cv2
import numpy as np

from .blade import load_mask_optional, load_motion

NAMED_COLORS = {
    "red": (40, 40, 255),
    "blue": (255, 90, 60),
    "green": (70, 220, 80),
}


def parse_color(spec):
    key = spec.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    if key.startswith("#") and len(key) == 7:
        r = int(key[1:3], 16)
        g = int(key[3:5], 16)
        b = int(key[5:7], 16)
        return (b, g, r)
    raise ValueError(f"Unrecognized color: {spec!r}. Use red, blue, green, or #RRGGBB.")


# ---------------------------------------------------------------------------
# B1.1 -- linear-light compositing
# ---------------------------------------------------------------------------
# Everything below composites in linear light: convert the plate sRGB ->
# linear, accumulate every glow contribution additively in float32
# (allowing values > 1), then tonemap/clip once and convert back to sRGB at
# the very end of a frame. This replaces the old chain of three
# `screen_blend` gamma-space passes with a uint8 round-trip between each --
# the textbook amateur over-bloom signature.

_GAMMA = 2.2


def _srgb_to_linear(img_u8):
    return (img_u8.astype(np.float32) / 255.0) ** _GAMMA


def _linear_to_srgb(img_linear):
    x = np.clip(img_linear, 0.0, 1.0)
    return np.clip(np.round((x ** (1.0 / _GAMMA)) * 255.0), 0, 255).astype(np.uint8)


def _soft_tonemap(x, knee=0.85):
    """Identity below `knee`, an exponential shoulder above it.

    Plate content that no glow ever touches stays under the knee for any
    normally-exposed footage, so it renders unchanged; only additive
    highlights near/above white get a soft roll-off instead of a hard clip
    (which is where banding comes from)."""
    span = 1.0 - knee
    shoulder = knee + span * (1.0 - np.exp(-(x - knee) / span))
    return np.where(x > knee, shoulder, x)


# ---------------------------------------------------------------------------
# B1.2 -- blade reconstruction as a capsule, not the raw silhouette
# ---------------------------------------------------------------------------
# ILM deliberately drew blades longer than the prop with a rounded tip.
# Rebuild the blade from BladeGeometry -- constant width, rounded tip,
# extended past the fitted tip, tapered at the hilt end -- instead of
# tracing the mask, which otherwise carries the prop's knob and wooden
# taper straight through ("the glowing bat" this phase exists to fix).
# `--no-blade-extend` (`blade_extend=False`) falls back to the raw mask.

def _capsule_mask(shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac):
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    hilt = np.asarray(hilt, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    half_w = max(0.5, width / 2.0)

    seg = tip - hilt
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        cv2.circle(out, (round(hilt[0]), round(hilt[1])), max(1, round(half_w)), 255, -1)
        return out

    axis = seg / length
    perp = np.array([-axis[1], axis[0]])
    extended_tip = tip + axis * (length * extend_frac)
    taper_len = min(length * hilt_taper_frac, length * 0.9)
    body_start = hilt + axis * taper_len

    # Constant-width body between the (tapered) hilt end and the extended tip.
    body = np.array([
        body_start + perp * half_w,
        extended_tip + perp * half_w,
        extended_tip - perp * half_w,
        body_start - perp * half_w,
    ])
    cv2.fillConvexPoly(out, np.round(body).astype(np.int32), 255)

    # Rounded tip cap.
    tip_pt = (round(extended_tip[0]), round(extended_tip[1]))
    cv2.circle(out, tip_pt, max(1, round(half_w)), 255, -1)

    # Tapered (not hard-cut) hilt wedge, narrowing towards the hilt point.
    hilt_half_w = half_w * hilt_taper_min_frac
    wedge = np.array([
        hilt + perp * hilt_half_w,
        body_start + perp * half_w,
        body_start - perp * half_w,
        hilt - perp * hilt_half_w,
    ])
    cv2.fillConvexPoly(out, np.round(wedge).astype(np.int32), 255)

    return out


def _build_blade_shape(mask, frame_shape, tip, hilt, width, blade_extend,
                        extend_frac, hilt_taper_frac, hilt_taper_min_frac):
    have_geometry = (
        blade_extend
        and tip is not None and hilt is not None
        and not (np.any(np.isnan(tip)) or np.any(np.isnan(hilt)))
    )
    if have_geometry:
        return _capsule_mask(frame_shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac)

    mask_u8 = mask.astype(np.uint8) * 255
    if mask_u8.shape[:2] != tuple(frame_shape[:2]):
        mask_u8 = cv2.resize(mask_u8, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_u8


def _stabilize_tip_hilt(tip, hilt, axis):
    """Continuity-correct the per-frame tip/hilt/axis sequence.

    Phase A's tip/hilt call (`classify_tip_by_taper`) is a single-frame
    heuristic with no memory of its own: near a symmetric taper, it can
    flip which end is "tip" between adjacent frames that barely moved.
    Uncorrected, that would make the capsule's rounded cap -- and B1.7's
    velocity vector, which is read off `tip` frame-to-frame -- jump ~180
    degrees for no physical reason. Fix it once here by flipping a frame's
    labeling whenever its axis points opposite the previous valid frame's.
    """
    tip = tip.copy()
    hilt = hilt.copy()
    axis = axis.copy()
    prev = None
    for i in range(len(axis)):
        a = axis[i]
        if np.any(np.isnan(a)):
            continue
        if prev is not None and np.dot(a, prev) < 0:
            axis[i] = -a
            tip[i], hilt[i] = hilt[i].copy(), tip[i].copy()
        prev = axis[i]
    return tip, hilt, axis


# ---------------------------------------------------------------------------
# B1.3 -- three distinct elements, with an eroded core
# ---------------------------------------------------------------------------
# core: eroded, pure white, tight blur -- must be narrower than the blade.
# colour band: blade width, the chosen colour, blurred just enough to soften
# the capsule edge. The wide, coloured "glow" element is B1.4 below.

def _make_core(blade01, kernel, blur_sigma):
    eroded = cv2.erode(blade01, kernel) if kernel is not None else blade01
    return cv2.GaussianBlur(eroded, (0, 0), blur_sigma)


def _make_colour_band(blade01, blur_sigma):
    return cv2.GaussianBlur(blade01, (0, 0), blur_sigma)


# ---------------------------------------------------------------------------
# B1.4 -- exponential falloff via stacked, level-crushed gaussians
# B1.8 -- chromatic bloom (slightly different per-channel radii) folded in
# ---------------------------------------------------------------------------
# Blur at 0.5x/1x/2x blade width, weight the scales so the falloff is
# exponential (w_i ~ exp(-r_i / tau)) rather than linear, and crush levels
# after blurring (clip(blurred * k, 0, 1)) then re-blur -- the crush is
# what turns a soft gaussian into a steep-edged broad halo.

def _make_wide_glow(blade01, width, color_lin, scales, tau, crush, chroma_frac):
    h, w = blade01.shape
    acc = np.zeros((h, w, 3), dtype=np.float32)
    weights = np.exp(-np.asarray(scales, dtype=np.float32) / max(tau, 1e-6))
    weights = weights / weights.sum()
    # Per-channel radius offset (+-chroma_frac) breaks the "too uniform"
    # look of a spatially identical bloom on every channel.
    chroma = np.array([1.0 - chroma_frac, 1.0, 1.0 + chroma_frac], dtype=np.float32)
    base = max(1.0, width)
    for scale, weight in zip(scales, weights, strict=False):
        for c in range(3):
            sigma = max(0.8, base * scale * chroma[c])
            blurred = cv2.GaussianBlur(blade01, (0, 0), sigma)
            crushed = np.clip(blurred * crush, 0.0, 1.0)
            level = cv2.GaussianBlur(crushed, (0, 0), max(0.6, sigma * 0.6))
            acc[..., c] += weight * level * color_lin[c]
    return acc


# ---------------------------------------------------------------------------
# B1.5 -- darken the plate before adding colour (Knoll)
# ---------------------------------------------------------------------------
# "On bright backgrounds you don't get any colour, because you're already
# so close to being white" -- John Knoll. This is specifically why a
# blue-on-sky render used to read white. Exposed as a standalone function
# so the darkening step is directly testable in isolation.

def knoll_darken(plate_linear, blade_u8, dilate_px, darken_factor, feather_sigma, dilate_kernel=None):
    """Multiply `plate_linear` by `darken_factor` inside a dilated,
    feathered region around `blade_u8`. Returns (darkened_plate,
    feathered_mask float32 0..1). `dilate_kernel`, if given, is used
    instead of building one from `dilate_px` (callers processing many
    frames should precompute and reuse one kernel)."""
    if dilate_kernel is None and dilate_px > 0:
        dilate_kernel = np.ones((dilate_px, dilate_px), np.uint8)
    dilated = cv2.dilate(blade_u8, dilate_kernel) if dilate_kernel is not None else blade_u8
    feathered = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (0, 0), feather_sigma)
    feathered = np.clip(feathered, 0.0, 1.0)
    factor = 1.0 - feathered[..., None] * (1.0 - darken_factor)
    return plate_linear * factor, feathered


# ---------------------------------------------------------------------------
# B1.6 -- temporal motion trail (state carried across frames in render_glow)
# ---------------------------------------------------------------------------
# trail = max(trail * decay, glow) each frame, composited under the
# current frame's own core/colour/glow. Because max() never lets a decayed
# old value beat a fresh one, using `trail` as the sole glow contribution
# for the frame already gives exactly that "current frame on top, faded
# ghosts underneath" result with no extra layering step.


# ---------------------------------------------------------------------------
# B1.7 -- directional motion blur along the velocity vector
# ---------------------------------------------------------------------------
# A line kernel oriented along the blade's frame-to-frame tip displacement,
# length proportional to speed, applied via cv2.filter2D to the glow layers
# only -- never the plate.

def _directional_kernel(velocity, gain, max_len):
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < 1e-3:
        return None
    length = int(np.clip(round(speed * gain), 1, max_len))
    if length < 2:
        return None
    direction = np.array(velocity, dtype=np.float32) / speed
    size = length * 2 + 1
    kernel = np.zeros((size, size), dtype=np.float32)
    center = length
    p1 = (center - direction[0] * length, center - direction[1] * length)
    p2 = (center + direction[0] * length, center + direction[1] * length)
    cv2.line(kernel, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), 1.0, 1)
    total = kernel.sum()
    if total <= 0:
        return None
    return kernel / total


# ---------------------------------------------------------------------------
# B1.9 -- light wrap (approximate, lowest priority)
# ---------------------------------------------------------------------------
# Blur the glow wide and multiply by a thin ring just outside the blade so
# nearby surfaces pick up a hint of colour. Honesty note: a proper light
# wrap needs a matte of the subjects next to the blade, which we don't
# have, and no amount of 2D blending can change which direction the light
# actually falls from -- this only adds a plausible colour bleed
# immediately around the blade's own silhouette edge.

def _light_wrap(blade_u8, color_lin, dilate_kernel, blur_sigma, strength):
    dilated = cv2.dilate(blade_u8, dilate_kernel)
    ring = cv2.subtract(dilated, blade_u8).astype(np.float32) / 255.0
    ring = cv2.GaussianBlur(ring, (0, 0), blur_sigma)
    out = np.empty((*ring.shape, 3), dtype=np.float32)
    for c in range(3):
        out[..., c] = ring * (color_lin[c] * strength)
    return out


def _robust_median(arr, default):
    """np.nanmedian, but returns `default` (silently, no RuntimeWarning)
    instead of NaN when every entry is NaN or the array is empty."""
    arr = np.asarray(arr, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return default
    return float(np.median(finite))


# ---------------------------------------------------------------------------
# Ignite/extinguish -- the blade grows out of the hilt at the start of its
# tracked appearance and shrinks back into it at the end, instead of just
# appearing/disappearing at full length. Exposed as a standalone function
# (matching knoll_darken) so the ramp math is directly testable without
# rendering a whole clip.
# ---------------------------------------------------------------------------

IGNITION_RAMP_SECONDS = 0.35


def ignition_fraction(n, first_active, last_active, ramp_frames):
    """Blade-length fraction (0..1) for frame `n`: ramps 0->1 over
    `ramp_frames` after `first_active`, holds at 1 through the steady
    middle, and ramps 1->0 over `ramp_frames` before `last_active`.

    On an active window shorter than 2*ramp_frames, the rise and fall
    overlap and the peak never reaches 1.0 -- a triangular taper rather
    than a plateau, so a short appearance never looks like it snapped to
    full length. Returns 1.0 (no-op) when there's no active window at all
    or no ramp to apply, so callers can pass this through unconditionally.
    """
    if ramp_frames <= 0 or first_active is None or last_active is None:
        return 1.0
    rise = (n - first_active + 1) / ramp_frames
    fall = (last_active - n + 1) / ramp_frames
    return max(0.0, min(1.0, rise, fall))


def _apply_ignition(tip, hilt, frac):
    """Lerp `tip` toward `hilt` by `frac` (1.0 = full length, 0.0 =
    collapsed onto the hilt). A no-op when tip/hilt aren't valid geometry
    (None or NaN) -- callers fall back to the raw mask in that case exactly
    as they did before this existed."""
    if tip is None or hilt is None or np.any(np.isnan(tip)) or np.any(np.isnan(hilt)):
        return tip
    tip = np.asarray(tip, dtype=np.float64)
    hilt = np.asarray(hilt, dtype=np.float64)
    return hilt + (tip - hilt) * frac


def _composite_blade_contribution(
    frame_shape, mask, tip, hilt, velocity,
    canonical_width, blade_extend, ignition_frac,
    tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
    core_erode_kernel, core_sigma, colour_sigma,
    color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
    spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
):
    """One object's core/colour/wide-glow/motion-blur contribution for one
    frame, confined to a local bounding box. Returns full-frame-sized arrays
    so callers can sum/OR them directly without tracking per-object offsets.

    Args:
    - ignition_frac: this object's own ignite/extinguish length fraction for
      this frame, from `ignition_fraction()`.

    Returns (full_fx, blade_u8) where:
    - full_fx: np.ndarray[h,w,3] float32, zero everywhere outside this
      object's local bounding box, in additive linear light
    - blade_u8: np.ndarray[h,w] uint8, zero everywhere the blade shape
      doesn't cover
    """
    h, w = frame_shape[:2]
    full_fx = np.zeros((h, w, 3), dtype=np.float32)
    blade_u8 = np.zeros((h, w), dtype=np.uint8)

    if mask is None or not mask.any():
        return full_fx, blade_u8

    effective_tip = tip
    if blade_extend:
        effective_tip = _apply_ignition(tip, hilt, ignition_frac)

    blade_u8 = _build_blade_shape(
        mask, frame_shape, effective_tip, hilt, canonical_width, blade_extend,
        tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
    )
    ys, xs = np.nonzero(blade_u8)
    if not len(xs):
        return full_fx, blade_u8

    x0 = max(0, int(xs.min()) - bbox_margin)
    y0 = max(0, int(ys.min()) - bbox_margin)
    x1 = min(w, int(xs.max()) + bbox_margin + 1)
    y1 = min(h, int(ys.max()) + bbox_margin + 1)

    blade01_local = blade_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    core_local = _make_core(blade01_local, core_erode_kernel, core_sigma)
    colour_local = _make_colour_band(blade01_local, colour_sigma)
    glow_local = _make_wide_glow(
        blade01_local, canonical_width, color_lin,
        glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
    ) * spill_strength

    fx_local = np.repeat(core_local[..., None], 3, axis=2)
    fx_local += colour_local[..., None] * color_lin[None, None, :]
    fx_local += glow_local

    kernel = _directional_kernel(velocity, motion_blur_gain, motion_blur_max_len)
    if kernel is not None:
        fx_local = cv2.filter2D(fx_local, -1, kernel)

    full_fx[y0:y1, x0:x1] = fx_local
    return full_fx, blade_u8


def render_glow(
    frames_dir,
    masks_dir,
    video_meta_path,
    output_frames_dir,
    motion_path,
    color=(40, 40, 255),
    spill_strength=0.35,
    blade_extend=True,
    # B1.2 capsule shape
    tip_extend_frac=0.10,
    hilt_taper_frac=0.12,
    hilt_taper_min_frac=0.35,
    # B1.3 core / colour band
    core_erode_frac=0.45,
    core_blur_frac=0.18,
    colour_blur_frac=0.35,
    # B1.4 wide glow, exponential falloff
    glow_scales=(0.5, 1.0, 2.0),
    glow_falloff_tau=1.0,
    glow_crush=4.0,
    # B1.5 Knoll darken
    knoll_darken_factor=0.7,
    knoll_dilate_frac=1.5,
    knoll_feather_frac=0.8,
    # B1.6 trail
    trail_decay=0.7,
    # Ignite/extinguish
    ignition_ramp_seconds=IGNITION_RAMP_SECONDS,
    # B1.7 directional motion blur
    motion_blur_gain=0.35,
    motion_blur_max_len=24,
    # B1.8 chromatic bloom + flicker
    chromatic_bloom_frac=0.10,
    flicker_strength=0.04,
    rng_seed=12345,
    # B1.9 light wrap
    light_wrap_strength=0.15,
    light_wrap_dilate_frac=1.5,
    progress_cb=None,
):
    """Render the glow for every tracked frame as a lossless PNG sequence
    (B1.10) into `output_frames_dir`, named ``{idx:05d}.png``.

    Reads blade geometry from `motion_path` (see
    ``lightsaber_fx.pipeline.blade.load_motion``) instead of writing a
    centroid track -- motion is now produced upstream by the "motion"
    pipeline stage. Each numbered step below is documented against
    docs/design-notes.md's B1 table.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    os.makedirs(output_frames_dir, exist_ok=True)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    with open(video_meta_path) as f:
        fps = float(f.readline())

    # Ignite/extinguish needs to know the clip's first/last frame with a
    # blade before the main loop reaches them, so it's a cheap separate pass
    # over mask presence rather than something the main loop can discover
    # about itself as it goes.
    mask_present = []
    for fname in frame_files:
        idx = int(os.path.splitext(fname)[0])
        m = load_mask_optional(masks_dir, idx)
        mask_present.append(m is not None and m.any())
    active_indices = [n for n, present in enumerate(mask_present) if present]
    first_active = active_indices[0] if active_indices else None
    last_active = active_indices[-1] if active_indices else None
    ignition_ramp_frames = round(fps * ignition_ramp_seconds)

    motion = load_motion(motion_path)
    tip_arr = np.asarray(motion.get("tip", np.zeros((0, 2))), dtype=np.float64)
    hilt_arr = np.asarray(motion.get("hilt", np.zeros((0, 2))), dtype=np.float64)
    axis_arr = np.asarray(motion.get("axis", np.zeros((0, 2))), dtype=np.float64)
    width_arr = np.asarray(motion.get("width", np.zeros(0)), dtype=np.float64)
    length_arr = np.asarray(motion.get("length", np.zeros(0)), dtype=np.float64)

    # Sign-continuity fix (see _stabilize_tip_hilt) -- do this before
    # anything reads tip/hilt/axis, since both the capsule and the B1.7
    # velocity vector depend on a stable tip/hilt labeling.
    tip_arr, hilt_arr, axis_arr = _stabilize_tip_hilt(tip_arr, hilt_arr, axis_arr)

    velocity = np.zeros_like(tip_arr)
    if len(tip_arr) > 1:
        velocity[1:] = np.diff(tip_arr, axis=0)
    velocity = np.nan_to_num(velocity, nan=0.0)

    # Blade width/length are pinned to the clip's median (not read per
    # frame) so blur sigmas and kernels below can be built once, outside
    # the frame loop, instead of being recomputed every frame from a
    # fluctuating per-frame estimate.
    canonical_width = _robust_median(width_arr, default=6.0)
    if canonical_width <= 0:
        canonical_width = 6.0
    canonical_length = _robust_median(length_arr, default=canonical_width * 4.0)
    if canonical_length <= 0:
        canonical_length = canonical_width * 4.0

    color_lin = (np.asarray(color, dtype=np.float32) / 255.0) ** _GAMMA

    first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first.shape[:2]

    core_erode_px = max(1, round(canonical_width * core_erode_frac))
    core_erode_kernel = np.ones((core_erode_px, core_erode_px), np.uint8)
    core_sigma = max(0.6, canonical_width * core_blur_frac)
    colour_sigma = max(0.6, canonical_width * colour_blur_frac)

    knoll_dilate_px = max(1, round(canonical_width * knoll_dilate_frac))
    knoll_dilate_kernel = np.ones((knoll_dilate_px, knoll_dilate_px), np.uint8)
    knoll_feather_sigma = max(1.0, canonical_width * knoll_feather_frac)

    wrap_dilate_px = max(1, round(canonical_width * light_wrap_dilate_frac))
    wrap_dilate_kernel = np.ones((wrap_dilate_px, wrap_dilate_px), np.uint8)
    wrap_blur_sigma = max(1.0, canonical_width)

    # Bounding-box margin (constant, computed once): large enough that the
    # widest glow blur, its crush re-blur, the directional-blur kernel, and
    # the tip extension are never truncated by the crop -- a clipped glow
    # would be a visible bug, not just a missed optimisation.
    max_scale = max(glow_scales) * (1.0 + chromatic_bloom_frac)
    blur_reach = int(np.ceil(canonical_width * max_scale * 3.5))
    extend_reach = int(np.ceil(canonical_length * tip_extend_frac))
    bbox_margin = blur_reach + motion_blur_max_len + extend_reach + 5

    rng = np.random.default_rng(rng_seed)
    trail = np.zeros((h, w, 3), dtype=np.float32)

    n_motion = len(tip_arr)
    total = len(frame_files)

    for n, fname in enumerate(frame_files):
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))
        mask = load_mask_optional(masks_dir, idx)

        plate_lin = _srgb_to_linear(frame)

        row = n if n < n_motion else None
        tip_i = tip_arr[row] if row is not None else None
        hilt_i = hilt_arr[row] if row is not None else None
        vel_i = velocity[row] if row is not None else np.zeros(2)

        frac = ignition_fraction(n, first_active, last_active, ignition_ramp_frames)
        full_fx, blade_u8 = _composite_blade_contribution(
            frame.shape, mask, tip_i, hilt_i, vel_i,
            canonical_width, blade_extend, frac,
            tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
            core_erode_kernel, core_sigma, colour_sigma,
            color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
            spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
        )

        # B1.6 -- decay always runs (even on a no-mask frame), so a trail
        # left behind by a lost-then-reacquired blade fades out normally
        # instead of freezing.
        trail = np.maximum(trail * trail_decay, full_fx)

        # B1.5 -- darken before colour is added.
        darkened_plate, _ = knoll_darken(
            plate_lin, blade_u8, knoll_dilate_px, knoll_darken_factor, knoll_feather_sigma,
            dilate_kernel=knoll_dilate_kernel,
        )
        # B1.9 -- approximate light wrap.
        wrap = _light_wrap(blade_u8, color_lin, wrap_dilate_kernel, wrap_blur_sigma, light_wrap_strength)

        # B1.8 -- per-frame intensity flicker (seeded, reproducible).
        jitter = 1.0 + float(rng.uniform(-flicker_strength, flicker_strength))

        combined = darkened_plate + (trail + wrap) * jitter
        combined = _soft_tonemap(combined)
        out = _linear_to_srgb(combined)

        cv2.imwrite(
            os.path.join(output_frames_dir, f"{idx:05d}.png"),
            out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        report((n + 1) / total * 100, f"frame {n + 1}/{total}")


def render_glow_multi(
    frames_dir,
    objects,
    video_meta_path,
    output_frames_dir,
    blade_extend=True,
    tip_extend_frac=0.10,
    hilt_taper_frac=0.12,
    hilt_taper_min_frac=0.35,
    core_erode_frac=0.45,
    core_blur_frac=0.18,
    colour_blur_frac=0.35,
    glow_scales=(0.5, 1.0, 2.0),
    glow_falloff_tau=1.0,
    glow_crush=4.0,
    knoll_darken_factor=0.7,
    knoll_dilate_frac=1.5,
    knoll_feather_frac=0.8,
    trail_decay=0.7,
    ignition_ramp_seconds=IGNITION_RAMP_SECONDS,
    motion_blur_gain=0.35,
    motion_blur_max_len=24,
    chromatic_bloom_frac=0.10,
    flicker_strength=0.04,
    rng_seed=12345,
    light_wrap_strength=0.15,
    light_wrap_dilate_frac=1.5,
    progress_cb=None,
):
    """Like `render_glow`, but for `len(objects)` (1-4) simultaneously
    tracked sabers, each with its own mask/motion/color/intensity, summed
    into one composited PNG sequence. See `_composite_blade_contribution`'s
    docstring for how a single object's contribution is computed; this
    function's job is purely to do that once per object per frame, sum the
    results, and run the shared (object-count-agnostic) trail/knoll-darken/
    tonemap steps exactly once per frame -- the same steps `render_glow`
    already runs on its own single contribution.

    Light wrap is the one exception: it is computed per object, from that
    object's own blade shape in that object's own color. Wrapping the
    union of every blade in every color would tint each blade's halo with
    every other saber's color -- a red-vs-blue duel would give both blades
    a magenta-ish halo -- and it is what makes the N=1 case identical to
    `render_glow` rather than merely close to it.
    """
    if not 1 <= len(objects) <= 4:
        raise ValueError("render_glow_multi supports 1-4 objects")

    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    os.makedirs(output_frames_dir, exist_ok=True)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    with open(video_meta_path) as f:
        fps = float(f.readline())

    first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first.shape[:2]
    total = len(frame_files)
    ignition_ramp_frames = round(fps * ignition_ramp_seconds)

    # Per-object setup: everything render_glow computes once from its one
    # shared masks_dir/motion_path, computed once per object here instead.
    prepared = []
    max_canonical_width = 1.0
    for obj in objects:
        masks_dir = obj["masks_dir"]
        motion = load_motion(obj["motion_path"])
        tip_arr = np.asarray(motion.get("tip", np.zeros((0, 2))), dtype=np.float64)
        hilt_arr = np.asarray(motion.get("hilt", np.zeros((0, 2))), dtype=np.float64)
        axis_arr = np.asarray(motion.get("axis", np.zeros((0, 2))), dtype=np.float64)
        width_arr = np.asarray(motion.get("width", np.zeros(0)), dtype=np.float64)
        tip_arr, hilt_arr, axis_arr = _stabilize_tip_hilt(tip_arr, hilt_arr, axis_arr)
        velocity = np.zeros_like(tip_arr)
        if len(tip_arr) > 1:
            velocity[1:] = np.diff(tip_arr, axis=0)
        velocity = np.nan_to_num(velocity, nan=0.0)

        canonical_width = _robust_median(width_arr, default=6.0)
        if canonical_width <= 0:
            canonical_width = 6.0
        max_canonical_width = max(max_canonical_width, canonical_width)

        mask_present = []
        for fname in frame_files:
            idx = int(os.path.splitext(fname)[0])
            m = load_mask_optional(masks_dir, idx)
            mask_present.append(m is not None and m.any())
        active_indices = [n for n, present in enumerate(mask_present) if present]

        color_lin = (np.asarray(obj["color"], dtype=np.float32) / 255.0) ** _GAMMA
        core_erode_px = max(1, round(canonical_width * core_erode_frac))
        prepared.append({
            "masks_dir": masks_dir,
            "tip_arr": tip_arr, "hilt_arr": hilt_arr, "velocity": velocity,
            "n_motion": len(tip_arr),
            "canonical_width": canonical_width,
            "first_active": active_indices[0] if active_indices else None,
            "last_active": active_indices[-1] if active_indices else None,
            "color_lin": color_lin,
            "spill_strength": obj["intensity"],
            "core_erode_kernel": np.ones((core_erode_px, core_erode_px), np.uint8),
            "core_sigma": max(0.6, canonical_width * core_blur_frac),
            "colour_sigma": max(0.6, canonical_width * colour_blur_frac),
        })

    knoll_dilate_px = max(1, round(max_canonical_width * knoll_dilate_frac))
    knoll_dilate_kernel = np.ones((knoll_dilate_px, knoll_dilate_px), np.uint8)
    knoll_feather_sigma = max(1.0, max_canonical_width * knoll_feather_frac)
    wrap_dilate_px = max(1, round(max_canonical_width * light_wrap_dilate_frac))
    wrap_dilate_kernel = np.ones((wrap_dilate_px, wrap_dilate_px), np.uint8)
    wrap_blur_sigma = max(1.0, max_canonical_width)

    max_scale = max(glow_scales) * (1.0 + chromatic_bloom_frac)
    blur_reach = int(np.ceil(max_canonical_width * max_scale * 3.5))
    bbox_margin = blur_reach + motion_blur_max_len + 5

    rng = np.random.default_rng(rng_seed)
    trail = np.zeros((h, w, 3), dtype=np.float32)

    for n, fname in enumerate(frame_files):
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))
        plate_lin = _srgb_to_linear(frame)

        combined_fx = np.zeros((h, w, 3), dtype=np.float32)
        combined_blade_u8 = np.zeros((h, w), dtype=np.uint8)
        per_object_blade_u8 = []

        for obj_state in prepared:
            mask = load_mask_optional(obj_state["masks_dir"], idx)
            row = n if n < obj_state["n_motion"] else None
            tip_i = obj_state["tip_arr"][row] if row is not None else None
            hilt_i = obj_state["hilt_arr"][row] if row is not None else None
            vel_i = obj_state["velocity"][row] if row is not None else np.zeros(2)
            frac = ignition_fraction(
                n, obj_state["first_active"], obj_state["last_active"], ignition_ramp_frames,
            )

            full_fx, blade_u8 = _composite_blade_contribution(
                frame.shape, mask, tip_i, hilt_i, vel_i,
                obj_state["canonical_width"], blade_extend, frac,
                tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
                obj_state["core_erode_kernel"], obj_state["core_sigma"], obj_state["colour_sigma"],
                obj_state["color_lin"], glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
                obj_state["spill_strength"], motion_blur_gain, motion_blur_max_len, bbox_margin,
            )
            combined_fx += full_fx
            combined_blade_u8 = np.maximum(combined_blade_u8, blade_u8)
            per_object_blade_u8.append(blade_u8)

        trail = np.maximum(trail * trail_decay, combined_fx)
        darkened_plate, _ = knoll_darken(
            plate_lin, combined_blade_u8, knoll_dilate_px, knoll_darken_factor, knoll_feather_sigma,
            dilate_kernel=knoll_dilate_kernel,
        )
        wrap_total = np.zeros((h, w, 3), dtype=np.float32)
        for obj_state, blade_u8 in zip(prepared, per_object_blade_u8, strict=False):
            wrap_total += _light_wrap(
                blade_u8, obj_state["color_lin"], wrap_dilate_kernel,
                wrap_blur_sigma, light_wrap_strength,
            )
        jitter = 1.0 + float(rng.uniform(-flicker_strength, flicker_strength))
        combined = darkened_plate + (trail + wrap_total) * jitter
        combined = _soft_tonemap(combined)
        out = _linear_to_srgb(combined)
        cv2.imwrite(
            os.path.join(output_frames_dir, f"{idx:05d}.png"),
            out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        report((n + 1) / total * 100, f"frame {n + 1}/{total}")
