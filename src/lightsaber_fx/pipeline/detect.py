"""Find the swung object automatically, so the first click is optional.

The object you want to turn into a blade is *the thing being swung*: the
fastest-moving object in the clip. That is a far more specific target than
"a bat-shaped thing", and it rules out the distractors that defeat a purely
shape-based search. A fence rail, the horizon, a chalk line and a roof edge
are all long, straight and high-contrast -- and all perfectly still, so they
score zero here no matter how bat-shaped they look.

Motion decides *where* to look. SAM2 decides *what is there*. `fit_blade`
decides whether that is blade-like enough to propose.

Splitting it that way matters, and the first version of this module got it
wrong. It scored elongation on the *optical-flow* region and gated on that,
which measures the wrong thing: flow magnitude on a rotating object scales
with radius, so a bat pivoting about the hands lights up only near its tip.
The resulting motion blob has an elongation around 2, and a shape gate set
anywhere sensible for a blade rejects it -- a swing dominated by rotation
rather than travel would silently never be detected. A seed point on the
tip is perfectly good for prompting, though, and SAM2 returns the whole
object's silhouette from it. So elongation is now measured on the SAM2 mask,
which is the actual object, and is also exactly what the renderer will later
have to draw.

The pipeline:

1. Sample pairs of adjacent frames at `n_samples` positions spread across
   the whole clip, so a clip that opens with the batter standing still is
   still searched where the action is.
2. Compute dense optical flow per pair and subtract the *median* flow
   vector. That median stands in for camera motion: a pan moves every pixel
   by roughly the same amount, so subtracting it leaves motion relative to
   the scene. It handles pans and tilts, not zoom or roll -- a known limit.
3. Keep the fastest pixels, split them into components, and emit the top few
   as seed points, fastest first. No shape gate here; see above.
4. Segment each seed with SAM2's image predictor, asking for several mask
   granularities (a bat alone, versus a bat merged with the hands), fit each
   with `fit_blade`, and take the most elongated. The first seed that yields
   a blade-like mask wins.

What comes back is a proposal, not a decision. It carries the frame index,
prompt points that are guaranteed to lie on the mask, and the mask itself
for display. The caller is expected to show it and let the user accept or
override: detection will sometimes be wrong, and a wrong guess that costs a
silent full render is worse than a click. `detect_blade` returns None rather
than a low-confidence guess.
"""

import cv2
import numpy as np

from .blade import fit_blade

# Work at this long edge for the flow pass regardless of source resolution.
# Optical flow at 720p or 1080p costs several times more for no benefit: we
# need the location of a large moving object, not sub-pixel accuracy.
FLOW_LONG_EDGE = 480

# A SAM2 mask must be at least this elongated (length/width, from the same
# fit the renderer uses) to be called a blade. A bat, broom or sword in frame
# measures roughly 5-15; a torso, a head or a hand measures 1-3. Below this
# we return None and let the user click rather than propose something wrong.
MIN_ELONGATION = 3.5

# Area bounds as a fraction of the frame. Applied to flow components (below
# the floor is flow speckle) and again to SAM2 masks (above the ceiling the
# "object" is the background, the whole person, or a failed segmentation).
MIN_MOTION_AREA_FRAC = 0.0004
MAX_MOTION_AREA_FRAC = 0.25
MAX_MASK_AREA_FRAC = 0.25

# Keep only the fastest pixels of each frame pair. High, because the swung
# object is expected to be dramatically faster than the person swinging it.
MOTION_PERCENTILE = 97.0

# Absolute floor, in flow pixels per frame at FLOW_LONG_EDGE. A percentile is
# a *relative* threshold, so on a clip where nothing moves it lands in the
# codec noise and happily reports the "fastest" 3% of a still frame: measured
# on a static test clip, that produced a 2318-pixel component with a mean
# speed of 0.00. This floor is what lets "nothing is moving" be answered as
# such. For scale, the winning seeds on the real test clips measure 5.0 and
# 16.1, and compression noise sits below 0.3.
MIN_SEED_SPEED = 0.5

# How many motion seeds to hand to SAM2 before giving up. Each costs one
# image-embedding pass, so this is the main cost knob. Five is enough to get
# past a couple of false leads (a foot, a ball) without a long stall.
MAX_SEEDS = 5


class MotionSeed:
    """A place worth asking SAM2 about: a point on the fastest-moving thing
    found at one sampled moment, in source-resolution coordinates."""

    def __init__(self, frame_index, point, speed, area):
        self.frame_index = int(frame_index)
        self.point = point
        self.speed = float(speed)
        self.area = int(area)

    def __repr__(self):
        return (
            f"MotionSeed(frame_index={self.frame_index}, point={self.point}, "
            f"speed={self.speed:.2f}, area={self.area})"
        )


class BladeProposal:
    """A detected candidate, with what a caller needs to show it and what the
    tracker needs to act on it.

    `points`/`labels` are in source-resolution coordinates, in the format
    `track_object` takes. `frame_index` is the frame they refer to, which is
    usually *not* frame 0 -- pass it as `track_object`'s `prompt_frame`.
    `mask` is SAM2's mask for the object at that frame, at source resolution,
    for overlaying in the confirmation step.
    """

    def __init__(self, frame_index, points, labels, mask, elongation, seed):
        self.frame_index = int(frame_index)
        self.points = points
        self.labels = labels
        self.mask = mask
        self.elongation = float(elongation)
        self.seed = seed

    def __repr__(self):
        return (
            f"BladeProposal(frame_index={self.frame_index}, points={self.points}, "
            f"elongation={self.elongation:.1f})"
        )


def _sample_positions(n_frames, n_samples):
    """Frame indices to examine, spread across the clip.

    Each position needs a successor to compute flow against, so the last
    usable index is `n_frames - 2`. Returns at most `n_samples` unique
    indices, and an empty list for a clip too short to have any motion.
    """
    last = n_frames - 2
    if last < 0:
        return []
    if n_samples >= last + 1:
        return list(range(last + 1))
    return sorted({round(p) for p in np.linspace(0, last, n_samples)})


def _read_frame(cap, index):
    """Read one BGR frame by index, or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _relative_motion(prev_gray, next_gray):
    """Per-pixel motion magnitude with global (camera) motion removed."""
    flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(
        prev_gray, next_gray, None
    )
    flow = flow - np.median(flow.reshape(-1, 2), axis=0)
    return np.linalg.norm(flow, axis=2)


def _hot_components(magnitude, min_area, max_area):
    """Split the fastest-moving pixels into plausible object components.

    Yields `(component_mask, mean_speed, area)`. Deliberately has no shape
    gate: on a rotating object the flow only lights up near the tip, so
    shape-gating here rejects exactly the swings this is meant to catch.
    There is an absolute speed gate, though -- see `MIN_SEED_SPEED`.
    """
    threshold = np.percentile(magnitude, MOTION_PERCENTILE)
    if not np.isfinite(threshold) or threshold < MIN_SEED_SPEED:
        return
    hot = (magnitude >= threshold).astype(np.uint8)
    # Close before opening: a motion-blurred object often breaks into a
    # dashed line of hot pixels, and closing rejoins it into one component
    # before the open removes isolated speckle.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, kernel)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels_img, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        component = labels_img == label
        speed = float(magnitude[component].mean())
        if speed < MIN_SEED_SPEED:
            continue
        yield component, speed, area


def _fastest_pixel(component, magnitude):
    """The single fastest pixel of a component, as (x, y).

    Prompting the fastest point rather than the centroid matters for a bent
    or curved object: the centroid of a banana-shaped flow region can fall
    off it entirely, and an include point on background is the one mistake
    that produces an empty mask and a render with no glow in it.
    """
    masked = np.where(component, magnitude, -np.inf)
    y, x = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return int(x), int(y)


def propose_motion_seeds(video_path, n_samples=12, max_seeds=MAX_SEEDS, progress_cb=None):
    """Rank places in `video_path` where something is moving fast, fastest
    first. Pure OpenCV on the CPU -- no GPU contention with tracking."""
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not read a frame from {video_path}")
    try:
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        positions = _sample_positions(n_frames, n_samples)
        if not positions or width <= 0 or height <= 0:
            report(100, "clip too short to detect motion")
            return []

        scale = min(1.0, FLOW_LONG_EDGE / max(width, height))
        flow_area = (width * scale) * (height * scale)
        min_area = max(12.0, MIN_MOTION_AREA_FRAC * flow_area)
        max_area = MAX_MOTION_AREA_FRAC * flow_area

        seeds = []
        for i, index in enumerate(positions):
            report(i / len(positions) * 100, f"scanning frame {index + 1}/{n_frames}")
            frames = [_read_frame(cap, index), _read_frame(cap, index + 1)]
            if any(f is None for f in frames):
                continue
            grays = []
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if scale != 1.0:
                    gray = cv2.resize(gray, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_AREA)
                grays.append(gray)
            magnitude = _relative_motion(*grays)
            for component, speed, area in _hot_components(magnitude, min_area, max_area):
                x, y = _fastest_pixel(component, magnitude)
                seeds.append(MotionSeed(
                    frame_index=index,
                    point=[round(x / scale), round(y / scale)],
                    speed=speed,
                    area=area,
                ))
    finally:
        cap.release()

    seeds.sort(key=lambda s: -s.speed)
    return seeds[:max_seeds]


def _build_image_predictor(checkpoint_path, config_name, device):
    """Isolated so tests can replace the whole SAM2 dependency in one place."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    return SAM2ImagePredictor(build_sam2(config_name, checkpoint_path, device=device))


def _best_blade_mask(predictor, frame_bgr, point, max_mask_area):
    """Segment at `point` and return the most blade-like mask, or None.

    Asks SAM2 for several granularities because a point on a bat plausibly
    means the bat, or the bat plus the hands gripping it, or the whole
    batter. Scoring each with `fit_blade` and keeping the most elongated is
    what picks the bat out of that set.
    """
    predictor.set_image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    masks, _scores, _logits = predictor.predict(
        point_coords=np.array([point], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    best = None
    for mask in np.asarray(masks):
        mask = mask.astype(bool)
        area = int(mask.sum())
        if area == 0 or area > max_mask_area:
            continue
        geometry = fit_blade(mask)
        if geometry is None:
            continue
        elongation = geometry.length / max(geometry.width, 1.0)
        if best is None or elongation > best[0]:
            best = (elongation, mask)
    return best


def _points_on_axis(mask, fractions=(0.3, 0.5, 0.7)):
    """Prompt points guaranteed to lie on `mask`.

    Projecting the mask's own pixels onto its major axis and taking the pixel
    nearest each target fraction means every returned point is a real
    foreground pixel, and they are spread along the object rather than
    clustered at one end.
    """
    ys, xs = np.nonzero(mask)
    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    centered = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ vt[0]
    lo, hi = projection.min(), projection.max()

    points = []
    for fraction in fractions:
        target = lo + fraction * (hi - lo)
        nearest = int(np.argmin(np.abs(projection - target)))
        point = [round(coords[nearest, 0]), round(coords[nearest, 1])]
        if point not in points:
            points.append(point)
    return points


def detect_blade(
    video_path,
    checkpoint_path,
    config_name,
    device,
    n_samples=12,
    max_seeds=MAX_SEEDS,
    progress_cb=None,
):
    """Propose the swung object in `video_path`, or return None.

    Returns a `BladeProposal` whose `frame_index` should be passed to
    `track_object` as `prompt_frame` -- the points refer to that frame, not
    to frame 0.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    seeds = propose_motion_seeds(
        video_path, n_samples=n_samples, max_seeds=max_seeds,
        progress_cb=lambda pct, message: report(pct * 0.5, message),
    )
    if not seeds:
        report(100, "found nothing moving")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not read a frame from {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_mask_area = MAX_MASK_AREA_FRAC * width * height
        predictor = _build_image_predictor(checkpoint_path, config_name, device)

        for i, seed in enumerate(seeds):
            report(50 + i / len(seeds) * 50, f"checking frame {seed.frame_index + 1}")
            frame = _read_frame(cap, seed.frame_index)
            if frame is None:
                continue
            best = _best_blade_mask(predictor, frame, seed.point, max_mask_area)
            if best is None:
                continue
            elongation, mask = best
            if elongation < MIN_ELONGATION:
                continue
            report(100, f"found a blade in frame {seed.frame_index + 1} "
                        f"(elongation {elongation:.1f})")
            points = _points_on_axis(mask)
            return BladeProposal(
                frame_index=seed.frame_index,
                points=points,
                labels=[1] * len(points),
                mask=mask,
                elongation=elongation,
                seed=seed,
            )
    finally:
        cap.release()

    report(100, "nothing blade-like found")
    return None
