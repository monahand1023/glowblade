import cv2
import numpy as np
import pytest

from lightsaber_fx.pipeline.detect import (
    MIN_ELONGATION,
    BladeProposal,
    MotionSeed,
    _points_on_axis,
    _sample_positions,
    detect_blade,
    propose_motion_seeds,
)

from .conftest import (
    BAR_LENGTH,
    HEIGHT,
    PIVOT,
    WIDTH,
    _rotating_bar,
    _static_distractors,
    _write_video,
)


def _on_the_bar(x, y, slack=18):
    """Is (x, y) within the disc the bar sweeps?"""
    return np.hypot(x - PIVOT[0], y - PIVOT[1]) <= BAR_LENGTH + slack


# --------------------------------------------------------------------------
# Motion seeds: pure OpenCV, no SAM2, no checkpoint.
# --------------------------------------------------------------------------

def test_motion_seeds_land_on_a_rotating_bar_despite_its_low_flow_elongation(
    rotating_bar_video
):
    # Regression test for the first design's flaw. A rotation-only swing
    # produces a flow blob near the tip with elongation ~2, so gating the
    # *flow* region on shape silently rejected every pivoting swing. Seeds
    # deliberately carry no shape gate; shape is judged later, on the SAM2
    # mask, which is the actual object.
    seeds = propose_motion_seeds(str(rotating_bar_video))

    assert seeds, "a clearly moving bar produced no motion seeds"
    assert _on_the_bar(*seeds[0].point), f"fastest seed {seeds[0].point} is off the bar"


def test_motion_seeds_ignore_long_static_lines(rotating_bar_video):
    # Every seed, not just the best, must be on the moving object: the
    # static horizon at y=60, fence at y~195 and post at x=30 are all more
    # elongated than the bar and must never be proposed.
    seeds = propose_motion_seeds(str(rotating_bar_video))

    for seed in seeds:
        assert _on_the_bar(*seed.point), f"seed {seed.point} is on static scenery"


def test_motion_seeds_are_ordered_fastest_first(rotating_bar_video):
    seeds = propose_motion_seeds(str(rotating_bar_video))

    speeds = [seed.speed for seed in seeds]
    assert speeds == sorted(speeds, reverse=True)


def test_motion_seeds_survive_a_camera_pan(tmp_path):
    # A pan moves every pixel, so without compensation the whole frame reads
    # as motion and the static scenery outscores the bar. The median-flow
    # subtraction in _relative_motion is what makes this case work.
    path = tmp_path / "pan.mp4"
    _write_video(path, lambda frame, i: (_static_distractors(frame, shift=i * 3),
                                         _rotating_bar(frame, i)))

    seeds = propose_motion_seeds(str(path))

    assert seeds
    assert _on_the_bar(*seeds[0].point), f"seed {seeds[0].point} is off the bar under a pan"


def test_motion_seeds_are_empty_when_nothing_moves(tmp_path):
    path = tmp_path / "still.mp4"
    _write_video(path, lambda frame, i: _static_distractors(frame))

    assert propose_motion_seeds(str(path)) == []


def test_motion_seeds_are_capped(rotating_bar_video):
    assert len(propose_motion_seeds(str(rotating_bar_video), max_seeds=2)) <= 2


def test_motion_seeds_reject_an_unreadable_file(tmp_path):
    path = tmp_path / "not-a-video.mp4"
    path.write_bytes(b"nope")

    with pytest.raises(ValueError, match="Could not read a frame"):
        propose_motion_seeds(str(path))


# --------------------------------------------------------------------------
# detect_blade: SAM2 replaced wholesale at _build_image_predictor.
# --------------------------------------------------------------------------

class _FakePredictor:
    """Stands in for SAM2's image predictor. `masks_for` maps a prompt point
    to the mask stack SAM2 would return for it, so a test can say "asking
    about this point yields a bat" or "...yields a torso"."""

    def __init__(self, masks_for, calls=None):
        self.masks_for = masks_for
        self.calls = calls if calls is not None else []
        self._image = None

    def set_image(self, image):
        self._image = image

    def predict(self, point_coords, point_labels, multimask_output=True):
        point = (int(point_coords[0][0]), int(point_coords[0][1]))
        self.calls.append(point)
        masks = self.masks_for(point)
        return np.asarray(masks), np.ones(len(masks)), None


def _bar_mask_through(point, thickness=7):
    """A bar-shaped mask running from the fixture's pivot through `point`.

    Built from the seed point rather than at fixed coordinates so it behaves
    like a real SAM2 mask of the moving bar: it contains the prompt point and
    it overlaps the pixels the flow found moving. Fixed-coordinate fakes
    passed the old code but are rejected by the cover and moving-fraction
    gates -- correctly, since a mask that contains neither its own prompt nor
    any motion is exactly the background-segmentation failure those gates
    exist to catch.
    """
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    direction = np.array(point, dtype=float) - np.array(PIVOT, dtype=float)
    norm = np.linalg.norm(direction)
    if norm < 1:
        direction, norm = np.array([0.0, -1.0]), 1.0
    far = np.array(PIVOT, dtype=float) + direction / norm * (BAR_LENGTH + 10)
    cv2.line(mask, PIVOT, (int(far[0]), int(far[1])), 1, thickness)
    return mask.astype(bool)


def _blob_mask(point, radius=22):
    """A round mask centred on the prompt point: moving, and on the object,
    but nothing like a blade. Separates the shape gate from the motion gate."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(mask, (int(point[0]), int(point[1])), radius, 1, -1)
    return mask.astype(bool)


def _install_fake_sam2(monkeypatch, masks_for):
    calls = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.detect._build_image_predictor",
        lambda checkpoint_path, config_name, device: _FakePredictor(masks_for, calls),
    )
    return calls


def test_detect_blade_returns_the_elongated_mask_sam2_gives_back(
    monkeypatch, rotating_bar_video
):
    _install_fake_sam2(monkeypatch, lambda point: [_blob_mask(point), _bar_mask_through(point)])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert proposal.elongation >= MIN_ELONGATION
    # The proposal must describe the bar mask, not the blob SAM2 also offered.
    assert proposal.mask.sum() == _bar_mask_through(proposal.seed.point).sum()


def test_detect_blade_points_all_land_on_the_proposed_mask(
    monkeypatch, rotating_bar_video
):
    # An include point on background is the one mistake that produces an
    # empty track and a render with no glow in it, so every proposed point
    # must be a foreground pixel of the mask being proposed -- not a
    # geometric construction like a centroid, which for a bent object can
    # fall outside it.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask_through(point)])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    for x, y in proposal.points:
        assert proposal.mask[y, x], f"point ({x},{y}) is not on the proposed mask"


def test_detect_blade_labels_are_all_includes(monkeypatch, rotating_bar_video):
    # Detection proposes what the object *is*, never what to carve out of
    # it. An exclude point that lands on the object is how a track ends up
    # empty, so detection never emits one.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask_through(point)])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert proposal.labels == [1] * len(proposal.points)


def test_detect_blade_returns_none_when_no_mask_is_blade_like(
    monkeypatch, rotating_bar_video
):
    # Motion alone is not enough. A moving foot, a thrown ball or a walking
    # person all move without being blade-shaped, and the honest answer is
    # "I could not find it" -- which the caller turns into "click it
    # yourself" rather than a confident wrong guess.
    _install_fake_sam2(monkeypatch, lambda point: [_blob_mask(point), _blob_mask(point, radius=30)])

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_detect_blade_rejects_a_mask_covering_most_of_the_frame(
    monkeypatch, rotating_bar_video
):
    # A segmentation that swallows the background can be extremely
    # "elongated" by the length/width fit while being useless, so mask area
    # is bounded as well as shape.
    everything = np.ones((HEIGHT, WIDTH), dtype=bool)
    _install_fake_sam2(monkeypatch, lambda point: [everything])

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_detect_blade_scores_every_seed_rather_than_taking_the_first(
    monkeypatch, rotating_bar_video
):
    # Seeds are ordered by speed, and the fastest-moving thing is not reliably
    # the most blade-like thing. An earlier version returned the first seed
    # that cleared the bar, and on a golf clip that meant proposing the *sky*
    # (elongation 8.7, from the fastest seed) while the club shaft (21.0, from
    # a slower one) was never looked at. So: every seed must be consulted, and
    # the best candidate must win -- not the first acceptable one.
    seeds = propose_motion_seeds(str(rotating_bar_video))
    if len(seeds) < 2:
        pytest.skip("fixture produced a single motion seed; nothing to compare")
    first_point = tuple(seeds[0].point)

    def masks_for(point):
        # The first (fastest) seed offers a thin-but-shorter bar; a later seed
        # offers a longer one. The longer one must win.
        thickness = 12 if point == first_point else 6
        return [_bar_mask_through(point, thickness=thickness)]

    calls = _install_fake_sam2(monkeypatch, masks_for)
    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert len(calls) == len(seeds), "did not consult every seed"
    assert tuple(proposal.seed.point) != first_point, (
        "kept the fastest seed's worse candidate instead of the best one"
    )


def test_detect_blade_reports_the_frame_its_points_refer_to(
    monkeypatch, rotating_bar_video
):
    # The points are only meaningful together with the frame they were found
    # in: a swung object is somewhere else entirely a few frames later. On
    # the real 10 s clip the best frame is 135 of 300, so reading the points
    # against frame 0 puts them on whatever happens to be there -- a mistake
    # I made myself. The frame index travels with the points, and
    # track_object takes it as prompt_frame.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask_through(point)])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert isinstance(proposal.frame_index, int)
    assert proposal.frame_index == proposal.seed.frame_index


def test_detect_blade_returns_none_for_a_clip_too_short_to_have_motion(
    monkeypatch, tmp_path
):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("built a SAM2 predictor for a clip with no motion")

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.detect._build_image_predictor", fail_if_called
    )
    path = tmp_path / "one-frame.mp4"
    _write_video(path, lambda frame, i: _rotating_bar(frame, i), n_frames=1)

    assert detect_blade(str(path), "ckpt", "cfg", "cpu") is None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def test_sample_positions_never_indexes_past_the_last_pair():
    for n_frames in range(12):
        for position in _sample_positions(n_frames, 12):
            assert position + 1 <= n_frames - 1


def test_sample_positions_spreads_across_the_whole_clip():
    # Sampling only the start would miss the swing in any clip that opens
    # with the batter standing still -- which is most of them.
    positions = _sample_positions(300, 12)

    assert len(positions) == 12
    assert positions[0] == 0
    assert positions[-1] == 298
    assert max(np.diff(positions)) <= 28


def test_sample_positions_handles_a_clip_shorter_than_the_sample_count():
    assert _sample_positions(4, 12) == [0, 1, 2]
    assert _sample_positions(1, 12) == []
    assert _sample_positions(0, 12) == []


def test_points_on_axis_spreads_points_along_a_diagonal_object():
    mask = np.zeros((80, 80), dtype=bool)
    for t in range(60):
        mask[10 + t, 10 + t] = True

    points = _points_on_axis(mask)

    assert len(points) == 3
    for x, y in points:
        assert mask[y, x], f"({x},{y}) is off the object"
    spread = max(p[0] for p in points) - min(p[0] for p in points)
    assert spread >= 20


def test_points_on_axis_raises_a_clear_error_for_an_empty_mask():
    # Confirmed as a real, reachable crash on real footage: an empty mask
    # reaching np.linalg.svd's PCA raised an uncaught IndexError three
    # lines down (index 0 is out of bounds for axis 0 with size 0), only
    # survived because an unrelated outer try/except happened to catch it
    # (see reacquire._retrack_one_object). Every current caller already
    # filters out an empty mask before calling this; this guards a future
    # caller's equivalent mistake with a clear, immediate error instead.
    mask = np.zeros((80, 80), dtype=bool)

    with pytest.raises(ValueError, match="no foreground pixels"):
        _points_on_axis(mask)


def test_motion_seed_and_proposal_reprs_are_readable():
    # These get printed in CLI output and agent reports; a default repr
    # there is useless.
    seed = MotionSeed(frame_index=7, point=[10, 20], speed=3.5, area=99)
    proposal = BladeProposal(
        frame_index=7, points=[[10, 20]], labels=[1],
        mask=np.zeros((4, 4), dtype=bool), elongation=6.25, seed=seed,
    )

    assert "frame_index=7" in repr(seed)
    assert "speed=3.50" in repr(seed)
    assert "elongation=6.2" in repr(proposal)


# --------------------------------------------------------------------------
# The gates that stop SAM2's background segmentations being proposed.
# Each is pinned to the real clip that motivated it.
# --------------------------------------------------------------------------

def test_detect_blade_rejects_a_mask_off_in_the_background(
    monkeypatch, rotating_bar_video
):
    # End-to-end rejection of the shape the real failures took: elongated,
    # blade-like, and nowhere near the moving object. Several gates can catch
    # this one, and that is fine here -- this test asserts the *outcome*. The
    # gates are isolated individually in the _candidate_masks tests below,
    # because neutering any single one of them leaves this test still
    # passing, which would make it useless as a pin for that gate.
    def masks_for(point):
        # Elongated and blade-like, but deliberately nowhere near the prompt.
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.line(mask, (5, HEIGHT - 5), (120, HEIGHT - 5), 1, 6)
        return [mask.astype(bool)]

    _install_fake_sam2(monkeypatch, masks_for)

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_detect_blade_rejects_a_speckled_background_mask(monkeypatch, rotating_bar_video):
    # Outcome-level: the ragged, shattered mask SAM2 returns for soft
    # out-of-focus background must not be proposed. Isolated gate coverage is
    # in the _candidate_masks tests below.
    def masks_for(point):
        mask = _bar_mask_through(point)
        # Scatter far more specks than any real object mask produced.
        rng = np.random.default_rng(0)
        ys = rng.integers(0, HEIGHT, 60)
        xs = rng.integers(0, WIDTH, 60)
        for y, x in zip(ys, xs, strict=False):
            mask[y, x] = True
        return [mask]

    _install_fake_sam2(monkeypatch, masks_for)

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_detect_blade_rejects_a_mask_whose_pixels_are_not_moving(
    monkeypatch, rotating_bar_video
):
    # This one DOES isolate its gate: verified by neutering the
    # moving-fraction check and confirming this test then fails. It is the
    # golf clip's failure -- a smear of blurred treeline beside the club,
    # 21.0 on elongation, of which only 0.20 of its pixels were moving,
    # against 0.54-1.00 for correct proposals.
    #
    # The mask contains the prompt and is solid and blade-shaped, so cover,
    # shape and speckle all pass; it runs perpendicular to the bar out into
    # still background, so only the motion check can reject it.
    def masks_for(point):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.line(mask, (int(point[0]), int(point[1])), (WIDTH - 2, int(point[1])), 1, 5)
        return [mask.astype(bool)]

    _install_fake_sam2(monkeypatch, masks_for)

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_detect_blade_rejects_a_frame_spanning_mask(
    monkeypatch, rotating_bar_video
):
    # Outcome-level: a mask sweeping most of the frame must not be proposed.
    # Isolated gate coverage is in the _candidate_masks tests below.
    def masks_for(point):
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        # A long, solid, blade-shaped sweep covering far more than the bar.
        cv2.line(mask, (2, int(point[1])), (WIDTH - 2, int(point[1])), 1, 40)
        return [mask.astype(bool)]

    _install_fake_sam2(monkeypatch, masks_for)

    assert detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu") is None


def test_min_elongation_is_above_what_a_standing_person_measures():
    # Pins the reason the floor was raised. A standing human body fitted 3.7
    # on the sword clip, which the old 3.5 floor accepted -- so the floor has
    # to sit above the range a person can reach, not merely above a hand.
    from lightsaber_fx.pipeline.detect import MIN_ELONGATION as floor

    assert floor > 4.0, "a standing person fits at ~3.7; the floor must clear it"


def test_moving_fraction_is_permissive_when_no_flow_map_is_available():
    # A MotionSeed built by hand (by a caller or a test) has no flow map, and
    # must not be silently rejected by a check it cannot feed.
    from lightsaber_fx.pipeline.detect import _moving_fraction

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 4] = True

    assert _moving_fraction(mask, None) == 1.0


def test_moving_fraction_rescales_a_flow_map_of_a_different_size():
    # The flow runs at FLOW_LONG_EDGE while masks are at source resolution, so
    # these two are routinely different shapes. Comparing them without a
    # resize would raise, or worse, index the wrong pixels.
    from lightsaber_fx.pipeline.detect import _moving_fraction

    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 20] = True
    hot = np.zeros((20, 20), dtype=bool)
    hot[5:15, 10] = True

    assert _moving_fraction(mask, hot) > 0.5


def test_speckle_count_ignores_the_largest_component():
    from lightsaber_fx.pipeline.detect import _speckle_count

    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 20] = True          # the object
    assert _speckle_count(mask) == 0

    mask[2, 2] = True              # one speck
    mask[2, 38] = True             # another
    assert _speckle_count(mask) == 2


# --------------------------------------------------------------------------
# Isolated gate coverage.
#
# Driving _candidate_masks directly, with `hot` covering the whole frame so
# the moving-fraction check can never be the reason a candidate is rejected,
# and a hand-built seed so cover and scale are under the test's control. Each
# test below violates exactly one gate; neutering that gate makes exactly
# that test fail, which is the property the end-to-end tests above could not
# provide.
# --------------------------------------------------------------------------

def _candidates_for(mask, *, point=(50, 50), motion_area=400.0, max_mask_area=1e9):
    from lightsaber_fx.pipeline.detect import MotionSeed, _candidate_masks

    seed = MotionSeed(
        frame_index=0, point=list(point), speed=5.0, area=int(motion_area),
        hot=np.ones(mask.shape, dtype=bool),   # everything is "moving"
    )
    predictor = _FakePredictor(lambda _p: [mask])
    frame = np.zeros((*mask.shape, 3), dtype=np.uint8)
    return list(
        _candidate_masks(predictor, frame, seed, max_mask_area, motion_area)
    )


def _solid_bar_through(point, shape=(100, 100), length=40, thickness=5):
    """A solid, blade-shaped mask centred on `point`."""
    mask = np.zeros(shape, dtype=np.uint8)
    x, y = int(point[0]), int(point[1])
    cv2.line(mask, (x, y - length // 2), (x, y + length // 2), 1, thickness)
    return mask.astype(bool)


def test_candidate_masks_accepts_a_solid_blade_on_the_prompt():
    # The control. Without this, every rejection test below could be passing
    # because the harness rejects everything.
    assert _candidates_for(_solid_bar_through((50, 50))), "the control candidate was rejected"


def test_candidate_masks_rejects_a_mask_not_containing_the_prompt():
    # Isolates the cover gate: identical to the control except the prompt
    # pixel is punched out, so shape, speckle, scale and motion all still
    # pass. SAM2 really does return masks that exclude their own prompt --
    # on a sword clip that candidate was the whole swordsman.
    mask = _solid_bar_through((50, 50))
    mask[45:56, 48:53] = False          # hole over the prompt
    assert mask[50, 50] == False

    assert _candidates_for(mask) == []


def test_candidate_masks_rejects_a_speckled_mask():
    # Isolates the speckle gate: a solid bar on the prompt, moving, correctly
    # scaled, plus more disconnected specks than any real object mask
    # produced (measured 2-9 for correct proposals, 41-47 for background).
    mask = _solid_bar_through((50, 50))
    rng = np.random.default_rng(1)
    for y, x in zip(rng.integers(0, 100, 40), rng.integers(70, 100, 40), strict=False):
        mask[y, x] = True

    assert _candidates_for(mask, motion_area=1e9) == []


def test_candidate_masks_rejects_a_mask_far_larger_than_its_motion():
    # Isolates the scale gate: a solid bar on the prompt, fully moving, few
    # specks -- but the motion component that seeded it was tiny.
    mask = _solid_bar_through((50, 50))

    assert _candidates_for(mask, motion_area=1.0) == []


def test_candidate_masks_rejects_a_mask_over_the_absolute_area_cap():
    mask = np.ones((100, 100), dtype=bool)

    assert _candidates_for(mask, max_mask_area=500) == []
