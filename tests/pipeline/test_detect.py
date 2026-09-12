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

WIDTH, HEIGHT = 320, 240
FPS = 30.0
PIVOT = (160, 140)
BAR_LENGTH = 70


def _write_video(path, draw_frame, n_frames=40):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for i in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        draw_frame(frame, i)
        writer.write(frame)
    writer.release()


def _static_distractors(frame, shift=0):
    """The things a shape-only search gets wrong: long, straight,
    high-contrast and perfectly still. A fence rail and a horizon are far
    more elongated than any bat, so a detector scoring on shape alone picks
    one of them. Scoring on motion is what makes them score zero."""
    cv2.line(frame, (-shift, 60), (WIDTH - shift, 60), (200, 200, 200), 3)
    cv2.line(frame, (-shift, 200), (WIDTH - shift, 190), (180, 180, 180), 4)
    cv2.line(frame, (30 - shift, 0), (30 - shift, HEIGHT), (170, 170, 170), 3)


def _rotating_bar(frame, i, n_frames=40):
    """An elongated object pivoting about one end -- rotation only, no
    travel. This is the case that broke the first version of the detector:
    optical-flow magnitude scales with radius, so only the bar's tip lights
    up, and the resulting flow blob has an elongation around 2. Any shape
    gate applied to the *flow* region rejects it."""
    angle = -np.pi / 2 + (i / n_frames) * np.pi
    tip = (
        int(PIVOT[0] + BAR_LENGTH * np.cos(angle)),
        int(PIVOT[1] + BAR_LENGTH * np.sin(angle)),
    )
    cv2.line(frame, PIVOT, tip, (240, 240, 240), 7)
    return tip


def _on_the_bar(x, y, slack=18):
    """Is (x, y) within the disc the bar sweeps?"""
    return np.hypot(x - PIVOT[0], y - PIVOT[1]) <= BAR_LENGTH + slack


@pytest.fixture
def rotating_bar_video(tmp_path):
    path = tmp_path / "swing.mp4"
    _write_video(path, lambda frame, i: (_static_distractors(frame), _rotating_bar(frame, i)))
    return path


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


def _bar_mask(vertical=True, thickness=6, length=90):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    if vertical:
        mask[60:60 + length, 150:150 + thickness] = True
    else:
        mask[150:150 + thickness, 60:60 + length] = True
    return mask


def _blob_mask(radius=25):
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    mask[(yy - 120) ** 2 + (xx - 160) ** 2 <= radius ** 2] = True
    return mask


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
    _install_fake_sam2(monkeypatch, lambda point: [_blob_mask(), _bar_mask()])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert proposal.elongation >= MIN_ELONGATION
    # The proposal must describe the bar mask, not the blob SAM2 also offered.
    assert proposal.mask.sum() == _bar_mask().sum()


def test_detect_blade_points_all_land_on_the_proposed_mask(
    monkeypatch, rotating_bar_video
):
    # An include point on background is the one mistake that produces an
    # empty track and a render with no glow in it, so every proposed point
    # must be a foreground pixel of the mask being proposed -- not a
    # geometric construction like a centroid, which for a bent object can
    # fall outside it.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask(vertical=False)])

    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    for x, y in proposal.points:
        assert proposal.mask[y, x], f"point ({x},{y}) is not on the proposed mask"


def test_detect_blade_labels_are_all_includes(monkeypatch, rotating_bar_video):
    # Detection proposes what the object *is*, never what to carve out of
    # it. An exclude point that lands on the object is how a track ends up
    # empty, so detection never emits one.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask()])

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
    _install_fake_sam2(monkeypatch, lambda point: [_blob_mask(), _blob_mask(radius=30)])

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


def test_detect_blade_tries_later_seeds_when_the_first_is_not_a_blade(
    monkeypatch, rotating_bar_video
):
    # The fastest-moving thing is not always the object -- a ball can beat a
    # bat. Falling through to the next seed is what makes that survivable,
    # so this asserts more than one seed was actually consulted rather than
    # just that the answer came out right.
    seeds = propose_motion_seeds(str(rotating_bar_video))
    if len(seeds) < 2:
        pytest.skip("fixture produced a single motion seed; nothing to fall through to")
    first_point = tuple(seeds[0].point)

    def masks_for(point):
        return [_blob_mask()] if point == first_point else [_bar_mask()]

    calls = _install_fake_sam2(monkeypatch, masks_for)
    proposal = detect_blade(str(rotating_bar_video), "ckpt", "cfg", "cpu")

    assert proposal is not None
    assert len(calls) >= 2, "gave up after the first seed instead of trying the next"
    assert calls[0] == first_point


def test_detect_blade_reports_the_frame_its_points_refer_to(
    monkeypatch, rotating_bar_video
):
    # The points are only meaningful together with the frame they were found
    # in: a swung object is somewhere else entirely a few frames later. On
    # the real 10 s clip the best frame is 135 of 300, so reading the points
    # against frame 0 puts them on whatever happens to be there -- a mistake
    # I made myself. The frame index travels with the points, and
    # track_object takes it as prompt_frame.
    _install_fake_sam2(monkeypatch, lambda point: [_bar_mask()])

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
