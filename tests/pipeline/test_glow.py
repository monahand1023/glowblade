import numpy as np
import pytest

from lightsaber_fx.pipeline.glow import parse_color, render_glow


def test_parse_color_named_colors():
    assert parse_color("red") == (40, 40, 255)
    assert parse_color("Blue") == (255, 90, 60)
    assert parse_color("GREEN") == (70, 220, 80)


def test_parse_color_hex():
    assert parse_color("#0000FF") == (255, 0, 0)  # pure blue, BGR order


def test_parse_color_rejects_unknown():
    with pytest.raises(ValueError):
        parse_color("not-a-color")


def test_render_glow_writes_video_and_motion(tmp_path, synthetic_track_fixture):
    # NOTE (Phase A): render_glow itself is untouched and still writes the
    # legacy (N, 2) centroid-only contract to motion_out_path -- glow.py is
    # Phase B1's file, out of scope here. The new enriched contract
    # (BladeGeometry per frame: centroid/tip/hilt/axis/length/width/angle,
    # written by lightsaber_fx.pipeline.blade.save_motion as motion.npz) is
    # covered by tests/pipeline/test_blade.py and by the runner-level
    # end-to-end test in tests/pipeline/test_runner.py, and is produced
    # independently in runner.py from the tracked masks. This assertion is
    # still true and is intentionally left as-is; see fx-a-report.md.
    output_video = tmp_path / "glow_video.mp4"
    motion_out = tmp_path / "motion.npy"
    progress_calls = []

    render_glow(
        synthetic_track_fixture["frames_dir"],
        synthetic_track_fixture["masks_dir"],
        synthetic_track_fixture["video_meta_path"],
        str(output_video),
        str(motion_out),
        progress_cb=lambda pct, msg: progress_calls.append(pct),
    )

    assert output_video.exists() and output_video.stat().st_size > 0
    motion = np.load(motion_out)
    assert motion.shape == (synthetic_track_fixture["n_frames"], 2)
    assert progress_calls[-1] == 100
