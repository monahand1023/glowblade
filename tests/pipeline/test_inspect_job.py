import numpy as np
import pytest

from lightsaber_fx.pipeline.blade import BladeGeometry, save_motion
from lightsaber_fx.pipeline.inspect_job import _analyze_object

FRAME_WIDTH = 1000


def _geometry(length, width, x=100.0, y=100.0):
    """A BladeGeometry with just the fields `_analyze_object` reads set to
    specific values -- the rest are geometrically irrelevant placeholders."""
    return BladeGeometry(
        centroid=(x, y), axis=(1.0, 0.0),
        tip=(x + length / 2, y), hilt=(x - length / 2, y),
        length=length, width=width, angle=0.0,
    )


def _write_motion(tmp_path, geometries, name="motion.npz"):
    path = tmp_path / name
    save_motion(str(path), geometries)
    return str(path)


def test_analyze_object_no_anomaly_for_a_consistently_elongated_blade(tmp_path):
    # length=200, width=20 -> elongation 10, comfortably above MIN_ELONGATION (6).
    geometries = [_geometry(length=200.0, width=20.0) for _ in range(10)]
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    assert report["anomalies"] == []


def test_analyze_object_flags_a_mask_that_is_consistently_blob_shaped(tmp_path):
    # length=40, width=35 -> elongation ~1.1, well under MIN_ELONGATION: a
    # compact blob (e.g. a patch of tunic), not a blade -- job 01's actual
    # failure mode, which the width/span checks alone did not catch because
    # a torso-sized blob is neither too wide nor too spanning for the frame.
    geometries = [_geometry(length=40.0, width=35.0) for _ in range(10)]
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    assert any("elongation" in a for a in report["anomalies"])


def test_analyze_object_ignores_a_minority_of_low_elongation_frames(tmp_path):
    # 8 healthy frames (elongation 10) + 2 collapsed (elongation ~1.1): a
    # blade can legitimately foreshorten toward the camera for a frame or
    # two mid-swing, and that alone shouldn't read as "tracking the wrong
    # thing" -- only a sustained collapse should.
    geometries = [_geometry(length=200.0, width=20.0) for _ in range(8)]
    geometries += [_geometry(length=40.0, width=35.0) for _ in range(2)]
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    assert not any("elongation" in a for a in report["anomalies"])


def test_analyze_object_reports_elongation_stats(tmp_path):
    geometries = [_geometry(length=200.0, width=20.0) for _ in range(5)]
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    assert report["mean_elongation"] == pytest.approx(10.0)
    assert report["low_elongation_frac"] == pytest.approx(0.0)


def test_analyze_object_elongation_check_ignores_zero_width_frames_without_crashing(tmp_path):
    # width=0.0 is a real degenerate case _median_perpendicular_extent can
    # return (e.g. a single-row mask) -- length/width must not raise, and a
    # single such frame shouldn't tip the object into a false "collapsed"
    # verdict on its own (it carries no elongation evidence either way).
    geometries = [_geometry(length=200.0, width=20.0) for _ in range(9)]
    geometries.append(_geometry(length=100.0, width=0.0))
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    assert report["anomalies"] == []
    assert np.isfinite(report["mean_elongation"])


def test_analyze_object_elongation_message_names_the_fraction(tmp_path):
    geometries = [_geometry(length=40.0, width=35.0) for _ in range(10)]
    path = _write_motion(tmp_path, geometries)

    report = _analyze_object(path, FRAME_WIDTH)

    elongation_msgs = [a for a in report["anomalies"] if "elongation" in a]
    assert len(elongation_msgs) == 1
    assert "100%" in elongation_msgs[0] or "1.1" in elongation_msgs[0]
