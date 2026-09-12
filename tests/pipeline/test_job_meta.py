import os
import time

import numpy as np
import pytest

from lightsaber_fx.pipeline.blade import compute_motion, save_mask
from lightsaber_fx.pipeline.job_meta import (
    JobNotRerenderableError,
    describe_job,
    read_job_meta,
    require_rerenderable,
    write_job_meta,
)


def _build_full_job(tmp_path, source_video_path, n_frames=3, job_name="job"):
    """A job dir with masks/, motion.npz, video_meta.txt, and job_meta.json
    all present -- the fully re-renderable case."""
    job_dir = tmp_path / job_name
    job_dir.mkdir()
    masks_dir = job_dir / "masks"
    for i in range(n_frames):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        save_mask(str(masks_dir), i, mask)
    compute_motion(str(masks_dir), str(job_dir / "motion.npz"))
    (job_dir / "video_meta.txt").write_text(f"24.0\n{n_frames}\n")
    write_job_meta(str(job_dir), source_video=str(source_video_path))
    return job_dir


# ---------------------------------------------------------------------------
# write_job_meta / read_job_meta
# ---------------------------------------------------------------------------


def test_write_job_meta_then_read_job_meta_roundtrips_source_video_and_created_at(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    before = time.time()
    write_job_meta(str(job_dir), source_video=str(video))
    after = time.time()

    meta = read_job_meta(str(job_dir))
    assert meta["source_video"] == os.path.abspath(str(video))
    assert before - 1 <= meta["created_at"] <= after + 1


def test_write_job_meta_stores_absolute_path_even_if_given_relative(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    write_job_meta(str(job_dir), source_video="clip.mp4")

    meta = read_job_meta(str(job_dir))
    assert meta["source_video"] == os.path.abspath("clip.mp4")


def test_read_job_meta_returns_none_when_absent(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    assert read_job_meta(str(job_dir)) is None


# ---------------------------------------------------------------------------
# describe_job
# ---------------------------------------------------------------------------


def test_describe_job_reports_rerenderable_true_when_all_artifacts_present(tmp_path, tiny_video_path):
    job_dir = _build_full_job(tmp_path, tiny_video_path, n_frames=3)

    info = describe_job(str(job_dir))

    assert info.rerenderable is True
    assert info.reason is None
    assert info.job_id == "job"
    assert info.source_video == os.path.abspath(str(tiny_video_path))
    assert info.frame_count == 3
    assert info.created_at is not None


def test_describe_job_not_rerenderable_when_masks_missing(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "video_meta.txt").write_text("24.0\n3\n")
    write_job_meta(str(job_dir), source_video=str(tiny_video_path))
    # No masks/ dir at all, and no motion.npz.

    info = describe_job(str(job_dir))

    assert info.rerenderable is False
    assert "masks" in info.reason


def test_describe_job_not_rerenderable_when_masks_dir_present_but_empty(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "masks").mkdir()
    (job_dir / "video_meta.txt").write_text("24.0\n3\n")
    write_job_meta(str(job_dir), source_video=str(tiny_video_path))

    info = describe_job(str(job_dir))

    assert info.rerenderable is False
    assert "masks" in info.reason


def test_describe_job_not_rerenderable_when_motion_missing(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    masks_dir = job_dir / "masks"
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    save_mask(str(masks_dir), 0, mask)
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(job_dir), source_video=str(tiny_video_path))
    # No motion.npz written.

    info = describe_job(str(job_dir))

    assert info.rerenderable is False
    assert "motion.npz" in info.reason


def test_describe_job_not_rerenderable_when_source_clip_missing(tmp_path, tiny_video_path):
    job_dir = _build_full_job(tmp_path, tiny_video_path)
    tiny_video_path.unlink()

    info = describe_job(str(job_dir))

    assert info.rerenderable is False
    assert "moved or deleted" in info.reason


def test_describe_job_not_rerenderable_when_job_predates_rerender_support(tmp_path):
    # No job_meta.json at all -- simulates a job dir from before this
    # feature existed (or one W1 already touched, since W1 never wrote it).
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    masks_dir = job_dir / "masks"
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    save_mask(str(masks_dir), 0, mask)
    compute_motion(str(masks_dir), str(job_dir / "motion.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")

    info = describe_job(str(job_dir))

    assert info.rerenderable is False
    assert "source clip path" in info.reason
    assert info.source_video is None


def test_describe_job_accepts_legacy_npy_masks(tmp_path, tiny_video_path):
    # W1 kept a fallback for uncompressed .npy masks specifically so older
    # job dirs stay re-renderable -- this is the direct test of that promise.
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    masks_dir = job_dir / "masks"
    masks_dir.mkdir()
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    np.save(str(masks_dir / "00000.npy"), mask)
    compute_motion(str(masks_dir), str(job_dir / "motion.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(job_dir), source_video=str(tiny_video_path))

    info = describe_job(str(job_dir))

    assert info.rerenderable is True


def test_describe_job_frame_count_read_from_video_meta_txt(tmp_path, tiny_video_path):
    job_dir = _build_full_job(tmp_path, tiny_video_path, n_frames=7)
    info = describe_job(str(job_dir))
    assert info.frame_count == 7


def test_describe_job_reports_multiple_reasons_when_multiple_things_missing(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    # Nothing at all present.
    info = describe_job(str(job_dir))
    assert info.rerenderable is False
    assert "masks" in info.reason
    assert "motion.npz" in info.reason
    assert "video_meta.txt" in info.reason
    assert "source clip path" in info.reason


# ---------------------------------------------------------------------------
# require_rerenderable
# ---------------------------------------------------------------------------


def test_require_rerenderable_raises_with_clear_reason(tmp_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(JobNotRerenderableError) as exc_info:
        require_rerenderable(str(job_dir))

    assert "masks" in str(exc_info.value)


def test_require_rerenderable_returns_info_when_ok(tmp_path, tiny_video_path):
    job_dir = _build_full_job(tmp_path, tiny_video_path)

    info = require_rerenderable(str(job_dir))

    assert info.rerenderable is True
    assert info.source_video == os.path.abspath(str(tiny_video_path))
