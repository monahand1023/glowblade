import glowblade.paths as paths_module


def test_get_data_dir_creates_directory(tmp_path, monkeypatch):
    target = tmp_path / "app-data"
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(target))

    result = paths_module.get_data_dir()

    assert result == target
    assert target.is_dir()


def test_new_job_dir_creates_unique_subdirectory(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path))

    job_dir = paths_module.new_job_dir("abc123")

    assert job_dir == tmp_path / "jobs" / "abc123"
    assert job_dir.is_dir()


def test_clean_jobs_removes_all_job_directories_and_returns_count(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path))
    paths_module.new_job_dir("job-a")
    paths_module.new_job_dir("job-b")

    removed = paths_module.clean_jobs()

    assert removed == 2
    assert list(paths_module.get_jobs_dir().iterdir()) == []


def test_checkpoint_and_sam2_src_paths_live_under_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path))

    assert paths_module.get_sam2_src_dir() == tmp_path / "sam2-src"
    assert paths_module.get_checkpoint_path() == tmp_path / "checkpoints" / "sam2.1_hiera_small.pt"
