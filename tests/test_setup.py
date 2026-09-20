import glowblade.setup as lfx_setup


def test_bootstrap_clones_installs_and_downloads_when_nothing_present(tmp_path, monkeypatch):
    sam2_src = tmp_path / "sam2-src"
    checkpoint = tmp_path / "checkpoints" / "sam2.1_hiera_small.pt"
    monkeypatch.setattr(lfx_setup, "get_sam2_src_dir", lambda: sam2_src)
    monkeypatch.setattr(lfx_setup, "get_checkpoint_path", lambda: checkpoint)

    calls = []
    monkeypatch.setattr(
        lfx_setup.subprocess, "run",
        lambda cmd, **kwargs: calls.append(cmd) or None,
    )

    lfx_setup.bootstrap()

    assert any(cmd[:2] == ["git", "clone"] for cmd in calls)
    assert any("pip" in cmd for cmd in calls if "-m" in cmd)
    assert any(cmd[0] == "curl" for cmd in calls)


def test_bootstrap_skips_clone_and_download_when_already_present(tmp_path, monkeypatch):
    sam2_src = tmp_path / "sam2-src"
    sam2_src.mkdir()
    checkpoint = tmp_path / "checkpoints" / "sam2.1_hiera_small.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"already here")
    monkeypatch.setattr(lfx_setup, "get_sam2_src_dir", lambda: sam2_src)
    monkeypatch.setattr(lfx_setup, "get_checkpoint_path", lambda: checkpoint)

    calls = []
    monkeypatch.setattr(
        lfx_setup.subprocess, "run",
        lambda cmd, **kwargs: calls.append(cmd) or None,
    )

    lfx_setup.bootstrap()

    assert not any(cmd[:2] == ["git", "clone"] for cmd in calls)
    assert not any(cmd[0] == "curl" for cmd in calls)


def test_bootstrap_force_reclones_and_redownloads(tmp_path, monkeypatch):
    sam2_src = tmp_path / "sam2-src"
    sam2_src.mkdir()
    (sam2_src / "marker.txt").write_text("old clone")
    checkpoint = tmp_path / "checkpoints" / "sam2.1_hiera_small.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"old checkpoint")
    monkeypatch.setattr(lfx_setup, "get_sam2_src_dir", lambda: sam2_src)
    monkeypatch.setattr(lfx_setup, "get_checkpoint_path", lambda: checkpoint)

    calls = []
    monkeypatch.setattr(
        lfx_setup.subprocess, "run",
        lambda cmd, **kwargs: calls.append(cmd) or None,
    )

    lfx_setup.bootstrap(force=True)

    assert not sam2_src.exists() or not (sam2_src / "marker.txt").exists()
    assert any(cmd[:2] == ["git", "clone"] for cmd in calls)
    assert any(cmd[0] == "curl" for cmd in calls)
