import lightsaber_fx.device as device_module


def test_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)

    assert device_module.select_device() == "mps"


def test_falls_back_to_cuda_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: True)

    assert device_module.select_device() == "cuda"


def test_falls_back_to_cpu_when_nothing_available(monkeypatch):
    monkeypatch.setattr(device_module.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(device_module.torch.cuda, "is_available", lambda: False)

    assert device_module.select_device() == "cpu"
