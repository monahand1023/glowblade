import numpy as np
import soundfile as sf

from lightsaber_fx.pipeline.audio import (
    SR,
    synth_ignition,
    synth_power_down,
    synthesize_audio,
)


def test_synth_ignition_has_expected_length_and_energy():
    audio = synth_ignition(0.5)
    assert len(audio) == int(SR * 0.5)
    assert np.sqrt(np.mean(audio ** 2)) > 0.01


def test_synth_power_down_has_expected_length_and_energy():
    audio = synth_power_down(0.5)
    assert len(audio) == int(SR * 0.5)
    assert np.sqrt(np.mean(audio ** 2)) > 0.01


def test_synthesize_audio_layers_ignition_and_power_down_at_clip_edges(tmp_path):
    np.random.seed(0)
    fps = 30.0
    n_frames = 90  # 3.0s clip, long enough for a quiet middle window
    motion = np.tile(np.array([100.0, 100.0]), (n_frames, 1))  # static -> zero speed -> no whooshes
    motion_path = tmp_path / "motion.npy"
    np.save(motion_path, motion)
    meta_path = tmp_path / "video_meta.txt"
    meta_path.write_text(f"{fps}\n{n_frames}\n")
    out_wav = tmp_path / "out.wav"

    synthesize_audio(str(motion_path), str(meta_path), str(out_wav))

    audio, sr = sf.read(str(out_wav))
    assert sr == SR
    start_rms = np.sqrt(np.mean(audio[: int(0.4 * SR)] ** 2))
    middle_rms = np.sqrt(np.mean(audio[int(1.3 * SR):int(1.7 * SR)] ** 2))
    end_rms = np.sqrt(np.mean(audio[-int(0.4 * SR):] ** 2))

    assert start_rms > middle_rms * 1.5
    assert end_rms > middle_rms * 1.5
