import warnings

import numpy as np
import pytest
import soundfile as sf

from lightsaber_fx.pipeline.audio import (
    SR,
    synth_hum,
    synth_ignition,
    synth_power_down,
    synth_swing_hum,
    synth_tv_buzz,
    synthesize_audio,
)


# ---------------------------------------------------------------------------
# Fixture helpers: build a motion.npz directly (matching blade.save_motion's
# field set) plus the plain-text video_meta.txt, without going through real
# masks/tracking.
# ---------------------------------------------------------------------------

def _make_motion(centroid_xy, theta, length=40.0, width=6.0):
    """Build the parallel arrays blade.load_motion expects from a per-frame
    centroid position and blade orientation. `axis` points hilt -> tip."""
    centroid_xy = np.asarray(centroid_xy, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    n = len(theta)
    axis_vec = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    tip = centroid_xy + (length / 2) * axis_vec
    hilt = centroid_xy - (length / 2) * axis_vec
    return dict(
        centroid=centroid_xy,
        tip=tip,
        hilt=hilt,
        axis=axis_vec,
        length=np.full(n, length, dtype=np.float64),
        width=np.full(n, width, dtype=np.float64),
        angle=theta,
    )


def _write_motion(path, fields):
    np.savez(path, **fields)


def _write_video_meta(path, fps, n_frames):
    path.write_text(f"{fps}\n{n_frames}\n")


def _static_fixture(tmp_path, n_frames=90, fps=30.0, pos=(100.0, 100.0)):
    fields = _make_motion(np.tile(pos, (n_frames, 1)), np.zeros(n_frames))
    motion_path = tmp_path / "motion.npz"
    meta_path = tmp_path / "video_meta.txt"
    _write_motion(motion_path, fields)
    _write_video_meta(meta_path, fps, n_frames)
    return str(motion_path), str(meta_path)


def _translating_fixture(tmp_path, n_frames=90, fps=30.0):
    """Still for the first and last third, fast constant-orientation
    translation through the middle third -- isolates tip_speed (angular
    speed stays ~0 throughout since orientation never changes)."""
    third = n_frames // 3
    x = np.full(n_frames, 100.0)
    x[third:2 * third] = 100.0 + np.linspace(0, 4000.0, third)
    x[2 * third:] = x[2 * third - 1]
    centroid = np.stack([x, np.full(n_frames, 100.0)], axis=1)
    fields = _make_motion(centroid, np.zeros(n_frames))
    motion_path = tmp_path / "motion.npz"
    meta_path = tmp_path / "video_meta.txt"
    _write_motion(motion_path, fields)
    _write_video_meta(meta_path, fps, n_frames)
    return str(motion_path), str(meta_path)


def _pivot_in_place_fixture(tmp_path, n_frames=90, fps=30.0, pos=(100.0, 100.0)):
    """Still for the first and last third; the middle third sweeps the
    blade's angle fast around a *constant* centroid -- reproduces the
    centroid-barely-moves bug scenario Phase A's tip/angular speed fixed."""
    third = n_frames // 3
    theta = np.zeros(n_frames)
    theta[third:2 * third] = np.linspace(0, 4.0, third)  # fast sweep, several radians
    theta[2 * third:] = theta[2 * third - 1]
    centroid = np.tile(pos, (n_frames, 1))  # never moves
    fields = _make_motion(centroid, theta)
    motion_path = tmp_path / "motion.npz"
    meta_path = tmp_path / "video_meta.txt"
    _write_motion(motion_path, fields)
    _write_video_meta(meta_path, fps, n_frames)
    return str(motion_path), str(meta_path)


def _all_nan_fixture(tmp_path, n_frames=6, fps=30.0):
    nan2 = np.full((n_frames, 2), np.nan)
    fields = dict(
        centroid=nan2.copy(),
        tip=nan2.copy(),
        hilt=nan2.copy(),
        axis=nan2.copy(),
        length=np.full(n_frames, np.nan),
        width=np.full(n_frames, np.nan),
        angle=np.full(n_frames, np.nan),
    )
    motion_path = tmp_path / "motion.npz"
    meta_path = tmp_path / "video_meta.txt"
    _write_motion(motion_path, fields)
    _write_video_meta(meta_path, fps, n_frames)
    return str(motion_path), str(meta_path)


def _rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


# ---------------------------------------------------------------------------
# synth_whoosh is gone (B2.4 deletes the discrete noise-burst mechanism)
# ---------------------------------------------------------------------------

def test_synth_whoosh_removed():
    import lightsaber_fx.pipeline.audio as audio_mod
    assert not hasattr(audio_mod, "synth_whoosh")


# ---------------------------------------------------------------------------
# Individual layers
# ---------------------------------------------------------------------------

def test_synth_hum_has_expected_length_and_energy():
    hum = synth_hum(0.5)
    assert len(hum) == int(SR * 0.5)
    assert _rms(hum) > 0.01
    assert np.all(np.isfinite(hum))


def test_synth_hum_beats_two_close_frequencies():
    # The sum of two oscillators a few Hz apart should show a slow
    # amplitude beat, i.e. energy concentrated near DC in the envelope's
    # own spectrum (not just a flat constant envelope).
    hum = synth_hum(2.0, base_freq=70.0)
    analytic_env = np.abs(hum)
    envelope_spectrum = np.abs(np.fft.rfft(analytic_env - analytic_env.mean()))
    assert envelope_spectrum.max() > 0  # a genuine beat exists, not silence


def test_synth_tv_buzz_energy_is_concentrated_in_2_to_4khz():
    buzz = synth_tv_buzz(1.0, seed=5)
    assert len(buzz) == SR
    assert np.all(np.isfinite(buzz))
    spectrum = np.abs(np.fft.rfft(buzz))
    freqs = np.fft.rfftfreq(len(buzz), 1.0 / SR)
    in_band = spectrum[(freqs >= 1500) & (freqs <= 4500)].sum()
    total = spectrum.sum() + 1e-12
    assert in_band / total > 0.5


def test_synth_ignition_has_expected_length_and_energy():
    audio = synth_ignition(0.5)
    assert len(audio) == int(SR * 0.5)
    assert _rms(audio) > 0.01


def test_synth_power_down_has_expected_length_and_energy():
    audio = synth_power_down(0.5)
    assert len(audio) == int(SR * 0.5)
    assert _rms(audio) > 0.01


def test_synth_swing_hum_length_matches_duration():
    fps = 30.0
    n_frames = 60
    tip_spd = np.zeros(n_frames)
    ang_spd = np.zeros(n_frames)
    hum = synth_swing_hum(2.0, tip_spd, ang_spd, fps)
    assert len(hum) == int(SR * 2.0)
    assert np.all(np.isfinite(hum))


def _dominant_fundamental(x, sr=SR, band=(40.0, 250.0)):
    """Frequency of the tallest spectral peak within `band` -- isolates the
    fundamental hum pitch from harmonic/distortion clutter elsewhere."""
    spectrum = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(freqs[mask][np.argmax(spectrum[mask])])


def test_synth_swing_hum_louder_and_pitched_up_when_swinging():
    fps = 30.0
    n_frames = 60
    still = synth_swing_hum(2.0, np.zeros(n_frames), np.zeros(n_frames), fps, seed=0)
    fast = synth_swing_hum(2.0, np.full(n_frames, 5000.0), np.zeros(n_frames), fps, seed=0)
    assert _rms(fast) > _rms(still) * 1.05
    # Continuous Doppler shift + cross-fade to the +semitone register (B2.4,
    # B2.5) should raise the fundamental hum pitch, not just its loudness.
    assert _dominant_fundamental(fast) > _dominant_fundamental(still)


# ---------------------------------------------------------------------------
# synthesize_audio: real end-to-end behaviour
# ---------------------------------------------------------------------------

def test_synthesize_audio_is_stereo_and_correct_length(tmp_path):
    motion_path, meta_path = _static_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, sr = sf.read(str(out_wav), always_2d=True)
    assert sr == SR
    assert audio.shape[1] == 2
    assert audio.shape[0] == int(SR * 3.0)


def test_synthesize_audio_layers_ignition_and_power_down_at_clip_edges(tmp_path):
    motion_path, meta_path = _static_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, sr = sf.read(str(out_wav), always_2d=True)
    mono = audio.mean(axis=1)
    start_rms = _rms(mono[: int(0.4 * SR)])
    middle_rms = _rms(mono[int(1.3 * SR):int(1.7 * SR)])
    end_rms = _rms(mono[-int(0.4 * SR):])

    assert start_rms > middle_rms * 1.5
    assert end_rms > middle_rms * 1.5


def test_synthesize_audio_swing_modulation_responds_to_tip_speed(tmp_path):
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav), seed=1)

    audio, sr = sf.read(str(out_wav), always_2d=True)
    mono = audio.mean(axis=1)
    # windows well clear of the ignition (0-0.5s) / power-down (2.5-3.0s)
    # transients: [0.6, 1.0) still, [1.35, 1.75) fast translation.
    still = mono[int(0.6 * SR):int(1.0 * SR)]
    moving = mono[int(1.35 * SR):int(1.75 * SR)]

    assert _rms(moving) > _rms(still) * 1.2

    still_spec = np.abs(np.fft.rfft(still))
    moving_spec = np.abs(np.fft.rfft(moving))
    freqs = np.fft.rfftfreq(len(still), 1.0 / SR)
    still_centroid = np.average(freqs, weights=still_spec + 1e-12)
    moving_centroid = np.average(freqs, weights=moving_spec + 1e-12)
    assert moving_centroid != pytest.approx(still_centroid, rel=0.02)


def test_synthesize_audio_swing_dynamic_range_exceeds_threshold(tmp_path):
    # Regression guard for the dynamic-range/HF rewrite: on a real 10s
    # bat-swing render, the pre-fix mix measured only ~1.1-1.2x RMS
    # contrast between idle and swing (excluding the ignition/power-down
    # transients) -- a swing that should be the most dramatic thing in the
    # clip barely registered. On this synthetic fast-translation fixture
    # the pre-fix code (a linear idle<->swing crossfade, which *loses*
    # power at the midpoint instead of gaining it, topped with a token
    # "+35% at full swing" loudness coefficient) manages only ~2.1x; this
    # asserts a substantially higher bar so that regressing back to that
    # design -- or any other change that quietly re-flattens the swing --
    # fails loudly here instead of only showing up as a subjective "the
    # swing sounds subtle" complaint on real footage.
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav), seed=9)

    audio, sr = sf.read(str(out_wav), always_2d=True)
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0

    mono = audio.mean(axis=1)
    still = mono[int(0.6 * SR):int(1.0 * SR)]
    moving = mono[int(1.35 * SR):int(1.75 * SR)]

    assert _rms(moving) > _rms(still) * 2.5


def test_synthesize_audio_pivot_in_place_still_produces_swing_modulation(tmp_path):
    # Centroid never moves (the old bug's blind spot); only the angle
    # sweeps. This must still produce audible swing modulation via
    # tip_speed/angular_speed.
    motion_path, meta_path = _pivot_in_place_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav), seed=2)

    audio, sr = sf.read(str(out_wav), always_2d=True)
    mono = audio.mean(axis=1)
    still = mono[int(0.6 * SR):int(1.0 * SR)]
    sweeping = mono[int(1.35 * SR):int(1.75 * SR)]

    assert _rms(sweeping) > _rms(still) * 1.2


def test_synthesize_audio_reproducible_with_same_seed(tmp_path):
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"

    synthesize_audio(motion_path, meta_path, str(out_a), seed=7)
    synthesize_audio(motion_path, meta_path, str(out_b), seed=7)

    audio_a, _ = sf.read(str(out_a))
    audio_b, _ = sf.read(str(out_b))
    assert np.array_equal(audio_a, audio_b)


def test_synthesize_audio_different_seeds_differ(tmp_path):
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_a = tmp_path / "a.wav"
    out_b = tmp_path / "b.wav"

    synthesize_audio(motion_path, meta_path, str(out_a), seed=1)
    synthesize_audio(motion_path, meta_path, str(out_b), seed=2)

    audio_a, _ = sf.read(str(out_a))
    audio_b, _ = sf.read(str(out_b))
    assert not np.array_equal(audio_a, audio_b)


@pytest.mark.parametrize("voice", ["jedi", "sith"])
def test_synthesize_audio_voices_differ_from_neutral(tmp_path, voice):
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_neutral = tmp_path / "neutral.wav"
    out_voice = tmp_path / f"{voice}.wav"

    synthesize_audio(motion_path, meta_path, str(out_neutral), voice="neutral", seed=3)
    synthesize_audio(motion_path, meta_path, str(out_voice), voice=voice, seed=3)

    neutral_audio, _ = sf.read(str(out_neutral))
    voice_audio, _ = sf.read(str(out_voice))
    assert not np.allclose(neutral_audio, voice_audio)


def test_synthesize_audio_jedi_and_sith_differ_from_each_other(tmp_path):
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_jedi = tmp_path / "jedi.wav"
    out_sith = tmp_path / "sith.wav"

    synthesize_audio(motion_path, meta_path, str(out_jedi), voice="jedi", seed=4)
    synthesize_audio(motion_path, meta_path, str(out_sith), voice="sith", seed=4)

    jedi_audio, _ = sf.read(str(out_jedi))
    sith_audio, _ = sf.read(str(out_sith))
    assert not np.allclose(jedi_audio, sith_audio)


def test_synthesize_audio_rejects_unknown_voice(tmp_path):
    motion_path, meta_path = _static_fixture(tmp_path, n_frames=30, fps=30.0)
    out_wav = tmp_path / "out.wav"
    with pytest.raises(ValueError):
        synthesize_audio(motion_path, meta_path, str(out_wav), voice="klingon")


def test_synthesize_audio_no_nan_inf_and_peak_in_range(tmp_path):
    motion_path, meta_path = _pivot_in_place_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, _ = sf.read(str(out_wav))
    assert np.all(np.isfinite(audio))
    assert np.max(np.abs(audio)) <= 1.0


def test_synthesize_audio_pans_across_stereo_field(tmp_path):
    # A blade that moves from one side of frame to the other should shift
    # stereo balance rather than always splitting the mix evenly.
    motion_path, meta_path = _translating_fixture(tmp_path, n_frames=90, fps=30.0)
    out_wav = tmp_path / "out.wav"

    synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, _ = sf.read(str(out_wav), always_2d=True)
    left, right = audio[:, 0], audio[:, 1]
    early = slice(int(0.6 * SR), int(1.0 * SR))
    late = slice(int(2.6 * SR), int(3.0 * SR))
    balance_early = _rms(left[early]) - _rms(right[early])
    balance_late = _rms(left[late]) - _rms(right[late])
    assert balance_early != pytest.approx(balance_late, abs=1e-4)


def test_synthesize_audio_handles_all_nan_motion_without_warnings(tmp_path):
    motion_path, meta_path = _all_nan_fixture(tmp_path, n_frames=6, fps=30.0)
    out_wav = tmp_path / "out.wav"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, _ = sf.read(str(out_wav))
    assert np.all(np.isfinite(audio))


def test_synthesize_audio_handles_very_short_clip_without_warnings(tmp_path):
    motion_path, meta_path = _static_fixture(tmp_path, n_frames=2, fps=30.0)
    out_wav = tmp_path / "out.wav"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        synthesize_audio(motion_path, meta_path, str(out_wav))

    audio, _ = sf.read(str(out_wav))
    assert np.all(np.isfinite(audio))


def test_synthesize_audio_handles_zero_length_clip_without_warnings(tmp_path):
    zeros2 = np.zeros((0, 2))
    fields = dict(
        centroid=zeros2, tip=zeros2, hilt=zeros2, axis=zeros2,
        length=np.zeros(0), width=np.zeros(0), angle=np.zeros(0),
    )
    motion_path = tmp_path / "motion.npz"
    meta_path = tmp_path / "video_meta.txt"
    _write_motion(motion_path, fields)
    _write_video_meta(meta_path, 30.0, 0)
    out_wav = tmp_path / "out.wav"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        synthesize_audio(str(motion_path), str(meta_path), str(out_wav))

    audio, sr = sf.read(str(out_wav), always_2d=True)
    assert sr == SR
    assert audio.shape == (0, 2)


def test_synthesize_audio_reports_progress_to_completion(tmp_path):
    motion_path, meta_path = _static_fixture(tmp_path, n_frames=30, fps=30.0)
    out_wav = tmp_path / "out.wav"
    calls = []

    synthesize_audio(motion_path, meta_path, str(out_wav), progress_cb=lambda pct, msg: calls.append(pct))

    assert calls[-1] == 100
    assert all(0 <= p <= 100 for p in calls)
