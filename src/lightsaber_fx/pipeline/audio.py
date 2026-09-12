"""Fully-synthesized lightsaber audio (numpy/scipy only -- no sampled or
external audio ever enters this module, which is a deliberate property of
the project).

Design notes (Phase B2 rewrite)
--------------------------------
Burtt's canonical sound is two layers: an idling film projector's interlock
motor (a low tonal hum beating against a second motor) *plus* electromagnetic
buzz picked up by walking a mic behind a TV picture tube. This module
synthesizes both:

- ``synth_hum`` -- the projector half: two detuned oscillators that beat
  against each other, an octave-up oscillator for definition, and a
  resonant-filtered "dark tone" layer, all with slow random pitch drift
  ("steady yet unstable, fluctuating with mechanical imperfections").
- ``synth_tv_buzz`` -- the TV half, previously entirely absent: bandpass
  noise around 2-4 kHz, amplitude-modulated by an irregular LFO, plus sparse
  crackle transients.

Burtt made swings by playing the hum through a speaker and physically
swinging a mic around it -- Doppler pitch/amplitude modulation *of the hum*.
ProffieOS "SmoothSwing V2" formalizes this as cross-fading a low/high pitched
hum pair by swing strength. ``synth_swing_hum`` implements that: a continuous
Doppler-style frequency multiplier driven by ``tip_speed``/``angular_speed``
(never the mask centroid, which barely moves when a blade pivots about its
middle), cross-faded between a base and a +5..+7-semitone register, itself
cross-faded against the plain idle hum by an overall (more smoothed) swing
strength. This *replaces* the old discretely-triggered white-noise burst
mechanism entirely -- ``synth_whoosh`` is gone; a discrete trigger can't
represent a continuous physical motion, which is exactly why the old
whooshes read as thin and mechanical.

Every oscillator here uses cumulative-phase accumulation
(``phase = 2*pi*cumsum(f_t)/SR``) rather than a fixed ``sin(2*pi*f*t)``, so
frequency can vary sample-to-sample without phase discontinuities -- this is
what makes continuous Doppler modulation and drifting pitch possible at all.
"""

import numpy as np
import soundfile as sf
from scipy import signal

from .blade import angular_speed, load_motion, tip_speed

SR = 44100


# ---------------------------------------------------------------------------
# Voicing (B2.9): `voice` is a parameter to the synthesis functions below,
# not a forked code path. `neutral` must sound like the current-but-improved
# saber; `jedi` is higher/smoother/less distorted; `sith` is deeper/heavier/
# more distorted.
# ---------------------------------------------------------------------------

_VOICE_PARAMS = {
    "neutral": {"pitch_mult": 1.00, "drive": 1.4, "buzz_gain": 1.00, "brightness": 1.00},
    "jedi":    {"pitch_mult": 1.09, "drive": 0.7, "buzz_gain": 0.65, "brightness": 1.15},
    "sith":    {"pitch_mult": 0.82, "drive": 2.6, "buzz_gain": 1.45, "brightness": 0.80},
}


def _voice_params(voice):
    try:
        return _VOICE_PARAMS[voice]
    except KeyError:
        raise ValueError(f"Unrecognized voice: {voice!r}. Use neutral, jedi, or sith.")


# ---------------------------------------------------------------------------
# Low-level DSP helpers
# ---------------------------------------------------------------------------

def _n_samples(duration, sr=SR):
    return max(0, int(round(sr * duration)))


def _osc(freq_t, sr=SR):
    """Sine oscillator driven by a per-sample instantaneous-frequency array
    (B2.1). Cumulative phase accumulation lets frequency vary continuously
    over time -- fixed `sin(2*pi*f*t)` cannot do this, which is why every
    time-varying effect below (drift, Doppler, sweeps) depends on this."""
    freq_t = np.asarray(freq_t, dtype=np.float64)
    if len(freq_t) == 0:
        return np.zeros(0, dtype=np.float64)
    phase = 2.0 * np.pi * np.cumsum(freq_t) / sr
    return np.sin(phase)


def _slow_drift(n, sr, rng, depth=0.5, cutoff=0.5):
    """Slow, irregular wander (Hz) from low-passed noise -- "steady yet
    unstable, fluctuating with mechanical imperfections" (B2.2)."""
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    noise = rng.standard_normal(n)
    sos = signal.butter(2, cutoff, btype="low", fs=sr, output="sos")
    drift = signal.sosfilt(sos, noise)
    peak = np.max(np.abs(drift))
    if peak > 0:
        drift = drift / peak
    return drift * depth


def _bandpass(x, sr, low, high, order=2):
    if len(x) == 0:
        return x
    low = max(1.0, low)
    high = min(sr / 2.0 - 1.0, high)
    if low >= high:
        return x
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def _highpass(x, sr, cutoff, order=2):
    if len(x) == 0:
        return x
    cutoff = min(max(cutoff, 1.0), sr / 2.0 - 1.0)
    sos = signal.butter(order, cutoff, btype="high", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def _time_varying_lowpass(x, fc_t, sr=SR):
    """One-pole lowpass whose cutoff (Hz, per-sample array) varies over
    time -- a static `scipy.signal` filter can't express this, so it's a
    direct one-pole recursion. Used for the ignition/retraction
    filter-cutoff sweep (B2.6)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    fc_t = np.clip(np.asarray(fc_t, dtype=np.float64), 1.0, sr / 2.0 - 1.0)
    alpha = 1.0 - np.exp(-2.0 * np.pi * fc_t / sr)
    y = np.empty(n, dtype=np.float64)
    prev = 0.0
    for i in range(n):
        prev += alpha[i] * (x[i] - prev)
        y[i] = prev
    return y


def _waveshape(x, drive):
    """Soft-clip waveshaper distortion, unity-gain-normalized at |x|=1 so
    `drive` purely controls harmonic content, not level. Used for the `sith`
    voice's heavier distortion (B2.9)."""
    if len(x) == 0 or drive <= 0:
        return x
    norm = np.tanh(drive)
    if norm <= 1e-9:
        return x
    return np.tanh(drive * x) / norm


def _soft_limit(x, ceiling=0.95):
    """Soft (tanh) limiter (B2.7): unlike a hard peak-normalize divide, this
    only compresses the loudest peaks, so quieter passages keep their
    dynamics instead of being dragged down by one transient."""
    return ceiling * np.tanh(x / ceiling)


def _normalize_speed(speed, ref_percentile=90.0):
    """Map a non-negative speed array to a roughly-[0, 1.5] "how hard is
    this motion" scale, robust to outliers, without a hard discrete
    threshold."""
    speed = np.asarray(speed, dtype=np.float64)
    positive = speed[speed > 0]
    if len(positive) == 0:
        return np.zeros_like(speed)
    ref = np.percentile(positive, ref_percentile)
    if ref <= 0:
        return np.zeros_like(speed)
    return np.clip(speed / ref, 0.0, 1.5)


def _interp_and_smooth(frame_values, fps, n_samples, sr=SR, smooth_ms=60.0):
    """Interpolate a per-frame signal up to sample rate (`np.interp`) and
    smooth it with a moving average, so modulation driven by ~30fps motion
    data isn't audibly steppy (B2.4)."""
    frame_values = np.asarray(frame_values, dtype=np.float64)
    n_frames = len(frame_values)
    if n_frames == 0 or n_samples <= 0:
        return np.zeros(max(0, n_samples), dtype=np.float64)
    fps = max(float(fps), 1e-6)
    values = np.nan_to_num(frame_values, nan=0.0)
    frame_t = np.arange(n_frames) / fps
    sample_t = np.arange(n_samples) / sr
    interped = np.interp(sample_t, frame_t, values, left=values[0], right=values[-1])
    # Clamp the kernel to the signal length: `np.convolve(..., mode="same")`
    # returns a length-max(M, N) array when the kernel is longer than the
    # signal, which would silently break every caller's length assumption
    # on short clips.
    win = min(max(1, int(round(smooth_ms / 1000.0 * sr))), n_samples)
    if win > 1 and n_samples > 1:
        kernel = np.ones(win, dtype=np.float64) / win
        interped = np.convolve(interped, kernel, mode="same")
    return interped


def _pan_positions(x_frames):
    """Normalize a per-frame x-position array to a [-1, 1] pan track using
    the clip's own observed range (no frame width is available to this
    module -- see B2.8)."""
    x = np.asarray(x_frames, dtype=np.float64)
    if len(x) == 0:
        return x
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.zeros_like(x)
    lo, hi = float(valid.min()), float(valid.max())
    span = hi - lo
    center = (lo + hi) / 2.0
    x_filled = np.where(np.isnan(x), center, x)
    if span <= 1e-6:
        return np.zeros_like(x_filled)
    pan = (x_filled - center) / (span / 2.0)
    return np.clip(pan, -1.0, 1.0)


def _apply_stereo_pan(mono, pan):
    """Equal-power pan law: `pan` in [-1 (left), 1 (right)]."""
    pan = np.clip(np.asarray(pan, dtype=np.float64), -1.0, 1.0)
    theta = (pan + 1.0) * (np.pi / 4.0)
    left = mono * np.cos(theta)
    right = mono * np.sin(theta)
    return np.stack([left, right], axis=-1)


# ---------------------------------------------------------------------------
# Synthesis layers
# ---------------------------------------------------------------------------

def _hum_core(f0_t, params, rng, sr=SR):
    """Shared hum timbre (B2.2): two oscillators a few Hz apart (`f0_t` and
    `f0_t * 71.5/70`) summed so a beat emerges naturally -- Burtt's hum came
    from one projector motor beating against another -- plus an octave-up
    oscillator for definition and a resonant-filtered dark tone (~776 Hz,
    high resonance, slow ~9.5 Hz LFO for movement). Frequencies carry slow
    random drift rather than an explicit tremolo LFO.

    `f0_t` is a per-sample base-frequency array, not a scalar, so this same
    core can be reused unmodified for `synth_hum`'s idle tone *and* for
    `synth_swing_hum`'s Doppler-shifted low/high registers (B2.4, B2.5) --
    rendering "the hum itself" pitched differently, rather than a separate,
    thinner-sounding proxy tone for swings.
    """
    n = len(f0_t)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    t = np.arange(n) / sr

    f_high = f0_t * (71.5 / 70.0)

    drift_a = _slow_drift(n, sr, rng, depth=0.6)
    drift_b = _slow_drift(n, sr, rng, depth=0.6)

    osc_a = _osc(f0_t + drift_a, sr)
    osc_b = _osc(f_high + drift_b, sr)
    beat = 0.55 * osc_a + 0.45 * osc_b

    octave = _osc(f0_t * 2.0 + drift_a, sr)
    tone = beat + 0.25 * params["brightness"] * octave

    lfo = 1.0 + 0.18 * np.sin(2 * np.pi * 9.5 * t)
    resonant_center = 776.0 * params["pitch_mult"]
    bandwidth = resonant_center / 7.0  # high resonance -> narrow band
    dark = _bandpass(
        tone, sr, resonant_center - bandwidth / 2, resonant_center + bandwidth / 2, order=2
    ) * lfo

    return tone * 0.7 + dark * 0.5 * params["brightness"]


def synth_hum(duration, base_freq=70.0, voice="neutral", seed=0, sr=SR):
    """The idle hum: `_hum_core` at a constant base frequency (B2.1, B2.2)."""
    params = _voice_params(voice)
    n = _n_samples(duration, sr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    f0 = np.full(n, base_freq * params["pitch_mult"], dtype=np.float64)
    return _waveshape(_hum_core(f0, params, rng, sr), params["drive"] * 0.35)


def synth_tv_buzz(duration, voice="neutral", seed=1, sr=SR):
    """The missing TV-interference layer (B2.3): bandpass noise centered
    2-4 kHz, amplitude-modulated by an irregular (two incommensurate rates)
    LFO, plus sparse short crackle transients. High-passed as a whole to
    keep the low end clear for the hum (B2.7)."""
    params = _voice_params(voice)
    n = _n_samples(duration, sr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr

    noise = rng.uniform(-1.0, 1.0, n)
    buzz = _bandpass(noise, sr, 2000.0, 4000.0, order=4)

    lfo = 0.5 + 0.3 * np.sin(2 * np.pi * 6.3 * t + 0.7) + 0.2 * np.sin(2 * np.pi * 2.1 * t + 2.4)
    lfo = np.clip(lfo, 0.0, None)
    buzz = buzz * lfo
    peak = np.max(np.abs(buzz))
    if peak > 0:
        buzz = buzz / peak

    n_crackles = max(1, int(round(duration * 3)))
    crackle_len = max(1, int(0.003 * sr))
    crackle_track = np.zeros(n, dtype=np.float64)
    for pos in rng.integers(0, n, size=n_crackles):
        end = min(n, pos + crackle_len)
        length = end - pos
        if length <= 0:
            continue
        decay = np.exp(-np.arange(length) / max(1.0, crackle_len / 4.0))
        crackle_track[pos:end] += rng.uniform(-1.0, 1.0, length) * decay

    out = (buzz * 0.7 + crackle_track * 0.3) * params["buzz_gain"]
    return _highpass(out, sr, cutoff=800.0)


def synth_swing_hum(
    duration,
    tip_speed_frames,
    angular_speed_frames,
    fps,
    base_freq=70.0,
    voice="neutral",
    seed=0,
    sr=SR,
    doppler_k=0.22,
    high_semitones=6.0,
):
    """Swings as pitch-shifted hum, continuously modulated (B2.4, B2.5) --
    the single biggest fix, replacing the deleted discrete noise-burst
    whoosh entirely.

    Drives everything from `tip_speed`/`angular_speed` (never centroid
    speed, which barely moves when a blade pivots about its middle):

    - the per-frame speed arrays are combined into a single normalized
      "swing strength" in [0, 1] (whichever of tip/angular speed shows more
      motion wins -- this is what catches a pivot-in-place swing that tip
      speed alone might miss, or vice versa);
    - a continuous Doppler-style frequency multiplier (B2.5) is applied to
      two *full hum-timbre* registers via `_hum_core` (base, and base
      pitched up by `high_semitones`, i.e. Burtt/ProffieOS's ~1.3-1.5x) --
      reusing the real hum texture (beat, octave, resonant dark tone) for
      both, so a swing keeps the saber's voice rather than thinning out to
      a bare oscillator;
    - those two registers are cross-faded per-sample by a fast-smoothed
      swing-strength track (the ProffieOS "SmoothSwing" low/high crossfade);
    - the resulting swing tone is cross-faded against the plain idle hum by
      a more heavily-smoothed "overall" swing strength, so a still saber
      reads as idle hum and a fast swing reads as the pitched-up register,
      continuously rather than via a discrete trigger.
    """
    params = _voice_params(voice)
    n = _n_samples(duration, sr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    rng_idle = np.random.default_rng(seed)
    f0_idle = np.full(n, base_freq * params["pitch_mult"], dtype=np.float64)
    idle = _waveshape(_hum_core(f0_idle, params, rng_idle, sr), params["drive"] * 0.35)

    tip_norm = _normalize_speed(tip_speed_frames)
    ang_norm = _normalize_speed(angular_speed_frames)
    swing_frames = np.clip(np.maximum(tip_norm, ang_norm), 0.0, 1.0)

    swing_inner = _interp_and_smooth(swing_frames, fps, n, sr=sr, smooth_ms=50.0)
    swing_outer = _interp_and_smooth(swing_frames, fps, n, sr=sr, smooth_ms=250.0)

    speed_factor = 1.0 + doppler_k * swing_inner  # continuous Doppler (B2.5)
    f0 = base_freq * params["pitch_mult"] * speed_factor
    high_mult = 2.0 ** (high_semitones / 12.0)

    rng_swing = np.random.default_rng(seed + 97)
    low_register = _hum_core(f0, params, rng_swing, sr)
    high_register = _hum_core(f0 * high_mult, params, rng_swing, sr)
    registers = low_register * (1.0 - swing_inner) + high_register * swing_inner
    swing_tone = _waveshape(registers, params["drive"] * 0.35)

    hum = idle * (1.0 - swing_outer) + swing_tone * swing_outer
    # Swings are louder as well as brighter -- amplitude follows the same
    # "overall" strength track used for the timbral crossfade.
    hum = hum * (1.0 + 0.35 * swing_outer)
    return hum


def synth_ignition(duration=0.5, voice="neutral", seed=2, sr=SR):
    """Ignition sweep (B2.6): the pitch sweep from the original
    implementation, plus a short broadband click transient at t=0 and a
    filter-cutoff sweep (the blade "opening up") alongside the pitch sweep.
    """
    params = _voice_params(voice)
    n = _n_samples(duration, sr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr

    f0, f1 = 40.0 * params["pitch_mult"], 90.0 * params["pitch_mult"]
    freq = f0 + (f1 - f0) * (t / duration)
    sweep = _osc(freq, sr)

    noise = _highpass(rng.uniform(-1.0, 1.0, n), sr, cutoff=40.0)
    thump_env = np.exp(-3 * t / duration)
    thump = noise * thump_env

    fc_sweep = 300.0 + (5000.0 - 300.0) * (t / duration) ** 1.2
    body = _time_varying_lowpass(sweep * 0.8 + thump * 0.6, fc_sweep, sr)

    click_len = min(n, max(1, int(0.003 * sr)))
    click = np.zeros(n, dtype=np.float64)
    click[:click_len] = rng.uniform(-1.0, 1.0, click_len) * np.exp(
        -np.arange(click_len) / max(1.0, click_len / 3.0)
    )

    rise_env = t / duration
    sig = body * rise_env + click * 0.6
    return _waveshape(sig, params["drive"] * 0.3)


def synth_power_down(duration=0.5, voice="neutral", seed=3, sr=SR):
    """Retraction sweep (B2.6): the pitch sweep from the original
    implementation, plus a filter-cutoff sweep (falling, mirroring
    ignition's rise) alongside the pitch sweep."""
    params = _voice_params(voice)
    n = _n_samples(duration, sr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr

    f0, f1 = 90.0 * params["pitch_mult"], 30.0 * params["pitch_mult"]
    freq = f0 + (f1 - f0) * (t / duration)
    sweep = _osc(freq, sr)

    noise = _highpass(rng.uniform(-1.0, 1.0, n), sr, cutoff=40.0)
    thump_env = np.exp(-3 * (duration - t) / duration)
    thump = noise * thump_env

    fc_sweep = 5000.0 + (300.0 - 5000.0) * (t / duration) ** 0.8
    body = _time_varying_lowpass(sweep * 0.8 + thump * 0.6, fc_sweep, sr)

    fall_env = 1 - (t / duration)
    sig = body * fall_env
    return _waveshape(sig, params["drive"] * 0.3)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def synthesize_audio(
    motion_path,
    video_meta_path,
    out_wav_path,
    voice="neutral",
    seed=0,
    progress_cb=None,
):
    """Render the full lightsaber soundtrack for one job.

    `motion_path` is a `motion.npz` written by `blade.compute_motion`/
    `blade.save_motion`, read via `blade.load_motion`; swings are driven by
    `blade.tip_speed`/`blade.angular_speed`, not the mask centroid.
    `progress_cb(pct, message)` reports step-level progress. `voice`
    ("neutral" | "jedi" | "sith") and `seed` (for reproducible renders) are
    forwarded to every synthesis layer.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    _voice_params(voice)  # fail fast on an invalid voice

    with open(video_meta_path) as f:
        fps = float(f.readline())
        n_frames = int(f.readline())
    duration = n_frames / fps if fps > 0 else 0.0
    n = _n_samples(duration, SR)

    motion = load_motion(motion_path)
    t_spd = tip_speed(motion, fps)
    a_spd = angular_speed(motion, fps)
    report(10, "computed swing dynamics")

    hum = synth_swing_hum(duration, t_spd, a_spd, fps, voice=voice, seed=seed, sr=SR)
    report(35, "synthesized swing-modulated hum")

    buzz = synth_tv_buzz(duration, voice=voice, seed=seed + 11, sr=SR)
    report(50, "synthesized TV-interference buzz")

    ignition_dur = min(0.5, duration / 2) if duration > 0 else 0.0
    power_down_dur = min(0.5, duration / 2) if duration > 0 else 0.0
    ignition = synth_ignition(ignition_dur, voice=voice, seed=seed + 2, sr=SR)
    power_down = synth_power_down(power_down_dur, voice=voice, seed=seed + 3, sr=SR)
    report(70, "synthesized ignition and power-down")

    # --- B2.7 mix discipline: gain-stage deliberately rather than summing
    # raw layers, cross-fade the hum around the ignition/power-down
    # transients instead of just adding them on top, then soft-limit.
    audio = np.zeros(n, dtype=np.float64)

    hum_gain = np.full(n, 0.30, dtype=np.float64)
    ign_n = len(ignition)
    if ign_n and n:
        ramp_len = min(ign_n, n)
        hum_gain[:ramp_len] *= (np.arange(ramp_len) / ign_n) ** 2
    pd_n = len(power_down)
    if pd_n and n:
        ramp_len = min(pd_n, n)
        hum_gain[n - ramp_len:] *= ((np.arange(ramp_len) / pd_n) ** 2)[::-1]

    audio += hum * hum_gain
    audio += buzz * 0.10  # low-level, per Burtt's "buzzy, sparkling" half

    if ign_n:
        audio[:ign_n] += ignition * 0.85
    if pd_n and n:
        audio[n - pd_n:] += power_down * 0.85
    report(85, "mixed layers")

    centroid_x = motion["centroid"][:, 0] if len(motion.get("centroid", [])) else np.zeros(0)
    pan_track = _interp_and_smooth(_pan_positions(centroid_x), fps, n, sr=SR, smooth_ms=80.0)
    stereo = _apply_stereo_pan(audio, pan_track)  # B2.8

    stereo = _soft_limit(stereo, ceiling=0.95)
    stereo = np.nan_to_num(stereo, nan=0.0, posinf=0.95, neginf=-0.95)

    sf.write(out_wav_path, stereo.astype(np.float32), SR)
    report(100, "wrote stereo audio")
