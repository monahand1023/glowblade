import numpy as np
import soundfile as sf

SR = 44100


def synth_hum(duration, base_freq=70.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    tone = (
        0.6 * np.sin(2 * np.pi * base_freq * t)
        + 0.3 * np.sin(2 * np.pi * base_freq * 2.01 * t)
        + 0.15 * np.sin(2 * np.pi * base_freq * 3.98 * t)
    )
    wobble = 1.0 + 0.05 * np.sin(2 * np.pi * 5.5 * t)
    return tone * wobble


def synth_whoosh(duration, peak_freq=900.0):
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    noise = np.random.uniform(-1, 1, n)
    envelope = np.sin(np.pi * t / duration)
    return noise * envelope


def synth_ignition(duration=0.5):
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 40 + (90 - 40) * (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sweep = np.sin(phase)
    noise = np.random.uniform(-1, 1, n)
    thump_env = np.exp(-3 * t / duration)
    thump = noise * thump_env
    rise_env = t / duration
    return sweep * rise_env * 0.8 + thump * 0.6


def synth_power_down(duration=0.5):
    n = int(SR * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    freq = 90 + (30 - 90) * (t / duration)
    phase = 2 * np.pi * np.cumsum(freq) / SR
    sweep = np.sin(phase)
    noise = np.random.uniform(-1, 1, n)
    thump_env = np.exp(-3 * (duration - t) / duration)
    thump = noise * thump_env
    fall_env = 1 - (t / duration)
    return sweep * fall_env * 0.8 + thump * 0.6


def synthesize_audio(motion_path, video_meta_path, out_wav_path, progress_cb=None):
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    with open(video_meta_path) as f:
        fps = float(f.readline())
        n_frames = int(f.readline())
    duration = n_frames / fps

    motion = np.load(motion_path)
    speed = np.zeros(len(motion))
    diffs = np.diff(motion, axis=0)
    frame_speed = np.linalg.norm(diffs, axis=1)
    speed[1:] = np.nan_to_num(frame_speed, nan=0.0)

    audio = synth_hum(duration) * 0.25
    report(20, "synthesized hum")

    positive = speed[speed > 0]
    threshold = np.percentile(positive, 80) if len(positive) else 1e9

    i = 0
    while i < len(speed):
        if speed[i] > threshold:
            t_start = i / fps
            whoosh_dur = 0.35
            gain = min(1.0, speed[i] / (threshold * 2))
            whoosh = synth_whoosh(whoosh_dur) * gain
            s = int(t_start * SR)
            e = min(len(audio), s + len(whoosh))
            audio[s:e] += whoosh[: e - s] * 0.7
            i += int(fps * whoosh_dur)
        else:
            i += 1
    report(60, "added swing whooshes")

    ignition_dur = min(0.5, duration / 2)
    ignition = synth_ignition(ignition_dur)
    audio[: len(ignition)] += ignition * 0.8

    power_down_dur = min(0.5, duration / 2)
    power_down = synth_power_down(power_down_dur)
    audio[len(audio) - len(power_down):] += power_down * 0.8
    report(90, "added ignition and power-down")

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    sf.write(out_wav_path, audio, SR)
    report(100, "wrote audio")
