"""
Step 3: Synthesize a lightsaber hum + swing whooshes from scratch (no
sampled/copyrighted sound) and time the whooshes to the sword's actual
motion, computed in Step 2.
"""
import numpy as np
import soundfile as sf

MOTION_FILE = "motion.npy"
VIDEO_META = "video_meta.txt"
OUT_WAV = "saber_audio.wav"
SR = 44100


def synth_hum(duration, base_freq=70.0):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    # Layered detuned sines + a slow amplitude wobble -> classic idle-hum character
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
    envelope = np.sin(np.pi * t / duration)  # rise and fall
    return noise * envelope


def main():
    with open(VIDEO_META) as f:
        fps = float(f.readline())
        n_frames = int(f.readline())
    duration = n_frames / fps

    motion = np.load(MOTION_FILE)
    speed = np.zeros(len(motion))
    diffs = np.diff(motion, axis=0)
    frame_speed = np.linalg.norm(diffs, axis=1)
    speed[1:] = np.nan_to_num(frame_speed, nan=0.0)

    audio = synth_hum(duration) * 0.25

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
            i += int(fps * whoosh_dur)  # don't re-trigger mid-whoosh
        else:
            i += 1

    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    sf.write(OUT_WAV, audio, SR)
    print(f"Wrote {OUT_WAV} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
