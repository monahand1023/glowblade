import os
import shutil

from .audio import synthesize_audio
from .blade import compute_motion
from .frames import extract_frames
from .glow import parse_color, render_glow
from .mux import encode
from .track import track_object


def run_pipeline(
    input_video,
    points,
    labels,
    output_path,
    job_dir,
    checkpoint_path,
    device,
    config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
    color="red",
    intensity=0.35,
    blade_extend=True,
    voice="neutral",
    progress_cb=None,
):
    def stage_cb(stage):
        def cb(pct, message):
            if progress_cb:
                progress_cb(stage, pct, message)
        return cb

    color_bgr = parse_color(color)

    frames_dir = os.path.join(job_dir, "frames")
    masks_dir = os.path.join(job_dir, "masks")
    video_meta_path = os.path.join(job_dir, "video_meta.txt")
    motion_path = os.path.join(job_dir, "motion.npz")
    # Lossless PNG sequence written by the glow stage (B1.10) and consumed,
    # once, by the final encode below -- a pure intermediate with no
    # debugging value of its own (unlike frames/masks, which the CLI keeps
    # on request to diagnose a bad track), so it is always removed after a
    # successful run rather than gated behind --keep-intermediate.
    glow_frames_dir = os.path.join(job_dir, "glow_frames")
    audio_path = os.path.join(job_dir, "saber_audio.wav")

    if progress_cb:
        progress_cb("extract", 0, "extracting frames")
    fps, n_frames = extract_frames(input_video, frames_dir)
    with open(video_meta_path, "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    track_object(
        frames_dir, masks_dir, points, labels,
        checkpoint_path, config_name, device, n_frames,
        progress_cb=stage_cb("track"),
    )

    compute_motion(masks_dir, motion_path, progress_cb=stage_cb("motion"))

    render_glow(
        frames_dir, masks_dir, video_meta_path, glow_frames_dir, motion_path,
        color=color_bgr, spill_strength=intensity, blade_extend=blade_extend,
        progress_cb=stage_cb("glow"),
    )

    synthesize_audio(
        motion_path, video_meta_path, audio_path,
        voice=voice, progress_cb=stage_cb("audio"),
    )

    encode(glow_frames_dir, fps, audio_path, output_path, progress_cb=stage_cb("mux"))

    shutil.rmtree(glow_frames_dir, ignore_errors=True)

    return output_path
