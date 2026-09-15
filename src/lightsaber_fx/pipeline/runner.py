import os
import shutil

from . import job_meta
from .audio import mix_hums, synthesize_audio
from .blade import compute_motion
from .frames import extract_frames
from .glow import parse_color, render_glow, render_glow_multi
from .mux import encode
from .track import track_object, track_objects


def _job_paths(job_dir):
    """The fixed set of on-disk paths every job dir uses. Both `run_pipeline`
    and `rerender_pipeline` build the same paths from the same job_dir, so
    there's exactly one place that knows a job's internal layout."""
    return {
        "frames_dir": os.path.join(job_dir, "frames"),
        "masks_dir": os.path.join(job_dir, "masks"),
        "video_meta_path": os.path.join(job_dir, "video_meta.txt"),
        "motion_path": os.path.join(job_dir, "motion.npz"),
        # Lossless PNG sequence written by the glow stage (B1.10) and consumed,
        # once, by the final encode below -- a pure intermediate with no
        # debugging value of its own (unlike frames/masks, which are kept on
        # request/by default -- to diagnose a bad track, or so `rerender` can
        # reuse them), so it is always removed after a successful run rather
        # than gated behind --keep-intermediate.
        "glow_frames_dir": os.path.join(job_dir, "glow_frames"),
        "audio_path": os.path.join(job_dir, "saber_audio.wav"),
    }


def _make_stage_cb(progress_cb):
    def stage_cb(stage):
        def cb(pct, message):
            if progress_cb:
                progress_cb(stage, pct, message)
        return cb
    return stage_cb


def _render_from_masks(job_dir, paths, fps, output_path, color_bgr, intensity, blade_extend, voice, stage_cb):
    """The tail both `run_pipeline` and `rerender_pipeline` share: glow ->
    audio -> mux against whatever frames/masks/motion.npz currently sit in
    `job_dir`, plus cleanup of the glow stage's own PNG-sequence
    intermediate.

    Kept in exactly one place on purpose -- a full render and a rerender
    diverging on stage order/arguments here is exactly the kind of drift a
    reviewer flagged earlier in this project."""
    render_glow(
        paths["frames_dir"], paths["masks_dir"], paths["video_meta_path"],
        paths["glow_frames_dir"], paths["motion_path"],
        color=color_bgr, spill_strength=intensity, blade_extend=blade_extend,
        progress_cb=stage_cb("glow"),
    )

    synthesize_audio(
        paths["motion_path"], paths["video_meta_path"], paths["audio_path"],
        voice=voice, progress_cb=stage_cb("audio"),
    )

    encode(paths["glow_frames_dir"], fps, paths["audio_path"], output_path, progress_cb=stage_cb("mux"))

    shutil.rmtree(paths["glow_frames_dir"], ignore_errors=True)

    return output_path


_LOW_COVERAGE_FRAC = 0.5


def _require_usable_track(n_tracked, n_with_blade, report):
    """Stop a doomed render at the motion stage instead of at the end of it.

    A track that found nothing still costs the full glow stage -- on a 10 s
    720p clip that is around 160 seconds, nearly as much as the tracking
    itself -- and then emits a video that looks exactly like the input, with
    nothing anywhere saying why. That is a bad failure: expensive, silent,
    and easy to misread as a bug in the compositing.

    The usual cause is the click points. An include point that missed the
    object, or an exclude point that landed *on* it, produces empty masks
    from the first frame, and every later frame inherits that. So the error
    names the points as the thing to check rather than reporting a bare
    count.

    Zero is the only unambiguous case, so it's the only one that raises.
    Partial coverage is legitimate -- an object can leave frame and come
    back -- so low coverage reports a warning through the normal progress
    channel (the CLI prints it, the web app streams it) and the render
    proceeds.
    """
    if n_tracked and not n_with_blade:
        raise RuntimeError(
            f"Tracking produced no blade in any of {n_tracked} frames, so there is "
            "nothing to render. This almost always means the click points were "
            "wrong: an include point that wasn't on the object, or an exclude "
            "point that landed on it. Check the points against the first frame "
            "and try again."
        )
    if n_with_blade and n_with_blade < _LOW_COVERAGE_FRAC * n_tracked:
        report(
            100,
            f"warning: a blade was found in only {n_with_blade} of {n_tracked} "
            "frames -- the mask was lost for most of the clip, so expect the glow "
            "to flicker or disappear. Rendering anyway.",
        )


def run_pipeline(
    input_video,
    points,
    labels,
    output_path,
    job_dir,
    checkpoint_path,
    device,
    prompt_frame=0,
    config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
    color="red",
    intensity=0.35,
    blade_extend=True,
    voice="neutral",
    progress_cb=None,
):
    stage_cb = _make_stage_cb(progress_cb)
    color_bgr = parse_color(color)

    paths = _job_paths(job_dir)

    if progress_cb:
        progress_cb("extract", 0, "extracting frames")
    fps, n_frames = extract_frames(input_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    # Recorded so a later `rerender` can re-extract frames from the same
    # source clip without keeping frames/ around in the meantime (see
    # docs/design-notes.md) -- this is the one piece of bookkeeping a job
    # needs beyond its masks/motion.npz to be re-renderable.
    job_meta.write_job_meta(job_dir, source_video=input_video)
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    track_object(
        paths["frames_dir"], paths["masks_dir"], points, labels,
        checkpoint_path, config_name, device, n_frames,
        prompt_frame=prompt_frame,
        progress_cb=stage_cb("track"),
    )

    n_tracked, n_with_blade = compute_motion(
        paths["masks_dir"], paths["motion_path"], progress_cb=stage_cb("motion"),
    )
    _require_usable_track(n_tracked, n_with_blade, stage_cb("motion"))

    return _render_from_masks(
        job_dir, paths, fps, output_path, color_bgr, intensity, blade_extend, voice, stage_cb,
    )


def _multi_job_paths(job_dir, object_ids):
    """Per-object mask/motion paths for a multi-saber job, plus the shared
    (object-count-agnostic) paths every job already uses."""
    return {
        "frames_dir": os.path.join(job_dir, "frames"),
        "video_meta_path": os.path.join(job_dir, "video_meta.txt"),
        "glow_frames_dir": os.path.join(job_dir, "glow_frames"),
        "masks_dirs": {oid: os.path.join(job_dir, "masks", str(oid)) for oid in object_ids},
        "motion_paths": {oid: os.path.join(job_dir, "motion", f"{oid}.npz") for oid in object_ids},
        "audio_paths": {oid: os.path.join(job_dir, f"saber_audio_{oid}.wav") for oid in object_ids},
        "mixed_audio_path": os.path.join(job_dir, "saber_audio.wav"),
    }


def _render_multi_from_masks(job_dir, paths, object_ids, sabers, color_bgrs, fps, output_path, blade_extend, stage_cb):
    """The tail both `run_pipeline_multi` and `rerender_pipeline_multi` share:
    glow -> per-object audio -> mix -> mux against whatever frames/masks/
    motion currently sit in `job_dir`, plus cleanup of the glow stage's own
    PNG-sequence intermediate. Mirrors `_render_from_masks`'s role for the
    single-object pair, and exists for the same reason: a full render and a
    rerender diverging on stage order/arguments here is exactly the kind of
    drift a reviewer flagged earlier in this project."""
    objects = [
        {
            "masks_dir": paths["masks_dirs"][oid],
            "motion_path": paths["motion_paths"][oid],
            "color": color_bgrs[i],
            "intensity": sabers[i]["intensity"],
        }
        for i, oid in enumerate(object_ids)
    ]
    render_glow_multi(
        paths["frames_dir"], objects, paths["video_meta_path"], paths["glow_frames_dir"],
        blade_extend=blade_extend, progress_cb=stage_cb("glow"),
    )

    for i, oid in enumerate(object_ids):
        synthesize_audio(
            paths["motion_paths"][oid], paths["video_meta_path"], paths["audio_paths"][oid],
            voice=sabers[i]["voice"], progress_cb=stage_cb("audio"),
        )
    mix_hums(list(paths["audio_paths"].values()), paths["mixed_audio_path"])

    encode(paths["glow_frames_dir"], fps, paths["mixed_audio_path"], output_path, progress_cb=stage_cb("mux"))
    shutil.rmtree(paths["glow_frames_dir"], ignore_errors=True)

    return output_path


def run_pipeline_multi(
    input_video,
    sabers,
    output_path,
    job_dir,
    checkpoint_path,
    device,
    config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
    blade_extend=True,
    progress_cb=None,
):
    """Like `run_pipeline`, but for 1-4 simultaneously tracked sabers, each
    with its own color/intensity/voice. Every saber is prompted at frame 0
    (see Global Constraints in the multi-saber backend plan) and tracked
    together in one SAM2 session via `track_objects`.
    """
    stage_cb = _make_stage_cb(progress_cb)
    color_bgrs = [parse_color(s["color"]) for s in sabers]  # validate every color up front

    object_ids = list(range(len(sabers)))
    paths = _multi_job_paths(job_dir, object_ids)
    os.makedirs(os.path.dirname(paths["masks_dirs"][0]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["motion_paths"][0]), exist_ok=True)

    if progress_cb:
        progress_cb("extract", 0, "extracting frames")
    fps, n_frames = extract_frames(input_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    job_meta.write_job_meta(job_dir, source_video=input_video, object_ids=object_ids)
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    prompts = [
        {"obj_id": oid, "masks_dir": paths["masks_dirs"][oid], "points": s["points"], "labels": s["labels"]}
        for oid, s in zip(object_ids, sabers)
    ]
    track_objects(
        paths["frames_dir"], prompts, checkpoint_path, config_name, device, n_frames,
        progress_cb=stage_cb("track"),
    )

    for oid in object_ids:
        n_tracked, n_with_blade = compute_motion(
            paths["masks_dirs"][oid], paths["motion_paths"][oid], progress_cb=stage_cb("motion"),
        )
        _require_usable_track(n_tracked, n_with_blade, stage_cb("motion"))

    return _render_multi_from_masks(
        job_dir, paths, object_ids, sabers, color_bgrs, fps, output_path, blade_extend, stage_cb,
    )


def rerender_pipeline(
    job_dir,
    output_path,
    color="red",
    intensity=0.35,
    blade_extend=True,
    voice="neutral",
    progress_cb=None,
):
    """Re-render an existing job with a new color/intensity/voice/
    blade_extend, reusing its cached tracking masks instead of re-running
    SAM2 -- the stage that dominates a full render's runtime (see
    docs/design-notes.md, "Two storage trade-offs"). Runs
    extract -> glow -> audio -> mux;
    `track_object` never runs.

    `compute_motion` is skipped too, but not because it's slow -- it's
    cheap. It's skipped because motion.npz depends only on the tracked
    masks, and none of this function's parameters (color/intensity/voice/
    blade_extend) can change a mask. The motion.npz already sitting in
    `job_dir` from the original render is exactly what a fresh compute would
    produce, so `_render_from_masks` below reads it as-is via
    `paths["motion_path"]` -- there is no call to `compute_motion` anywhere
    in this function. If this looks like an omission, it isn't: see
    docs/design-notes.md.

    Frames are deliberately NOT cached between renders -- re-extracted here
    from the job's recorded source clip -- because W1 measured a real job
    dir at masks/ <3% of its size and frames/ at ~93%: the masks are cheap
    to keep and expensive to recompute (SAM2 tracking), while frames are the
    opposite (cheap to recompute, expensive to keep). See
    `job_meta.require_rerenderable` for the check (and clear error) that a
    job still has what this needs -- most commonly, the source clip must
    still exist at the path it was rendered from.
    """
    stage_cb = _make_stage_cb(progress_cb)
    # Validate before touching the job dir or the (possibly moved/deleted)
    # source clip, same as run_pipeline validates color before the
    # potentially minutes-long extract/track stages ever run.
    color_bgr = parse_color(color)

    info = job_meta.require_rerenderable(job_dir)

    paths = _job_paths(job_dir)

    if progress_cb:
        progress_cb("extract", 0, "re-extracting frames from source clip")
    fps, n_frames = extract_frames(info.source_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    return _render_from_masks(
        job_dir, paths, fps, output_path, color_bgr, intensity, blade_extend, voice, stage_cb,
    )


def rerender_pipeline_multi(
    job_dir,
    output_path,
    sabers,
    blade_extend=True,
    progress_cb=None,
):
    """Re-render an existing multi-saber job with new color/intensity/voice
    per saber, reusing every object's cached tracking masks -- `track_objects`
    never runs. Mirrors `rerender_pipeline`'s re-extract-frames-but-reuse-
    masks trade-off, generalized to N objects.
    """
    stage_cb = _make_stage_cb(progress_cb)
    color_bgrs = [parse_color(s["color"]) for s in sabers]  # validate before touching the job dir

    info = job_meta.require_rerenderable(job_dir)
    object_ids = info.object_ids
    if object_ids is None:
        raise ValueError("This job has no recorded object_ids -- it isn't a multi-saber job")
    if len(sabers) != len(object_ids):
        raise ValueError(
            f"This job has {len(object_ids)} tracked object(s), but {len(sabers)} saber(s) were given"
        )

    paths = _multi_job_paths(job_dir, object_ids)

    if progress_cb:
        progress_cb("extract", 0, "re-extracting frames from source clip")
    fps, n_frames = extract_frames(info.source_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    return _render_multi_from_masks(
        job_dir, paths, object_ids, sabers, color_bgrs, fps, output_path, blade_extend, stage_cb,
    )
