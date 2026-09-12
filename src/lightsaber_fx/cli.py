import shutil
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone

import click

from . import paths
from .device import select_device
from .pipeline import job_meta
from .pipeline.detect import detect_blade
from .pipeline.frames import extract_frame_at
from .pipeline.runner import rerender_pipeline, run_pipeline
from .pipeline.track import pick_points_interactive
from .progress import EtaTracker, format_duration
from .setup import bootstrap


@click.group()
@click.version_option()
def main():
    """lightsaber-fx: turn a home video into a lightsaber VFX clip."""


@main.command()
@click.option("--force", is_flag=True, help="Redo SAM2 clone/install/checkpoint download even if already present.")
def setup(force):
    """One-time setup: clone SAM2, install it, download the checkpoint."""
    bootstrap(force=force)
    click.echo("Setup complete.")


def _render_options(f):
    """Shared --output/--color/--intensity/--blade-extend/--voice options for
    both `run` and `rerender` -- one definition (and one set of validation:
    `click.FloatRange`, `click.Choice`) rather than the two commands
    quietly drifting apart on what a valid --intensity or --voice is."""
    f = click.option("--output", default="final.mp4", help="Output video path.")(f)
    f = click.option("--color", default="red", help="Blade color: red, blue, green, or #RRGGBB.")(f)
    f = click.option(
        "--intensity", default=0.35, type=click.FloatRange(0.0, 1.0), help="Light-spill strength, 0.0-1.0."
    )(f)
    f = click.option(
        "--blade-extend/--no-blade-extend", default=True,
        help="Rebuild the blade as an extended capsule (default), or fall back to tracing the raw mask.",
    )(f)
    f = click.option(
        "--voice", default="neutral", type=click.Choice(["jedi", "sith", "neutral"]),
        help="Hum/swing character. Independent of --color.",
    )(f)
    return f


def _echo_progress(eta):
    def progress_cb(stage, pct, message):
        elapsed, remaining = eta.update(stage, pct)
        timing = f"elapsed {format_duration(elapsed)}"
        if remaining is not None:
            timing += f", ~{format_duration(remaining)} left"
        click.echo(f"[{stage}] {pct:5.1f}% {message} ({timing})")
    return progress_cb


@main.command()
@click.argument("input_video", type=click.Path(exists=True))
@_render_options
@click.option(
    "--keep-intermediate", is_flag=True,
    help="Also keep the job's frames/ after rendering. masks/ is always kept now "
         "(W1 made it small) so `lightsaber-fx rerender` can reuse this job later "
         "without re-tracking; frames/ is still the disk hog, so it stays opt-in.",
)
@click.option(
    "--auto/--no-auto", default=True, show_default=True,
    help="Look for the swung object automatically before asking you to click. "
         "It finds the fastest-moving elongated thing in the clip and proposes "
         "it; press Enter to accept or click to choose your own. --no-auto skips "
         "the search and shows you the first frame straight away.",
)
def run(input_video, output, color, intensity, blade_extend, voice, keep_intermediate, auto):
    """Run the full pipeline on INPUT_VIDEO.

    By default it looks for the swung object itself and shows you what it
    found to accept or override. Pass --no-auto to go straight to clicking.
    """
    if not paths.get_checkpoint_path().exists():
        raise click.ClickException("SAM2 is not installed yet — run `lightsaber-fx setup` first.")

    device = select_device()
    click.echo(f"Using device: {device}")
    if device == "cpu":
        click.echo("No GPU/MPS acceleration available — running on CPU, this will be much slower.")

    job_id = uuid.uuid4().hex[:8]
    job_dir = paths.new_job_dir(job_id)

    proposal = None
    if auto:
        click.echo("Looking for the swung object...")
        proposal = detect_blade(
            input_video, str(paths.get_checkpoint_path()),
            "configs/sam2.1/sam2.1_hiera_s.yaml", device,
        )
        if proposal is None:
            click.echo("Couldn't find one automatically — click it yourself.")
        else:
            click.echo(
                f"Found a candidate in frame {proposal.frame_index + 1} "
                f"(elongation {proposal.elongation:.1f}). "
                "Press Enter in the popup to accept it, or click to choose your own."
            )

    # The frame shown is the one the points refer to. With a proposal that is
    # the frame it was found in, which is usually mid-swing rather than the
    # first frame -- so the picker, the overlay and `prompt_frame` all have to
    # agree on it.
    prompt_frame = proposal.frame_index if proposal is not None else 0
    preview_path = job_dir / f"frame{prompt_frame}_preview.jpg"
    extract_frame_at(input_video, prompt_frame, str(preview_path))

    if proposal is None:
        click.echo("Click the object in the popup window. Shift-click to exclude a spot. Press Enter when done.")
    points, labels = pick_points_interactive(str(preview_path), proposal=proposal)
    if not points:
        click.echo("No points selected, aborting.")
        raise SystemExit(1)

    result = run_pipeline(
        input_video=input_video,
        points=points,
        labels=labels,
        prompt_frame=prompt_frame,
        output_path=output,
        job_dir=str(job_dir),
        checkpoint_path=str(paths.get_checkpoint_path()),
        device=device,
        color=color,
        intensity=intensity,
        blade_extend=blade_extend,
        voice=voice,
        progress_cb=_echo_progress(EtaTracker()),
    )
    click.echo(
        f"Wrote {result}. Job {job_id} kept at {job_dir} -- "
        f"`lightsaber-fx rerender {job_id}` re-renders it with a new "
        "color/intensity/voice without re-tracking."
    )

    # masks/ is always kept now -- W1 shrank it from gigabytes to a few MB,
    # and keeping it is what makes `rerender` possible later without
    # re-running SAM2. frames/ (near-lossless JPEGs) is still the disk hog,
    # so it stays gated behind --keep-intermediate as before; `rerender`
    # re-extracts frames from the source clip when it needs them.
    if not keep_intermediate:
        shutil.rmtree(job_dir / "frames", ignore_errors=True)


@main.command()
@click.argument("job_id")
@_render_options
def rerender(job_id, output, color, intensity, blade_extend, voice):
    """Re-render JOB_ID with a new --color/--intensity/--voice/
    --blade-extend, reusing its cached tracking masks instead of re-running
    SAM2 (the slow stage). Only extract, glow, audio, and mux run again.

    JOB_ID needs its masks/, motion.npz, video_meta.txt, and a resolvable
    path to its original source clip still present -- frames/ is NOT
    required, they are re-extracted from the source clip. If that source
    clip has since been moved, renamed, or deleted, this fails with a clear
    error naming it instead of a traceback -- that is the expected price of
    not keeping frames/ around for every job. Run `lightsaber-fx jobs` to
    see which jobs still qualify, and why others don't.
    """
    job_dir = paths.get_jobs_dir() / job_id
    if not job_dir.is_dir():
        raise click.ClickException(f"No such job: {job_id!r}. Run `lightsaber-fx jobs` to see what's available.")

    try:
        result = rerender_pipeline(
            job_dir=str(job_dir),
            output_path=output,
            color=color,
            intensity=intensity,
            blade_extend=blade_extend,
            voice=voice,
            progress_cb=_echo_progress(EtaTracker()),
        )
    except job_meta.JobNotRerenderableError as exc:
        raise click.ClickException(str(exc))
    click.echo(f"Wrote {result}")


@main.command(name="jobs")
def list_jobs():
    """List past render jobs and whether `rerender` can reuse each one.

    A job is re-renderable if it still has masks/, motion.npz,
    video_meta.txt, and its source clip is still at the path it was
    rendered from -- frames/ is not required, since `rerender` re-extracts
    them from that source clip.
    """
    job_dirs = sorted((p for p in paths.get_jobs_dir().iterdir() if p.is_dir()), key=lambda p: p.name)
    if not job_dirs:
        click.echo("No jobs found.")
        return

    for job_dir in job_dirs:
        info = job_meta.describe_job(str(job_dir))
        if info.rerenderable:
            created = (
                datetime.fromtimestamp(info.created_at, tz=timezone.utc)
                .astimezone()
                .strftime("%Y-%m-%d %H:%M:%S")
                if info.created_at is not None else "unknown"
            )
            click.echo(
                f"{info.job_id}  [re-renderable]  source={info.source_video}  "
                f"frames={info.frame_count}  created={created}"
            )
        else:
            click.echo(f"{info.job_id}  [NOT re-renderable: {info.reason}]")


@main.command()
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=8000, type=int)
@click.option("--open-browser", is_flag=True)
def serve(host, port, open_browser):
    """Start the local web app."""
    import uvicorn

    if host not in ("127.0.0.1", "localhost", "::1"):
        click.echo(
            f"Warning: binding to {host} exposes this app to your network. "
            "It has no authentication and accepts arbitrary uploads — anyone who can "
            "reach this address can fill your disk and read renders."
        )

    if open_browser:
        def opener():
            time.sleep(1.0)
            webbrowser.open(f"http://{host}:{port}/")

        threading.Thread(target=opener, daemon=True).start()

    uvicorn.run("lightsaber_fx.web.server:app", host=host, port=port)


@main.command()
def clean():
    """Delete all past render job directories.

    This removes every job's frames/masks/intermediate files, including any
    render currently in progress -- there is no cross-process lock, so don't
    run this while another `lightsaber-fx run` or `serve` job is rendering.
    """
    count = paths.clean_jobs()
    click.echo(f"Removed {count} job director{'y' if count == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
