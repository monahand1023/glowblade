import shutil
import threading
import time
import uuid
import webbrowser

import click

from . import paths
from .device import select_device
from .pipeline.frames import extract_first_frame
from .pipeline.runner import run_pipeline
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


@main.command()
@click.argument("input_video", type=click.Path(exists=True))
@click.option("--output", default="final.mp4", help="Output video path.")
@click.option("--color", default="red", help="Blade color: red, blue, green, or #RRGGBB.")
@click.option(
    "--intensity", default=0.35, type=click.FloatRange(0.0, 1.0), help="Light-spill strength, 0.0-1.0."
)
@click.option("--keep-intermediate", is_flag=True, help="Keep the job's frames/masks/intermediate files.")
def run(input_video, output, color, intensity, keep_intermediate):
    """Run the full pipeline on INPUT_VIDEO, prompting you to click the object to track."""
    if not paths.get_checkpoint_path().exists():
        raise click.ClickException("SAM2 is not installed yet — run `lightsaber-fx setup` first.")

    job_id = uuid.uuid4().hex[:8]
    job_dir = paths.new_job_dir(job_id)
    preview_path = job_dir / "frame0_preview.jpg"
    extract_first_frame(input_video, str(preview_path))

    click.echo("Click the object in the popup window. Shift-click to exclude a spot. Press Enter when done.")
    points, labels = pick_points_interactive(str(preview_path))
    if not points:
        click.echo("No points selected, aborting.")
        raise SystemExit(1)

    eta = EtaTracker()

    def progress_cb(stage, pct, message):
        elapsed, remaining = eta.update(stage, pct)
        timing = f"elapsed {format_duration(elapsed)}"
        if remaining is not None:
            timing += f", ~{format_duration(remaining)} left"
        click.echo(f"[{stage}] {pct:5.1f}% {message} ({timing})")

    device = select_device()
    click.echo(f"Using device: {device}")
    if device == "cpu":
        click.echo("No GPU/MPS acceleration available — running on CPU, this will be much slower.")
    result = run_pipeline(
        input_video=input_video,
        points=points,
        labels=labels,
        output_path=output,
        job_dir=str(job_dir),
        checkpoint_path=str(paths.get_checkpoint_path()),
        device=device,
        color=color,
        intensity=intensity,
        progress_cb=progress_cb,
    )
    click.echo(f"Wrote {result}")

    if not keep_intermediate:
        shutil.rmtree(job_dir / "frames", ignore_errors=True)
        shutil.rmtree(job_dir / "masks", ignore_errors=True)


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
