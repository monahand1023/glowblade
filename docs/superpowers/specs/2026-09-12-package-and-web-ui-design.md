# lightsaber_fx: packaging as a CLI + local web app

Status: approved
Date: 2026-09-12

## Problem

The pipeline (`track_sword.py` -> `apply_glow.py` -> `generate_audio.py` -> manual
`ffmpeg` mux) is three standalone scripts with hardcoded relative paths and
module-level constants, run manually in sequence from inside the project
directory. There's no way to run it as one command, no progress feedback beyond
stdout prints, and no way to use it without a terminal.

Goal: package it as (a) a CLI that orchestrates the whole pipeline in one
command, and (b) a local web app where a user can drag and drop a video, click
to select the object to track, watch progress, and play the result. Both need
to ship together in a way someone other than Dan can install and run on their
own machine.

## Scope

This is a **local-first, single-user tool**, packaged so other people can also
install and run it on their own machines. It is explicitly not a hosted
multi-user service:

- No auth, no accounts.
- No concurrent job queue -- exactly one render job at a time.
- No cloud storage -- everything lives under a per-user local app-data
  directory.
- No JS build tooling -- the web frontend is static HTML/CSS/JS served
  directly by the same Python process as the CLI.

## Architecture

One Python package (`lightsaber_fx`) with a shared pipeline core, plus two
thin entry points on top of it:

```
CLI (click)  ---\
                  >---  lightsaber_fx.pipeline.runner.run_pipeline()
Web (FastAPI) --/
```

Both entry points call the exact same orchestration function. No pipeline
logic is duplicated between them.

- **CLI**: `click`-based. Runs the pipeline synchronously in the terminal
  with plain percentage progress. Point selection reuses the existing `cv2`
  desktop popup (click = include, shift-click = exclude) -- unchanged from
  today's `track_sword.py`.
- **Web**: FastAPI + `uvicorn`, serving a single static page. Progress is
  streamed to the browser via Server-Sent Events (SSE) -- no WebSocket
  dependency needed. Point selection is a `<canvas>` click-picker
  (click = include, shift-click = exclude) mirroring the desktop UX.
- **Job model**: exactly one job at a time, tracked as in-process state. A
  second upload/run request while a job is active is rejected (web: HTTP 409
  with a "busy" message; CLI: it's a blocking command, so this is naturally
  impossible from a single invocation).

This is a deliberately smaller-scope choice than a "real" hosted app (React
frontend, WebSockets, Redis-backed job queue) because the actual requirement
is a personal tool other people can also run locally -- that heavier stack
would add packaging complexity (Node toolchain, external services) without
serving a real need here.

## Data layout

Everything the tool needs at runtime lives under a per-user app-data
directory (resolved via `platformdirs.user_data_dir("lightsaber-fx")`, e.g.
`~/Library/Application Support/lightsaber-fx` on macOS):

```
<data_dir>/
  sam2-src/                  # SAM2 clone + editable install source
  checkpoints/
    sam2.1_hiera_small.pt
  jobs/
    <job_id>/
      input.mp4
      frames/
      masks/
      video_meta.txt
      motion.npy
      glow_video.mp4
      saber_audio.wav
      final.mp4
```

This removes the manual symlink trick used during initial setup entirely --
pipeline code references absolute paths under `<data_dir>`, not a relative
`checkpoints/` folder inside a project directory. It also makes the SAM2
shadow-import bug (discovered and fixed earlier: SAM2's own code refuses to
load if cloned as a sibling directory of code that imports it) structurally
impossible, since `sam2-src/` never sits near any script that imports `sam2`.

Job directories are not automatically deleted -- disk usage under `jobs/`
grows with each render until `lightsaber-fx clean` is run manually, which
deletes everything under `jobs/` (not `sam2-src/` or `checkpoints/`, which
are one-time setup artifacts).

## Package layout

```
lightsaber_fx/
  pyproject.toml                    # console_scripts: lightsaber-fx
  src/lightsaber_fx/
    paths.py                        # app-data dir resolution, job dir helpers
    device.py                       # mps -> cuda -> cpu fallback
    pipeline/
      frames.py                     # extract_frames()
      track.py                      # SAM2 point-prompt tracking -> masks
      glow.py                       # glow rendering (color/intensity parameterized)
      audio.py                      # hum + whoosh + ignition + power-down synthesis
      mux.py                        # ffmpeg mux wrapper (subprocess)
      runner.py                     # run_pipeline() orchestrator w/ progress_cb
    setup.py                        # SAM2 bootstrap (clone, install, checkpoint)
    cli.py                          # click CLI: setup / run / serve / clean
    web/
      server.py                     # FastAPI app + routes
      jobs.py                       # single-job state machine, background thread
      static/
        index.html
        app.js
        style.css
  tests/
    test_pipeline.py
    test_cli.py
    test_web.py
```

## Pipeline refactor

The three scripts become importable functions instead of `if __name__ ==
"__main__"` scripts with module-level constants:

- `extract_frames(video_path, frames_dir) -> (fps, n_frames)`
- `track_object(frames_dir, masks_dir, points, labels, checkpoint_path, config_name, device, progress_cb) -> None`
- `render_glow(frames_dir, masks_dir, video_meta_path, output_video_path, motion_out_path, color, core_blur, inner_glow_blur, spill_blur, spill_strength, progress_cb) -> None`
- `synthesize_audio(motion_path, video_meta_path, out_wav_path, progress_cb) -> None`
- `mux(video_path, audio_path, output_path) -> None`
- `run_pipeline(input_video, points, labels, output_path, color, intensity, progress_cb) -> Path`

`progress_cb(stage: str, pct: float, message: str)` replaces today's
`print()` calls. Stages: `extract`, `track`, `glow`, `audio`, `mux`. The CLI's
callback prints percentages to the terminal; the web job runner's callback
pushes onto an SSE queue.

### Audio: ignition + power-down

`pipeline/audio.py` adds two synthesis functions alongside the existing
`synth_hum()` / `synth_whoosh()`, using the same fully-synthesized approach
(sine layers + shaped noise -- no sampled audio, no copyright concern):

- `synth_ignition()`: ~0.4-0.6s rising frequency sweep + a low-pitched noise
  "thump" (same noise-burst technique as `synth_whoosh`, pitched down and
  stretched), layered in at t=0.
- `synth_power_down()`: the reverse -- a falling sweep + thump, layered in at
  the end of the clip.

Since this pipeline tracks an object already present in frame 0 (there's no
moment in the source footage where a blade visibly switches on), t=0 and
clip-end are the natural sync points -- not object-appearance detection.
Both are always on, no CLI flag or web toggle (matches how the hum and
whooshes have no on/off control today).

### Tuning exposed as parameters

Per approved scope, `color` and `intensity` become real parameters (not just
code constants):

- CLI: `--color red|blue|green|#RRGGBB`, `--intensity 0.0-1.0`
- Web: a color swatch selector + intensity slider in the form alongside the
  click-picker

`CORE_BLUR` / `INNER_GLOW_BLUR` / `SPILL_BLUR` stay as internal constants --
not asked for, not exposed.

## CLI

`click`-based, four subcommands:

- `lightsaber-fx setup [--force]` -- automates the manual bootstrap done this
  session: clone SAM2 to `<data_dir>/sam2-src`, `pip install -e` it, `curl`
  *only* the small checkpoint directly into `<data_dir>/checkpoints/`
  (bypassing `download_ckpts.sh`, which ignores its argument and downloads
  all four checkpoints regardless). Idempotent -- skips steps already done
  unless `--force`.
- `lightsaber-fx run INPUT.mp4 [--output out.mp4] [--color red] [--intensity 0.35] [--keep-intermediate]`
  -- opens the existing `cv2` click/shift-click popup for point selection,
  then runs `run_pipeline()` with terminal progress output.
- `lightsaber-fx serve [--host 127.0.0.1] [--port 8000] [--open-browser]` --
  starts the FastAPI/uvicorn server.
- `lightsaber-fx clean` -- deletes all job directories under `<data_dir>/jobs/`.

## Web app

Endpoints:

- `GET /` -- serves the static SPA shell.
- `POST /api/upload` -- multipart video upload; creates a job dir, extracts
  frame 0, returns `{job_id, frame0_url, width, height}`. Returns 409 if a
  job is already running.
- `GET /api/jobs/{job_id}/frame0` -- serves the frame-0 JPEG for the picker.
- `POST /api/jobs/{job_id}/points` -- body `{points: [[x,y,label], ...],
  color, intensity}`; requires at least one include point; starts the
  background job thread; returns `{status: "started"}`.
- `GET /api/jobs/{job_id}/events` -- SSE stream of `{stage, pct, message}`,
  terminating with `{stage: "done", result_url}` or `{stage: "error",
  message}`.
- `GET /api/jobs/{job_id}/result` -- serves `final.mp4`.

Frontend flow (single page, vanilla JS, no framework):

1. Drag-and-drop zone (or file picker) -> upload -> frame 0 shown on a
   `<canvas>`.
2. Click = include point (green dot); shift-click = exclude point (red dot),
   matching the desktop tool. Color swatch + intensity slider above a
   "Track & Render" button, enabled once >=1 include point exists.
3. Submit -> open an `EventSource` against the SSE endpoint -> progress bar +
   current-stage label update live.
4. On completion -> `<video controls>` player against the result endpoint,
   plus a download link.
5. On error -> the error message is shown plainly.

## Device selection

`device.py` picks `mps` (Apple Silicon) -> `cuda` (NVIDIA, for other users'
machines) -> `cpu` (with a printed/UI warning that it will be slow),
extending today's `mps`-or-`cpu`-only check in `track_sword.py`.

## Testing

- `tests/test_pipeline.py`: codifies the synthetic frame/mask fixture smoke
  test performed manually this session, covering `render_glow`,
  `synthesize_audio` (including ignition/power-down), and `mux`.
- `track_object` is not meaningfully unit-testable without the real SAM2
  checkpoint present -- skip automatically if `<data_dir>/checkpoints/` is
  empty.
- `tests/test_cli.py`: `click.testing.CliRunner` for argument parsing and
  `--help` on all subcommands.
- `tests/test_web.py`: FastAPI `TestClient` covering upload -> points -> SSE
  shape, with `run_pipeline` swapped for a fast stub via dependency
  injection -- no GPU/checkpoint required in CI.

## Distribution

`pyproject.toml` with a `lightsaber-fx` console-script entry point. Installed
via `pip install .` or `pipx install .` from a clone of the (currently
private) repo -- not published to PyPI in this pass. Static web assets ship
as package data.

## Out of scope (explicitly deferred)

- Hosting this anywhere reachable over the network, multi-user auth, job
  queuing for concurrent renders.
- Exposing `CORE_BLUR`/`INNER_GLOW_BLUR`/`SPILL_BLUR` as parameters.
- An ignition/power-down on/off toggle.
- Job history/persistence UI beyond the current job.
- PyPI publishing.
