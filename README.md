# Lightsaber FX pipeline (SAM2 tracking + glow compositing)

No tape markers, no frame-by-frame painting. Point SAM2 at the sword once
and it tracks it through the whole clip; the glow, light-spill onto nearby
objects, and audio are all generated from that tracking data.

## Setup (one-time)

1. Python 3.10-3.12 in a virtualenv (PyTorch wheels lag behind the very
   latest Python releases -- 3.12 is the safe choice; skip 3.13/3.14
   until torch publishes wheels for them).
2. Non-Python prerequisites on `PATH`: `ffmpeg` (the final mux step execs
   it directly -- `brew install ffmpeg` on macOS, `apt install ffmpeg` on
   Debian/Ubuntu), plus `git` and `curl`, which `lightsaber-fx setup`
   shells out to for cloning SAM2 and downloading the checkpoint.
3. `pip install -e ".[dev]"`
4. `lightsaber-fx setup` -- clones SAM2, installs it, and downloads the
   small checkpoint. Everything it needs (SAM2 source, the checkpoint,
   and per-run job files) lives under a per-user app-data directory --
   `~/Library/Application Support/lightsaber-fx/` on macOS -- not
   inside this repo.

## Run: command line

```
lightsaber-fx run your_clip.mp4 --color red --intensity 0.35
```

A window opens on the first frame -- click the object once (shift-click to
exclude a spot, e.g. a hand or hilt), press Enter. The rest of the pipeline
(tracking, glow, synthesized hum/whoosh/ignition/power-down audio, and the
ffmpeg mux) runs automatically, printing progress, and writes `final.mp4`
(or the path given to `--output`).

## Run: web app

```
lightsaber-fx serve --open-browser
```

Drag a video onto the page, click the object on the first frame (shift-click
to exclude), pick a color and intensity, and click "Track & Render". Progress
streams live; the finished clip plays in the browser with a download link.

## Tuning

- **Color / intensity**: `--color` / `--intensity` on the CLI, or the color
  dropdown and intensity slider in the web UI.
- **Glow shape**: `core_blur`, `inner_glow_blur`, and `spill_blur` are
  keyword parameters on `render_glow()` in
  `src/lightsaber_fx/pipeline/glow.py` (defaults `5`, `25`, `95`) -- there's
  no CLI/web option for these yet, so change the defaults there if needed.
- **Whoosh sensitivity**: `synthesize_audio()` in
  `src/lightsaber_fx/pipeline/audio.py` flags a frame as a swing when its
  motion speed exceeds the 80th percentile of all positive per-frame speeds
  (an inline `np.percentile(positive, 80)` call) -- edit that percentile if
  swings are over- or under-triggering.
- **Fast swings / motion blur losing the mask**: this still applies --
  SAM2 tracking only gets one click on frame 0; very fast motion can lose
  the mask partway through a long clip.

## Audio

Fully synthesized (sine layers + shaped noise) -- no sampled or copyrighted
sound. A track now includes an ignition swell at the start, the running
hum/whoosh timed to the tracked object's motion, and a power-down at the end.

## Maintenance

- `lightsaber-fx clean` deletes all past render job directories (frames,
  masks, intermediate audio/video) to reclaim disk space. It does not touch
  the SAM2 install or the checkpoint -- rerun `lightsaber-fx setup --force`
  to redo those.
- These intermediates add up fast: masks are stored as uncompressed boolean
  `.npy` arrays (~2MB per 1080p frame), so a single 60-second render at 30fps
  can leave several GB of frames/masks behind. Run `clean` periodically,
  especially after experimenting with several renders of the same clip.
