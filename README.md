# lightsaber_fx

Turn a home video of someone swinging a stick — a bat, a broom, a toy sword —
into a glowing-blade VFX clip with matching sound, from a single mouse click.

No tape markers, no green screen, no frame-by-frame painting. You click the
object once in the first frame; [SAM2](https://github.com/facebookresearch/sam2)
tracks it through the rest of the clip, and the glow, the light it spills onto
nearby surfaces, and the audio are all generated from that tracking data.

The audio is synthesized from scratch — layered sines and shaped noise — so
there is no sampled or copyrighted sound anywhere in the output.

Two ways to use it:

- **A browser app.** `lightsaber-fx serve`, then drag a clip onto the page,
  click the object, watch progress stream, play the result in place.
- **A command line.** `lightsaber-fx run clip.mp4`, click the object in a popup
  window, and the rest runs unattended.

Everything runs locally on your machine. Nothing is uploaded anywhere.

---

## Requirements

| | |
|---|---|
| **Python** | 3.10, 3.11, or 3.12. **Not 3.13+** — PyTorch does not publish wheels for it yet. |
| **ffmpeg** | Required, on your `PATH`. The final step shells out to it by name. |
| **git + curl** | Required, on your `PATH`. `lightsaber-fx setup` uses them to fetch SAM2 and its model. |
| **Disk** | ~400 MB for SAM2 and its model, plus a few GB per render for intermediates (see [Disk usage](#disk-usage)). |
| **GPU** | Optional but strongly recommended. Apple Silicon (MPS) and NVIDIA (CUDA) are both used automatically; CPU-only works but is *much* slower. |

Installing ffmpeg:

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Debian / Ubuntu
```

---

## Install

```bash
git clone https://github.com/<your-user>/lightsaber_fx.git
cd lightsaber_fx

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
lightsaber-fx setup
```

`lightsaber-fx setup` is a one-time step that clones SAM2, installs it, and
downloads the small model checkpoint (~176 MB). It is safe to re-run — it skips
whatever it already has. Use `--force` to redo it from scratch.

None of that lands in this repo. It goes in a per-user application-data
directory, alongside the working files from each render:

```
~/Library/Application Support/lightsaber-fx/     # macOS
~/.local/share/lightsaber-fx/                    # Linux
├── sam2-src/         # the SAM2 clone, installed into your venv
├── checkpoints/      # sam2.1_hiera_small.pt
└── jobs/<job-id>/    # per-render working files (see Disk usage)
```

Verify the install:

```bash
lightsaber-fx --version
pytest -q                # 59 tests; the SAM2 tracking test is skipped if setup hasn't run
```

---

## Getting started

### Pick a clip that will work well

The tracker follows one object from one click, so the clips that work best are
the ones where that object stays distinguishable:

- **Short.** Start with 2–10 seconds. Runtime scales with frame count, and
  you'll want to iterate.
- **Good contrast** between the object and what's behind it. A bat against open
  sky is ideal; a brown stick against a brown fence is the hard case.
- **Not too fast.** Heavy motion blur can smear the object badly enough that
  the mask drifts partway through. Slow-motion footage is excellent.
- **Object stays in frame.** If it leaves the edge and comes back, the mask may
  not recover.

### Your first render: the browser app

```bash
lightsaber-fx serve --open-browser
```

Then, in the page:

1. **Drag your clip onto the drop zone** (or click it to pick a file). The first
   frame appears on a canvas.
2. **Click once on the object** you want to glow. A green dot marks it. Click
   the middle of the thickest part — not the very tip, and not where it
   overlaps a hand.
3. **Shift-click anything you want excluded** — a hand, a glove, a hilt. Red
   dots mark those. This is how you stop the glow bleeding onto the person
   holding the object. Optional, but it noticeably improves the result.
4. **Pick a colour and intensity**, then click **Track & Render**.
5. **Watch the progress bar.** It names the stage it's in and counts frames, so
   you can see the slow part (tracking) working.
6. **The finished clip plays in the page** when it's done, with a download link.

One render happens at a time. If you submit a second while one is going, you'll
get a "busy" message rather than two jobs fighting over your GPU.

### Your first render: the command line

```bash
lightsaber-fx run clip.mp4
```

A window opens showing the first frame. Click the object (**shift-click** to
exclude a spot), then press **Enter**. Everything after that is unattended, and
progress prints per stage. The result is written to `final.mp4`.

```bash
lightsaber-fx run clip.mp4 \
  --output blue_broom.mp4 \
  --color blue \
  --intensity 0.5 \
  --keep-intermediate
```

| Option | Default | Notes |
|---|---|---|
| `--output PATH` | `final.mp4` | Written relative to the current directory. |
| `--color` | `red` | `red`, `blue`, `green`, or any `#RRGGBB` hex value. |
| `--intensity` | `0.35` | `0.0`–`1.0`. How strongly the blade lights up its surroundings. Values outside the range are rejected immediately. |
| `--keep-intermediate` | off | Keep the extracted frames and masks after rendering (useful for debugging a bad track). |

In the picker window, note that the only way to finish is **Enter**, and the
only way to abort is **Ctrl-C** — closing the window doesn't do it, and there's
currently no undo for a misplaced point. If you misclick, Ctrl-C and re-run.

### How long it takes

Tracking dominates, and it scales with frame count. Measured on an Apple
Silicon Mac using MPS:

| Clip | Tracking | Whole pipeline |
|---|---|---|
| 2 s, 60 frames, 640×360 | ~35 s | ~1 min |
| 10 s, 300 frames, 1280×720 | ~3 min | ~4 min |

On CPU, expect several times that — the CLI warns you when it falls back.

---

## How it works

`lightsaber-fx run` and the web app both call the same pipeline. Five stages:

1. **extract** — the clip is exploded into per-frame JPEGs.
2. **track** — SAM2 takes your click points on frame 0 and propagates a mask
   for that object through every frame. This is the expensive stage.
3. **glow** — for each frame, the mask becomes three layers screen-blended over
   the original: a tight white-hot core, a coloured glow hugging the object,
   and a wide soft spill that brightens nearby surfaces. The object's centre of
   mass per frame is recorded here too.
4. **audio** — a continuous hum, plus a whoosh placed at each frame where the
   tracked motion exceeds the 80th percentile of its own speed, plus an
   ignition swell at the start and a power-down at the end. All synthesized.
5. **mux** — ffmpeg combines the rendered video and the audio into the output.

---

## Tuning

Exposed directly:

- **Colour and intensity** — `--color` / `--intensity` on the CLI, or the
  dropdown and slider in the web UI.

Code-level, in `src/lightsaber_fx/pipeline/`:

- **Glow shape** — `core_blur`, `inner_glow_blur`, and `spill_blur` are keyword
  parameters on `render_glow()` in `glow.py` (defaults `5`, `25`, `95`). Larger
  values bloom wider.
- **Whoosh sensitivity** — `synthesize_audio()` in `audio.py` treats a frame as
  a swing when its speed exceeds the 80th percentile of all positive per-frame
  speeds (an inline `np.percentile(positive, 80)`). Raise the percentile for
  fewer whooshes.
- **Model size** — `setup.py` fetches `sam2.1_hiera_small`. Larger SAM2
  checkpoints track better and run slower; switching means changing both the
  checkpoint URL and the matching config path.

---

## Troubleshooting

**`SAM2 is not installed yet — run 'lightsaber-fx setup' first.`**
Exactly what it says. This is checked before any work happens.

**`FileNotFoundError: 'ffmpeg'`, at the very end of a long render**
ffmpeg isn't on your `PATH`. Install it (see [Requirements](#requirements)).
Everything up to the final mux is preserved in the job directory.

**The mask drifts off the object partway through**
Usually fast motion or low contrast. Try a slower or higher-contrast clip, or
add more include points on frame 0 and shift-click exclude points on whatever
it's leaking onto.

**The glow covers a hand or the hilt**
Shift-click those spots to exclude them and re-run.

**`UserWarning: cannot import name '_C' from 'sam2'`**
Harmless. SAM2 ships an optional compiled extension that isn't built here;
upstream documents it as safe to ignore, and it doesn't affect tracking. The
test suite filters it.

**Everything is extremely slow**
You're probably on CPU. The CLI prints which device it selected, and warns when
it falls back.

**The downloaded clip won't play somewhere you shared it**
The mux copies the video stream rather than re-encoding, which keeps the render
fast but leaves it in a codec some players dislike. Re-encode if you need to
share widely:
```bash
ffmpeg -i final.mp4 -c:v libx264 -pix_fmt yuv420p -c:a aac shareable.mp4
```

---

## Disk usage

Each render keeps its working files so a failed or interesting run can be
inspected. They are not small: masks are uncompressed boolean arrays, roughly
2 MB per 1080p frame, so a 60-second 30 fps render can leave several GB of
frames and masks behind.

```bash
lightsaber-fx clean     # delete all job directories
```

`clean` leaves the SAM2 install and the model checkpoint alone. It has no
cross-process lock, so don't run it while a render is in progress. The CLI
already deletes its own intermediates unless you pass `--keep-intermediate`;
the web app keeps them until you run `clean`.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q                              # 59 tests
pytest tests/pipeline/test_glow.py -v  # one file
```

The suite runs in a few seconds and needs no GPU: the one test that exercises
SAM2 for real is skipped automatically if the checkpoint isn't installed, and
the expensive tracking step is stubbed everywhere else. Everything else is
tested against real artifacts — real videos written and read back, real WAVs
measured for the ignition and power-down transients, real `ffprobe` checks on
the muxed output.

```
src/lightsaber_fx/
├── cli.py              # click CLI: setup / run / serve / clean
├── paths.py            # where SAM2, the checkpoint, and job dirs live
├── device.py           # mps -> cuda -> cpu selection
├── setup.py            # the one-time SAM2 bootstrap
├── pipeline/
│   ├── frames.py       # 1. extract
│   ├── track.py        # 2. click-picker + SAM2 propagation
│   ├── glow.py         # 3. glow compositing + colour parsing
│   ├── audio.py        # 4. hum / whoosh / ignition / power-down
│   ├── mux.py          # 5. ffmpeg
│   └── runner.py       # run_pipeline(): the single orchestrator
└── web/
    ├── server.py       # FastAPI routes
    ├── jobs.py         # one-job-at-a-time manager
    └── static/         # the browser UI (plain HTML/CSS/JS, no build step)
```

Both entry points call `pipeline.runner.run_pipeline()` — there is no second
copy of the stage sequencing. Each pipeline stage is a plain function taking
explicit paths and options, so they're usable on their own.

A note on the web app's scope: it is deliberately local-first and single-user.
There is no authentication, no job queue, and it accepts arbitrary uploads, so
`--host` defaults to `127.0.0.1`. Binding it to a public interface is not
something this app is built for, and it warns you if you try.

---

## Credits

Object tracking is [SAM 2](https://github.com/facebookresearch/sam2) by Meta
AI, fetched and installed by `lightsaber-fx setup` under its own license and
not redistributed here. Compositing uses OpenCV; audio synthesis uses NumPy and
SoundFile; the web app is FastAPI.

This project is a fan-made visual-effects tool and is not affiliated with,
endorsed by, or connected to Lucasfilm or Disney.
