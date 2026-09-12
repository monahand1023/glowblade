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
pytest -q                # the SAM2 tracking test is skipped if setup hasn't run
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
  --voice sith \
  --keep-intermediate
```

| Option | Default | Notes |
|---|---|---|
| `--output PATH` | `final.mp4` | Written relative to the current directory. |
| `--color` | `red` | `red`, `blue`, `green`, or any `#RRGGBB` hex value. |
| `--intensity` | `0.35` | `0.0`–`1.0`. How strongly the blade lights up its surroundings. Values outside the range are rejected immediately. |
| `--blade-extend` / `--no-blade-extend` | extend on | Rebuilds the blade as a capsule extending past the tracked object's tip (what makes a bat or broom read as a blade rather than a glowing prop). `--no-blade-extend` falls back to tracing the raw tracked silhouette instead — useful for an object that isn't elongated. |
| `--voice` | `neutral` | `neutral`, `jedi`, or `sith`. Changes the hum/swing character only — independent of `--color`, so picking red never silently changes the soundtrack. |
| `--keep-intermediate` | off | Also keep the extracted `frames/` after rendering (useful for debugging a bad track). The tracking masks are kept either way — they are tiny and `rerender` needs them. The rendered PNG sequence used for the final encode is always deleted after a successful run; it has no debugging value once encoded. |

In the picker window, note that the only way to finish is **Enter**, and the
only way to abort is **Ctrl-C** — closing the window doesn't do it, and there's
currently no undo for a misplaced point. If you misclick, Ctrl-C and re-run.

### Trying a different colour without re-tracking

Tracking is most of the runtime, and nothing about the colour, intensity, voice
or blade shape can change the mask — so changing your mind about any of those
shouldn't cost you another full render. It doesn't:

```bash
lightsaber-fx jobs                          # which past jobs can be reused, and why others can't
lightsaber-fx rerender a1b2c3d4 --color green --voice sith
```

`rerender` reuses the cached masks and re-runs only extract, glow, audio and the
encode. On the 2-second test clip that is **7.9 s instead of 42.8 s** — the
34.7 s tracking stage is skipped entirely.

The browser app does the same thing: when a render finishes, the colour,
intensity and voice controls stay on screen with a **Re-render** button, so you
can iterate without re-uploading or re-clicking.

What a job needs to stay re-renderable: its `masks/`, `motion.npz`,
`video_meta.txt`, and **its original source clip still at the same path**.
Notably it does *not* need `frames/` — those are re-extracted, because frames
are cheap to recompute (a few seconds) and expensive to keep (hundreds of MB),
while masks are the exact opposite. The cost of that trade is that **moving or
deleting the source clip makes a job un-re-renderable**; `rerender` tells you so
by name rather than failing obscurely, and `jobs` shows it up front.

### How long it takes

Tracking still dominates and scales with frame count, but **glow and the
final encode are no longer negligible** — the fidelity upgrade deliberately
traded speed for quality there (see [Tuning](#tuning)). Measured on an Apple
Silicon Mac using MPS, 2 s / 60 frames / 640×360:

| Stage | Time |
|---|---|
| extract | <0.1 s |
| track | ~32 s |
| motion | <0.1 s |
| glow | ~16 s |
| audio | <0.1 s |
| mux (encode) | <0.2 s |
| **whole pipeline** | **~54 s** |

| Clip | Tracking | Whole pipeline |
|---|---|---|
| 2 s, 60 frames, 640×360 | ~32 s | ~1 min |
| 10 s, 300 frames, 1280×720 | ~2m 47s | ~5m 16s |

Both rows are measured, not estimated. Note that glow scales with the blade's
on-screen size, not just the frame count: the wide multi-scale blur is confined
to a bounding box around the blade, so a long blade swung across the frame costs
several times more per frame than a small one.

On CPU, expect several times that — the CLI warns you when it falls back.

---

## How it works

`lightsaber-fx run` and the web app both call the same pipeline. Six stages:

1. **extract** — the clip is exploded into per-frame, near-lossless JPEGs
   (quality ~100). Everything downstream composites on these, so this stage
   deliberately doesn't save space the way a normal JPEG export would.
2. **track** — SAM2 takes your click points on frame 0 and propagates a mask
   for that object through every frame. This is the expensive stage.
3. **motion** — a fast, pure-numpy stage that fits each frame's mask to a
   `BladeGeometry` (centroid, tip/hilt endpoints, axis, length, width) via
   PCA, and writes `motion.npz`. Both later stages read it: **glow** rebuilds
   the blade from this geometry instead of tracing the raw mask, and **audio**
   drives swings from the blade's actual tip/angular speed instead of the
   mask's centroid (which barely moves when a blade pivots in place).
4. **glow** — the blade is reconstructed as a capsule (constant width,
   rounded tip, extended past the tracked tip) rather than the raw silhouette,
   composited in linear light as three elements — an eroded white-hot core, a
   coloured band, and a wide exponential-falloff glow — over a plate that's
   darkened first (so colour doesn't wash out on a bright background), plus a
   temporal trail, directional motion blur, chromatic bloom, flicker, and an
   approximate light wrap. Writes a lossless PNG sequence, one stage-internal
   intermediate this pipeline is happy to spend disk and time on.
5. **audio** — a two-layer hum (a beating oscillator pair plus a resonant
   "dark tone", both with slow pitch drift) and a TV-interference buzz layer,
   continuously cross-faded into a pitched-up register as the blade swings
   faster (no discrete whoosh trigger), with an ignition swell at the start
   and a power-down at the end. Panned in stereo by the blade's on-screen x
   position. All synthesized, never sampled.
6. **mux** — a single ffmpeg pass encodes the PNG sequence to H.264 and muxes
   in the stereo audio as AAC, in one lossy generation instead of three.

---

## Tuning

Exposed directly:

- **Colour and intensity** — `--color` / `--intensity` on the CLI, or the
  dropdown and slider in the web UI.
- **Blade extension** — `--blade-extend` / `--no-blade-extend` on the CLI, or
  the checkbox in the web UI. On (default) rebuilds the blade as an extended
  capsule; off traces the raw tracked mask, which suits an object that isn't
  elongated.
- **Voice** — `--voice neutral|jedi|sith` on the CLI, or the dropdown in the
  web UI. Changes only the hum/swing character (pitch, distortion, buzz
  level); colour and voice are independent, so switching colour never changes
  the sound.

This is a **quality-over-speed pipeline by design**: frames are extracted at
near-lossless JPEG quality, the glow stage composites in linear light through
several Gaussian passes per frame, and the final encode is a single
high-quality `libx264 -crf 16` pass rather than a fast intermediate. The glow
and encode stages are the ones that got slower on purpose in exchange for a
visibly cleaner result — there's no `--fast` escape hatch.

Code-level, in `src/lightsaber_fx/pipeline/`:

- **Glow shape** — `render_glow()` in `glow.py` exposes tuning knobs for every
  layer (capsule extension/taper, core/colour-band blur, the three-scale glow
  falloff, Knoll darkening strength, trail decay, motion-blur gain, chromatic
  bloom, flicker, light wrap) as keyword parameters — see the function's
  docstring and defaults for the full list. Most are expressed as fractions of
  the clip's own median blade width/length, not absolute pixels, so they scale
  with the source footage.
- **Swing sensitivity** — `synth_swing_hum()` in `audio.py` normalizes
  `tip_speed`/`angular_speed` against their own 90th-percentile-of-positive
  values (`_normalize_speed`'s `ref_percentile`); raise it for a swing sound
  that only kicks in on faster motion.
- **Encode quality** — `-crf 16` is hardcoded in `mux.py`'s `encode()`; lower
  is higher quality (and larger) output.
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
Output is H.264/`yuv420p` with `+faststart` and stereo AAC audio — the
combination most players and platforms accept natively — so this shouldn't
come up. If it does, it's worth checking the exact codec/profile with
`ffprobe -show_streams final.mp4` before assuming it's this pipeline's fault.

---

## Disk usage

Each render keeps its tracking masks so you can `rerender` it later. Those are
cheap: they compress about **355×** (measured on real tracked footage — blade
masks are overwhelmingly empty), which works out to well under a megabyte for a
whole clip. Keeping them is effectively free, which is why it is the default.

The extracted `frames/` are the expensive part — near-lossless JPEGs, hundreds
of MB for a long clip. The CLI deletes them after a successful render unless you
pass `--keep-intermediate`, and `rerender` re-extracts them from your source
clip when it needs them. The browser app currently keeps them until you run
`clean`.

```bash
lightsaber-fx clean     # delete all job directories
```

`clean` leaves the SAM2 install and the model checkpoint alone — rerun
`lightsaber-fx setup --force` if you need to replace those. It has no
cross-process lock, so don't run it while a render is in progress. Note that
cleaning a job also makes it un-re-renderable, since it removes the masks.

---

## Development

```bash
pip install -e ".[dev]"
pytest -q
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
├── cli.py              # click CLI: setup / run / serve / clean / rerender / jobs
├── paths.py            # where SAM2, the checkpoint, and job dirs live
├── device.py           # mps -> cuda -> cpu selection
├── progress.py         # per-stage elapsed/ETA tracking
├── setup.py            # the one-time SAM2 bootstrap
├── pipeline/
│   ├── frames.py       # 1. extract
│   ├── track.py        # 2. click-picker + SAM2 propagation
│   ├── blade.py        # 3. motion: per-frame blade geometry fit + mask I/O
│   ├── glow.py         # 4. glow compositing + colour parsing
│   ├── audio.py        # 5. hum / swing / ignition / power-down
│   ├── mux.py          # 6. single ffmpeg encode() pass
│   ├── job_meta.py     # per-job source-clip record; rerender eligibility
│   └── runner.py       # run_pipeline() / rerender_pipeline()
└── web/
    ├── server.py       # FastAPI routes
    ├── jobs.py         # one-job-at-a-time manager
    └── static/         # the browser UI (plain HTML/CSS/JS, no build step)
```

The code uses short labels — `B1.4`, `B2.5`, `W1` — where a comment would
otherwise have to re-explain a whole compositing or synthesis technique.
[`docs/design-notes.md`](docs/design-notes.md) is what those point at: the
film-compositing and sound-design practice behind each step, the measurements
the storage design rests on, and a write-up of the one interesting bug (a
linear crossfade that made swings quieter instead of louder). The original
design spec for the packaging and web UI is in
[`docs/superpowers/specs/`](docs/superpowers/specs/).

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
