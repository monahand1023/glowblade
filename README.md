# Glowblade

Turn a home video of someone swinging a stick — a bat, a broom, a toy sword —
into a glowing-blade VFX clip with matching sound, usually without clicking
anything.

No tape markers, no green screen, no frame-by-frame painting. It finds the
swung object by itself and shows you what it found;
[SAM2](https://github.com/facebookresearch/sam2) tracks it through the rest of
the clip, and the glow, the light it spills onto nearby surfaces, and the audio
are all generated from that tracking data. If it guesses wrong — or declines to
guess, which it does rather than propose something it isn't sure of — one click
overrides it.

The audio is synthesized from scratch — layered sines and shaped noise — so
there is no sampled or copyrighted sound anywhere in the output.

Two ways to use it:

- **A browser app.** `glowblade serve`, then drag a clip onto the page,
  confirm what it found, watch progress stream, play the result in place.
- **A command line.** `glowblade run clip.mp4`, press Enter to accept what
  it found in a popup window, and the rest runs unattended.

Runs locally on your machine by default. If you set a `GEMINI_API_KEY` (or
`GOOGLE_API_KEY`) environment variable, one video frame per upload is sent to
Google's Gemini API to automatically find multiple swung objects at once;
without a key, detection falls back to the fully local motion-based search
and nothing leaves your machine.

---

## Requirements

| | |
|---|---|
| **Python** | 3.10, 3.11, or 3.12. **Not 3.13+** — PyTorch does not publish wheels for it yet. |
| **ffmpeg** | Required, on your `PATH`. The final step shells out to it by name. |
| **git + curl** | Required, on your `PATH`. `glowblade setup` uses them to fetch SAM2 and its model. |
| **Disk** | ~400 MB for SAM2 and its model, plus a few GB per render for intermediates (see [Disk usage](#disk-usage)). |
| **GPU** | Optional but strongly recommended. Apple Silicon (MPS) and NVIDIA (CUDA) are both used automatically; CPU-only works but is *much* slower. |
| **`GEMINI_API_KEY`** (or `GOOGLE_API_KEY`) | Optional. When set, detection asks [Gemini](https://ai.google.dev/) (via the `google-genai` package, installed automatically) to find every swung object in one frame, so multiple objects can be detected and tracked at once. Without it, detection falls back to the fully local, motion-based search and nothing leaves your machine — only one object is found per clip in that mode. |

Installing ffmpeg:

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Debian / Ubuntu
```

---

## Install

```bash
git clone https://github.com/<your-user>/glowblade.git
cd glowblade

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
glowblade setup
```

`glowblade setup` is a one-time step that clones SAM2, installs it, and
downloads the small model checkpoint (~176 MB). It is safe to re-run — it skips
whatever it already has. Use `--force` to redo it from scratch.

None of that lands in this repo. It goes in a per-user application-data
directory, alongside the working files from each render:

```
~/Library/Application Support/glowblade/     # macOS
~/.local/share/glowblade/                    # Linux
├── sam2-src/         # the SAM2 clone, installed into your venv
├── checkpoints/      # sam2.1_hiera_small.pt
└── jobs/<job-id>/    # per-render working files (see Disk usage)
```

Verify the install:

```bash
glowblade --version
pytest -q                # the SAM2 tracking test is skipped if setup hasn't run
```

---

## Getting started

### Pick a clip that will work well

The tracker follows one object all the way through, so the clips that work
best are the ones where that object stays distinguishable:

- **One straight object.** This is the biggest constraint, and it is
  structural rather than a tuning problem. The blade is rebuilt as a capsule
  along a single fitted axis, which is exactly right for a bat, a broom, a
  stick or a sword — all genuinely one axis. It is wrong for anything
  L-shaped. A golf club is a shaft with a head roughly perpendicular to it,
  so fitting one axis to that mask returns a direction the club doesn't
  actually have, and the blade reads as a streak crossing the club rather
  than lying along it. See the tested-clips table below: everything with one
  real axis works, and the golf club is the only thing that doesn't.
- **Short.** Start with 2–10 seconds. Runtime scales with frame count, and
  you'll want to iterate.
- **Good contrast** between the object and what's behind it. A bat against open
  sky is ideal; a brown stick against a brown fence is the hard case.
- **Big enough in frame.** A thin object in a wide shot is hard for automatic
  detection and hard for the tracker. On a sword-demonstration clip shot wide,
  with two figures and a busy background, detection declines to guess and asks
  you to click.
- **Not too fast.** Heavy motion blur can smear the object badly enough that
  the mask collapses partway through. On a full-speed baseball swing the mask
  shrinks to a fraction of its usual size at the fastest frames and the blade
  briefly smears. Slow-motion footage is excellent.
- **Object stays in frame.** If it leaves the edge and comes back, the mask may
  not recover. A tight close-up where the object swings in and out of shot
  will flicker — on one golf clip 34 of 120 frames had no object in them at
  all, which is the footage, not a failure.

### What it has actually been tested on

Five clips, all rendered end to end. "Auto" is whether detection proposed the
right object without being clicked; "masks" is how many frames the tracker
held the object for.

| Clip | Auto | Masks | Render |
|---|---|---|---|
| Baseball bat, 2 s, 640×360 | yes | 60/60 | good |
| Baseball bat, 10 s, 1280×720 | yes | 300/300 | good, except the mask collapses at the fastest frames of the swing |
| Broom, indoor, 4 s | yes | 96/96 | good |
| Sword demonstration, wide shot, 4 s | **no** — declines, so you click | 89/100 | good; the tracker holds a small object (median mask 748 px) fine once pointed at it |
| Golf club, close-up, 4 s | yes | 86/120 | **half** — lands on the club, but the blade axis is unstable because a club is L-shaped, and the club leaves this tight frame for 34 frames |

The honest summary: the effect is good on anything that is genuinely one
straight object, automatic detection handles four of the five, and it declines
rather than guessing wrong on the hard one.

### Multiple objects

If a `GEMINI_API_KEY` is set, detection can find and track up to four objects
at once, each with its own colour, intensity and voice. Once found, each
object is tracked independently by the same SAM2 tracker described above, so
everything in "Pick a clip that will work well" applies per object.

Two things worth knowing, both confirmed by actually testing on real footage
rather than assumed:

- **Busy, heavily-occluded multi-person scenes can make the tracker drift
  onto the wrong nearby object mid-clip.** On a chaotic four-person sword
  fight, detection correctly found both blades, but by partway through the
  clip the tracker on one had drifted off the sword entirely and the other
  had jumped onto a third person's weapon. The same points fed through the
  single-object pipeline on a cleaner two-person fencing clip tracked
  perfectly for the full clip with no drift — so this is a scene-difficulty
  problem (similar objects crossing and occluding each other), not a defect
  in multi-object detection or tracking itself. Prefer clips with clearer
  separation between people for multi-object renders, same as the
  single-object contrast/size guidance above.
- **The two detection paths don't have equal coverage.** On a fencing clip,
  Gemini found both blades instantly; the local motion-only fallback (no API
  key) found nothing at all on the same clip — a fast rapier thrust
  apparently doesn't produce the motion signature the fallback's heuristic
  looks for. Tracking quality is identical either way once a starting point
  exists (verified by feeding the same points through both paths); it's
  *finding* that starting point where the two differ.

### Your first render: the browser app

```bash
glowblade serve --open-browser
```

Then, in the page:

1. **Drag your clip onto the drop zone** (or click it to pick a file).
2. **Wait a second or two while it looks for the object.** If it finds one, the
   page jumps to the frame where the object was moving fastest, tints the
   detected shape green, and marks the points it will track from. That frame is
   usually mid-swing rather than the first frame — that is deliberate, and the
   tracker works outwards from there in both directions.
3. **Accept or override.** If the green shape is the thing you want glowing,
   just click **Track & Render**. If it isn't — or if it found nothing — click
   the object yourself. Your first click discards the detection entirely
   rather than adding to it.
4. **Shift-click anything you want excluded** — a hand, a glove, a hilt. Red
   dots mark those. This is how you stop the glow bleeding onto the person
   holding the object. Optional, but it noticeably improves the result.
5. **Pick a colour, intensity and voice**, then click **Track & Render**.
6. **Watch the progress bar.** It names the stage it's in, counts frames, and
   shows elapsed time and an estimate of what's left.
7. **The finished clip plays in the page** when it's done, with a download
   link — and the controls stay put with a **Re-render** button, so trying
   another colour doesn't re-run the tracking.

One render happens at a time. If you submit a second while one is going, you'll
get a "busy" message rather than two jobs fighting over your GPU.

### Your first render: the command line

```bash
glowblade run clip.mp4
```

It looks for the swung object first, then opens a window showing what it
found, with the detected shape tinted green. Press **Enter** to accept it, or
click the object yourself to override (**shift-click** to exclude a spot).
Everything after that is unattended, and progress prints per stage with an ETA.
The result is written to `final.mp4`.

Pass `--no-auto` to skip the search and go straight to clicking the first
frame.

```bash
glowblade run clip.mp4 \
  --output blue_broom.mp4 \
  --color blue \
  --intensity 0.5 \
  --voice deep \
  --keep-intermediate
```

| Option | Default | Notes |
|---|---|---|
| `--output PATH` | `final.mp4` | Written relative to the current directory. |
| `--color` | `red` | `red`, `blue`, `green`, or any `#RRGGBB` hex value. |
| `--intensity` | `0.35` | `0.0`–`1.0`. How strongly the blade lights up its surroundings. Values outside the range are rejected immediately. |
| `--blade-extend` / `--no-blade-extend` | extend on | Rebuilds the blade as a capsule extending past the tracked object's tip (what makes a bat or broom read as a blade rather than a glowing prop). `--no-blade-extend` falls back to tracing the raw tracked silhouette instead — useful for an object that isn't elongated. |
| `--voice` | `neutral` | `neutral`, `bright`, or `deep`. Changes the hum/swing character only — independent of `--color`, so picking red never silently changes the soundtrack. |
| `--keep-intermediate` | off | Also keep the extracted `frames/` after rendering (useful for debugging a bad track). The tracking masks are kept either way — they are tiny and `rerender` needs them. The rendered PNG sequence used for the final encode is always deleted after a successful run; it has no debugging value once encoded. |
| `--auto` / `--no-auto` | auto on | Look for the swung object before asking you to click. `--no-auto` skips the search (a couple of seconds) and shows you the first frame straight away. |

In the picker window, note that the only way to finish is **Enter**, and the
only way to abort is **Ctrl-C** — closing the window doesn't do it, and there's
currently no undo for a misplaced point. If you misclick, Ctrl-C and re-run.

### Trying a different colour without re-tracking

Tracking is most of the runtime, and nothing about the colour, intensity, voice
or blade shape can change the mask — so changing your mind about any of those
shouldn't cost you another full render. It doesn't:

```bash
glowblade jobs                          # which past jobs can be reused, and why others can't
glowblade rerender a1b2c3d4 --color green --voice deep
```

`rerender` reuses the cached masks and re-runs only extract, glow, audio and the
encode. On the 2-second test clip that is **7.2 s instead of 41 s** — the 33.6 s
tracking stage is skipped entirely. On a 10-second 720p clip the saving is real
but smaller (1m 37s instead of 5m 45s), because glow is most of what is left
once tracking is gone; see [How long it takes](#how-long-it-takes).

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

Measured on an Apple Silicon Mac using MPS. These are real timings, not
estimates, but the machine had ordinary background load (Spotlight indexing and
a backup running, load average 8–20), so treat them as representative rather
than as a clean benchmark.

| Stage | 2 s / 60 frames / 640×360 | 10 s / 300 frames / 1280×720 |
|---|---|---|
| detect | 2.0 s | 4.1 s |
| extract | 1.3 s | 10.1 s |
| track | 33.6 s | 168.3 s |
| motion | 0.1 s | 3.8 s |
| glow | 5.9 s | 159.6 s |
| audio | 0.1 s | 0.6 s |
| mux (encode) | 0.2 s | 2.9 s |
| **whole pipeline** | **41 s** | **5m 45s** |
| `rerender` (same job, new colour) | **7.2 s** | **1m 37s** |

Two things worth reading off that table:

**Tracking dominates on short clips, but not on long ones.** At 720p the glow
stage (159.6 s) costs almost as much as tracking (168.3 s). Glow scales with
the blade's on-screen *size* as well as the frame count — the wide multi-scale
blur is confined to a bounding box around the blade, so a long blade swung
across a 720p frame costs several times more per frame than a small one in a
360p frame. The fidelity upgrade deliberately traded speed for quality here
(see [Tuning](#tuning)).

**That is also why `rerender` saves less on a long clip** — 5.7× on the 2 s
clip but 3.6× on the 10 s one. It skips tracking entirely, but glow is most of
what remains.

On CPU, expect several times all of this — the CLI warns you when it falls
back.

---

## How it works

`glowblade run` and the web app both call the same pipeline. Six stages:

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
   continuously cross-faded into a louder, pitched-up register as the blade
   swings faster (no discrete whoosh trigger), with an ignition swell at the
   start and a power-down at the end. A full-speed swing is the loudest
   sustained thing in the clip — about 2.9× the idle hum's level, measured,
   and louder than the ignition transient. Panned in stereo by the blade's
   on-screen x position. All synthesized, never sampled.
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
- **Voice** — `--voice neutral|bright|deep` on the CLI, or the dropdown in the
  web UI. Changes only the hum/swing character (pitch, distortion, buzz
  level); colour and voice are independent, so switching colour never changes
  the sound.

This is a **quality-over-speed pipeline by design**: frames are extracted at
near-lossless JPEG quality, the glow stage composites in linear light through
several Gaussian passes per frame, and the final encode is a single
high-quality `libx264 -crf 16` pass rather than a fast intermediate. The glow
and encode stages are the ones that got slower on purpose in exchange for a
visibly cleaner result — there's no `--fast` escape hatch.

Code-level, in `src/glowblade/pipeline/`:

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
- **Swing loudness** — also in `synth_swing_hum()`, `hum * (1.0 + 1.6 *
  swing_outer)` sets how much louder a full-speed swing is than the idle hum.
  It is deliberately large: a swing should be the loudest sustained thing in
  the clip. Lower it for a subtler result. Pushing it much higher runs into
  the `tanh` limiter, which gives the extra gain straight back.
- **Encode quality** — `-crf 16` is hardcoded in `mux.py`'s `encode()`; lower
  is higher quality (and larger) output.
- **Model size** — `setup.py` fetches `sam2.1_hiera_small`. Larger SAM2
  checkpoints track better and run slower; switching means changing both the
  checkpoint URL and the matching config path.

---

## Troubleshooting

**`SAM2 is not installed yet — run 'glowblade setup' first.`**
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
glowblade clean     # delete all job directories
```

`clean` leaves the SAM2 install and the model checkpoint alone — rerun
`glowblade setup --force` if you need to replace those. It has no
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
src/glowblade/
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
AI, fetched and installed by `glowblade setup` under its own license and
not redistributed here. Compositing uses OpenCV; audio synthesis uses NumPy and
SoundFile; the web app is FastAPI.

This project is a fan-made visual-effects tool and is not affiliated with,
endorsed by, or connected to Lucasfilm or Disney.
