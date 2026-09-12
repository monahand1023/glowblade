# Lightsaber FX pipeline (SAM2 + red glow, Apple Silicon)

No tape markers, no frame-by-frame painting. Point SAM2 at the sword once
and it tracks it through the whole clip; the glow, light-spill onto nearby
objects, and audio are all generated from that tracking data.

## Setup (one-time)

1. Python 3.10-3.12 in a virtualenv (PyTorch wheels lag behind the very
   latest Python releases — 3.12 is the safe choice; skip 3.13/3.14
   until torch publishes wheels for them).
2. `pip install -r requirements.txt`
3. Install SAM2 itself (not a simple PyPI package). **Clone it outside
   this project directory**, not as a `sam2/` subfolder next to these
   scripts — SAM2's own `build_sam.py` refuses to import if it detects
   its repo cloned as a sibling of the code that imports it (it looks
   like the package shadowing itself) and raises a `RuntimeError`:
   ```
   git clone https://github.com/facebookresearch/sam2.git ../sam2-src
   cd ../sam2-src && pip install -e .
   ```
   (The `[demo]`/`[demo]`-style extras from older SAM2 docs no longer
   exist upstream — `notebooks` and `interactive-demo` are the current
   extras, and neither is needed for this pipeline.)
4. Download a checkpoint. On an M-series Mac, start with the **small**
   model — it's noticeably faster than base/large under MPS or CPU
   fallback. `download_ckpts.sh` ignores its argument and always
   downloads all four checkpoints (~1.5GB) — pull just the one you want
   directly instead:
   ```
   cd ../sam2-src/checkpoints
   curl -L -O https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
   ```
   Then symlink it into this project so `track_sword.py`'s
   `SAM2_CHECKPOINT` path resolves:
   ```
   mkdir -p checkpoints
   ln -s ../sam2-src/checkpoints/sam2.1_hiera_small.pt checkpoints/sam2.1_hiera_small.pt
   ```
   `SAM2_CONFIG` in `track_sword.py` must be the full path Hydra expects
   relative to the installed `sam2` package, e.g.
   `configs/sam2.1/sam2.1_hiera_s.yaml` — a bare filename like
   `sam2.1_hiera_s.yaml` fails with `MissingConfigException`.

## Run

1. Put your clip as `input.mp4` next to these scripts.
2. `python track_sword.py` — a window opens on the first frame. Click the
   blade once. If the mask also grabs the handle or hand, shift-click on
   that spot to exclude it, then press Enter. This step is the slow one —
   SAM2 is propagating the mask through every frame.
3. `python apply_glow.py` — renders `glow_video.mp4`: a white-hot core,
   a colored glow around the blade, and a soft wide "spill" that
   brightens nearby surfaces. Also saves the blade's per-frame position
   to `motion.npy` for the audio step.
4. `python generate_audio.py` — synthesizes a hum + swing whooshes timed
   to the sword's actual motion, saved as `saber_audio.wav`. It's fully
   synthesized (sine layers + shaped noise), not sampled from anything,
   so there's no copyright concern reusing it.
5. Mux video + audio:
   ```
   ffmpeg -i glow_video.mp4 -i saber_audio.wav -c:v copy -c:a aac -shortest final.mp4
   ```

## Tuning

- **Color**: `BLADE_COLOR` in `apply_glow.py` (OpenCV uses BGR, not RGB).
  Currently set to red.
- **Glow shape/intensity**: `CORE_BLUR`, `INNER_GLOW_BLUR`, `SPILL_BLUR`,
  `SPILL_STRENGTH` in the same file.
- **Whoosh sensitivity**: the `threshold` logic in `generate_audio.py`
  currently fires on the fastest ~20% of frame-to-frame motion — raise
  the percentile to make it fire less often.
- **Fast swings / motion blur losing the mask**: add more prompt points
  at a later frame — `add_new_points_or_box(state, frame_idx=<n>, ...)`
  isn't limited to frame 0.
- **Speed**: downscale `input.mp4` to 720p before tracking if MPS
  fallback-to-CPU makes SAM2 too slow on a long clip.
