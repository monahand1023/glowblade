# Design notes

The code carries short labels — `B1.4`, `B2.5`, `W1`, `B-fix` — where a
comment would otherwise have to re-explain a whole technique. This file is
what those labels point at. It is the "why" behind the renderer; the module
docstrings are the "what".

Nothing here is required reading to *use* the tool. Start with the
[README](../README.md) for that.

## How to read the labels

| Label | What it refers to |
|---|---|
| **Phase A** | The blade-geometry foundation: fitting an axis, tip, hilt, length and width to each tracked mask (`pipeline/blade.py`). Everything downstream is a function of that geometry rather than of the raw mask. |
| **B1.x** | The glow renderer's compositing techniques (`pipeline/glow.py`). Table below. |
| **B2.x** | The audio synthesizer's layers (`pipeline/audio.py`). Table below. |
| **Phase C** | Encoding: a single `libx264 -crf 16` pass with stereo AAC, replacing an OpenCV `mpeg4` write followed by a stream copy. |
| **W1** | Mask storage. See "Two storage trade-offs" below. |
| **W2** | `rerender` — re-running glow/audio/encode against a finished job's cached masks. See "Two storage trade-offs" below. |
| **B-fix** | A follow-up rewrite of the swing's dynamic range and high-frequency content. See "The swing loudness bug" below. |

## B1 — the glow renderer

The tracked mask is the silhouette of a *bat*, not of a blade. Painting it
with colour produces a glowing bat. Every step below exists to turn that
silhouette into something that reads as a lit blade, and each is standard
practice in film compositing rather than an invention of this project.

| | Technique | Why |
|---|---|---|
| **B1.1** | Linear-light compositing | Light adds linearly; sRGB pixel values do not. Layers are linearized (`(v/255)**2.2`), summed, then re-encoded. Adding in gamma space is what makes naive glows look milky and grey instead of hot. |
| **B1.2** | Blade reconstruction as a capsule | The blade is rebuilt from Phase A's axis as a rounded-cap capsule extended past the tracked tip, not traced from the silhouette. Prop blades in film are shorter than the blades drawn over them, and real sabers have a rounded tip; both are geometry the tracker cannot supply. |
| **B1.3** | Three distinct elements | A hot, nearly-white core (an eroded, tightly-blurred capsule), a saturated colour band around it, and a wide glow. ILM's saber work is built from these three separable elements; collapsing them into one blurred mask is what makes a core as wide as its blade. |
| **B1.4** | Stacked, level-crushed gaussians | Real glow falls off roughly exponentially, which a single gaussian does not do. Three blurs at 0.5x/1x/2x the blade width, each crushed and weighted, approximate it cheaply. |
| **B1.5** | Darken the plate before adding colour | John Knoll's note on the prequel sabers: pull the underlying plate *down* inside the blade region before adding the colour on top, or bright backgrounds wash the colour out to white. Implemented as a feathered, dilated darkening pass. |
| **B1.6** | Temporal motion trail | `trail = max(trail * decay, glow)` carried frame to frame. A fast swing leaves a persistence streak; a still blade leaves none. Decay runs even on frames with no mask, so a trail fades out rather than freezing when tracking drops. |
| **B1.7** | Directional motion blur | A linear blur along the per-frame velocity vector, length scaled by tip speed. Distinct from B1.6: the trail is history, this is intra-frame smear. |
| **B1.8** | Chromatic bloom and flicker | Slightly different blur radii per channel, so the glow's outer edge separates into colour the way a real lens does, plus a small seeded per-frame intensity flicker so the blade is never perfectly static. |
| **B1.9** | Light wrap | A cheap approximation of the blade's light bending around foreground edges. Lowest-priority element; kept subtle. |
| **B1.10** | Lossless PNG sequence | The glow stage writes numbered PNGs and the encode stage reads them. Writing a lossy intermediate and then encoding it again compounds artifacts for no gain. |

## B2 — the audio synthesizer

Every sound is generated from scratch, so nothing sampled or copyrighted
ends up in the output. The reference point is the documented construction
of the original sound rather than an imitation of any recording.

| | Layer | Why |
|---|---|---|
| **B2.1** | Cumulative-phase oscillators | Phase is accumulated (`2*pi*cumsum(f_t)/sr`) rather than evaluated as `sin(2*pi*f*t)`, so frequency can change sample to sample without a discontinuity. Continuous Doppler modulation is impossible without this. |
| **B2.2** | The hum as two detuned oscillators | Ben Burtt's hum came from an idling film-projector interlock motor — steady but never quite constant. Two oscillators a few Hz apart beat against each other, and a slow low-passed drift wanders the pitch, giving "steady yet unstable" without an explicit tremolo. |
| **B2.3** | A television-interference buzz layer | The other half of Burtt's original: the electromagnetic buzz picked up from a TV picture tube. Bandpass noise around 2–4 kHz. Without it the hum is a clean synth tone rather than a saber. |
| **B2.4** | Swings as pitched-up hum, not whooshes | Burtt recorded swings by moving a microphone past a speaker playing the hum — so a swing *is* the hum, Doppler-shifted, not a separate noise burst. Two registers (low and high) are cross-faded by swing strength, the same structure ProffieOS calls "SmoothSwing". |
| **B2.5** | Continuous Doppler | A frequency multiplier driven by instantaneous tip and angular speed from the tracking data, applied per sample. Continuous, so there is no discrete "swing triggered" moment. |
| **B2.6** | Ignition and power-down as filter/pitch sweeps | Both are the hum under a swept pitch and a swept resonant low-pass, up for ignition and down for retraction, rather than separate one-shot samples. |
| **B2.7** | Gain-staging and a soft limiter | Layers are mixed at deliberate levels and the sum passes through a `tanh` limiter with a 0.95 ceiling. A hard peak-normalize divide would let one transient set the level for the whole clip. |
| **B2.8** | Stereo panning from the blade's position | The blade's horizontal position across the frame, normalized to the clip's own observed range, drives an equal-power pan. |
| **B2.9** | Voices | `neutral`, `jedi`, `sith` change base pitch, detune and waveshaper drive. Deliberately independent of `--color`, so choosing red does not silently change the soundtrack. |

## The swing loudness bug (B-fix)

Worth recording because the mechanism is not obvious and the same mistake
is easy to make again.

B2's swing structure was right, but on real footage a swing was inaudible:
idle-to-swing RMS contrast measured 1.19x, so the swing — which should be
the loudest sustained thing in a clip — sat at about half the ignition's
level.

The cause was not the swing-strength signal (it already spanned 0.00–1.00
on real motion data) and not the limiter (the mix peaked at 0.34 against a
0.95 ceiling, so it was doing nothing). It was the crossfade. `idle` and
`swing_tone` are both full-strength hum textures of near-identical RMS, so
a **linear** crossfade between them changes timbre far more than level —
and because the two are decorrelated, at the midpoint their sum is about
0.71x either endpoint. The blend was *losing* power in the middle while a
`* (1 + 0.35 * swing)` coefficient tried to add some back.

The fix is an equal-power (quarter-cosine) crossfade law, a much larger
explicit loudness boost on top of it, swing-scaled waveshaper drive, and a
modest swing-scaled boost to the buzz layer. Contrast went to 2.85x, and
the swing now exceeds the ignition transient instead of being half of it.

Two measurement lessons came out of it:

- **A magnitude-weighted spectral centroid weights by FFT bin count**, so a
  broadband layer dominates it regardless of level. The buzz layer made the
  centroid report that swings got *darker*, both before and after the fix.
  Power-weighted (mag²) is the meaningful version.
- **Even the power-weighted centroid is the wrong statistic here.** A swing
  does gain real high-frequency content in absolute terms — swing-vs-idle
  band *energy* went from 1.35x to 2.47x at 1.2–4 kHz and from 1.63x to
  3.08x at 4–12 kHz — but its boosted fundamental grows faster still, so
  every centroid measure falls. A swing reads as louder and fuller, not
  thinner and sharper. That is the intended character, but it means "did it
  get brighter" has to be asked as absolute band energy, and the answer
  depends on whether you quote energy or amplitude (the same measurements
  are 1.16x → 1.57x and 1.28x → 1.75x in amplitude).

## Two storage trade-offs (W1, W2)

Both follow from one measurement: **tracking dominates runtime, and nothing
a user typically wants to change can affect the mask.**

**W1 — masks are stored compressed.** A tracked blade mask is a thin,
mostly-empty band, on the order of 2% foreground pixels. Measured on a
representative mask:

| Resolution | `np.save` | `np.packbits` | `np.savez_compressed` |
|---|---|---|---|
| 720p | 0.922 MB | 0.115 MB (8x) | **0.003 MB (356x)** |
| 1080p | 2.074 MB | 0.259 MB (8x) | **0.004 MB (467x)** |

Bit-packing only ever buys the fixed 8x; compression exploits the sparsity.
That is what makes "always keep the masks" affordable rather than a disk
hazard. Loading still accepts the older uncompressed `.npy` format, so job
directories created before the change remain usable.

**W2 — masks are cached, frames are re-extracted.** Once masks are tiny,
the extracted `frames/` become the dominant disk cost. Comparing cost to
store against cost to recompute settles which to keep:

| Artifact | Cost to recompute | Cost to store |
|---|---|---|
| masks | expensive — minutes of SAM2 tracking | trivial (~10 MB at 1080p/60s) |
| frames | cheap — seconds of extraction | large (~900 MB at 1080p/60s) |

So `rerender` keeps the masks and `motion.npz` and re-extracts the frames
from the source clip. The cost of that choice is a real failure mode: a job
stops being re-renderable if its source clip is moved or deleted, which is
why the source path is recorded per job (`pipeline/job_meta.py`) and why
`lightsaber-fx jobs` reports *why* a job cannot be reused.

`motion.npz` is not recomputed either, and not because it is slow — it is
cheap. It depends only on the masks, and no `rerender` parameter can change
a mask, so the file already in the job directory is exactly what a fresh
computation would produce.

## References

- Ben Burtt on building the saber hum from a projector motor and a
  television's picture-tube buzz, and on recording swings by moving a
  microphone past a speaker — discussed across the *Star Wars* sound
  documentaries and in interviews collected in *Sound Design* (David
  Sonnenschein) and Michael Coleman's *Soundworks Collection* interviews.
- John Knoll on darkening the plate inside the blade region before adding
  colour, from ILM's prequel-era saber compositing.
- [ProffieOS](https://github.com/profezzorn/ProffieOS) — the "SmoothSwing
  V2" low/high hum cross-fade, the closest documented description of
  swing-as-modulated-hum in an open codebase.
- [SAM2](https://github.com/facebookresearch/sam2) — the tracker that makes
  the single-click workflow possible.
