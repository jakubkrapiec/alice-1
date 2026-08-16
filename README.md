# CRF→VMAF Predictor

**Predict the CRF that hits a target VMAF — before you encode.**

A single LightGBM model that, given cheap content features of a 10-second video
segment **plus an optional 2-second probe encode** (v2.0, recommended), a target resolution,
a codec and a desired VMAF score, predicts the CRF value that will produce that
quality. Covers **x264, x265, VP9 and AV1 (SVT-AV1)**
at **720p, 1080p, 1440p and 2160p**, for target **VMAF 60–95**.

Trained on **1.73 M samples derived from 53.5k measured CRF→VMAF curves** across
**18,098 unique source videos** (CC0/CC-BY stock footage and standard test
sequences). Validated end-to-end on real encodes of 8 unseen videos.

---

## Table of contents

- [What the model does](#what-the-model-does)
- [Quick start](#quick-start)
- [Input features](#input-features)
- [Critical: encoder settings the model assumes](#critical-encoder-settings-the-model-assumes)
- [Calibrating to your encoder settings](#calibrating-to-your-encoder-settings-recommended)
- [Training data](#training-data)
- [Evaluation](#evaluation)
- [Strengths](#strengths)
- [Limitations & known failure modes](#limitations--known-failure-modes)
- [Package contents](#package-contents)
- [How the model was built (pipeline)](#how-the-model-was-built-pipeline)
- [Retraining / drift](#retraining--drift)
- [Version history](#version-history)
- [Data sources & credits](#data-sources--credits)
- [License](#license)

---

## What the model does

Per-title and per-scene encoding pipelines need an answer to:

> *"For this chunk of content, at this resolution, with this codec — which CRF
> gives me VMAF ≈ X?"*

Answering it by brute force means encoding each segment at several CRFs and
measuring VMAF (expensive). This model answers it in **milliseconds** from
lightweight content features that cost **one decode + one filter pass** to compute
(no encoding required).

- **Input:** 20 features — content statistics (SI/TI/motion), fps, source &
  target resolution, codec, target VMAF, **and 4 probe-encode features**
  (`probe_vmaf`, `probe_vmaf2`, `probe_slope`, `probe_log_br`) measured from a
  2 s probe encode at the target resolution (two fixed CRFs per codec); recommended but optional — `predict.py --no-probe` fills neutral defaults at v1.x-level accuracy.
- **Output:** predicted CRF (float; round + clamp to the codec's legal range
  before encoding), plus an 80% prediction interval (q10–q90) from the
  bundled quantile models.
- **Model:** LightGBM GBDT quantile family (v2.0.0): point prediction =
  median (q50), 1,266 trees, 511 leaves, text format (`model.txt`, 59.6 MB);
  interval bounds in `model_q10.txt` / `model_q90.txt`. LightGBM 4.7.
  Test **VMAF MAE 1.35** (was 3.67 in v1.x — a 2.6× improvement brought by
  the probe features); CRF MAE 0.62.

Typical use in a per-title ladder generator or a per-scene constrained-quality
encoder controller.

## Quick start

Requirements: Python 3.10+, `lightgbm`, `numpy`, `pandas`, an `ffmpeg`
binary with `libvmaf` (`vmafmotion` filter) on `$PATH`, **and the `vmaf` CLI
on `$PATH`** (needed for the v2.0 probe encode; override with the `VMAF` env var).

```bash
pip install -r requirements.txt

# Predict CRF for a target quality:
python predict.py input.mp4 --codec av1 --target-height 2160 --target-vmaf 90
# -> e.g. "predicted CRF: 41  (raw 40.8)"
#         "80% interval (q10-q90): CRF 36..47  (raw 35.60..47.12)"
# (point prediction only: --no-interval)
```

Minimal programmatic use:

```python
import lightgbm as lgb
import pandas as pd
from predict import extract_features, build_feature_row, FEATURES, CODEC_CATS, CRF_RANGE

model = lgb.Booster(model_file="model.txt")
# analyzes the whole file; pass start=/duration= to analyze one segment
feats, meta = extract_features("input.mp4", width=3840, height=2160)
row = build_feature_row(feats, meta, codec="av1", target_vmaf=90,
                        target_width=3840, target_height=2160)
df = pd.DataFrame([row])
df["codec"] = pd.Categorical(df["codec"], categories=CODEC_CATS)  # order matters!
crf = float(model.predict(df[FEATURES])[0])
lo, hi = CRF_RANGE["av1"]
crf = int(round(min(max(crf, lo), hi)))
```

> **Categorical feature gotcha:** the model was trained with `codec` as a pandas
> categorical with alphabetical categories `["av1", "vp9", "x264", "x265"]`.
> At inference you must construct the categorical the same way (as in the
> example). Passing raw integer codes raises
> `train and valid dataset categorical_feature do not match`.

## Input features

All content features are computed **on the target-resolution segment** (after
bicubic scaling), over ~10 s of content. This mirrors training exactly —
features computed any other way (full video, source resolution, different
scaler) shift the distribution and degrade accuracy.

| # | Feature | Description |
|---|---------|-------------|
| 1 | `si_mean` | Mean per-frame spatial information: luma-plane mean of the Sobel edge filter (ffmpeg `sobel` + `signalstats` `YAVG`). Proxy for spatial complexity/detail. |
| 2 | `si_std` | Per-frame std-dev of the above (within-segment spatial variability). |
| 3 | `ti_mean` | Mean per-frame temporal information: luma mean of the frame-difference signal (ffmpeg `tblend=all_mode=difference` + `signalstats` `YAVG`). Proxy for motion amount. |
| 4 | `ti_std` | Per-frame std-dev of TI. |
| 5 | `vmafmotion` | Mean of libvmaf's `motion` feature (SAD-based) over the segment (`vmafmotion` filter). |
| 6 | `fps` | Frames per second of the source. |
| 7 | `target_height` | Output height in pixels (720/1080/1440/2160). |
| 8 | `target_width` | Output width in pixels (1280/1920/2560/3840). |
| 9 | `source_height` | Source height in pixels. |
| 10 | `codec` | Categorical: `av1`, `vp9`, `x264`, `x265`. |
| 11 | `target_vmaf` | Desired VMAF score (60–95). |
| 12–16 | derived | `si_x_ti=si_mean·ti_mean`, `motion_x_ti=vmafmotion·ti_mean`, `si_cv=si_std/(si_mean+1e-6)`, `ti_cv=ti_std/(ti_mean+1e-6)`, `res_ratio=source_height/target_height` |

*16 features since v1.1 — five were pruned after ablation showed zero
information loss (`segment_duration`, `log_si`/`log_ti`/`log_motion`,
`pixels`); see [Version history](#version-history).*

`predict.py` computes all of them for you.

## Video length

The model was trained on ~10 s segments, but it **works for videos of any
length ≥ ~3 s**:

- The content features are temporal aggregates (mean/std of per-frame
  statistics), so they do not depend on how long the analyzed window is.
  By default `predict.py` analyzes the **whole file**; use `--start` /
  `--duration` to analyze a single segment instead (per-scene use case).
- The model has **no duration input at all** — `segment_duration` was pruned
  in v1.1 after a 150k-row ablation (freezing real durations 2–13 s to a
  constant shifted predictions by 0.07 CRF on average, 93% identical after
  rounding) and a full retrain confirmed zero information loss.
- Windows **< 3 s** produce noisy frame statistics; `predict.py` prints a
  warning and you should treat the result as rough guidance only.

## Critical: encoder settings the model assumes

A CRF value is only meaningful **relative to a specific encoder, preset and
speed configuration**. The model was trained on — and therefore predicts for —
exactly these settings:

| Codec | Encoder & settings | CRF range |
|-------|--------------------|-----------|
| `x264` | `libx264 -preset veryfast` | 0–51 |
| `x265` | `libx265 -preset veryfast` | 0–51 |
| `vp9` | `libvpx-vp9 -b:v 0 -cpu-used 6 -row-mt 1 -deadline good` | 0–63 |
| `av1` | `libsvtav1 -preset 10` (`-preset 12` at 2160p) | 0–63 |

Scaling before encode: `scale=W:H:force_original_aspect_ratio=decrease` +
pad to WxH, `format=yuv420p`. Training used ffmpeg 6.1.1 (Ubuntu 24.04).

If your pipeline uses different presets/speeds (e.g. x264 `medium`, SVT-AV1
`preset 6`), the CRF→VMAF mapping shifts and predictions will be off — treat the
model as a prior and calibrate with a few probe encodes, or retrain.

**AV1 preset note.** Training encodes used SVT-AV1 `-preset 10` at 720p–1440p
and the faster `-preset 12` at 2160p. The preset regime is fully determined by
(codec, target_height), so the model separates it internally, and the 2160p AV1
labels were genuinely measured under preset 12. But if *your* 2160p AV1 encode
uses a slower preset (e.g. 6–10), expect actual VMAF to land roughly 3–5 points
**above** target (preset 12 gives ~3–5 VMAF less per CRF than preset 10); a
faster preset than 10 at ≤1440p shifts the result the other way.

**VMAF models:** labels ≤1080p use `vmaf_v0.6.1`; 1440p/2160p use
`vmaf_4k_v0.6.1`. VMAF scores from the 4K model are not on exactly the same
perceptual anchor as the 1080p model — keep this in mind when comparing targets
across resolutions.

## Calibrating to your encoder settings (recommended)

If your encoder configuration differs from the training settings above — a
different preset, speed flags, or an encoder-version upgrade — run a one-time
**probe calibration**. A preset/version change shifts the CRF→VMAF curve
roughly horizontally, so a single measured offset removes most of the
systematic error while the model keeps doing the content-dependent work.

*(For a plain preset change, the measured `--preset` offsets below need no
probes at all — calibration is for everything beyond that.)*

```bash
# 1. one-time: >= 20 diverse probe videos, ~3 encodes each (takes minutes)
python3 calibrate.py --model model.txt --codec x264 \
    --encoder-args "-c:v libx264 -preset medium" \
    --target-vmaf 85 --height 1080 \
    --videos probes/*.mp4 --output calibration.json

# 2. then just pass the calibration file to predict.py
python3 predict.py input.mp4 --codec x264 --target-height 1080 \
    --target-vmaf 85 --calibration calibration.json
```

How it works: for each probe video the tool predicts the CRF, encodes a 10 s
window at three ladder points around the prediction using **your**
`--encoder-args`, measures VMAF against the scaled source (same scaling and
VMAF model as training), inverts at the target, and stores the **median
delta** between actual and predicted CRF. `predict.py` then applies the
offset (`CRF_final = round(prediction + delta)`).

Validation of the tool itself (this repo's validation videos, x264 @720p,
target 85): calibrating against the *training* preset yields delta ≈ −0.1
(zero, as expected); switching to `-preset medium` yields delta ≈ +3.0.

Notes:

- Use ≥ 20 diverse probe videos for a stable median; the tool prints the IQR
  so you can judge the spread (content-dependence of the shift is small but
  non-zero — about ±0.5 CRF in our tests).
- Calibrate per codec × resolution you deploy. One file can hold multiple
  entries; `calibrate.py` replaces entries keyed by codec+height+target.
- A calibration measured at a nearby `target_vmaf` is reused automatically
  (preset shifts are roughly constant across targets).
- `calibrate.py` needs the `vmaf` CLI (libvmaf command-line tools) on PATH —
  the same tool used by the training pipeline. Binary names can be overridden
  with the `FFMPEG` / `FFPROBE` / `VMAF` environment variables.

## Encoding with a non-baseline preset (`--preset`)

The model predicts the CRF for the **training-baseline preset** (x264/x265
`veryfast`, vp9 `cpu-used 6`, av1 `p10` (`p12` at 2160p)). If you encode
with a different measured preset, pass `--preset` and `predict.py` applies the
offset measured on the Track 2 preset-delta dataset (5,471 real encodes,
27 clips):

```bash
python3 predict.py input.mp4 --codec x265 --target-height 1080 \
    --target-vmaf 90 --preset medium        # +1.25 CRF vs veryfast
```

Available presets — x264/x265: `veryfast` `fast` `medium` `slow`; vp9: `cu6`
`cu4` `cu2`; av1: `p10` `p12` `p8` `p6` (bare numbers work too: `--preset 4`
= `cu4` for vp9, `--preset 8` = `p8` for av1). Slower presets shift the CRF
up (a more efficient encoder reaches the same VMAF at a higher CRF); the
offset shifts the q10–q90 interval as well.

Accuracy of the offsets (leave-one-clip-out validation): median offset error
**0.44 CRF** (MAE 0.73) vs 2.25 when the preset is ignored entirely. Tightest
for vp9/x265 (≤0.5), widest for av1 (up to ~2 at 1440p/2160p — the printed
IQR tells you how much to trust each cell).

Precedence: `--calibration` / `--crf-offset` **replace** the preset offset —
a calibration measured with your exact encoder settings already contains the
preset effect.

## Training data

- **18,098 unique source videos** (full list: `training_videos.txt`) —
  Pixabay (dominant), Mixkit, Pexels, Internet Archive, Wikimedia Commons,
  Xiph (derf / aomctc / extra), Blender Foundation. SDR, 8-bit, 720p–4K,
  ~6–60 s clips, wide content mix (nature, city, people, sports, CGI, screen
  content, film grain).
- **Two measurement datasets:**
  - *main:* 38,151 segments @720p, x264 only, jittered 6-point CRF ladder
    (anchors 18/43 + 4 random interior points per segment, deterministic per
    segment) → dense real coverage of CRF 18–43;
  - *multicodec:* 3,891 (segment × resolution) rows, 4 codecs, fixed 5-point
    ladders: x264/x265 CRF {18,23,28,33/34,40}, vp9/av1 CRF {30,40,50,58,63}.
- **Related dataset (not used in training):** *preset_delta* v1.0 (2026-08-12) — 5,471 rows: 27 validation clips × 4 codecs × full preset ladders × 4 resolutions, measured VMAF + bitrate + encode time per cell; `s3://kubakra-videos/ml-crf-vmaf/training_data/preset_delta_v1.0/`.
- **Labels:** per (segment × codec × resolution), a decreasing logistic curve
  is fit to the 5–6 measured (CRF, VMAF) points (fallback: linear interp;
  99.8% of fits logistic, mean fit RMSE ≈ 0.55 VMAF). The curve is inverted
  analytically for each integer target VMAF 60–95. **Labels are emitted only
  when the inverted CRF lies inside the measured ladder** — no extrapolated
  labels (this was a hard-won fix: an earlier iteration silently dropped
  vp9/AV1 labels requiring CRF > 52, which biased those codecs by ~+4 VMAF).
- **Table:** 1,731,985 rows — 1,427,912 x264 / 109,343 x265 / 106,515 vp9 /
  88,215 av1; 1,435,530 @720p / 134,702 @1080p / 97,005 @1440p / 64,748 @2160p.
- **Split:** by *source video* (never by row): train 80% / val 10% / test 10%
  (deterministic MD5 hash of `source_key`) — no content leakage between splits.
- **Model selection:** 6-config hyperparameter sweep (best: 511 leaves,
  lr 0.03, feature_fraction 0.8, L2 1.0, early stopping — 154 rounds in
  v1.0, 137 in v1.1 after feature pruning).
  A Bilibili-style two-stage correction model was trained and **rejected**:
  it found no learnable structure in stage-1 residuals (early-stopped after
  1 iteration).

## Evaluation

### Label-space (held-out test sources, 180,610 rows)

Predicted CRF is mapped back to VMAF via the segment's fitted curve and
compared with the target:

| Metric | Value |
|--------|-------|
| VMAF MAE | **3.60** |
| within ±1 VMAF | 25.1% |
| within ±2 VMAF | 42.2% |
| within ±5 VMAF | 75.1% |

*(v1.1, 16-feature model: VMAF MAE 3.65 on the same test split —
statistically equivalent to the 21-feature v1.0. A v1.3 retrain on an
extended table (v3 + hardcontent) scored 3.67 — no improvement, rejected.
**v1.5.0:** the median (q50) model scores **3.52** on the same split (±2 hit
43.4% vs 41.5%), beating the L2 model on every codec slice — `metrics.json` →
`v1.5_quantile_switch`; see [Version history](#version-history).)*

### End-to-end validation (real encodes, 8 unseen videos)

The honest test: predict CRF → actually encode → measure VMAF → compare with
target. 8 videos (5× 4K Pixabay, 1× 4K Pexels, 2× 1080p Mixkit — CGI,
underwater, nature, sports, urban, screen content), 28 (video × resolution)
units × 4 codecs × target VMAFs {75, 85, 95} (`validation_videos.txt`,
`metrics.json`).

**v1.5.0 head-to-head** — every cell encoded twice: at the v1.5.0 q50
prediction and at the previous L2 model's prediction (**672 real encodes**,
307 paired cells with both measurements valid):

| | MAE | bias | ±2 | ±5 |
|---|---|---|---|---|
| **v1.5.0 q50 — overall** | **4.59** | **+0.15** | 34% | 66% |
| v1.4.0 L2 — overall | 5.15 | +0.74 | 32% | 60% |
| q50 — target 75 | 6.13 | +1.56 | 25% | 53% |
| q50 — target 85 | 4.79 | −0.25 | 32% | 58% |
| q50 — target 95 | 2.79 | −0.91 | 44% | 89% |

Paired cells: q50 MAE 4.57 vs L2 5.12 (**−10.7%**); q50 wins 133, L2 wins 70,
ties 104. q50 is better on every codec (MAE, q50 vs L2: x264 4.31 vs 4.95,
x265 4.25 vs 4.67, vp9 4.38 vs 4.92, av1 5.40 vs 6.04) and every resolution.

Per codec (q50 bias): x264 +0.79, x265 +0.58, vp9 +0.56, av1 −1.28.
Per resolution (q50 bias): 720p −0.87, 1080p −0.88, 1440p +0.99, 2160p +2.86.

The q10–q90 interval averages ~7.8 CRF wide on validation content; the point
prediction lands inside it for 91% of cells (nominal coverage 80% — treat as
approximate on av1/vp9, where training data is thin).

Positive bias = encoded quality came out *higher* than requested (bitrate
"wasted" on over-quality); negative = undershoot.

## Strengths

- **Well-calibrated at practically relevant targets.** For target VMAF ≥ 85
  the model is essentially unbiased (target 85: +0.03). High-target accuracy is
  the best regime (target 95: MAE 2.98).
- **No systematic codec bias.** All four codecs are within ±1.4 VMAF bias on
  real encodes (vs +3.8/+4.3 for vp9/AV1 before the label fix).
- **Cheap features.** One decode + one filter pass per segment; no GPU; the
  model itself is milliseconds.
- **Broad coverage.** 4 codecs × 4 resolutions × VMAF 60–95, trained on
  ~18k diverse sources with real (not synthetic) CRF→VMAF measurements.
- **Source-disjoint evaluation.** Both the test split and the E2E validation
  use videos the model never saw.

## Limitations & known failure modes

- **Low targets (≤ 80) are the weak regime.** MAE ~7 at target 75. Partly
  *irreducible*: for easy content even the maximum CRF yields VMAF ≈ 80+, so
  "VMAF 75" cannot be reached within the legal CRF range at all — the model
  correctly saturates at max CRF, but the error is still counted. If you need
  VMAF ≤ 75, expect best-effort behaviour rather than accuracy.
- **CGI / synthetic / very smooth content** produces the largest errors
  (worst E2E case: a CGI tunnel loop, mean |err| ≈ 10.9). Highly unusual
  content (heavy grain, very dark scenes) was rare in training.
- **Analysis window.** Content features are aggregates over the analyzed
  window (whole file by default). The model knows nothing about scene
  changes *within* the window — for long, mixed-content videos, per-scene
  prediction (`--start`/`--duration` per shot) is more accurate than one
  whole-file prediction.
- **Encoder-version sensitivity.** Predictions assume the exact encoder
  builds/presets above (see [Critical](#critical-encoder-settings-the-model-assumes)).
  Encoder upgrades (especially SVT-AV1) can silently shift the CRF→VMAF
  mapping — see [Retraining / drift](#retraining--drift).
- **VMAF itself is a noisy label** (it is an SVM model; ±0.5 noise floor), and
  1440p/2160p labels use the 4K VMAF model — scores are not perfectly
  cross-resolution comparable.
- **CRF ladder coverage censors the label space.** A label exists only where
  the measured CRF ladder brackets the target (39k / 2.2% of attempted labels
  dropped as out-of-ladder). Share of source videos that have *any* label at
  the extreme targets — target 95: x264 38%, av1 64%, vp9 83%, x265 93%;
  target 60: av1 39%, x265 66%, vp9 70%, x264 99%. Practical meaning:
  x264@veryfast usually needs CRF < 18 for VMAF 95 (hard-content @95 is thin),
  and AV1 at max CRF 63 still beats VMAF 60 on easy content (easy-content @60
  is thin). Predictions near these edges are best-effort.
- **Small E2E sample.** The end-to-end numbers come from 8 videos; treat them
  as indicative, not as tight confidence intervals.
- **SDR 8-bit only.** No HDR, no 10-bit, no film-grain synthesis flags in
  training encodes.
- **Accuracy ceiling.** With aggregate SI/TI/motion features, residual error
  is dominated by content variation the features cannot see. Richer features
  (VIF/ADM descriptors, SI/TI percentiles) are the known next lever.

## Reviewed and rejected (v1.1.1)

Two changes suggested during external review were implemented and measured on
the fixed test split (same data/params/seed as the shipped model):

- **Per-slice sample weighting** (weights inversely proportional to
  codec × resolution frequency): global VMAF MAE 3.65 → 3.91, x264 bias
  −0.03 → −0.80; only the smallest slice (x265) improved (+0.97 → +0.21).
  Rejected: it trades accuracy on 82% of production traffic for a marginal gain
  on one minority slice. A gentler sqrt-weighting variant landed between the
  two (MAE 3.79) and was rejected on the same grounds.
- **Aspect-ratio / padding-fraction feature** (`pad_fraction` computed from
  source and target geometry): VMAF MAE 3.6524 → 3.6457 (−0.2%), within noise;
  per-codec biases unchanged or marginally worse. Rejected for this corpus
  (predominantly ~16:9; mean pad fraction 4.2%). If your inputs are vertical or
  letterboxed at scale, this is the first feature to re-add — `source_width` is
  already present in the training schema.

Also verified in response to review: the residual AV1 bias is **not** explained
by the 2160p preset switch — AV1 bias is *largest* at 720p (+2.3, preset 10)
and smaller at 2160p (+1.6, preset 12), the opposite of what the preset story
predicts. It tracks data thinness (~594 sources per AV1 slice vs 18k for
x264@720p), not the preset boundary.

## Package contents

| File | Description |
|------|-------------|
| `model.txt` | LightGBM quantile-median (q50) model (text format), 377 trees — the predictor. |
| `model_q10.txt` | Quantile q10 model — lower bound of the 80% prediction interval. |
| `model_q90.txt` | Quantile q90 model — upper bound of the 80% prediction interval. |
| `README.md` | This document. |
| `predict.py` | Self-contained inference example: feature extraction (ffmpeg) + prediction, CLI included. |
| `calibrate.py` | Probe-based calibration for custom encoder presets/builds; writes `calibration.json` consumed by `predict.py --calibration`. |
| `preset_offsets.json` | Measured CRF offsets per codec × preset × resolution (from the Track 2 preset-delta dataset); consumed by `predict.py --preset`. |
| `features.json` | Machine-readable feature spec: order, derived formulas, codec categories, CRF ranges. |
| `training_videos.txt` | All 18,098 source videos used for training (one `source_key` per line). |
| `validation_videos.txt` | The 8 videos used for end-to-end validation (disjoint from training). |
| `metrics.json` | Full evaluation results (label-space test + E2E validation). |
| `metadata.json` | Version, build date, checksums, library versions. |
| `requirements.txt` | Python dependencies for `predict.py`. |
| `LICENSE` | License of this package. |

## How the model was built (pipeline)

1. **Corpus:** ~21.7k CC-licensed videos collected from Pixabay, Mixkit,
   Pexels, Internet Archive, Wikimedia Commons, Xiph, Blender. 18,098 of them
   produced valid training data; the rest were dropped by deduplication on
   source key, corrupt/unreadable files, and duration/probe/encode failures.
2. **Measure:** for each 10 s segment and each codec × resolution, encode at a
   5–6 point CRF ladder and measure VMAF against the scaled source
   (streaming VMAF through FIFO pipes; ~78k encodes total across datasets).
3. **Features:** SI/TI (Sobel / frame-diff signalstats) + libvmaf motion on the
   scaled segment in a single ffmpeg pass.
4. **Labels:** logistic CRF→VMAF curve per segment; invert for integer targets
   60–95; keep only in-ladder inversions.
5. **Train:** LightGBM, 21 features, source-disjoint 80/10/10 split,
   config sweep, early stopping.
6. **Validate:** label-space metrics + full end-to-end encode-and-measure
   validation on unseen videos (this step caught the label-generation bug
   fixed in v3).

## Retraining / drift

Encoder libraries change their CRF semantics over time. Recommended practice:

- For a small sample of production encodes, measure the actual VMAF and compare
  with the target the model was asked for. A growing systematic bias = model
  drift → retrain (at least the affected codec).
- Content mix shifts (new platforms, HDR, new genres) also justify retraining.
- The whole pipeline is deterministic and re-runnable; the expensive part is
  re-measuring the CRF ladders (CPU-bound encoding).

## Version history

| Version | Changes |
|---------|---------|
| 1.0.0 | Initial release: 21 features, 154 trees. |
| 1.0.1 | `predict.py`: whole-file analysis by default; `segment_duration` fed as constant 10.0; warning for windows < 3 s. Model unchanged. |
| 1.1.0 | **Feature pruning, model retrained** (137 trees): removed `segment_duration` (near-zero variance in training), `log_si`/`log_ti`/`log_motion` (monotonic transforms — carry no information for tree models) and `pixels` (exact product of two kept features). 21 → 16 features. Label-space test VMAF MAE 3.65 vs 3.60 — statistically equivalent; an aggressive 14-feature variant (also dropping `res_ratio`, `motion_x_ti`) scored identically (3.64), but those features carry genuine information and were kept. E2E numbers above were measured on the 21-feature model and remain the best available estimate. |
| 1.1.1 | Docs-only after external review: AV1 preset 10/12 note, CRF-ladder coverage table, corpus attrition note (21.7k → 18,098), 'Reviewed and rejected' section. Model unchanged. |
| 1.2.0 | `calibrate.py` added: probe-based calibration for custom encoder settings/presets/builds; `predict.py` gains `--calibration` and `--crf-offset`; `FFMPEG`/`FFPROBE`/`VMAF` env overrides. Model unchanged. |
| 1.3.0 | **Model unchanged** — v4 retrain evaluated and **rejected**. The retraining pipeline was rebuilt for reproducibility (v4 table = v3 rows reproduced 1:1 + 14,359 hardcontent label rows from 410 segments × 16 stress-test videos, all 4 codecs) and the retrained model was benchmarked head-to-head against the production model on the identical test split: overall CRF MAE 1.722 vs 1.712, label-space VMAF MAE 3.67 vs 3.65 — no improvement (slightly negative on every non-hardcontent slice), so the production model ships unchanged. Full experiment record in `metrics.json` (`v1.3_rejected_v4_retrain`); pipeline + artifacts archived at `s3://kubakra-videos/ml-crf-vmaf/v4/`. Known gap quantified: the model lands ~5 VMAF below target on hardcontent (bias −5.1) — a dedicated hardcontent measurement dataset is in progress for a future release. |
| 1.4.0 | **Model and code unchanged** — preset_delta (Track 2) measurement dataset released as v1.0: 5,471 rows = 27 validation clips × 4 codecs × full preset ladders (x264/x265 veryfast/fast/medium/slow, vp9 cu2/cu4/cu6, av1 p6/p8/p10/p12) × 4 resolutions, with measured VMAF, bitrate and encode time per cell. `val_coral_b` excluded (reference decode failure); 4 rows with `vmaf=-1` dropped. Dataset + generator + manifest archived at `s3://kubakra-videos/ml-crf-vmaf/training_data/preset_delta_v1.0/`; record in `metrics.json` (`preset_delta_v1.0_dataset`). |
| 1.5.0 | **Quantile model family — model replaced.** Point prediction switched from L2-mean to the **median (q50)**: label-space VMAF MAE 3.52 vs 3.65 (−3.7%, better on every codec slice), E2E head-to-head on the 8 unseen validation videos (672 real encodes): paired MAE 4.57 vs 5.12 (**−10.7%**), bias +0.15 vs +0.74, better on every codec and resolution. New `model_q10.txt`/`model_q90.txt` ship an 80% prediction interval, printed by default (`--no-interval` to suppress); calibration/`--crf-offset` shift the interval too. Track 3 experiments rejected: monotone constraint on `target_vmaf` (+0.7% MAE cost) and per-codec models (av1/x265 regressions) — record in `metrics.json` (`v1.5_quantile_switch`). **Recalibrate** if you have a `calibration.json` from ≤1.4.0 — deltas were measured against the L2 point model. |
| 1.6.0 | **Model unchanged** — new `--preset` flag + `preset_offsets.json`: measured CRF offsets for non-baseline encoder presets from the Track 2 preset_delta v1.0 dataset (44 cells codec×preset×resolution; delta = CRF(preset) − CRF(baseline) at equal VMAF, median over targets 75–95 and clips). Leave-one-clip-out validation: offset residual MAE 0.75 CRF (median 0.46) vs 2.29 when ignoring the preset. Offsets shift the q10–q90 interval too; `--calibration`/`--crf-offset` take precedence. Also: `ffmpeg -nostdin` fix for scripted use. Record in `metrics.json` (`v1.6_preset_offsets`). |

## Data sources & credits

- Video: **Pixabay**, **Mixkit**, **Pexels**, **Internet Archive**,
  **Wikimedia Commons**, **Xiph.Org Foundation** (derf / AOM CTC / extra
  test sequences), **Blender Foundation** (CC-licensed open movies).
- Quality metric: **Netflix VMAF** (libvmaf), models `vmaf_v0.6.1` and
  `vmaf_4k_v0.6.1`.
- Encoders: x264, x265, libvpx, SVT-AV1 via ffmpeg.

Please respect the licenses of the underlying video content if you
redistribute any of it (this package contains only file *names*, no video
data).

## License

See `LICENSE`. The model and code in this package are released under the terms
stated there; video content remains property of its respective sources.
