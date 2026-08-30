---
license: mit
library_name: lightgbm
tags:
  - video-encoding
  - vmaf
  - crf
  - lightgbm
  - x264
  - x265
  - vp9
  - av1
  - per-title-encoding
  - video-quality
datasets:
  - jakubkrapiec/crf-vmaf-training-data
metrics:
  - name: VMAF MAE
    type: vmaf_mae
    value: 1.47
  - name: CRF MAE
    type: crf_mae
    value: 0.53
---

# Alice-1: a CRF -> VMAF prediction model

A LightGBM model that predicts the CRF value needed to hit a target VMAF, given cheap content features of a video segment, a target resolution, and a codec. Covers x264, x265, VP9 and AV1 (SVT-AV1) at 720p, 1080p, 1440p and 2160p, for target VMAF 60–95.

Trained on 1.76M samples derived from ~54k measured CRF -> VMAF curves across 18,114 unique source videos (CC0/CC-BY stock footage and standard test sequences).

---

## Table of contents

- [What the model does](#what-the-model-does)
- [Quick start](#quick-start)
- [Input features](#input-features)
- [Video length](#video-length)
- [Encoder settings the model assumes](#encoder-settings-the-model-assumes)
- [Calibrating to your encoder settings](#calibrating-to-your-encoder-settings-strongly-recommended)
- [Encoding with a non-baseline preset](#encoding-with-a-non-baseline-preset---preset)
- [Training data](#training-data)
- [Evaluation](#evaluation)
- [Strengths](#strengths)
- [Limitations & known failure modes](#limitations--known-failure-modes)
- [Package contents](#package-contents)
- [Retraining / drift](#retraining--drift)
- [Data sources & credits](#data-sources--credits)

---

## What the model does

Per-title and per-scene encoding pipelines need to know, for a given chunk of content, resolution and codec, which CRF produces a given VMAF. Solving this by brute force (even by using binary search) means encoding each segment at several CRFs and measuring VMAF, which is expensive. This model answers it in milliseconds from lightweight content features that cost one decode plus one filter pass to compute, with no encoding required.

- **Input:** 20 features - content statistics (SI/TI/motion), fps, source and target resolution, codec, target VMAF, and 4 probe-encode features (`probe_vmaf`, `probe_vmaf2`, `probe_slope`, `probe_log_br`) measured from a 2 s probe encode at the target resolution (v2.0; recommended but optional - `predict.py --no-probe` fills neutral defaults at v1.x-level accuracy).
- **Output:** predicted CRF (float; round and clamp to the codec's legal range before encoding), plus an 80% prediction interval (q10–q90) from the bundled quantile models.
- **Model:** LightGBM GBDT quantile family. Point prediction is the median (q50), 1,266 trees, 511 leaves, text format (`model.txt`, 59.6 MB). Interval bounds live in `model_q10.txt` / `model_q90.txt`. Built with LightGBM 4.7.

Typical use case: a per-title ladder generator or a per-scene constrained-quality encoder controller.

## Quick start

Requirements: Python 3.10+, `lightgbm`, `numpy`, `pandas`, and an `ffmpeg` binary with `libvmaf` (`vmafmotion` filter) on `$PATH`. The v2.0 probe encode additionally needs the `vmaf` CLI on `$PATH` (or pass `--no-probe` to skip the probe).

```bash
pip install -r requirements.txt

# Predict CRF for a target quality:
python predict.py input.mp4 --codec av1 --target-height 2160 --target-vmaf 90
# -> "predicted CRF: 41  (raw 40.8)"
#    "80% interval (q10-q90): CRF 36..47  (raw 35.60..47.12)"
#    "calibrated band (split-conformal, 80% coverage): CRF 36..48"
# (point prediction only: --no-interval; raw uncalibrated band: --no-conformal)

# Machine-readable output:
python predict.py input.mp4 --codec x265 --target-height 1080 --target-vmaf 90 --json

# Batch: several (codec, target) jobs on the same file - features and
# probe encodes are computed once and reused (probe cached on disk):
cat jobs.json
# [{"codec": "x264", "target_vmaf": 90},
#  {"codec": "x265", "target_vmaf": 88},
#  {"codec": "av1",  "target_vmaf": 88}]
python predict.py input.mp4 --target-height 1080 --batch jobs.json --json
```

The two probe encodes run in parallel, and probe results are cached in
`~/.cache/crf-vmaf-predictor/probe_cache.json` (keyed by a content hash of
the file + codec + resolution + analysis start offset), so repeated
predictions on the same video skip the probe entirely. Override with
`--probe-cache PATH` or disable with `--no-probe-cache`.

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
df["codec"] = pd.Categorical(df["codec"], categories=CODEC_CATS)  # order matters
crf = float(model.predict(df[FEATURES])[0])
lo, hi = CRF_RANGE["av1"]
crf = int(round(min(max(crf, lo), hi)))
```

Note on categoricals: the model was trained with `codec` as a pandas categorical with alphabetical categories `["av1", "vp9", "x264", "x265"]`. At inference you must construct the categorical the same way as in the example above. Passing raw integer codes raises `train and valid dataset categorical_feature do not match`.

## Input features

All content features are computed on the target-resolution segment, after bicubic scaling. This mirrors training exactly - features computed any other way (source resolution, a different scaler) shift the distribution and degrade accuracy.

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
| 17–20 | probe (v2.0) | `probe_vmaf`, `probe_vmaf2` = VMAF of two 2 s probe encodes at fixed CRFs (x264/x265: 28/34, vp9/av1: 40/46, training-baseline presets), `probe_slope` = their VMAF slope per CRF point, `probe_log_br` = log1p of the first probe's bitrate (kbps) |

`predict.py` computes all of these for you. The probe features come from
`run_probe()` (two 2 s encodes + VMAF measurements, a few seconds of extra
work); with `--no-probe` they are filled with neutral defaults
(`probe_vmaf` = `probe_vmaf2` = target VMAF, slope 0, training-median
bitrate) and accuracy degrades to v1.x levels. Model files without `probe_*`
features (v1.x) never trigger the probe.

## Video length

The model was trained on ~10s segments but works for videos of any length ≥ ~3s. The content features are temporal aggregates (mean/std of per-frame statistics), so they don't depend on the length of the analyzed window. By default `predict.py` analyzes the whole file; use `--start` / `--duration` to analyze a single segment instead for per-scene use. Windows under 3s produce noisy frame statistics - `predict.py` prints a warning and the result should be treated as rough guidance only.

## Encoder settings the model assumes

A CRF value is only meaningful relative to a specific encoder, preset and speed configuration. The model was trained on, and therefore predicts for, exactly these settings:

| Codec | Encoder & settings | CRF range |
|-------|--------------------|-----------|
| `x264` | `libx264 -preset veryfast` | 0–51 |
| `x265` | `libx265 -preset veryfast` | 0–51 |
| `vp9` | `libvpx-vp9 -b:v 0 -cpu-used 6 -row-mt 1 -deadline good` | 0–63 |
| `av1` | `libsvtav1 -preset 10` (`-preset 12` at 2160p) | 0–63 |

Scaling before encode: `scale=W:H:force_original_aspect_ratio=decrease` + pad to WxH, `format=yuv420p`. Training used ffmpeg 6.1.1 (Ubuntu 24.04).

If your pipeline uses different presets or speeds (e.g. x264 `medium`, SVT-AV1 `preset 6`), the CRF -> VMAF mapping shifts and predictions will be off. Treat the model as a prior and calibrate with a few probe encodes, or retrain.

**AV1 preset note.** Training encodes used SVT-AV1 `-preset 10` at 720p–1440p and the faster `-preset 12` at 2160p. The preset regime is fully determined by (codec, target_height), so the model separates it internally, and the 2160p AV1 labels were measured under preset 12. If your 2160p AV1 encode uses a slower preset (e.g. 6–10), expect actual VMAF to land roughly 3–5 points above target (preset 12 gives ~3–5 VMAF less per CRF than preset 10); a faster preset than 10 at ≤1440p shifts the result the other way.

**VMAF models:** labels <=1080p use `vmaf_v0.6.1`; 1440p/2160p use `vmaf_4k_v0.6.1`. Scores from the 4K model aren't on exactly the same perceptual anchor as the 1080p model - keep this in mind when comparing targets across resolutions.

## Calibrating to your encoder settings (strongly recommended)

If your encoder configuration differs from the training settings above - a different preset, speed flags, or an encoder-version upgrade - run a one-time probe calibration. A preset or version change shifts the CRF -> VMAF curve roughly horizontally, so a single measured offset removes most of the systematic error while the model keeps doing the content-dependent work.

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

For each probe video, the tool predicts the CRF, encodes a 10s window at three ladder points around the prediction using your `--encoder-args`, measures VMAF against the scaled source (same scaling and VMAF model as training), inverts at the target, and stores the median delta between actual and predicted CRF. `predict.py` then applies the offset (`CRF_final = round(prediction + delta)`).


Notes:

- Use at least 20 diverse probe videos for a stable median. The tool prints the IQR so you can judge the spread - content-dependence of the shift is small but non-zero, about ±0.5 CRF in my tests.
- Calibrate per codec * resolution you deploy. One file can hold multiple entries; `calibrate.py` replaces entries keyed by codec+height+target.
- A calibration measured at a nearby `target_vmaf` is reused automatically, since preset shifts are roughly constant across targets.
- `calibrate.py` needs the `vmaf` CLI (libvmaf command-line tools) on PATH, the same tool used by the training pipeline. Binary names can be overridden with the `FFMPEG` / `FFPROBE` / `VMAF` environment variables.

## Encoding with a non-baseline preset (`--preset`)

The model predicts the CRF for the training-baseline preset (x264/x265 `veryfast`, vp9 `cpu-used 6`, av1 `preset 10`, or `preset 12` at 2160p). If you encode with a different preset, pass `--preset` and `predict.py` applies the offset measured on the Track 2 preset-delta dataset (5,471 real encodes, 27 clips):

```bash
python3 predict.py input.mp4 --codec x265 --target-height 1080 \
    --target-vmaf 90 --preset medium        # +1.25 CRF vs veryfast
```

Available presets - x264/x265: `veryfast` `fast` `medium` `slow`; vp9: `cu6` `cu4` `cu2`; av1: `p10` `p12` `p8` `p6` (bare numbers work too: `--preset 4` = `cu4` for vp9, `--preset 8` = `p8` for av1). Slower presets shift the CRF up (a more efficient encoder reaches the same VMAF at a higher CRF); the offset shifts the q10-q90 interval as well.

Accuracy of the offsets (leave-one-clip-out validation): median offset error **0.44 CRF** (MAE 0.73) vs 2.25 when the preset is ignored entirely. Tightest for vp9/x265 (<=0.5), widest for av1 (up to ~2 at 1440p/2160p - the printed IQR tells you how much to trust each cell).

Precedence: `--calibration` / `--crf-offset` **replace** the preset offset - a calibration measured with your exact encoder settings already contains the preset effect.

## Training data

The full training table (1.76M rows) and the probe-encode data (52k rows) are published as a Hugging Face dataset: **[jakubkrapiec/crf-vmaf-training-data](https://huggingface.co/datasets/jakubkrapiec/crf-vmaf-training-data)** (CC0).

- **18,114 unique source videos** (full list: `training_videos.txt`) - Pixabay (dominant), Mixkit, Pexels, Internet Archive, Wikimedia Commons, Xiph (derf / aomctc / extra), Blender Foundation. SDR, 8-bit, 720p–4K, ~6–60s clips, wide content mix (nature, city, people, sports, CGI, screen content, film grain).
- **Two measurement datasets:**
  - *main:* 38,151 segments @720p, x264 only, jittered 6-point CRF ladder (anchors 18/43 + 4 random interior points per segment, deterministic per segment), giving dense real coverage of CRF 18–43.
  - *multicodec:* 3,891 (segment * resolution) rows, 4 codecs, fixed 5-point ladders: x264/x265 CRF {18,23,28,33/34,40}, vp9/av1 CRF {30,40,50,58,63}.
- **Labels:** per (segment * codec * resolution), a decreasing logistic curve is fit to the 5–6 measured (CRF, VMAF) points (fallback: linear interpolation; 99.8% of fits logistic, mean fit RMSE ≈ 0.55 VMAF). The curve is inverted analytically for each integer target VMAF 60–95. Labels are emitted only when the inverted CRF lies inside the measured ladder - extrapolated labels are dropped. An earlier iteration extrapolated past the ladder for vp9/AV1 at CRF > 52, which biased those codecs by roughly +4 VMAF; the current pipeline avoids this.
- **Table:** 1,762,232 rows - 1,444,289 x264 / 123,213 x265 / 106,515 vp9 / 88,215 av1; 1,449,889 @720p / 134,702 @1080p / 106,164 @1440p / 71,477 @2160p.
- **Split:** by source video, never by row - train 80% / val 10% / test 10% (deterministic MD5 hash of `source_key`), so there's no content leakage between splits.
- **Model selection:** 6-config hyperparameter sweep (best: 511 leaves, lr 0.03, feature_fraction 0.8, L2 1.0, early stopping). A two-stage residual-correction model (in the style used by some Bilibili encoding papers) was tried and rejected: it found no learnable structure in the stage-1 residuals and early-stopped after one iteration.

## Evaluation

### Label-space (held-out test sources, 182,940 rows)

Predicted CRF is mapped back to VMAF via the segment's fitted curve and compared with the target. Test split: 10% of the 18,098 source videos, disjoint by source.

| Metric         | Value |
| -------------- | -------------------- |
| VMAF MAE       | 1.47                 |
| CRF MAE        | 0.53                 |
| within ±2 VMAF | 78.0%                |

Per-codec CRF MAE: x264 0.45, x265 0.53, vp9 1.07, av1 1.15. The probe
features cut the label-space VMAF MAE from 3.67 to ~1.4.


#### 80% prediction interval (q10–q90, label-space)

| Metric                 | Value           |
| ---------------------- | --------------- |
| Coverage (nominal 80%) | 69.8% raw, 80.9% calibrated |

`conformal_q10_q90.json` ships split-conformal per-codec corrections,
`predict.py` applies them by default and prints the calibrated band as
`crf_q10_cal`/`crf_q90_cal` in `--json` output. `--no-conformal` restores
the raw band. Per-codec test coverage after calibration: x264 80.3%,
x265 79.0%, vp9 79.1%, av1 85.0%.


### Runtime: probe vs `--no-probe` (predict.py wall time)

Measured on a synthetic 1080p30 clip (ffmpeg `testsrc2`), analyzing a
**10 s segment** (`--start 0 --duration 10`), Google Cloud `n2d-highcpu-16`
(AMD EPYC 7B13, europe-north1), 3 runs per cell, mean wall time, target
VMAF 90:

| Codec | with probe | `--no-probe` | probe overhead |
| ----- | ---------- | ------------ | -------------- |
| x264  | 17.9 s     | 13.9 s       | +4.0 s (+29%)  |
| x265  | 18.3 s     | 14.0 s       | +4.3 s (+31%)  |
| vp9   | 23.2 s     | 13.9 s       | +9.3 s (+67%)  |
| av1   | 19.2 s     | 13.9 s       | +5.3 s (+38%)  |


## Strengths

- Well-calibrated at practically relevant targets: at target VMAF 95 the end-to-end MAE is 1.54 with 97% of encodes within ±5.
- No systematic codec bias. All four codecs are within ±0.5 VMAF bias on real encodes.
- Cheap to run: one decode + one filter pass per segment, no GPU, and the model inference itself takes milliseconds.
- Broad coverage: 4 codecs * 4 resolutions * VMAF 60–95, trained on ~18k diverse sources with real (not synthetic) CRF -> VMAF measurements.
- Source-disjoint evaluation: both the test split and the end-to-end validation use videos the model never saw during training.

## Limitations & known failure modes

- Low targets (≤80) are the weak regime: end-to-end MAE 2.91 at target 75, but 1.88 once physically unreachable labels are excluded. A large share is irreducible: for easy content even the maximum CRF yields VMAF well above 75, so the target cannot be reached within the legal CRF range at all - the model correctly saturates at max CRF (reported by the saturation warning), but the error is still counted. If you need VMAF ≤ 75, expect best-effort behaviour rather than accuracy.
- CGI, synthetic, and very smooth content produce the largest errors (worst end-to-end case: a CGI tunnel loop, mean |err| ≈ 11).
- Content features are aggregated over the analyzed window (the whole file, by default), and the model has no visibility into scene changes within that window. For long, mixed-content videos, per-scene prediction (`--start`/`--duration` per shot) is more accurate than one whole-file prediction.
- Predictions assume the exact encoder builds and presets listed above (see [Encoder settings](#encoder-settings-the-model-assumes)). Encoder upgrades, especially to SVT-AV1, can shift the CR -> VMAF mapping without warning - see [Retraining / drift](#retraining--drift).
- VMAF itself is a noisy label (it's an SVM-based model with roughly ±0.5 noise floor), and 1440p/2160p labels use the 4K VMAF model, so scores aren't perfectly cross-resolution comparable.
- CRF ladder coverage censors the label space: a label exists only where the measured CRF ladder brackets the target. The share of source videos with any label at extreme targets varies a lot by codec. Predictions near the edges are best-effort.
- Training data is SDR, 8-bit only - no HDR, no 10-bit, no film-grain synthesis flags.
- With aggregate SI/TI/motion features, residual error is dominated by content variation the features can't see.

## Package contents

| File | Description |
|------|-------------|
| `model.txt` | LightGBM quantile-median (q50) model (text format), 1,266 trees - the predictor. |
| `model_q10.txt` | Quantile q10 model - lower bound of the 80% prediction interval. |
| `model_q90.txt` | Quantile q90 model - upper bound of the 80% prediction interval. |
| `README.md` | This document. |
| `predict.py` | Self-contained inference example: feature extraction (ffmpeg) + prediction, CLI included. |
| `calibrate.py` | Probe-based calibration for custom encoder presets/builds; writes `calibration.json` consumed by `predict.py --calibration`. |
| `preset_offsets.json` | Measured CRF offsets per codec * preset * resolution (from the Track 2 preset-delta dataset); consumed by `predict.py --preset`. |
| `conformal_q10_q90.json` | Split-conformal per-codec corrections widening the q10-q90 band to nominal 80% coverage; applied by `predict.py` by default (`--no-conformal` to skip). |
| `features.json` | Machine-readable feature spec: order, derived formulas, codec categories, CRF ranges. |
| `training_videos.txt` | All 18,098 source videos used for training (one `source_key` per line). |
| `metrics.json` | Full evaluation results (label-space test + end-to-end validation). |
| `metrics.json` | Latest evaluation results (label-space test, probe-feature metrics, E2E validation). |
| `metadata.json` | Version, build date, checksums, library versions. |
| `requirements.txt` | Python dependencies for `predict.py`. |
| `LICENSE` | License of this package. |
| `CHANGELOG` | The changelog. |

## Retraining / drift

Encoder libraries change their CRF semantics over time. Recommended practice:

- For a small sample of production encodes, measure the actual VMAF and compare it with the target the model was asked for. A growing systematic bias indicates model drift and warrants a retrain, at least for the affected codec.
- Content mix shifts (new platforms, HDR, new genres) also justify retraining.
- The pipeline is deterministic and re-runnable. The expensive part is re-measuring the CRF ladders, which is CPU-bound encoding work.

## Data sources & credits

- Video: Pixabay, Mixkit, Pexels, Internet Archive, Wikimedia Commons, Xiph.Org Foundation (derf / AOM CTC / extra test sequences), Blender Foundation (CC-licensed open movies).
- Quality metric: Netflix VMAF (libvmaf), models `vmaf_v0.6.1` and `vmaf_4k_v0.6.1`.
- Encoders: x264, x265, libvpx, SVT-AV1, via ffmpeg.
