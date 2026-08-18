# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Alice-1: a LightGBM model + inference scripts that predict the CRF value needed to hit a target VMAF, given cheap content features of a video segment, target resolution, and codec (x264/x265/vp9/av1). This is a model distribution package (published to Hugging Face Hub), not an application — the two scripts (`predict.py`, `calibrate.py`) plus the shipped model files (`model.txt`, `model_q10.txt`, `model_q90.txt`) *are* the product.

No test suite exists in this repo. There is no build step. Correctness is validated via the metrics in `metrics.json` (label-space test + end-to-end real-encode validation), not unit tests.

## Setup

```bash
pip install -r requirements.txt
```

Also requires on `$PATH`: `ffmpeg`/`ffprobe` (with `libvmaf` compiled in — `sobel`, `tblend`, `signalstats`, `vmafmotion` filters) and the `vmaf` CLI (needed for the v2.0 probe-encode features and for `calibrate.py`). Override binary names with the `FFMPEG` / `FFPROBE` / `VMAF` env vars.

## Common commands

```bash
# Predict CRF for a target quality
python predict.py input.mp4 --codec av1 --target-height 2160 --target-vmaf 90

# Skip the probe encode (faster, v1.x-level accuracy)
python predict.py input.mp4 --codec x264 --target-height 1080 --target-vmaf 90 --no-probe

# Analyze one segment instead of the whole file
python predict.py input.mp4 --codec x265 --target-height 1080 --target-vmaf 90 --start 30 --duration 10

# One-time calibration for a non-baseline encoder config (>= 20 probe videos)
python calibrate.py --model model.txt --codec x264 \
    --encoder-args "-c:v libx264 -preset medium" \
    --target-vmaf 85 --height 1080 \
    --videos probes/*.mp4 --output calibration.json

# Use a calibration file
python predict.py input.mp4 --codec x264 --target-height 1080 --target-vmaf 85 --calibration calibration.json

# Use a measured non-baseline preset offset instead (mutually exclusive with --calibration)
python predict.py input.mp4 --codec x265 --target-height 1080 --target-vmaf 90 --preset medium
```

There's no test/lint/CI step to run locally — `.github/workflows/main.yml` only mirrors the repo to the Hugging Face Hub on push to `main`.

## Architecture

**Feature pipeline is the crux of this codebase.** `predict.py` and the training pipeline must compute features *identically* — same ffmpeg filters, same scaling, same feature order and derived formulas — or predictions silently drift off-distribution. `features.json` is the machine-readable source of truth for feature order, derived formulas, categorical encoding, and CRF ranges; keep it and `predict.py`'s `FEATURES`/`build_feature_row` in sync with any model change.

Key invariants when touching feature code:
- Content features (`si_*`, `ti_*`, `vmafmotion`) are computed **after** scaling to the *target* resolution, not source resolution — computing them at the wrong resolution shifts the whole distribution.
- `codec` must be built as a `pandas.Categorical` with the exact alphabetical categories `["av1", "vp9", "x264", "x265"]` (see `CODEC_CATS` in `predict.py`) — raw integer codes raise a LightGBM categorical mismatch error.
- The 4 probe features (`probe_vmaf`, `probe_vmaf2`, `probe_slope`, `probe_log_br`, added in v2.0) come from `run_probe()`: two 2s encodes at fixed per-codec CRFs at the training-baseline preset. `model_needs_probe()` detects whether a given model file expects these (v1.x models don't have them) so old and new model files both work through the same script. `--no-probe` fills neutral defaults (probe_vmaf/probe_vmaf2 = target VMAF, slope 0, training-median bitrate).

**Offset stack** (order of precedence, highest first): `--calibration` / `--crf-offset` > `--preset` offset > raw model prediction. A calibration measured on your exact encoder settings already contains the preset effect, so it always wins over `--preset`. This is implemented via `load_calibration()` and `load_preset_offsets()`/`baseline_preset()`/`normalize_preset()` in `predict.py`, driven by `preset_offsets.json` (measured deltas per codec × preset × resolution) and a user-supplied `calibration.json` (from `calibrate.py`, keyed by codec + height + target VMAF, reused across nearby targets since preset shifts are roughly constant across targets).

**Model files are the training-baseline encoder settings** — the CRF a given model predicts is only meaningful for the exact encoder/preset/speed it was trained on (see the table in README.md's "Encoder settings the model assumes"). Any code change that affects inference must not implicitly assume a different encoder config than what's documented in `features.json`'s `encoder_settings_assumed`.

`calibrate.py` is a separate, self-contained probe-and-measure tool (`calibrate_video()` → `invert_crf()` on a small CRF ladder around the model's prediction) that produces the `calibration.json` consumed by `predict.py --calibration`. It shares CRF range / resolution constants with `predict.py` but does not import from it — keep the two in sync manually if those constants change.

## Versioning and docs

This repo follows semver-ish releases tracked in `CHANGELOG.md`, with `metadata.json` holding version/build-date/checksums and `metrics.json` holding the full evaluation record (label-space + end-to-end) for every shipped version, including rejected retrain experiments. When changing the model or feature set, update all three (`CHANGELOG.md`, `metadata.json`, `metrics.json`) plus `features.json` and the relevant README.md sections — they're treated as a single source of truth for downstream consumers (the package is synced to Hugging Face Hub verbatim on every push to `main`).
