#!/usr/bin/env python3
"""CRF -> VMAF predictor - inference script.

Computes content features for a video (whole file by default, or a chosen
segment) at the target resolution - one ffmpeg pass, same filters as
training - and predicts the CRF that should produce the requested VMAF score.

Works for videos of any length >= ~3 s: the content features are temporal
aggregates (mean/std over frames). A warning is printed
for analysis windows < 3 s, where frame statistics become noisy.

Usage:
    python predict.py input.mp4 --codec x265 --target-height 1080 --target-vmaf 90
    python predict.py input.mp4 --codec av1 --target-height 2160 --target-vmaf 88 \
        --start 30 --duration 10          # optional: analyze one segment only

When model_q10.txt / model_q90.txt are present next to
the model file, an 80% prediction interval (CRF q10..q90) is printed too.
Suppress with --no-interval.

--preset applies a measured CRF offset (preset_offsets.json next to the
model) when you encode with a non-baseline preset; training baselines:
x264/x265 veryfast, vp9 cpu-used 6, av1 p10 (preset 12 at 2160p).
--calibration / --crf-offset take precedence - they already include the
preset effect.

The model can also consume probe-encode features: before
predicting, the script encodes two 2-second probe segments of your video
at the target resolution (fixed CRF per codec, training-baseline preset),
measures their VMAF + bitrate, and feeds the two points + slope to the
model. The probe is recommended but optional - pass --no-probe to skip it;
the probe features are then filled with neutral defaults
(probe_vmaf = probe_vmaf2 = target VMAF, slope 0, training-median bitrate)
and accuracy degrades back towards v1.x levels. v1.x model files (no probe
features) keep working unchanged.

Requires: ffmpeg + ffprobe on PATH (the vmafmotion filter must be compiled
in - it is in stock Ubuntu and ffmpeg.org builds); lightgbm, numpy, pandas.
Binary names can be overridden with the FFMPEG / FFPROBE env vars.
calibrate.py additionally needs the vmaf CLI; so does the v2.0 probe
(VMAF env var to override).
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
FFPROBE = os.environ.get("FFPROBE", "ffprobe")
VMAF_CLI = os.environ.get("VMAF", "vmaf")

# ---------------------------------------------------------------------------
# Model spec (must match training exactly) - 16 features, v1.1 pruned set
# ---------------------------------------------------------------------------
CODEC_CATS = ["av1", "vp9", "x264", "x265"]          # alphabetical == training
CRF_RANGE = {"x264": (0, 51), "x265": (0, 51), "vp9": (0, 63), "av1": (0, 63)}
RES_W = {720: 1280, 1080: 1920, 1440: 2560, 2160: 3840}

MIN_RELIABLE_SECONDS = 3.0

BASE_FEATS = ["si_mean", "si_std", "ti_mean", "ti_std", "vmafmotion",
              "fps", "target_height", "source_height", "target_width",
              "codec", "target_vmaf"]
DERIVED = ["si_x_ti", "motion_x_ti", "si_cv", "ti_cv", "res_ratio"]
FEATURES = BASE_FEATS + DERIVED   # 16 features, exact training order

# v2.0: probe-encode features appended at the end (exact training order:
# 16 base+derived, then probe_vmaf, probe_vmaf2, probe_slope, probe_log_br)
PROBE_KEYS = ["probe_vmaf", "probe_vmaf2", "probe_slope", "probe_log_br"]

# Training-set median of probe_log_br (52,316 rows, probe_data.jsonl) -
# used as the neutral fallback for --no-probe.
PROBE_LOG_BR_MEDIAN = 6.80

# Probe spec - MUST match collector/generate_probe_dataset.py exactly:
# 2 s from the analysis start, target WxH (scale decrease + pad, yuv420p),
# training-baseline presets, VMAF std model <=1080p / 4k model >1080p.
PROBE_SEC = 2.0
PROBE_SPEC = {
    "x264": {"crfs": (28, 34), "args": ["-c:v", "libx264", "-preset", "veryfast"]},
    "x265": {"crfs": (28, 34), "args": ["-c:v", "libx265", "-preset", "veryfast"]},
    "vp9":  {"crfs": (40, 46), "args": ["-c:v", "libvpx-vp9", "-b:v", "0",
                                         "-cpu-used", "6", "-row-mt", "1",
                                         "-deadline", "good"]},
    "av1":  {"crfs": (40, 46), "args": ["-c:v", "libsvtav1", "-preset", "10"],
             "args_4k": ["-c:v", "libsvtav1", "-preset", "12"]},
}


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def probe(video: str) -> dict:
    r = subprocess.run(
        [FFPROBE, "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", video],
        capture_output=True, text=True, timeout=60)
    r.check_returncode()
    return json.loads(r.stdout)


def parse_fps(rate: str) -> float:
    try:
        return float(Fraction(rate))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _ff_path(p) -> str:
    """Escape a filesystem path for embedding inside an ffmpeg filter graph.

    Filter syntax uses ':' as option separator and '\\' as escape char, so a
    raw Windows path (C:\\Users\\..) breaks parsing or corrupts the filename.
    Convert to forward slashes, then escape ':'.  POSIX paths pass through
    unchanged.
    """
    return str(p).replace("\\", "/").replace(":", "\\:")


# ---------------------------------------------------------------------------
# Feature extraction - EXACTLY as in training:
# plain bicubic scale to WxH (no aspect preservation / pad here),
# SI = sobel -> signalstats YAVG, TI = tblend difference -> signalstats YAVG,
# motion = libvmaf vmafmotion stats file.
# ---------------------------------------------------------------------------
def _parse_yavg(path: Path) -> list:
    vals = []
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").split("\n"):
        if "YAVG=" in line:
            try:
                vals.append(float(line.split("YAVG=")[-1].strip()))
            except (ValueError, IndexError):
                pass
    return vals


def _parse_motion(path: Path) -> list:
    vals = []
    if not path.exists():
        return vals
    for line in path.read_text(errors="replace").split("\n"):
        if "motion:" in line:
            try:
                vals.append(float(line.split("motion:")[-1].strip()))
            except (ValueError, IndexError):
                pass
    return vals


def extract_features(video: str, width: int, height: int,
                     start: float = None, duration: float = None,
                     threads: int = 4):
    """Returns (features_dict, meta_dict).

    start/duration: None -> analyze the whole file (default).
    meta has fps/source_width/height/analyzed_seconds.
    """
    pr = probe(video)
    vs = next((s for s in pr.get("streams", [])
               if s.get("codec_type") == "video"), None)
    if not vs:
        raise RuntimeError("no video stream found")
    file_dur = float(pr.get("format", {}).get("duration", 0) or 0)
    meta = {
        "fps": parse_fps(vs.get("r_frame_rate", "0/1")),
        "source_width": int(vs.get("width", 0)),
        "source_height": int(vs.get("height", 0)),
    }

    seek = []
    if start is not None:
        seek += ["-ss", str(start)]
    if duration is not None:
        seek += ["-t", str(duration)]
    meta["analyzed_seconds"] = (duration if duration is not None
                                else max(file_dur - (start or 0.0), 0.0))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        si_file, ti_file, vm_file = td / "si.txt", td / "ti.txt", td / "vm.txt"
        si_f, ti_f, vm_f = _ff_path(si_file), _ff_path(ti_file), _ff_path(vm_file)
        fc = (
            f"[0:v]scale={width}:{height}:flags=bicubic,split=3[a][b][c];"
            f"[a]sobel,signalstats,metadata=mode=print:file={si_f}:direct=1[aout];"
            f"[b]tblend=all_mode=difference,signalstats,"
            f"metadata=mode=print:file={ti_f}:direct=1[bout];"
            f"[c]vmafmotion=stats_file={vm_f}[cout]"
        )
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-nostdin",
             "-threads", str(threads), "-filter_threads", str(threads),
             "-filter_complex_threads", str(threads)]
            + seek + ["-i", video,
                      "-filter_complex", fc,
                      "-map", "[aout]", "-f", "null", "-",
                      "-map", "[bout]", "-f", "null", "-",
                      "-map", "[cout]", "-f", "null", "-"],
            capture_output=True, text=True, timeout=7200)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg feature extraction failed: "
                               f"{(r.stderr or '')[-300:]}")
        si_vals = _parse_yavg(si_file)
        ti_vals = _parse_yavg(ti_file)
        vm_vals = _parse_motion(vm_file)
        if not si_vals or not ti_vals or not vm_vals:
            raise RuntimeError("feature extraction produced no values")

    feats = {
        "si_mean": float(np.mean(si_vals)),
        "si_std": float(np.std(si_vals)) if len(si_vals) > 1 else 0.0,
        "ti_mean": float(np.mean(ti_vals)),
        "ti_std": float(np.std(ti_vals)) if len(ti_vals) > 1 else 0.0,
        "vmafmotion": round(float(np.mean(vm_vals)), 4),
    }
    return feats, meta


# ---------------------------------------------------------------------------
# Probe encode (v2.0) - mirrors collector/generate_probe_dataset.py
# ---------------------------------------------------------------------------
def _scale_vf(w: int, h: int) -> str:
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")


def _probe_one(video: str, start: float, w: int, h: int, codec: str,
               crf: int, ref_yuv: Path, td: Path, threads: int) -> dict:
    """One probe encode + VMAF/bitrate measurement. Returns dict."""
    cfg = PROBE_SPEC[codec]
    cfg_args = cfg["args"]
    if h >= 2160 and "args_4k" in cfg:
        cfg_args = cfg["args_4k"]
    enc_out = td / f"probe_{codec}_{crf}.mkv"
    dis_yuv = td / f"dis_{codec}_{crf}.yuv"
    vmaf_json = td / f"vmaf_{codec}_{crf}.json"
    r = subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-nostdin",
         "-threads", str(threads), "-filter_threads", str(threads),
         "-ss", f"{start:.3f}", "-i", video, "-t", f"{PROBE_SEC:.3f}",
         "-vf", _scale_vf(w, h)] + cfg_args +
        ["-crf", str(crf), "-an", "-threads", str(threads), str(enc_out)],
        capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"probe encode failed ({codec} crf={crf}): "
                           f"{(r.stderr or '')[-300:]}")
    dec = subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-nostdin", "-i", str(enc_out),
         "-pix_fmt", "yuv420p", "-f", "rawvideo", str(dis_yuv)],
        capture_output=True, text=True, timeout=300)
    if dec.returncode != 0:
        raise RuntimeError(f"probe decode failed ({codec} crf={crf}): "
                           f"{(dec.stderr or '')[-300:]}")
    vmaf_model = "version=vmaf_4k_v0.6.1" if h > 1080 else "version=vmaf_v0.6.1"
    vp = subprocess.run(
        [VMAF_CLI, "-r", str(ref_yuv), "-d", str(dis_yuv),
         "-w", str(w), "-h", str(h), "-p", "420", "-b", "8",
         "-m", vmaf_model, "--threads", str(threads),
         "--json", "-o", str(vmaf_json)],
        capture_output=True, text=True, timeout=900)
    if vp.returncode != 0 or not vmaf_json.exists():
        raise RuntimeError(f"vmaf CLI failed ({codec} crf={crf}): "
                           f"{(vp.stderr or '')[-300:]}")
    data = json.loads(vmaf_json.read_text())
    vs = [f["metrics"]["vmaf"] for f in data.get("frames", [])
          if "vmaf" in f.get("metrics", {})]
    if vs:
        vmaf = sum(vs) / len(vs)
    else:
        vmaf = float(data["pooled_metrics"]["vmaf"]["mean"])
    kbps = enc_out.stat().st_size * 8 / PROBE_SEC / 1000.0
    return {"crf": crf, "vmaf": vmaf, "bitrate_kbps": kbps}


def run_probe(video: str, codec: str, width: int, height: int,
              start: float = 0.0, threads: int = 4) -> dict:
    """Two probe encodes -> {probe_vmaf, probe_vmaf2, probe_slope, probe_log_br}.

    Recommended for v2.x models (skip with --no-probe). Uses PROBE_SEC
    seconds from `start` (same analysis start as feature extraction).
    """
    c1, c2 = PROBE_SPEC[codec]["crfs"]
    with tempfile.TemporaryDirectory() as tds:
        td = Path(tds)
        ref_yuv = td / "ref.yuv"
        r = subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-nostdin",
             "-ss", f"{start:.3f}", "-i", video, "-t", f"{PROBE_SEC:.3f}",
             "-vf", _scale_vf(width, height),
             "-pix_fmt", "yuv420p", "-f", "rawvideo", str(ref_yuv)],
            capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not ref_yuv.exists() or ref_yuv.stat().st_size == 0:
            raise RuntimeError(f"probe reference decode failed: "
                               f"{(r.stderr or '')[-300:]}")
        p1 = _probe_one(video, start, width, height, codec, c1, ref_yuv, td, threads)
        p2 = _probe_one(video, start, width, height, codec, c2, ref_yuv, td, threads)
    v1, v2 = p1["vmaf"], p2["vmaf"]
    return {
        "probe_vmaf": round(v1, 4),
        "probe_vmaf2": round(v2, 4),
        "probe_slope": round((v1 - v2) / (c2 - c1), 4),
        "probe_log_br": round(float(np.log1p(max(p1["bitrate_kbps"], 0))), 4),
    }


def model_needs_probe(model: lgb.Booster) -> bool:
    """v2.x models have probe_* features; v1.x don't."""
    return any(f.startswith("probe_") for f in model.feature_name())


# ---------------------------------------------------------------------------
# Feature row + prediction
# ---------------------------------------------------------------------------
def build_feature_row(feats: dict, meta: dict, codec: str, target_vmaf: float,
                      target_width: int, target_height: int) -> dict:
    if codec not in CRF_RANGE:
        raise ValueError(f"codec must be one of {sorted(CRF_RANGE)}")
    row = dict(feats)
    row.update({
        "fps": float(meta["fps"]),
        "target_height": int(target_height),
        "target_width": int(target_width),
        "source_height": int(meta["source_height"]),
        "codec": codec,
        "target_vmaf": float(target_vmaf),
    })
    # derived features (same formulas as training)
    row["si_x_ti"] = row["si_mean"] * row["ti_mean"]
    row["motion_x_ti"] = row["vmafmotion"] * row["ti_mean"]
    row["si_cv"] = row["si_std"] / (row["si_mean"] + 1e-6)
    row["ti_cv"] = row["ti_std"] / (row["ti_mean"] + 1e-6)
    row["res_ratio"] = row["source_height"] / row["target_height"]
    return row


def load_calibration(path: str, codec: str, target_height: int,
                     target_vmaf: float) -> tuple:
    """Returns (delta, entry) for the best-matching calibration entry.

    Matches on codec + target_height; among those picks the entry whose
    target_vmaf is nearest to the requested one (preset shifts are roughly
    constant across targets, so a nearby-target calibration is usable).
    """
    data = json.loads(Path(path).read_text())
    cands = [e for e in data.get("entries", [])
             if e["codec"] == codec and e["target_height"] == target_height]
    if not cands:
        raise ValueError(f"no calibration entry for {codec}@{target_height} "
                         f"in {path}")
    entry = min(cands, key=lambda e: abs(e["target_vmaf"] - target_vmaf))
    return float(entry["crf_delta"]), entry


def predict_crf(model: lgb.Booster, row: dict) -> tuple:
    """Returns (crf_int, crf_raw)."""
    df = pd.DataFrame([row])
    df["codec"] = pd.Categorical(df["codec"], categories=CODEC_CATS)
    raw = float(model.predict(df[model.feature_name()])[0])
    lo, hi = CRF_RANGE[row["codec"]]
    return int(round(min(max(raw, lo), hi))), raw


def load_interval_models(model_path: str):
    """Looks for <stem>_q10.txt / <stem>_q90.txt next to the model file.

    Returns (q10_booster, q90_booster) or None when either file is missing
    (older single-model packages keep working unchanged).
    """
    mp = Path(model_path)
    q10 = mp.parent / f"{mp.stem}_q10{mp.suffix}"
    q90 = mp.parent / f"{mp.stem}_q90{mp.suffix}"
    if q10.exists() and q90.exists():
        return (lgb.Booster(model_file=str(q10)),
                lgb.Booster(model_file=str(q90)))
    return None


def load_preset_offsets(model_path: str):
    """preset_offsets.json next to the model file, or None when absent."""
    p = Path(model_path).parent / "preset_offsets.json"
    return json.loads(p.read_text()) if p.exists() else None


def normalize_preset(codec: str, preset: str) -> str:
    """Accepts 'medium', 'slow', vp9 'cu4'/'4'/'cpu-used 4', av1 'p8'/'8'."""
    p = preset.strip().lower()
    if codec == "vp9":
        p = p.replace("cpu-used", "").replace("cpu", "").strip("- ")
        return f"cu{p}" if p.isdigit() else p
    if codec == "av1":
        return f"p{p}" if p.isdigit() else p
    return p


def baseline_preset(offsets: dict, codec: str, target_height: int) -> str:
    b = offsets["baseline_presets"][codec]
    return b[str(target_height)] if isinstance(b, dict) else b


def main():
    ap = argparse.ArgumentParser(description="Predict CRF for a target VMAF.")
    ap.add_argument("video")
    ap.add_argument("--codec", required=True, choices=sorted(CRF_RANGE))
    ap.add_argument("--target-height", type=int, required=True,
                    choices=sorted(RES_W))
    ap.add_argument("--target-vmaf", type=float, required=True)
    ap.add_argument("--start", type=float, default=None,
                    help="optional: analyze from this second (default: whole file)")
    ap.add_argument("--duration", type=float, default=None,
                    help="optional: analyze only this many seconds")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", default="model.txt")
    ap.add_argument("--no-probe", action="store_true",
                    help="v2.x models: skip the probe encode and fill the "
                         "probe_* features with neutral defaults - faster, "
                         "no vmaf CLI needed, but accuracy degrades towards "
                         "v1.x levels (vmaf_mae ~3.7 instead of ~1.35)")
    ap.add_argument("--no-interval", action="store_true",
                    help="point prediction only, skip the q10-q90 interval "
                         "even when model_q10.txt/model_q90.txt are present")
    ap.add_argument("--calibration", default=None,
                    help="calibration.json from calibrate.py - adds the "
                         "measured CRF offset for your encoder settings")
    ap.add_argument("--preset", default=None,
                    help="encoder preset you will encode with — x264/x265: "
                         "veryfast|fast|medium|slow, vp9: cu6|cu4|cu2, "
                         "av1: p10|p12|p8|p6. Applies the measured CRF offset "
                         "from preset_offsets.json (ignored when "
                         "--calibration/--crf-offset is given)")
    ap.add_argument("--crf-offset", type=float, default=None,
                    help="manual constant added to the raw predicted CRF "
                         "(ignored when --calibration is given)")
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    if not (60 <= args.target_vmaf <= 95):
        print("warning: target VMAF outside the trained range 60-95; "
              "expect degraded accuracy", file=sys.stderr)

    width = RES_W[args.target_height]
    feats, meta = extract_features(args.video, width, args.target_height,
                                   start=args.start, duration=args.duration,
                                   threads=args.threads)

    if meta["analyzed_seconds"] and meta["analyzed_seconds"] < MIN_RELIABLE_SECONDS:
        print(f"warning: analysis window is only {meta['analyzed_seconds']:.1f} s "
              f"(< {MIN_RELIABLE_SECONDS:g} s) - frame statistics are noisy, "
              f"expect reduced accuracy", file=sys.stderr)

    row = build_feature_row(feats, meta, args.codec, args.target_vmaf,
                            width, args.target_height)
    model = lgb.Booster(model_file=args.model)

    # v2.0: probe-encode - recommended but optional (--no-probe). Without
    # it the probe_* features get neutral defaults and accuracy degrades
    # towards v1.x levels. v1.x models (no probe_* features) never probe.
    if model_needs_probe(model):
        if args.no_probe:
            row.update({
                "probe_vmaf": args.target_vmaf,
                "probe_vmaf2": args.target_vmaf,
                "probe_slope": 0.0,
                "probe_log_br": PROBE_LOG_BR_MEDIAN,
            })
            print("warning: --no-probe - probe features filled with neutral "
                  "defaults; expect v1.x-level accuracy (vmaf_mae ~3.7 "
                  "instead of ~1.35)", file=sys.stderr)
        else:
            row.update(run_probe(args.video, args.codec, width,
                                 args.target_height,
                                 start=args.start or 0.0,
                                 threads=args.threads))
    crf, raw = predict_crf(model, row)

    # v1.5.0: 80% prediction interval from sibling quantile models
    q10_raw = q90_raw = None
    if not args.no_interval:
        im = load_interval_models(args.model)
        if im:
            _, q10_raw = predict_crf(im[0], row)
            _, q90_raw = predict_crf(im[1], row)

    # measured preset offset (preset_offsets.json next to the model);
    # the model predicts the training-baseline preset
    preset_note = None
    if args.preset:
        offsets = load_preset_offsets(args.model)
        if offsets is None:
            ap.error("--preset requires preset_offsets.json next to the model "
                     f"({Path(args.model).parent / 'preset_offsets.json'} "
                     "not found)")
        pn = normalize_preset(args.codec, args.preset)
        base = baseline_preset(offsets, args.codec, args.target_height)
        cell = None
        if pn == base:
            d = 0.0
        else:
            cell = offsets["cells"].get(f"{args.codec}|{pn}|{args.target_height}")
            if cell is None:
                avail = sorted({k.split("|")[1] for k in offsets["cells"]
                                if k.startswith(args.codec + "|")} | {base})
                ap.error(f"no measured offset for {args.codec}/{pn} @ "
                         f"{args.target_height}p; available for {args.codec}: "
                         + ", ".join(avail))
            d = float(cell["delta"])
        if (args.calibration or args.crf_offset is not None) and d != 0.0:
            print("warning: --calibration/--crf-offset already captures your "
                  "encoder settings (incl. preset) — ignoring the --preset "
                  "offset", file=sys.stderr)
            d = 0.0
        if d != 0.0:
            lo, hi = CRF_RANGE[args.codec]
            crf = int(round(min(max(raw + d, lo), hi)))
            if q10_raw is not None:
                q10_raw += d
                q90_raw += d
        preset_note = {"preset": pn, "baseline_preset": base, "crf_delta": d}
        if cell:
            preset_note["iqr"] = cell["iqr"]
            preset_note["n_clips"] = cell["n_clips"]

    cal_note = None
    if args.calibration:
        delta, entry = load_calibration(args.calibration, args.codec,
                                        args.target_height, args.target_vmaf)
        lo, hi = CRF_RANGE[args.codec]
        raw_cal = raw + delta
        crf = int(round(min(max(raw_cal, lo), hi)))
        if q10_raw is not None:
            q10_raw += delta
            q90_raw += delta
        cal_note = {"crf_delta": delta, "crf_uncalibrated": int(round(min(max(raw, lo), hi))),
                    "source_entry": {k: entry[k] for k in
                                     ("codec", "target_height", "target_vmaf",
                                      "encoder_args", "n_used")}}
    elif args.crf_offset is not None:
        lo, hi = CRF_RANGE[args.codec]
        raw_cal = raw + args.crf_offset
        crf = int(round(min(max(raw_cal, lo), hi)))
        if q10_raw is not None:
            q10_raw += args.crf_offset
            q90_raw += args.crf_offset
        cal_note = {"crf_delta": args.crf_offset,
                    "crf_uncalibrated": int(round(min(max(raw, lo), hi)))}

    # saturation detection (v2.0.1): computed on the FINAL effective raw
    # prediction — after --preset / --calibration / --crf-offset deltas — so
    # the warning reflects the CRF actually returned. An effective raw above
    # the ladder top means even the most aggressive allowed CRF cannot
    # degrade this content enough: the target VMAF is likely unreachable
    # (easy/synthetic content whose VMAF floor at max CRF sits above the
    # target). A real low-CRF probe_vmaf ~99 corroborates; the neutral
    # --no-probe fallback (probe_vmaf == target_vmaf) is not evidence.
    raw_eff = raw
    if cal_note is not None:
        raw_eff = raw + cal_note["crf_delta"]
    elif preset_note is not None:
        raw_eff = raw + preset_note["crf_delta"]
    lo, hi = CRF_RANGE[args.codec]
    saturated = raw_eff > hi
    probe_ran = model_needs_probe(model) and not args.no_probe
    if saturated:
        extra = ""
        if probe_ran and row.get("probe_vmaf", 0) >= 97:
            extra = (f" (low-CRF probe VMAF is {row['probe_vmaf']:.1f} "
                     "— this content barely degrades even at high quality)")
        print(f"warning: target VMAF {args.target_vmaf:g} is likely "
              f"unreachable for this video — effective raw prediction "
              f"{raw_eff:.1f} exceeds the {args.codec} ladder maximum CRF "
              f"{hi}{extra}. Returning CRF {hi}; actual VMAF will be higher "
              f"than the target.", file=sys.stderr)

    interval = None
    if q10_raw is not None:
        lo, hi = CRF_RANGE[args.codec]
        interval = {"crf_q10": int(round(min(max(q10_raw, lo), hi))),
                    "crf_q90": int(round(min(max(q90_raw, lo), hi))),
                    "crf_q10_raw": round(q10_raw, 3),
                    "crf_q90_raw": round(q90_raw, 3)}

    if args.json:
        out = {"crf": crf, "crf_raw": round(raw, 3),
               "codec": args.codec,
               "target_height": args.target_height,
               "target_vmaf": args.target_vmaf,
               "analyzed_seconds": round(meta["analyzed_seconds"], 2),
               "features": {k: row[k] for k in BASE_FEATS}}
        if interval:
            out["interval_q10_q90"] = interval
        if preset_note:
            out["preset"] = preset_note
        if cal_note:
            out["calibration"] = cal_note
        if saturated:
            out["saturation_warning"] = {
                "message": "target VMAF likely unreachable; effective raw "
                           "prediction clipped at ladder maximum",
                "crf_raw_effective": round(raw_eff, 3),
                "ladder_max_crf": hi,
                "probe_vmaf": row.get("probe_vmaf") if probe_ran else None,
            }
        print(json.dumps(out, indent=1))
    else:
        line = (f"predicted CRF: {crf}  (raw {raw:.2f})  "
                f"[{args.codec} @ {args.target_height}p, target VMAF "
                f"{args.target_vmaf:g}]")
        if interval:
            line += (f"\n80% interval (q10-q90): CRF {interval['crf_q10']}.."
                     f"{interval['crf_q90']}  (raw {interval['crf_q10_raw']:.2f}"
                     f"..{interval['crf_q90_raw']:.2f})")
        if preset_note:
            line += (f"\npreset: {preset_note['preset']} "
                     f"({preset_note['crf_delta']:+.2f} CRF vs baseline "
                     f"{preset_note['baseline_preset']}"
                     + (f", IQR {preset_note['iqr']:.2f}"
                        if preset_note.get("iqr") is not None else "")
                     + ")")
        if cal_note:
            line += (f"\ncalibration: {cal_note['crf_delta']:+.2f} CRF "
                     f"(uncalibrated {cal_note['crf_uncalibrated']})")
        if saturated:
            line += (f"\n⚠ saturation: effective raw {raw_eff:.1f} clipped "
                     f"at ladder max CRF {hi} — target VMAF likely "
                     f"unreachable for this content, actual VMAF will "
                     f"overshoot")
        print(line)


if __name__ == "__main__":
    main()
