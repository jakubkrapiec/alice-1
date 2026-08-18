#!/usr/bin/env python3
"""Probe-based calibration for encoder settings that differ from training.

The model was trained on specific encoder builds/presets (see README). A
different preset (or encoder version) shifts the CRF -> VMAF curve roughly
horizontally - the content-dependent part of the prediction still holds.
This tool measures that shift on YOUR encoder settings with a small set of
probe encodes and writes a calibration file that predict.py can apply:

    python3 calibrate.py --model model.txt --codec x264 \
        --encoder-args "-c:v libx264 -preset medium" \
        --target-vmaf 85 --height 1080 \
        --videos probes/*.mp4 --output calibration.json

    python3 predict.py input.mp4 --codec x264 --target-height 1080 \
        --target-vmaf 85 --calibration calibration.json

Protocol (per probe video):
  1. extract content features at the target resolution, predict CRF (p)
  2. encode a 10 s window at CRF {p-span, p, p+span} with --encoder-args
  3. measure VMAF of each encode against the scaled source (same scaling and
     VMAF model as training)
  4. invert the measured (CRF, VMAF) points at the target -> actual CRF (a)
  5. delta = a - p; the calibration value is the MEDIAN delta over probes

Use >= 20 diverse probe videos (different content, motion, grain) for a
stable median. Fewer probes still help but check the reported IQR.

Requires: ffmpeg + ffprobe + the vmaf CLI (libvmaf command-line tools,
https://github.com/Netflix/vmaf) on PATH - binary names can be overridden
with the FFMPEG / FFPROBE / VMAF environment variables. Python: lightgbm,
numpy, pandas; predict.py from this package (same directory).
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import (RES_W, CRF_RANGE, probe, extract_features,  # noqa: E402
                     build_feature_row, predict_crf, model_needs_probe,
                     PROBE_LOG_BR_MEDIAN)

FFMPEG = os.environ.get("FFMPEG", "ffmpeg")
VMAF = os.environ.get("VMAF", "vmaf")
ANALYSIS_SECONDS = 10.0


def run(cmd, timeout=7200):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd[:3])} ...): "
                           f"{(r.stderr or '')[-300:]}")
    return r


def scale_vf(w, h):
    return (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuv420p")


def make_ref(src, start, dur, w, h, out_dir, threads):
    ref = out_dir / "ref.yuv"
    run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", src, "-vf", scale_vf(w, h), "-an", "-pix_fmt", "yuv420p",
         "-f", "rawvideo", "-threads", str(threads), str(ref)])
    return ref


def encode_one(src, start, dur, w, h, encoder_args, crf, out_dir, tag, threads):
    dis = out_dir / f"dis_{tag}.mkv"
    run([FFMPEG, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", src, "-vf", scale_vf(w, h), "-an"]
        + list(encoder_args) + ["-crf", str(crf), "-threads", str(threads),
                                str(dis)])
    return dis


def vmaf_score(dis, ref, w, h, out_dir, tag, threads):
    """VMAF via the vmaf CLI on rawvideo YUV — exactly as in training:
    distorted decoded to yuv420p, mean over per-frame scores."""
    log = out_dir / f"vmaf_{tag}.json"
    model = "version=vmaf_v0.6.1" if h <= 1080 else "version=vmaf_4k_v0.6.1"
    dis_yuv = out_dir / f"dis_{tag}.yuv"
    try:
        run([FFMPEG, "-y", "-v", "error", "-i", str(dis),
             "-pix_fmt", "yuv420p", "-f", "rawvideo", str(dis_yuv)])
        run([VMAF, "-r", str(ref), "-d", str(dis_yuv),
             "-w", str(w), "-h", str(h), "-p", "420", "-b", "8",
             "-m", model, "--threads", str(threads),
             "--json", "-o", str(log)], timeout=1800)
    finally:
        dis_yuv.unlink(missing_ok=True)
    data = json.loads(log.read_text())
    frames = data.get("frames", [])
    if not frames:
        raise RuntimeError(f"empty VMAF result in {log}")
    m = frames[0]["metrics"]
    key = "vmaf" if "vmaf" in m else "integer_vmaf"
    vals = [f["metrics"][key] for f in frames]
    return sum(vals) / len(vals)


def invert_crf(points, target):
    """points: list of (crf, vmaf). Returns CRF at which vmaf==target,
    or None if the target is outside the measured range."""
    pts = sorted(set(points))
    if len(pts) < 2:
        return None
    crfs = [p[0] for p in pts]
    vmafs = [p[1] for p in pts]
    xs = vmafs[::-1]          # vmaf decreases with crf -> increasing after flip
    ys = crfs[::-1]
    if target < xs[0] - 1e-9 or target > xs[-1] + 1e-9:
        return None
    return float(np.interp(target, xs, ys))


def calibrate_video(video, args, width, model_path):
    """Returns dict with per-probe result (or skip reason)."""
    pr = probe(video)
    file_dur = float(pr.get("format", {}).get("duration", 0) or 0)
    if file_dur < 3.0:
        return {"video": str(video), "skipped": f"too short ({file_dur:.1f} s)"}
    dur = min(args.duration, file_dur)
    start = max(0.0, (file_dur - dur) * 0.25)

    model = lgb.Booster(model_file=model_path)
    feats, meta = extract_features(str(video), width, args.height,
                                   start=start, duration=dur, threads=2)
    row = build_feature_row(feats, meta, args.codec, args.target_vmaf,
                            width, args.height)
    if model_needs_probe(model):
        # Neutral defaults, as predict.py --no-probe uses: this initial
        # prediction only seeds the ladder center below - the ladder's
        # measured (crf, vmaf) points determine the actual calibration.
        row.update({
            "probe_vmaf": args.target_vmaf,
            "probe_vmaf2": args.target_vmaf,
            "probe_slope": 0.0,
            "probe_log_br": PROBE_LOG_BR_MEDIAN,
        })
    _, pred_raw = predict_crf(model, row)

    lo, hi = CRF_RANGE[args.codec]
    p = int(round(pred_raw))

    def ladder(span):
        pts = {min(max(p + d, lo), hi) for d in (-span, 0, span)}
        return sorted(pts)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ref = make_ref(str(video), start, dur, width, args.height, td,
                       args.threads)
        measured = []
        for attempt, span in enumerate((args.span, args.span * 2)):
            for crf in ladder(span):
                if any(c == crf for c, _ in measured):
                    continue
                tag = f"{crf}"
                dis = encode_one(str(video), start, dur, width, args.height,
                                 args.encoder_args, crf, td, tag, args.threads)
                v = vmaf_score(dis, ref, width, args.height, td, tag,
                               args.threads)
                measured.append((crf, v))
            actual = invert_crf(measured, args.target_vmaf)
            if actual is not None:
                break
        if actual is None:
            return {"video": str(video), "pred": round(pred_raw, 2),
                    "measured": measured,
                    "skipped": "target outside measured ladder"}
    return {"video": str(video), "pred": round(pred_raw, 3),
            "actual": round(actual, 3),
            "delta": round(actual - pred_raw, 3), "measured": measured}


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description="Calibrate the CRF predictor to your encoder settings.")
    ap.add_argument("--model", default="model.txt")
    ap.add_argument("--codec", required=True, choices=sorted(CRF_RANGE))
    ap.add_argument("--encoder-args", required=True,
                    help="full ffmpeg encoder options, e.g. "
                         "'-c:v libx264 -preset medium' or "
                         "'-c:v libsvtav1 -preset 6' (CRF is added by this "
                         "tool; for vp9 include -b:v 0)")
    ap.add_argument("--target-vmaf", type=float, required=True)
    ap.add_argument("--height", type=int, required=True, choices=sorted(RES_W),
                    help="target resolution height (720/1080/1440/2160)")
    ap.add_argument("--videos", nargs="+", required=True,
                    help="probe videos (>= 20 diverse clips recommended)")
    ap.add_argument("--output", default="calibration.json")
    ap.add_argument("--span", type=int, default=4,
                    help="ladder half-width in CRF around the prediction")
    ap.add_argument("--duration", type=float, default=ANALYSIS_SECONDS,
                    help="seconds per probe window (default 10)")
    ap.add_argument("--jobs", type=int, default=4,
                    help="probe videos processed in parallel")
    ap.add_argument("--threads", type=int, default=2,
                    help="ffmpeg threads per encode")
    args = ap.parse_args()

    if not (60 <= args.target_vmaf <= 95):
        print("warning: target VMAF outside the trained range 60-95",
              file=sys.stderr)
    args.encoder_args = shlex.split(args.encoder_args)
    if any(a == "-crf" for a in args.encoder_args):
        sys.exit("error: do not pass -crf inside --encoder-args; "
                 "the tool sets it per ladder point")

    width = RES_W[args.height]
    videos = sorted(str(v) for v in args.videos)
    print(f"calibrating {args.codec} @ {args.height}p, target VMAF "
          f"{args.target_vmaf:g}, {len(videos)} probes, "
          f"encoder: {' '.join(args.encoder_args)}", flush=True)

    results = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(calibrate_video, v, args, width, args.model): v
                for v in videos}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if "delta" in r:
                print(f"[{i}/{len(videos)}] {Path(r['video']).name}: "
                      f"pred={r['pred']:.2f} actual={r['actual']:.2f} "
                      f"delta={r['delta']:+.2f}", flush=True)
            else:
                print(f"[{i}/{len(videos)}] {Path(r['video']).name}: "
                      f"SKIPPED ({r['skipped']})", flush=True)

    deltas = [r["delta"] for r in results if "delta" in r]
    if not deltas:
        sys.exit("error: no usable probes — nothing to calibrate")
    deltas = np.array(deltas)
    entry = {
        "codec": args.codec,
        "target_height": args.height,
        "target_vmaf": args.target_vmaf,
        "encoder_args": " ".join(args.encoder_args),
        "crf_delta": round(float(np.median(deltas)), 3),
        "delta_mean": round(float(np.mean(deltas)), 3),
        "delta_std": round(float(np.std(deltas)), 3),
        "delta_iqr": [round(float(np.percentile(deltas, 25)), 3),
                      round(float(np.percentile(deltas, 75)), 3)],
        "n_used": int(len(deltas)),
        "n_total": len(results),
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model_md5": md5_of(args.model),
        "probes": results,
    }

    out = Path(args.output)
    if out.exists():
        data = json.loads(out.read_text())
    else:
        data = {"format_version": 1, "entries": []}
    key = (entry["codec"], entry["target_height"], entry["target_vmaf"])
    data["entries"] = [e for e in data["entries"]
                       if (e["codec"], e["target_height"], e["target_vmaf"])
                       != key]
    data["entries"].append(entry)
    out.write_text(json.dumps(data, indent=1))

    print(f"\ncrf_delta (median): {entry['crf_delta']:+.2f}  "
          f"[IQR {entry['delta_iqr'][0]:+.2f} .. {entry['delta_iqr'][1]:+.2f}]  "
          f"n={len(deltas)}/{len(results)}")
    if len(deltas) < 10:
        print("warning: fewer than 10 usable probes — the median is noisy; "
              ">= 20 diverse videos recommended", file=sys.stderr)
    print(f"written: {out}  (predict.py --calibration {out})")


if __name__ == "__main__":
    main()
