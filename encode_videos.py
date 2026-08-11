#!/usr/bin/env python3
"""Encode GOT-10k-style image sequences into videos, one video per sample.

Each sample is a folder of sequential JPEG frames. This script concatenates the
frames of every sample into a single video, in parallel across worker processes.
Original images are never modified.

Default layout (relative to --dataset):
    test_data/test/<sample>/*.jpg  ->  test_data_video/<sample>.mp4
    val_data/val/<sample>/*.jpg    ->  val_data_video/<sample>.mp4

Examples:
    python encode_videos.py                         # defaults: fps=30, lossless, all splits
    python encode_videos.py --fps 25 --workers 8
    python encode_videos.py --crf 17                # visually lossless, smaller files
    python encode_videos.py --split my_frames:my_videos --overwrite
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Default (frames_root, output_dir) pairs, relative to the dataset directory.
DEFAULT_SPLITS = [
    ("test_data/test", "test_data_video"),
    ("val_data/val", "val_data_video"),
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class Job:
    sample: str          # sample name, used for the output filename
    frames_dir: Path     # folder containing this sample's frames
    output: Path         # destination video path
    fps: int
    qp: int | None       # lossless quality; None when crf is used
    crf: int | None      # constant-rate-factor; None when qp is used
    ext: str             # image extension to glob


def encode_one(job: Job) -> tuple[str, str, str]:
    """Encode a single sample. Returns (sample, status, detail)."""
    if job.output.exists():
        return (job.sample, "skipped", "output exists")

    frames = sorted(job.frames_dir.glob(f"*{job.ext}"))
    if not frames:
        return (job.sample, "skipped", f"no *{job.ext} frames")

    job.output.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file first, then atomically rename. The muxer can't be
    # inferred from the ".part" extension, so pass the format explicitly via -f.
    tmp = job.output.with_suffix(job.output.suffix + ".part")
    muxer = {".mp4": "mp4", ".mkv": "matroska", ".mov": "mov"}.get(
        job.output.suffix.lower(), "mp4")

    quality = ["-qp", str(job.qp)] if job.crf is None else ["-crf", str(job.crf)]
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-framerate", str(job.fps),
        "-pattern_type", "glob", "-i", str(job.frames_dir / f"*{job.ext}"),
        # pad odd dimensions up to even so yuv420p is always valid
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264", *quality, "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-f", muxer, str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        lines = (exc.stderr or "").strip().splitlines()
        return (job.sample, "error", lines[-1] if lines else "ffmpeg failed")
    tmp.replace(job.output)  # atomic: partial files never masquerade as done
    return (job.sample, "done", f"{len(frames)} frames")


def discover_jobs(args) -> list[Job]:
    dataset = Path(args.dataset).resolve()
    jobs: list[Job] = []
    for spec in args.split:
        frames_root, out_name = spec.split(":", 1) if ":" in spec else (spec, spec + "_video")
        frames_root = (dataset / frames_root).resolve()
        out_dir = (dataset / out_name).resolve()
        if not frames_root.is_dir():
            print(f"warning: frames root not found, skipping: {frames_root}", file=sys.stderr)
            continue
        for sample_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
            jobs.append(Job(
                sample=sample_dir.name,
                frames_dir=sample_dir,
                output=out_dir / f"{sample_dir.name}.mp4",
                fps=args.fps,
                qp=args.qp,
                crf=args.crf,
                ext=args.ext,
            ))
    return jobs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default=str(Path(__file__).resolve().parent / "dataset"),
                   help="dataset directory containing the split folders (default: ./dataset)")
    p.add_argument("--split", action="append", metavar="FRAMES_ROOT[:OUTPUT_DIR]",
                   help="split to encode, relative to --dataset; repeatable. "
                        "Default: test_data/test:test_data_video and val_data/val:val_data_video")
    p.add_argument("--fps", type=int, default=30, help="output frame rate (default: 30)")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4,
                   help="parallel worker processes (default: number of CPUs)")
    p.add_argument("--crf", type=int, default=None,
                   help="use constant-rate-factor (e.g. 17) instead of lossless; smaller files")
    p.add_argument("--qp", type=int, default=0,
                   help="quantization parameter for lossless encode (default: 0 = lossless). "
                        "Ignored when --crf is set.")
    p.add_argument("--ext", default=".jpg", help="frame image extension (default: .jpg)")
    p.add_argument("--overwrite", action="store_true", help="re-encode even if output exists")
    args = p.parse_args(argv)
    if not args.split:
        args.split = [f"{f}:{o}" for f, o in DEFAULT_SPLITS]
    if args.crf is not None:
        args.qp = None
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("error: ffmpeg not found on PATH", file=sys.stderr)
        return 2

    jobs = discover_jobs(args)
    if not jobs:
        print("no samples found — nothing to do", file=sys.stderr)
        return 1
    if args.overwrite:
        for j in jobs:
            j.output.unlink(missing_ok=True)

    quality = "lossless (qp=0)" if args.crf is None else f"crf={args.crf}"
    print(f"{len(jobs)} samples | fps={args.fps} | {quality} | workers={args.workers}")

    counts = {"done": 0, "skipped": 0, "error": 0}
    start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(encode_one, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            sample, status, detail = fut.result()
            counts[status] += 1
            tag = {"done": "ok", "skipped": "--", "error": "ERR"}[status]
            print(f"[{i}/{len(jobs)}] {tag} {sample} ({detail})", flush=True)

    elapsed = time.time() - start
    print(f"\nfinished in {elapsed:.1f}s — "
          f"{counts['done']} encoded, {counts['skipped']} skipped, {counts['error']} errors")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
