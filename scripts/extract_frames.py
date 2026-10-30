#!/usr/bin/env python3
"""Stage A: video -> a controlled set of sharp frames.

AnySplat's own demo takes 1 frame/sec and accepts whatever it gets. That wastes
the capture: at 1 fps a blurry frame is 1/30th of your entire input, and motion
blur is unrecoverable downstream.

Instead this oversamples (default 4 fps), scores every frame by
variance-of-Laplacian, then splits the timeline into --max-frames equal bins and
keeps the SHARPEST frame in each bin. That holds frame count to the model's
budget while keeping even coverage of the camera path -- picking the globally
sharpest N instead would happily take them all from the one moment you held
still, leaving the rest of the scene unreconstructed.

Needs ffmpeg on PATH. Uses PIL + numpy only (no OpenCV), both already present
in AnySplat's environment.

Usage:
  python extract_frames.py capture.mp4 -o frames/ --max-frames 32
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def log(msg):
    print(f"[frames] {msg}", file=sys.stderr)


def laplacian_variance(path, work_size=512):
    """Focus measure: variance of the Laplacian. Higher = sharper."""
    img = Image.open(path).convert("L")
    img.thumbnail((work_size, work_size), Image.BILINEAR)
    a = np.asarray(img, np.float32)
    if min(a.shape) < 3:
        return 0.0
    # 4-neighbour Laplacian via slicing; no scipy dependency.
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("-o", "--output", required=True, help="output frame directory")
    ap.add_argument("--fps", type=float, default=4.0,
                    help="oversample rate before selection (default 4)")
    ap.add_argument("--max-frames", type=int, default=32,
                    help="frames to keep. AnySplat examples run 2-64; VRAM "
                         "scales badly, so lower this first on OOM")
    ap.add_argument("--min-sharpness", type=float, default=0.0,
                    help="absolute variance-of-Laplacian floor; frames below "
                         "are dropped even if that means fewer than --max-frames")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep the oversampled pre-selection frames for inspection")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        ap.error("ffmpeg not found on PATH (apt install ffmpeg / brew install ffmpeg)")
    if not Path(args.video).is_file():
        ap.error(f"no such video: {args.video}")

    out = Path(args.output)
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)
    for stale in raw.glob("*.png"):
        stale.unlink()

    log(f"extracting at {args.fps} fps -> {raw}")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", args.video,
         "-vf", f"fps={args.fps}", "-pix_fmt", "rgb24", str(raw / "%06d.png")],
        check=True,
    )

    cands = sorted(raw.glob("*.png"))
    if not cands:
        sys.exit("ffmpeg produced no frames -- is the video readable?")
    log(f"scoring {len(cands)} candidate frames for sharpness")

    scores = np.array([laplacian_variance(p) for p in cands])
    log(f"sharpness: min={scores.min():.1f} median={np.median(scores):.1f} "
        f"max={scores.max():.1f}")

    keep_mask = scores >= args.min_sharpness
    idx_ok = np.flatnonzero(keep_mask)
    if args.min_sharpness > 0:
        log(f"{len(cands) - len(idx_ok)} frames below --min-sharpness, dropped")
    if len(idx_ok) == 0:
        sys.exit("every frame fell below --min-sharpness")

    # Temporal binning: one (sharpest) frame per equal slice of the timeline.
    n_keep = min(args.max_frames, len(idx_ok))
    bins = np.array_split(idx_ok, n_keep)
    chosen = [int(b[np.argmax(scores[b])]) for b in bins if len(b)]

    for stale in out.glob("*.png"):
        stale.unlink()
    for n, i in enumerate(chosen):
        shutil.copy2(cands[i], out / f"{n:06d}.png")

    log(f"kept {len(chosen)} frames (1 sharpest per temporal bin) -> {out}")
    log(f"mean sharpness of kept frames: {scores[chosen].mean():.1f} "
        f"(vs {scores.mean():.1f} across all candidates)")

    if not args.keep_raw:
        shutil.rmtree(raw)
    else:
        log(f"oversampled frames left in {raw}")


if __name__ == "__main__":
    main()
