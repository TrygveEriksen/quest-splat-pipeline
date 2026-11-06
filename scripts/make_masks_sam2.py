#!/usr/bin/env python3
"""Optional helper: generate per-frame subject masks with SAM 2.

SAM 2's video predictor is the natural fit here -- prompt the subject once on
the first frame and it propagates the mask through the rest, which is exactly
the shape of this problem (one static subject, a slow orbit).

Install separately (it needs torch>=2.5.1, so it cannot share AnySplat's env):
scripts/install.sh sets up a "sam2" conda env and downloads
sam2.1_hiera_large.pt. Run this script with that env's python.

Prompt with a box around the subject in frame 0 (easiest), or with points:
    python make_masks_sam2.py --frames frames/ --out masks/ \\
        --checkpoint sam2.1_hiera_large.pt --box 120 90 330 400

Then:
    python mask_splats.py --ply out/gaussians.ply --poses out/poses.npz \\
        --masks masks/ -o out/masked.ply
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def log(msg):
    print(f"[sam2] {msg}", file=sys.stderr)


def as_jpeg_dir(frames):
    """SAM 2's init_state wants a directory of JPEG frames named numerically.

    extract_frames.py writes PNG, so mirror them into a temp JPEG directory
    rather than making the caller convert by hand.
    """
    paths = sorted(p for p in Path(frames).iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not paths:
        sys.exit(f"no frames in {frames}")
    if all(p.suffix.lower() in {".jpg", ".jpeg"} for p in paths):
        return str(frames), paths, None

    tmp = tempfile.mkdtemp(prefix="sam2_frames_")
    log(f"converting {len(paths)} frames to JPEG for SAM 2 -> {tmp}")
    for i, p in enumerate(paths):
        Image.open(p).convert("RGB").save(Path(tmp) / f"{i:05d}.jpg", quality=95)
    return tmp, paths, tmp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", required=True, help="SAM 2 .pt checkpoint")
    ap.add_argument("--model-cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--box", nargs=4, type=float, metavar=("X0", "Y0", "X1", "Y1"),
                    help="box around the subject in frame 0 (pixels)")
    ap.add_argument("--points", type=str,
                    help="'x,y;x,y' foreground points in frame 0")
    ap.add_argument("--prompt-frame", type=int, default=0)
    args = ap.parse_args()

    if not args.box and not args.points:
        ap.error("give --box or --points to say what the subject is")

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as e:
        sys.exit(f"SAM 2 not importable ({e}). Run this with the sam2 env's "
                 f"python (see scripts/install.sh).")

    frames_dir, orig_paths, tmp = as_jpeg_dir(args.frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"building predictor on {device}")
    predictor = build_sam2_video_predictor(args.model_cfg, args.checkpoint,
                                           device=device)

    try:
        state = predictor.init_state(video_path=frames_dir)

        kwargs = {}
        if args.box:
            kwargs["box"] = np.array(args.box, np.float32)
            log(f"prompting frame {args.prompt_frame} with box {args.box}")
        if args.points:
            pts = np.array([[float(x) for x in pair.split(",")]
                            for pair in args.points.split(";")], np.float32)
            kwargs["points"] = pts
            kwargs["labels"] = np.ones(len(pts), np.int32)   # all foreground
            log(f"prompting frame {args.prompt_frame} with {len(pts)} point(s)")

        predictor.add_new_points_or_box(
            state, frame_idx=args.prompt_frame, obj_id=1, **kwargs)

        written = 0
        for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits[0] > 0.0).cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            # Name masks after the ORIGINAL frames: mask_splats.py matches masks
            # to poses by sorted order, so the two sets must sort identically.
            stem = orig_paths[frame_idx].stem
            Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{stem}.png")
            written += 1

        log(f"wrote {written} masks -> {out}")
        if written != len(orig_paths):
            log(f"WARNING: {len(orig_paths)} frames but {written} masks. "
                f"mask_splats.py matches by sorted order and will warn too.")
        log("check a few masks by eye before running mask_splats.py -- a lost "
            "track mid-sequence is the usual failure, and it shows up as a "
            "subject with chunks missing")
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
