#!/usr/bin/env python3
"""Stage B2 (optional): keep only the Gaussians belonging to a segmented subject.

AnySplat has no mask input -- it is trained on RGB only -- but its Gaussians are
PIXEL-ALIGNED, one per input pixel per view. So a 2D segmentation can be turned
into a 3D selection by reprojection: push every Gaussian back into each input
view, look up the mask, and keep the ones that land inside it.

ORDERING: run this on the RAW gaussians.ply straight out of run_anysplat.py,
BEFORE fixup_ply.py. The predicted poses live in the model's coordinate frame,
so once fixup_ply has flipped the axes the poses no longer apply.

    run_anysplat.py -> mask_splats.py -> fixup_ply.py -> decimate_ply.py

WHY MULTI-VIEW AGREEMENT: a wall directly behind the subject projects *inside*
the mask from that viewpoint, so a single-view test drags the background along.
Geometry behind the subject in one view is not behind it in another, so
requiring agreement across the views that actually see a Gaussian rejects it.
--min-agreement controls the strictness, and the default of 0.8 is measured,
not guessed. On a synthetic ground-truth scene (90k subject splats inside a 500k
room, 24 cameras orbiting the subject with the far wall in frame):

    --min-agreement 0.50   precision  32.8%   recall 100%    <- unusable
    --min-agreement 0.80   precision  99.9%   recall 100%
    --min-agreement 0.95   precision 100.0%   recall 100%

A plain majority vote drags in ~185k background splats, because the wall behind
the subject really is inside the mask in a large minority of views. Those masks
were perfect, so expect real recall to fall off as agreement rises; 0.8 is the
balance point. If your subject comes out eaten away, lower it; if background
bleeds in, raise it.

Why this is worth doing on Quest 2: it is the biggest quality lever available.
Masking to the subject first means the whole ~150k splat budget goes to the
subject, instead of being spent on walls that decimation then throws away.

Mask convention: white/non-zero = KEEP. Any resolution (they are rescaled to the
projection frame). Matched to frames by sorted filename order.

Usage:
  python mask_splats.py --ply out/gaussians.ply --poses out/poses.npz \\
      --masks masks/ -o out/masked.ply
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from ply_io import read_splats, write_splats
from pose_utils import load_poses, detect_convention, project, log as _log


def log(msg):
    _log("mask", msg)


def morph(mask, pixels):
    """Erode (pixels < 0) or dilate (pixels > 0) a boolean mask, 4-neighbour.

    Erosion is the useful direction here: it trims the halo of mixed
    subject/background Gaussians that sits on the silhouette edge.
    """
    if pixels == 0:
        return mask
    grow = pixels > 0
    out = mask.copy()
    for _ in range(abs(int(pixels))):
        shifted = [out,
                   np.roll(out, 1, 0), np.roll(out, -1, 0),
                   np.roll(out, 1, 1), np.roll(out, -1, 1)]
        # Edges roll around, so clamp the wrapped rows/cols to the safe value.
        for s, (ax, where) in zip(shifted[1:], [(0, 0), (0, -1), (1, 0), (1, -1)]):
            idx = [slice(None)] * 2
            idx[ax] = where
            s[tuple(idx)] = out[tuple(idx)]
        out = np.logical_or.reduce(shifted) if grow else np.logical_and.reduce(shifted)
    return out


def load_masks(mask_dir, count):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    paths = sorted(p for p in Path(mask_dir).iterdir() if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"no mask images in {mask_dir}")
    if len(paths) != count:
        log(f"WARNING: {len(paths)} masks but {count} poses. They are matched by "
            f"sorted order, so a mismatch means they are misaligned. "
            f"Using the first {min(len(paths), count)}.")
    return paths[:count]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ply", required=True, help="RAW gaussians.ply (pre-fixup)")
    ap.add_argument("--poses", required=True, help="poses.npz from run_anysplat.py")
    ap.add_argument("--masks", required=True, help="directory of per-frame masks")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--convention", choices=["auto", "w2c", "c2w"], default="auto")
    ap.add_argument("--min-agreement", type=float, default=0.8,
                    help="fraction of observing views that must agree the "
                         "Gaussian is inside the mask (default 0.8, measured; "
                         "0.5 majority vote gives only ~33%% precision because "
                         "the wall behind the subject is inside the mask in a "
                         "large minority of views)")
    ap.add_argument("--min-views", type=int, default=2,
                    help="Gaussians seen by fewer views than this are dropped as "
                         "unjudgeable (default 2, so one stray view cannot decide)")
    ap.add_argument("--erode", type=int, default=0,
                    help="erode masks by N pixels to trim silhouette halo; "
                         "negative dilates")
    ap.add_argument("--threshold", type=int, default=127,
                    help="mask grey level above which a pixel counts as inside")
    ap.add_argument("--invert", action="store_true",
                    help="keep the BACKGROUND instead of the subject")
    ap.add_argument("--keep-unseen", action="store_true",
                    help="keep Gaussians no view observes (default: drop)")
    args = ap.parse_args()

    splats = read_splats(args.ply)
    n = len(splats)
    log(f"read {n:,} splats from {args.ply}")

    ext, K, w, h = load_poses(args.poses, args.image_size, tag="mask")
    conv = detect_convention(ext, K, splats.means, w, h, args.convention, tag="mask")
    mask_paths = load_masks(args.masks, ext.shape[0])

    means = splats.means.astype(np.float64)
    votes_inside = np.zeros(n, np.int32)
    votes_seen = np.zeros(n, np.int32)

    for i, mp in enumerate(mask_paths):
        m = np.asarray(Image.open(mp).convert("L"))
        keep_px = m > args.threshold
        if args.invert:
            keep_px = ~keep_px
        if args.erode:
            keep_px = morph(keep_px, -abs(args.erode) if args.erode > 0 else abs(args.erode))
        mh, mw = keep_px.shape

        u, v, z = project(means, ext[i], K[i], conv)
        in_frame = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not in_frame.any():
            continue

        # Mask resolution need not match the projection frame.
        mx = np.clip((u[in_frame] / w * mw).astype(np.int64), 0, mw - 1)
        my = np.clip((v[in_frame] / h * mh).astype(np.int64), 0, mh - 1)

        idx = np.flatnonzero(in_frame)
        votes_seen[idx] += 1
        votes_inside[idx] += keep_px[my, mx].astype(np.int32)

    seen = votes_seen >= max(1, args.min_views)
    agreement = np.divide(votes_inside, np.maximum(votes_seen, 1), dtype=np.float64)
    keep = seen & (agreement >= args.min_agreement)

    never_seen = votes_seen == 0
    if args.keep_unseen:
        keep |= never_seen

    log(f"views per splat: median {int(np.median(votes_seen))}, "
        f"max {int(votes_seen.max())}, never seen {int(never_seen.sum()):,}")
    log(f"agreement >= {args.min_agreement} and >= {args.min_views} views: "
        f"keeping {int(keep.sum()):,} / {n:,} ({keep.sum()/n:.1%})")

    if keep.sum() == 0:
        sys.exit("mask kept nothing. Check that white = subject, that the masks "
                 "are in the same sorted order as the frames, and that the "
                 "detected pose convention was conclusive.")

    write_splats(splats.select(keep), args.output)
    log(f"wrote {int(keep.sum()):,} splats -> {args.output}")
    log("next: fixup_ply.py (axes/opacity/scale), then decimate_ply.py")


if __name__ == "__main__":
    main()
