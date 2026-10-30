#!/usr/bin/env python3
"""Stage E: cut a splat scene down to something a Quest 2 can actually draw.

Published Quest 3 budgets are ~400-500k splats at 72 Hz. Quest 2 is roughly
half that machine (Adreno 650 vs 740), so plan for 150-250k and confirm by
profiling on device.

Three passes, in this order and for these reasons:

1. Opacity prune -- near-transparent splats cost full fill rate and contribute
   almost nothing. Cheapest possible win.
2. Size prune -- a handful of enormous splats (sky, background floaters) can
   each cover the whole viewport. These are the documented cause of the "gets
   slow when you walk close to it" complaint that every Quest splat project
   reports, so this pass matters more on a headset than on desktop.
3. Subsample to the target count -- this is the pass that actually buys frames,
   because on-device cost tracks splat count and overdraw.

Default method is 'voxel': auto-tune a grid size and keep the most important
splat per cell. That keeps coverage spatially even. Pure importance sampling
('importance') is also available but tends to thin out dark or matte regions
into visible holes.

Usage:
  python decimate_ply.py refined.ply -o quest.ply --target 150000
"""

import argparse
import sys

import numpy as np

from ply_io import read_splats, write_splats


def log(msg):
    print(f"[decimate] {msg}", file=sys.stderr)


def importance(splats):
    """alpha * footprint. Screen footprint grows with the square of the radius,
    so a splat's cost/benefit tracks alpha * scale^2."""
    lin = splats.linear_scales()
    radius = np.exp(np.log(np.maximum(lin, 1e-12)).mean(axis=1))   # geometric mean
    return splats.alpha() * radius ** 2


def voxel_keys(means, vsize):
    """Pack integer voxel coords into one int64 key. Exact (no hashing), as long
    as the grid fits in 21 bits per axis."""
    ijk = np.floor(means / vsize).astype(np.int64)
    ijk -= ijk.min(axis=0)
    dims = ijk.max(axis=0) + 1
    if np.any(dims > (1 << 21)):
        return None, dims
    return (ijk[:, 0] * (1 << 21) + ijk[:, 1]) * (1 << 21) + ijk[:, 2], dims


def voxel_select(splats, target, imp):
    """Bisect the voxel size until the surviving count lands near `target`."""
    means = splats.means.astype(np.float64)
    extent = means.max(axis=0) - means.min(axis=0)
    diag = float(np.linalg.norm(extent))

    # Count shrinks monotonically as voxel size grows, so bisect in log space.
    lo, hi = diag * 1e-5, diag
    best = None
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        keys, dims = voxel_keys(means, mid)
        if keys is None:
            lo = mid                      # grid too fine to index; coarsen
            continue
        count = len(np.unique(keys))
        if best is None or abs(count - target) < abs(best[1] - target):
            best = (mid, count)
        if count > target:
            lo = mid
        else:
            hi = mid
        if abs(count - target) <= max(500, target * 0.01):
            break

    vsize, count = best
    log(f"voxel size {vsize:.6g} -> {count:,} occupied cells (target {target:,})")

    keys, _ = voxel_keys(means, vsize)
    # Sort by descending importance, then take the first hit per voxel: that is
    # the most important splat in each cell.
    order = np.argsort(-imp, kind="stable")
    _, first = np.unique(keys[order], return_index=True)
    return order[first]


def importance_select(splats, target, imp):
    """Efraimidis-Spirakis A-Res: weighted sampling without replacement in one
    pass, via key = u^(1/w)."""
    rng = np.random.default_rng(0)
    w = np.maximum(imp, 1e-20)
    u = rng.random(len(w))
    keys = np.log(u) / w            # equivalent ordering to u^(1/w), no overflow
    return np.argsort(-keys)[:target]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--target", type=int, default=150_000,
                    help="splat count to aim for (Quest 2: start 150k)")
    ap.add_argument("--method", choices=["voxel", "importance"], default="voxel")
    ap.add_argument("--min-alpha", type=float, default=0.02,
                    help="drop splats whose sigmoid(opacity) is below this")
    ap.add_argument("--max-extent-frac", type=float, default=0.05,
                    help="drop splats whose radius exceeds this fraction of the "
                         "scene bbox diagonal (kills viewport-filling floaters)")
    args = ap.parse_args()

    splats = read_splats(args.input)
    n0 = len(splats)
    log(f"read {n0:,} splats from {args.input}")

    # --- pass 1: opacity ---
    alpha = splats.alpha()
    keep = alpha >= args.min_alpha
    log(f"opacity prune (alpha >= {args.min_alpha}): "
        f"-{n0 - int(keep.sum()):,} -> {int(keep.sum()):,}")
    splats = splats.select(keep)
    if len(splats) == 0:
        sys.exit("everything was pruned -- is --min-alpha too high, or did stage C "
                 "mis-handle the opacity encoding?")

    # --- pass 2: oversized splats ---
    extent = splats.means.max(axis=0) - splats.means.min(axis=0)
    diag = float(np.linalg.norm(extent))
    limit = diag * args.max_extent_frac
    radius = splats.linear_scales().max(axis=1)
    keep = radius <= limit
    log(f"size prune (radius <= {limit:.4g}, {args.max_extent_frac:.0%} of "
        f"{diag:.4g} diag): -{len(splats) - int(keep.sum()):,} -> {int(keep.sum()):,}")
    splats = splats.select(keep)

    # --- pass 3: subsample ---
    if len(splats) <= args.target:
        log(f"already at or under target ({len(splats):,} <= {args.target:,}), "
            f"no subsampling")
    else:
        imp = importance(splats)
        if args.method == "voxel":
            idx = voxel_select(splats, args.target, imp)
        else:
            idx = importance_select(splats, args.target, imp)
        log(f"{args.method} subsample: {len(splats):,} -> {len(idx):,}")
        splats = splats.select(np.sort(idx))

    write_splats(splats, args.output)
    mb = len(splats) * 48 / 1e6
    log(f"wrote {len(splats):,} splats -> {args.output}")
    log(f"kept {len(splats) / n0:.1%} of the input; "
        f"~{mb:.1f} MB GPU memory at 48 bytes/splat")


if __name__ == "__main__":
    main()
