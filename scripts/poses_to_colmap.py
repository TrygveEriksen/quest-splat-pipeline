#!/usr/bin/env python3
"""Stage D prerequisite: write a COLMAP-format dataset from AnySplat's predicted
poses, so gsplat's simple_trainer can refine the scene.

gsplat's `default` strategy expects a COLMAP directory (images/ + sparse/0/),
which AnySplat does not emit. The pose convention is undocumented, so it is
detected empirically -- see pose_utils.detect_convention.

Usage:
  python poses_to_colmap.py --poses out/poses.npz --ply out/gaussians.ply \\
      --frames frames/ --out colmap_scene/
  python src/post_opt/simple_trainer.py default --data_dir colmap_scene/
"""

import argparse
import shutil
from pathlib import Path

import numpy as np

from ply_io import read_splats, mat_to_quat
from pose_utils import load_poses, detect_convention, log as _log

SH_C0 = 0.28209479177387814


def log(msg):
    _log("colmap", msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", required=True, help="poses.npz from run_anysplat.py")
    ap.add_argument("--ply", required=True, help="gaussians.ply (init point cloud)")
    ap.add_argument("--frames", required=True, help="the input frame directory")
    ap.add_argument("--out", required=True, help="COLMAP dataset directory to create")
    ap.add_argument("--image-size", type=int, default=448,
                    help="resolution the model ran at (default 448)")
    ap.add_argument("--convention", choices=["auto", "w2c", "c2w"], default="auto")
    ap.add_argument("--max-points", type=int, default=100_000)
    args = ap.parse_args()

    ext, K, w, h = load_poses(args.poses, args.image_size, tag="colmap")
    k = ext.shape[0]

    splats = read_splats(args.ply)
    log(f"{len(splats):,} Gaussians available as the init point cloud")

    conv = detect_convention(ext, K, splats.means, w, h, args.convention, tag="colmap")
    # COLMAP stores world-to-camera, so invert if the model gave us c2w.
    w2c = ext if conv == "w2c" else np.linalg.inv(ext)

    out = Path(args.out)
    (out / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in exts)
    if len(frames) != k:
        log(f"WARNING: {len(frames)} frames but {k} poses. They must correspond "
            f"1:1 in sorted order; truncating to {min(len(frames), k)}.")
    n = min(len(frames), k)
    for p in frames[:n]:
        shutil.copy2(p, out / "images" / p.name)

    with open(out / "sparse" / "0" / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy]\n")
        for i in range(n):
            f.write(f"{i+1} PINHOLE {w} {h} {K[i,0,0]:.6f} {K[i,1,1]:.6f} "
                    f"{K[i,0,2]:.6f} {K[i,1,2]:.6f}\n")

    quats = mat_to_quat(w2c[:n, :3, :3])
    with open(out / "sparse" / "0" / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        for i in range(n):
            q, t = quats[i], w2c[i, :3, 3]
            f.write(f"{i+1} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i+1} {frames[i].name}\n")
            f.write("\n")          # no 2D observations; the band must still exist

    rng = np.random.default_rng(0)
    m = min(args.max_points, len(splats))
    sel = rng.choice(len(splats), m, replace=False)
    rgb = np.clip((0.5 + SH_C0 * splats.sh_dc[sel]) * 255.0, 0, 255).astype(int)
    with open(out / "sparse" / "0" / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for j in range(m):
            x, y, z = splats.means[sel[j]]
            r, g, b = rgb[j]
            f.write(f"{j+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")

    log(f"wrote {out}/ -- {n} images, {m:,} init points")
    log(f"next: python src/post_opt/simple_trainer.py default --data_dir {out}")


if __name__ == "__main__":
    main()
