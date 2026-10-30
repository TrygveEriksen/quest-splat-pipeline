#!/usr/bin/env python3
"""Stage C: turn AnySplat's PLY export into something a Unity splat importer
reads correctly.

Three independent corrections, each of which produces a distinctive failure if
skipped:

1. OPACITY.  AnySplat's src/model/ply_export.py writes the raw `opacities`
   tensor with no inverse-sigmoid, while INRIA-convention loaders apply
   sigmoid() on read. If the values are already activated ([0,1]), every loader
   double-activates them, squashing all alpha into ~0.5-0.73 -> the scene
   renders as uniform milky fog.

2. AXES.  AnySplat is distilled from VGGT and inherits an OpenCV-style camera
   frame (X right, Y down, Z forward, right-handed). Unity is left-handed, Y up.
   The flip is a mirror (determinant -1), so it CANNOT be applied as a Unity
   object transform without corrupting the per-splat quaternions -- it has to be
   baked into the covariances here. See _apply_axes for the maths.

3. SCALE.  Feed-forward output has arbitrary scale. In VR a wrongly-scaled
   scene reads as either a dollhouse or a cathedral, so pass --scale derived
   from one known real-world measurement in the capture.

Usage:
  python fixup_ply.py gaussians.ply -o unity_raw.ply --scale 2.4
"""

import argparse
import sys

import numpy as np

from ply_io import read_splats, write_splats, quat_to_mat, mat_to_quat

# Named presets. Each is the diagonal of a signed-permutation-free mirror/flip.
AXES_PRESETS = {
    "opencv-to-unity": (1.0, -1.0, 1.0),   # Y down -> Y up, flips handedness
    "flip-y": (1.0, -1.0, 1.0),
    "flip-z": (1.0, 1.0, -1.0),
    "flip-yz": (1.0, -1.0, -1.0),          # determinant +1, a pure rotation
    "none": (1.0, 1.0, 1.0),
}


def log(msg):
    print(f"[fixup] {msg}", file=sys.stderr)


def fix_opacity(splats, mode):
    """Decide whether the opacity column needs an inverse-sigmoid, and say so."""
    o = splats.opacity
    lo, hi = float(o.min()), float(o.max())
    log(f"opacity range: [{lo:.4f}, {hi:.4f}]  mean={float(o.mean()):.4f}")

    if mode == "raw":
        log("opacity: --opacity-mode raw, leaving column untouched")
        return
    if mode == "logit":
        decision = True
    else:  # auto
        # Logit-space opacities from a real scene span roughly -10..+10. Values
        # confined to [0,1] mean the column is already activated alpha.
        decision = lo >= 0.0 and hi <= 1.0
        log(f"opacity: auto-detected {'ACTIVATED [0,1]' if decision else 'LOGIT space'}")

    if decision:
        p = np.clip(o.astype(np.float64), 1e-6, 1.0 - 1e-6)
        splats.opacity = np.log(p / (1.0 - p)).astype(np.float32)
        log(f"opacity: applied logit() -> new range "
            f"[{float(splats.opacity.min()):.3f}, {float(splats.opacity.max()):.3f}]")
    else:
        log("opacity: already in logit space, left untouched")


def _apply_axes(splats, diag):
    """Apply M = diag(dx,dy,dz), dk in {+1,-1}, to means and covariances.

    A Gaussian's covariance is Sigma = R S S^T R^T. Under a linear map M,
    Sigma' = (M R) S S^T (M R)^T.

    If det(M) = +1 then M R is already a rotation and we just re-encode it.

    If det(M) = -1 then M R is a reflection, which has no quaternion. But
    S S^T is diagonal, so for any diagonal sign matrix D we have
    D S S^T D^T = S S^T, hence

        Sigma' = (M R D) (D^T S S^T D) (M R D)^T = (M R D) S S^T (M R D)^T

    and picking D with det(D) = -1 makes R' = M R D a proper rotation that
    reproduces the mirrored covariance exactly, with the scales unchanged.
    """
    M = np.diag(np.asarray(diag, np.float64))
    det = float(np.linalg.det(M))

    splats.means = (splats.means @ M.T).astype(np.float32)

    R = quat_to_mat(splats.rot)                      # (N,3,3)
    if det < 0:
        D = np.diag([1.0, 1.0, -1.0])                # det -1, restores a rotation
        Rp = M @ R @ D
    else:
        Rp = M @ R
    splats.rot = mat_to_quat(Rp).astype(np.float32)
    # Scales are untouched by construction -- see the docstring.
    log(f"axes: applied diag{tuple(diag)} (det={det:+.0f}"
        f"{', mirror compensated in quaternions' if det < 0 else ''})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--opacity-mode", choices=["auto", "logit", "raw"], default="auto",
                    help="auto (default) inspects the column range and decides")
    ap.add_argument("--axes", choices=sorted(AXES_PRESETS), default="opencv-to-unity",
                    help="use 'none' if your importer already does the flip")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="uniform world scale factor to reach metric size")
    ap.add_argument("--recenter", action="store_true",
                    help="move the median Gaussian to the origin")
    ap.add_argument("--keep-sh", action="store_true",
                    help="keep f_rest_* bands. NOT rotated by --axes, so colours "
                         "will be wrong at angles; Quest wants DC-only anyway")
    args = ap.parse_args()

    splats = read_splats(args.input)
    log(f"read {len(splats):,} splats from {args.input} "
        f"(f_rest columns: {splats.sh_rest_count})")

    fix_opacity(splats, args.opacity_mode)

    if splats.sh_rest_count:
        if args.keep_sh:
            log("WARNING: keeping f_rest_* but NOT rotating it -- view-dependent "
                "colour will be wrong after an axis flip")
        else:
            log(f"SH: stripping {splats.sh_rest_count} f_rest_* columns, keeping DC only")
            splats.sh_rest = None

    _apply_axes(splats, AXES_PRESETS[args.axes])

    if args.recenter:
        centre = np.median(splats.means, axis=0)
        splats.means = (splats.means - centre).astype(np.float32)
        log(f"recentred by {-centre}")

    if args.scale != 1.0:
        k = float(args.scale)
        if k <= 0:
            ap.error("--scale must be positive")
        splats.means = (splats.means * k).astype(np.float32)
        # scale_* is log-encoded, so a world scale of k is an ADDITION of log(k).
        splats.scales = (splats.scales + np.log(k)).astype(np.float32)
        log(f"scale: x{k} (means multiplied, log-scales offset by {np.log(k):+.4f})")

    extent = splats.means.max(0) - splats.means.min(0)
    log(f"final bbox extent: {extent[0]:.3f} x {extent[1]:.3f} x {extent[2]:.3f} units")
    write_splats(splats, args.output)
    log(f"wrote {len(splats):,} splats -> {args.output}")


if __name__ == "__main__":
    main()
