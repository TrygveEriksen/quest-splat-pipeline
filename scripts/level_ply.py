#!/usr/bin/env python3
"""Stage B3: stand a masked subject upright on the surface it was filmed on.

AnySplat works in the first camera's frame (OpenCV: x right, y down, z
forward), so "up" in its output is wherever the phone pointed in frame 0. An
object filmed from above comes out tilted by the camera's pitch, and can't sit
flat on a table in the headset.

The object was standing on something, though, and that surface is still in the
raw reconstruction -- masking only removed it. This fits a plane to the raw
Gaussians around the object (RANSAC, restricted to roughly-horizontal planes),
rotates so the plane is level, and moves the origin to the point under the
object's centre, on the plane. The Unity app then puts that origin on a real
table and the object rests on it.

    level_ply.py --raw raw/gaussians.ply --subject masked.ply -o leveled.ply

Run it on the files before fixup_ply.py; the output stays in the OpenCV frame
(y down), so `fixup_ply.py --axes none` applies unchanged. Don't pass
--recenter to fixup afterwards: that would throw the meaningful origin away.
Exits non-zero (writing nothing) when no convincing surface is found, so a
caller can fall back to plain recentring.
"""
import argparse
import sys

import numpy as np

from ply_io import Splats, mat_to_quat, quat_to_mat, read_splats, write_splats

UP = np.array([0.0, -1.0, 0.0])   # OpenCV y is down


def log(msg):
    print(f"[level] {msg}", flush=True)


def alpha(splats):
    """Raw AnySplat files hold activated alpha; fixed-up ones hold logits."""
    o = splats.opacity.astype(np.float64)
    if o.min() >= 0.0 and o.max() <= 1.0:
        return o
    return 1.0 / (1.0 + np.exp(-o))


def camera_up(poses, image_size, convention, means):
    """Gravity from the capture itself: people hold a phone upright, and over
    an orbit each frame's pitch toward the object cancels out, so the mean of
    the cameras' up vectors points up. Returns (unit vector, agreement 0-1)."""
    from pose_utils import detect_convention, load_poses
    ext, K, w, h = load_poses(poses, image_size, tag="level")
    conv = detect_convention(ext, K, means, w, h, convention, tag="level")
    R = ext[:, :3, :3] if conv == "c2w" else np.transpose(ext[:, :3, :3], (0, 2, 1))
    mean = (-R[:, :, 1]).mean(0)            # each camera's up (-y), in world
    return mean / np.linalg.norm(mean), float(np.linalg.norm(mean))


def fit_plane_near(points, prior, threshold, base, radius, iters=10):
    """Deterministic fit when the expected up is reliable: the surface is the
    densest layer of points just under the subject's base along `prior`;
    then refine its tilt by least squares over everything in that layer.
    Returns (normal, point_on_plane, inlier_mask) or None."""
    h = points @ prior
    window = (h > base - 0.5 * radius) & (h < base + 0.1 * radius)
    if window.sum() < 50:
        return None
    counts, edges = np.histogram(h[window], bins=100)
    d = float(edges[counts.argmax()] + edges[counts.argmax() + 1]) / 2
    normal = prior.copy()
    for _ in range(iters):
        inliers = np.abs(points @ normal - d) < threshold
        if inliers.sum() < 50:
            return None
        centroid = points[inliers].mean(0)
        normal = np.linalg.svd(points[inliers] - centroid, full_matrices=False)[2][-1]
        if normal @ prior < 0:
            normal = -normal
        d = float(centroid @ normal)
    return normal, centroid, np.abs(points @ normal - d) < threshold


def fit_plane(points, threshold, prior, max_tilt_deg, rng, iters=2000):
    """RANSAC over planes whose normal is within max_tilt of the prior up.
    Returns (normal, point_on_plane, inlier_mask) refined by SVD, or None."""
    n_pts = len(points)
    if n_pts < 50:
        return None
    cos_tilt = np.cos(np.radians(max_tilt_deg))
    best = (0, None)
    for _ in range(iters):
        a, b, c = points[rng.choice(n_pts, 3, replace=False)]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        if abs(normal @ prior) < cos_tilt:
            continue
        inliers = np.abs((points - a) @ normal) < threshold
        count = int(inliers.sum())
        if count > best[0]:
            best = (count, inliers)
    if best[1] is None:
        return None

    # Refine: least-squares plane through the inliers (smallest singular
    # vector), re-select inliers, repeat. Converges to the same plane whichever
    # random triple RANSAC happened to land on.
    inliers = best[1]
    for _ in range(5):
        centroid = points[inliers].mean(0)
        normal = np.linalg.svd(points[inliers] - centroid, full_matrices=False)[2][-1]
        inliers = np.abs((points - centroid) @ normal) < threshold
    return normal, centroid, inliers


def rotation_between(a, b):
    """Rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(a @ b)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # 180 degrees: any axis perpendicular to a.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="full AnySplat output (pre-mask)")
    ap.add_argument("--subject", required=True, help="masked subject from mask_splats.py")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--search-radius", type=float, default=2.5,
                    help="look for the surface within this many subject radii (default 2.5)")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="plane inlier distance, as a fraction of the subject radius")
    ap.add_argument("--poses", help="poses.npz from run_anysplat.py: estimates up from "
                                    "the cameras, which steadies the fit a lot")
    ap.add_argument("--image-size", type=int, default=448)
    ap.add_argument("--convention", choices=["auto", "w2c", "c2w"], default="auto")
    ap.add_argument("--max-tilt", type=float, default=None,
                    help="only accept planes within this many degrees of the expected up "
                         "(default 15 with --poses, 60 without)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subject = read_splats(args.subject)
    raw = read_splats(args.raw)
    pts = subject.means.astype(np.float64)
    center = np.median(pts, 0)
    radius = float(np.percentile(np.linalg.norm(pts - center, axis=1), 90))
    log(f"subject: {len(subject):,} splats, radius {radius:.3f}")

    # The raw cloud minus the subject itself (masked is an exact subset).
    key = lambda m: np.ascontiguousarray(m, np.float32).view("V12").ravel()
    others = ~np.isin(key(raw.means), key(subject.means))
    near = others & (np.linalg.norm(raw.means - center, axis=1) < args.search_radius * radius)
    # AnySplat builds surfaces from many faint Gaussians (alpha ~0.01-0.1), so
    # a fixed cutoff either keeps everything or almost nothing. Keep the more
    # opaque half of the neighbourhood instead: haze and floaters drop out.
    a = alpha(raw)
    solid = a >= np.median(a[near]) if near.any() else near
    cand = raw.means[near & solid].astype(np.float64)
    log(f"surface candidates around the subject: {len(cand):,}")

    # Expected up: from the cameras if we have them and they agree, else the
    # first camera's up with a wide window.
    prior, max_tilt, trusted = UP, args.max_tilt or 60.0, False
    if args.poses:
        up, agreement = camera_up(args.poses, args.image_size, args.convention, raw.means)
        log(f"camera up: {np.degrees(np.arccos(np.clip(up @ UP, -1, 1))):.1f} deg from "
            f"camera 0's (agreement {agreement:.2f})")
        if agreement >= 0.5:
            prior, max_tilt, trusted = up, args.max_tilt or 15.0, True
        else:
            log("cameras disagree about up (not an upright capture?); not using them")

    threshold = args.threshold * radius
    if trusted:
        # Deterministic: same input, same levelling.
        base = float(np.percentile(pts @ prior, 1))
        fit = fit_plane_near(cand, prior, threshold, base, radius)
    else:
        rng = np.random.default_rng(args.seed)
        if len(cand) > 200_000:
            cand = cand[rng.choice(len(cand), 200_000, replace=False)]
        fit = fit_plane(cand, threshold, prior, max_tilt, rng)
    if fit is None:
        sys.exit("[level] no roughly-horizontal surface found near the subject")
    normal, on_plane, inliers = fit

    # Point the normal toward the subject: the subject stands on top.
    if normal @ (center - on_plane) < 0:
        normal = -normal
    heights = (pts - on_plane) @ normal
    above = float((heights > -2 * threshold).mean())
    gap = float(np.percentile(heights, 2)) / radius   # how far the base is off the surface
    tilt = float(np.degrees(np.arccos(np.clip(normal @ UP, -1, 1))))
    off_prior = float(np.degrees(np.arccos(np.clip(normal @ prior, -1, 1))))
    log(f"surface: {int(inliers.sum()):,} inliers ({inliers.mean():.0%}), "
        f"tilted {tilt:.1f} deg from camera 0's up and {off_prior:.1f} deg from the expected up; "
        f"{above:.0%} of the subject above it, base {gap:+.2f} radii from it")
    # The neighbourhood also holds floor, walls and clutter, so the inlier share
    # is only a sanity floor; what matters is that the object rests on the plane.
    if inliers.mean() < 0.03 or above < 0.9 or abs(gap) > 0.2 or off_prior > max_tilt:
        sys.exit("[level] surface fit not convincing; leaving the subject as it is")

    # Level the surface, then put the origin under the subject at its own base.
    # The plane gives the tilt; the subject's lowest points give the height, as
    # a reconstructed surface is a few mm thick and would leave a visible gap.
    R = rotation_between(normal, UP)
    base = float(np.percentile(heights, 1))
    foot = center - (normal @ (center - on_plane) - base) * normal
    means = (pts - foot) @ R.T
    rot = mat_to_quat(R[None] @ quat_to_mat(subject.rot))

    out = Splats(means, subject.sh_dc, subject.opacity, subject.scales, rot, subject.sh_rest)
    write_splats(out, args.output)
    height = float((means @ UP).max())   # distance from the surface to the top
    log(f"levelled by {tilt:.1f} deg; origin on the surface; subject {height:.3f} tall")
    log(f"wrote {len(out):,} splats -> {args.output}")


if __name__ == "__main__":
    main()
