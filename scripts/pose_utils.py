"""Shared handling of AnySplat's predicted camera poses.

Used by poses_to_colmap.py (stage D) and mask_splats.py (stage B2). Both need
the same two undocumented things resolved before they can project anything:
whether the extrinsics are world-to-camera or camera-to-world, and whether the
intrinsics are in pixels or normalised. Keeping that in one place means a fix
benefits both.
"""

import sys

import numpy as np


def log(tag, msg):
    print(f"[{tag}] {msg}", file=sys.stderr)


def squeeze_batch(a, rank):
    """Drop a leading batch axis of size 1 until the array has `rank` dims."""
    a = np.asarray(a, np.float64)
    while a.ndim > rank and a.shape[0] == 1:
        a = a[0]
    if a.ndim != rank:
        raise ValueError(f"expected rank {rank}, got shape {a.shape}")
    return a


def to_4x4(ext):
    """Accept (K,3,4) or (K,4,4); return (K,4,4)."""
    if ext.shape[-2:] == (4, 4):
        return ext
    if ext.shape[-2:] == (3, 4):
        out = np.tile(np.eye(4), (ext.shape[0], 1, 1))
        out[:, :3, :4] = ext
        return out
    raise ValueError(f"unsupported extrinsic shape {ext.shape}")


def load_poses(path, image_size, tag="pose"):
    """Load poses.npz -> (ext4 (K,4,4), K (K,3,3) in PIXELS, width, height)."""
    z = np.load(path)
    if "extrinsic" not in z:
        raise SystemExit(f"{path} has no 'extrinsic' array (keys: {list(z.keys())})")
    ext = to_4x4(squeeze_batch(z["extrinsic"], 3))
    k = ext.shape[0]
    w = h = image_size
    log(tag, f"{k} camera poses, extrinsic shape {ext.shape}")

    if "intrinsic" in z:
        K = squeeze_batch(z["intrinsic"], 3)
        if K.shape[0] != k:
            raise SystemExit(f"intrinsic count {K.shape[0]} != extrinsic count {k}")
        # Normalised intrinsics have fx ~ O(1); pixel intrinsics ~ O(image size).
        if float(np.median(K[:, 0, 0])) < 2.0:
            log(tag, f"intrinsics look NORMALISED (median fx="
                     f"{float(np.median(K[:, 0, 0])):.4f}); scaling by {w}x{h}")
            K = K.copy()
            K[:, 0, :] *= w
            K[:, 1, :] *= h
        else:
            log(tag, f"intrinsics look like PIXELS (median fx="
                     f"{float(np.median(K[:, 0, 0])):.1f})")
    else:
        log(tag, "no 'intrinsic' array; assuming a 60 deg horizontal FOV pinhole")
        f = 0.5 * w / np.tan(np.radians(30.0))
        K = np.tile(np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]]), (k, 1, 1))

    return ext, K, w, h


def project(pts, ext_i, K_i, convention):
    """Project (N,3) world points through one camera.

    Returns (u, v, z). z > 0 means in front of the camera.
    """
    R, t = ext_i[:3, :3], ext_i[:3, 3]
    if convention == "w2c":
        cam = pts @ R.T + t                   # X_cam = R X + t
    else:
        cam = (pts - t) @ R                   # X_cam = R^T (X - t)
    z = cam[:, 2]
    zc = np.where(z > 1e-6, z, 1.0)
    u = K_i[0, 0] * cam[:, 0] / zc + K_i[0, 2]
    v = K_i[1, 1] * cam[:, 1] / zc + K_i[1, 2]
    return u, v, z


def pose_metrics(ext4, K, pts, w, h, convention, bins=32):
    """Score one candidate interpretation of the extrinsics.

    The decision is driven by IMAGE FILL: bin each camera's image plane into a
    coarse grid and measure what fraction of bins receive at least one
    projected point. AnySplat's Gaussians are pixel-aligned -- each is
    unprojected from some input pixel -- so under the correct convention every
    camera's image plane should be almost completely covered, while a wrong
    convention leaves much of the frame empty.

    Two weaker metrics are reported but deliberately NOT used to decide:
      * `mean_in_frame` (in-frustum fraction over all camera/point pairs) is
        diluted when each camera sees only part of the scene, and a wrong
        convention can beat the right one by luck.
      * `coverage` (seen by >= 1 camera) saturates at 100% for both candidates
        as soon as the capture is a full orbit.
    Both were tried as the primary signal and both misfired on real geometry:
    on an inward room scan they still favour the WRONG convention.

    NOTE on sampling: image fill is only meaningful if enough points reach each
    frame to fill `bins**2` cells. With too small a probe the metric degenerates
    into counting in-frame points, which is the exact failure mode it replaced.
    Hence the per-camera loop (memory O(N), not O(K*N)) so callers can afford a
    large probe.
    """
    n = pts.shape[0]
    min_pts_needed = bins * bins * 4
    seen_any = np.zeros(n, bool)
    fills, in_front_frac, in_frame_frac, starved = [], [], [], 0

    for i in range(ext4.shape[0]):
        u, v, z = project(pts, ext4[i], K[i], convention)
        in_front = z > 1e-6
        inside = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)

        seen_any |= inside
        in_front_frac.append(in_front.mean())
        in_frame_frac.append(inside.mean())

        cnt = int(inside.sum())
        if cnt == 0:
            fills.append(0.0)
            continue
        if cnt < min_pts_needed:
            starved += 1
        bx = np.clip((u[inside] / w * bins).astype(np.int64), 0, bins - 1)
        by = np.clip((v[inside] / h * bins).astype(np.int64), 0, bins - 1)
        fills.append(np.unique(by * bins + bx).size / float(bins * bins))

    return {
        "image_fill": float(np.mean(fills)),
        "coverage": float(seen_any.mean()),
        "mean_in_frame": float(np.mean(in_frame_frac)),
        "in_front": float(np.mean(in_front_frac)),
        "starved_cams": starved,
    }


def detect_convention(ext4, K, means, w, h, requested="auto", tag="pose", seed=0):
    """Resolve w2c vs c2w, reporting the evidence. Returns the chosen string."""
    rng = np.random.default_rng(seed)
    n_probe = min(300_000, means.shape[0])
    idx = rng.choice(means.shape[0], n_probe, replace=False)
    probe = means[idx].astype(np.float64)
    log(tag, f"probing conventions with {n_probe:,} points")

    scores = {c: pose_metrics(ext4, K, probe, w, h, c) for c in ("w2c", "c2w")}
    for name, m in scores.items():
        warn = f"  [{m['starved_cams']} cams point-starved]" if m["starved_cams"] else ""
        log(tag, f"  {name}: image_fill={m['image_fill']:.1%}  "
                 f"coverage={m['coverage']:.1%}  in_front={m['in_front']:.1%}  "
                 f"mean_in_frame={m['mean_in_frame']:.1%}{warn}")

    if requested != "auto":
        log(tag, f"using --convention {requested} (override)")
        return requested

    conv = "w2c" if scores["w2c"]["image_fill"] >= scores["c2w"]["image_fill"] else "c2w"
    best = max(s["image_fill"] for s in scores.values())
    other = min(s["image_fill"] for s in scores.values())
    log(tag, f"auto-detected convention: {conv} (image_fill {best:.1%})")

    # The RATIO is the signal, not the absolute level: a small subject framed in
    # a wide FOV legitimately fills little of the frame under the correct
    # convention, so a low absolute fill is not itself suspicious.
    margin = best / max(other, 1e-6)
    if margin < 1.5:
        log(tag, f"WARNING: the two conventions score within {margin:.2f}x of each "
                 f"other, so this detection is NOT conclusive. Try both "
                 f"(--convention w2c / c2w) and compare results.")
    else:
        log(tag, f"margin {margin:.1f}x over the alternative -- conclusive")
    if best < 0.10:
        log(tag, f"note: absolute image_fill is low ({best:.1%}); expected if the "
                 f"subject is small in a wide frame, suspicious otherwise")
    return conv
