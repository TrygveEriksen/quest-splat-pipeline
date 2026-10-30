"""Read/write 3D Gaussian Splat PLY files in the INRIA convention.

Shared by fixup_ply.py and decimate_ply.py. Only needs numpy + plyfile, both of
which are already in AnySplat's requirements.txt.

Storage convention (INRIA 3DGS, what every viewer/importer expects):
  x, y, z            world-space mean
  nx, ny, nz         unused, written as zeros
  f_dc_0..2          SH degree-0 (DC) colour coefficients
  f_rest_0..N        higher SH bands, optional
  opacity            LOGIT space (viewers apply sigmoid on read)
  scale_0..2         LOG space (viewers apply exp on read)
  rot_0..3           quaternion, w first
"""

import numpy as np
from plyfile import PlyData, PlyElement


class Splats:
    """A set of Gaussians, held in the same encoding the PLY uses."""

    def __init__(self, means, sh_dc, opacity, scales, rot, sh_rest=None):
        self.means = np.asarray(means, np.float32)       # (N,3) world space
        self.sh_dc = np.asarray(sh_dc, np.float32)       # (N,3)
        self.opacity = np.asarray(opacity, np.float32)   # (N,) logit space
        self.scales = np.asarray(scales, np.float32)     # (N,3) log space
        self.rot = np.asarray(rot, np.float32)           # (N,4) wxyz
        self.sh_rest = None if sh_rest is None else np.asarray(sh_rest, np.float32)

    def __len__(self):
        return self.means.shape[0]

    @property
    def sh_rest_count(self):
        return 0 if self.sh_rest is None else self.sh_rest.shape[1]

    def select(self, idx):
        return Splats(
            self.means[idx], self.sh_dc[idx], self.opacity[idx],
            self.scales[idx], self.rot[idx],
            None if self.sh_rest is None else self.sh_rest[idx],
        )

    def linear_scales(self):
        """exp() of the stored log scales, i.e. real world-space radii."""
        return np.exp(self.scales)

    def alpha(self):
        """sigmoid() of the stored logit opacity, i.e. real alpha in [0,1]."""
        return 1.0 / (1.0 + np.exp(-self.opacity))


def read_splats(path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = set(v.data.dtype.names)

    def col(name):
        return np.asarray(v[name], np.float32)

    for required in ("x", "y", "z", "opacity"):
        if required not in names:
            raise ValueError(f"{path}: not a Gaussian splat PLY (missing '{required}')")

    means = np.stack([col("x"), col("y"), col("z")], 1)
    opacity = col("opacity")

    if "f_dc_0" in names:
        sh_dc = np.stack([col(f"f_dc_{i}") for i in range(3)], 1)
    else:
        raise ValueError(f"{path}: missing f_dc_* colour coefficients")

    scales = np.stack([col(f"scale_{i}") for i in range(3)], 1)
    rot = np.stack([col(f"rot_{i}") for i in range(4)], 1)

    # f_rest_* must be collected in numeric order, not the set's arbitrary order.
    rest_names = sorted(
        (n for n in names if n.startswith("f_rest_")),
        key=lambda n: int(n.split("_")[-1]),
    )
    sh_rest = np.stack([col(n) for n in rest_names], 1) if rest_names else None

    return Splats(means, sh_dc, opacity, scales, rot, sh_rest)


def write_splats(splats, path):
    n = len(splats)
    fields = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    fields += [f"f_rest_{i}" for i in range(splats.sh_rest_count)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]

    arr = np.empty(n, dtype=[(f, "f4") for f in fields])
    arr["x"], arr["y"], arr["z"] = splats.means.T
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    for i in range(3):
        arr[f"f_dc_{i}"] = splats.sh_dc[:, i]
    for i in range(splats.sh_rest_count):
        arr[f"f_rest_{i}"] = splats.sh_rest[:, i]
    arr["opacity"] = splats.opacity
    for i in range(3):
        arr[f"scale_{i}"] = splats.scales[:, i]
    for i in range(4):
        arr[f"rot_{i}"] = splats.rot[:, i]

    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


# --- quaternion helpers (w, x, y, z order, matching the PLY layout) ---

def quat_to_mat(q):
    """(N,4) wxyz -> (N,3,3) rotation matrices."""
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def mat_to_quat(R):
    """(N,3,3) rotation matrices -> (N,4) wxyz. Shepperd's method: picks the
    branch with the largest denominator so it stays stable near 180 degrees."""
    R = np.asarray(R, np.float64)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    q = np.empty((R.shape[0], 4))
    b0 = tr > 0
    b1 = (~b0) & (m00 >= m11) & (m00 >= m22)
    b2 = (~b0) & (~b1) & (m11 >= m22)
    b3 = ~(b0 | b1 | b2)

    s = np.sqrt(np.maximum(tr[b0] + 1.0, 1e-12)) * 2
    q[b0, 0] = 0.25 * s
    q[b0, 1] = (m21[b0] - m12[b0]) / s
    q[b0, 2] = (m02[b0] - m20[b0]) / s
    q[b0, 3] = (m10[b0] - m01[b0]) / s

    s = np.sqrt(np.maximum(1.0 + m00[b1] - m11[b1] - m22[b1], 1e-12)) * 2
    q[b1, 0] = (m21[b1] - m12[b1]) / s
    q[b1, 1] = 0.25 * s
    q[b1, 2] = (m01[b1] + m10[b1]) / s
    q[b1, 3] = (m02[b1] + m20[b1]) / s

    s = np.sqrt(np.maximum(1.0 + m11[b2] - m00[b2] - m22[b2], 1e-12)) * 2
    q[b2, 0] = (m02[b2] - m20[b2]) / s
    q[b2, 1] = (m01[b2] + m10[b2]) / s
    q[b2, 2] = 0.25 * s
    q[b2, 3] = (m12[b2] + m21[b2]) / s

    s = np.sqrt(np.maximum(1.0 + m22[b3] - m00[b3] - m11[b3], 1e-12)) * 2
    q[b3, 0] = (m10[b3] - m01[b3]) / s
    q[b3, 1] = (m02[b3] + m20[b3]) / s
    q[b3, 2] = (m12[b3] + m21[b3]) / s
    q[b3, 3] = 0.25 * s

    return q / np.linalg.norm(q, axis=1, keepdims=True)
