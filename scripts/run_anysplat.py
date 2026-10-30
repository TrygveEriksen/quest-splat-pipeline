#!/usr/bin/env python3
"""Stage B: run AnySplat on a folder of frames, and keep the camera poses.

The repo's own inference.py hard-codes `image_folder = "examples/vrnerf/riverview"`
and throws away `pred_context_pose`. Stage D (refinement) needs those poses, so
this wrapper turns inference into a real CLI and dumps them alongside the PLY.

Run this on the NVIDIA box (Linux or WSL2), inside AnySplat's conda env.
Native Windows is not worth it: requirements.txt pins a gsplat wheel built for
pt22cu121-cp310 linux_x86_64.

Usage:
  python run_anysplat.py --frames frames/ --out out/ --anysplat-root ~/AnySplat
"""

import argparse
import json
import sys
from pathlib import Path


def log(msg):
    print(f"[anysplat] {msg}", file=sys.stderr)


def load_preprocessor(root):
    """Prefer AnySplat's own image preprocessing; fall back with a loud warning.

    Guessing the preprocessing is risky -- a plain square resize distorts aspect
    ratio, which shifts the implied intrinsics and degrades the reconstruction.
    So we try the repo's function first and only fall back if it has moved.
    """
    candidates = [
        ("src.utils.image", "process_image"),
        ("src.misc.image_io", "process_image"),
        ("src.dataset.shims.load_shim", "process_image"),
    ]
    import importlib
    for mod, fn in candidates:
        try:
            m = importlib.import_module(mod)
            if hasattr(m, fn):
                log(f"using AnySplat's own preprocessing: {mod}.{fn}")
                return getattr(m, fn), True
        except Exception:
            continue

    log("WARNING: could not import AnySplat's process_image. Falling back to a "
        "448x448 resize that preserves aspect ratio by centre-cropping. If "
        "results look wrong, locate process_image in the repo and point "
        "load_preprocessor() at it.")

    import numpy as np
    import torch
    from PIL import Image

    def fallback(path, size=448):
        img = Image.open(path).convert("RGB")
        w, h = img.size
        s = min(w, h)
        img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        img = img.resize((size, size), Image.LANCZOS)
        a = np.asarray(img, np.float32) / 255.0
        return torch.from_numpy(a).permute(2, 0, 1) * 2.0 - 1.0   # -> [-1,1]

    return fallback, False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory of input frames")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--anysplat-root", required=True,
                    help="clone of github.com/InternRobotics/AnySplat")
    ap.add_argument("--checkpoint", default=None,
                    help="override; default tries lhjiang/anysplat then anysplat_ckpt_v1")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="truncate input; lower this on CUDA OOM")
    ap.add_argument("--keep-sh", action="store_true",
                    help="write all SH bands (much larger). Default is DC-only, "
                         "which is what the Quest build wants anyway")
    args = ap.parse_args()

    root = Path(args.anysplat_root).expanduser().resolve()
    if not (root / "src").is_dir():
        ap.error(f"{root} does not look like the AnySplat repo (no src/)")
    sys.path.insert(0, str(root))

    import torch
    from src.model.model.anysplat import AnySplat
    from src.model.ply_export import export_ply

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg"}
    paths = sorted(p for p in Path(args.frames).iterdir() if p.suffix.lower() in exts)
    if not paths:
        ap.error(f"no images in {args.frames}")
    if args.max_frames:
        paths = paths[:args.max_frames]
    log(f"{len(paths)} input frames")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log("WARNING: no CUDA device visible -- this will be unusably slow")

    names = [args.checkpoint] if args.checkpoint else ["lhjiang/anysplat", "anysplat_ckpt_v1"]
    model = None
    for name in names:
        try:
            log(f"loading weights: {name}")
            model = AnySplat.from_pretrained(name)
            break
        except Exception as e:
            log(f"  failed ({type(e).__name__}: {e})")
    if model is None:
        sys.exit(f"could not load weights from any of {names}")

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    process_image, native = load_preprocessor(root)
    images = torch.stack([process_image(str(p)) for p in paths], 0)
    images = images.unsqueeze(0).to(device)          # [1, K, 3, 448, 448]
    log(f"input tensor {tuple(images.shape)}")

    try:
        with torch.no_grad():
            # The repo calls inference((images+1)*0.5), i.e. it expects [0,1].
            gaussians, pred_pose = model.inference((images + 1) * 0.5)
    except torch.cuda.OutOfMemoryError:
        sys.exit(f"CUDA OOM with {len(paths)} frames. Re-run with "
                 f"--max-frames {max(2, len(paths)//2)}, or re-extract fewer frames.")

    n = gaussians.means[0].shape[0]
    log(f"produced {n:,} Gaussians")
    if n > 1_000_000:
        log(f"note: {n:,} is far above a Quest 2 budget (~150-250k). "
            f"decimate_ply.py handles that in stage E.")

    ply_path = outdir / "gaussians.ply"
    export_ply(
        gaussians.means[0], gaussians.scales[0], gaussians.rotations[0],
        gaussians.harmonics[0], gaussians.opacities[0], ply_path,
        save_sh_dc_only=not args.keep_sh,
    )
    log(f"wrote {ply_path}")

    # Keep the poses: stage D cannot run without them, and re-running inference
    # just to recover them costs another full forward pass.
    import numpy as np
    pose_npz = outdir / "poses.npz"
    payload = {}
    for key in ("extrinsic", "intrinsic"):
        if key in pred_pose:
            payload[key] = pred_pose[key].detach().cpu().numpy()
            log(f"pose '{key}': shape {payload[key].shape}")
    np.savez(pose_npz, **payload)
    (outdir / "frames.json").write_text(json.dumps([p.name for p in paths], indent=2))
    log(f"wrote {pose_npz} and frames.json ({len(paths)} frame names, in input order)")
    log(f"preprocessing used: {'AnySplat native' if native else 'FALLBACK (verify!)'}")
    log("next: inspect gaussians.ply in https://playcanvas.com/supersplat/editor "
        "before running stage C")


if __name__ == "__main__":
    main()
