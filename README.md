# Video → feed-forward 3D Gaussians → Quest 2

A pipeline that takes a handheld video, runs a **feed-forward** model
(AnySplat — one forward pass, no COLMAP, no per-scene training from scratch),
and produces a 3D Gaussian Splat scene that renders on a **standalone Quest 2**
in Unity.

## Layout

```
scripts/
  ply_io.py             shared 3DGS PLY reader/writer + quaternion helpers
  pose_utils.py         shared pose loading, projection, convention detection
  extract_frames.py     A: video -> sharp, evenly-spread frames
  run_anysplat.py       B: inference CLI (+ dumps the camera poses)
  make_masks_sam2.py    B2: per-frame subject masks via SAM 2 (optional)
  mask_splats.py        B2: keep only Gaussians inside the masks (optional)
  fixup_ply.py          C: opacity / axes / metric scale corrections
  poses_to_colmap.py    D: COLMAP export for the refinement pass
  decimate_ply.py       E: prune + subsample to a Quest 2 budget
  pipeline.sh           A -> E end to end
unity/QuestSplatViewer/Assets/Scripts/
  SplatLoaderAdapter.cs the only file touching the UnitySplats package API
  SplatSceneBrowser.cs  runtime splat loading (adb push, no APK rebuild)
  SplatPlacementRig.cs  in-headset scale/placement, to validate metric scale
  PerfHud.cs            mean + p95 frame time vs the 72 Hz budget
docs/
  CAPTURE.md            how to shoot, and the budget table to fill in
  UNITY_SETUP.md        exact Unity/XR/URP settings for Quest 2
```

## Quick start

On the **NVIDIA box** (Linux or WSL2 — `requirements.txt` pins a Linux-only
gsplat wheel):

```bash
git clone https://github.com/InternRobotics/AnySplat && cd AnySplat
conda create -y -n anysplat python=3.10 && conda activate anysplat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Then:

```bash
SCALE=1.0 ./scripts/pipeline.sh capture.mp4 work/ ~/AnySplat 32 150000
```

To reconstruct only a segmented subject, generate masks and set `MASKS=`:

```bash
python scripts/make_masks_sam2.py --frames work/frames --out work/masks \
    --checkpoint sam2.1_hiera_large.pt --box 120 90 330 400
MASKS=work/masks SCALE=1.0 ./scripts/pipeline.sh capture.mp4 work/ ~/AnySplat
```

Check `work/quest.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor)
— **not foggy** (opacity correct), **not mirrored** (handedness correct) — then:

```bash
adb push work/quest.ply /sdcard/Android/data/<your.package>/files/
```

## Masking to a subject (optional, stage B2)

AnySplat has no mask input -- it is trained on RGB only -- but its Gaussians are
**pixel-aligned**, one per input pixel per view. So a 2D segmentation becomes a
3D selection by reprojection: push each Gaussian back into every input view,
look up the mask, and keep the ones that land inside it.

This runs on the **raw** PLY, before `fixup_ply.py`, because the predicted poses
are in the model's coordinate frame.

The subtlety is depth ambiguity: a wall directly behind the subject genuinely is
inside the mask from that viewpoint. Requiring agreement across the views that
observe a Gaussian rejects it, and the threshold matters a lot — measured on a
synthetic ground-truth scene (90k subject splats in a 500k room, 24 cameras
orbiting with the far wall in frame):

| `--min-agreement` | precision | recall |
|---|---|---|
| 0.50 (majority) | 32.8% | 100% |
| **0.80 (default)** | **99.9%** | **100%** |
| 0.95 | 100% | 100% |

A plain majority vote is unusable. Add `--erode 3` to trim the silhouette halo
(100% precision in the same test).

On Quest 2 this is the biggest quality lever available: masking first means the
whole ~150k budget goes to the subject, rather than being spent on walls that
decimation then discards.

## The three corrections that matter

Stage C exists because AnySplat's PLY export is not directly loadable:

1. **Opacity** is written raw, with no inverse-sigmoid, while INRIA-convention
   loaders apply `sigmoid()` on read. Double-activation squashes all alpha into
   ~0.5–0.73 → the scene renders as uniform milky fog. `fixup_ply.py` detects
   the encoding from the column range and corrects it.
2. **Axes**: AnySplat inherits VGGT's OpenCV-style frame (Y down, Z forward,
   right-handed); Unity is left-handed, Y up. The flip has determinant −1, so it
   *cannot* be applied as a Unity object transform without corrupting the
   per-splat quaternions. It's baked into the covariances instead — see
   `_apply_axes` for the derivation.
3. **Scale** is arbitrary. In VR a wrongly-scaled scene reads as a dollhouse or
   a cathedral, so `--scale` is derived from one real measurement.

## What's verified vs not

**Verified** (synthetic ground-truth tests, run locally):
- quaternion round-trip exact to 7e-16; mirrored covariances reproduced to
  float32 precision with scales provably unchanged
- opacity logit round-trip to 1.2e-08
- decimation: 600k → 149,784 against a 150k target, floaters culled
- pose-convention auto-detection: correct on both an object orbit and a room
  scan, with 5.3× and 7.6× margins
- subject masking: F1 99.9% against ground-truth labels at the default
  `--min-agreement 0.8`; 100% with `--erode 3`

**Not verified** (needs your hardware):
- anything touching AnySplat itself — no CUDA GPU here
- `make_masks_sam2.py` — SAM 2 is not installed here, so the mask *consumer* is
  tested against synthetic masks but the *generator* is not
- the C# against a compiled Unity project — no Unity install here. The package
  API is used via reflection in one adapter file so the shell builds before the
  renderer is wired up.
