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
  pick_box.py           A2: drag a box around the subject on frame 0
  make_masks_sam2.py    A3: per-frame subject masks via SAM 2
  mask_splats.py        B2: keep only Gaussians inside the masks
  fixup_ply.py          C: opacity / axes / metric scale corrections
  poses_to_colmap.py    D: COLMAP export for the refinement pass
  decimate_ply.py       E: prune + subsample to a budget (manual only, not in pipeline.sh)
  level_ply.py          B3: stand the subject upright on its surface, origin at its base
  pipeline.sh           A -> C end to end, conda envs handled itself
  install.sh            clones AnySplat + SAM 2, fetches weights, creates both envs
  splat_server.py       phone upload -> pick in the headset -> gaussians back (LAN)
  splat_server_upload.html  the phone's upload page
  worker_proto.py       stdio line protocol for the resident model workers
  deploy_quest.sh       install the APK + push a splat over USB
unity/QuestSplatViewer/Assets/Editor/
  QuestProjectSetup.cs  applies docs/UNITY_SETUP.md + builds the scene and APK
unity/QuestSplatViewer/Assets/Scripts/
  SplatLoaderAdapter.cs the only file touching the UnitySplats package API
  SplatSceneBrowser.cs  runtime splat loading (adb push, no APK rebuild)
  CaptureClient.cs      the headset side of splat_server.py
  SceneSurfaces.cs      Space Setup tables/floor as AR planes to set splats on
  SplatGrab.cs          right grip picks the splat up, release drops it on a surface
  SplatPlacementRig.cs  in-headset scale/placement, to validate metric scale
  PerfHud.cs            mean + p95 frame time vs the 72 Hz budget
docs/
  pipeline_diagram.html one-page diagram of the whole pipeline
  CAPTURE.md            how to shoot, and the budget table to fill in
  UNITY_SETUP.md        exact Unity/XR/URP settings for Quest 2
```

## Quick start

On the **NVIDIA box** (Linux or WSL2 — AnySplat pins a Linux-only gsplat
wheel), install everything with:

```bash
./scripts/install.sh            # base dir defaults to the parent of this repo
```

That creates, next to this repo: `AnySplat/`, `sam2/` (with
`checkpoints/sam2.1_hiera_large.pt`), `hf_cache/` (AnySplat + VGGT-1B weights,
~7.5 GB), and two conda envs — `anysplat` (py3.10, torch 2.2 cu121, ffmpeg) and
`sam2` (py3.11, torch 2.5.1 cu121). SAM 2 needs torch ≥ 2.5.1, which AnySplat's
pinned wheels can't use, hence the separate env. Deviations from AnySplat's own
`requirements.txt`: `pytorch3d` is skipped (training-only, needs nvcc),
`torch_scatter` comes as a prebuilt PyG wheel, and OpenCV is the `-headless`
build (no libGL).

Then, from any shell -- each step runs in its env via `conda run`, and the
pipeline finds `../AnySplat` and `../hf_cache` on its own:

```bash
./scripts/pipeline.sh capture.mp4                # 64 frames -> work/quest.ply
./scripts/pipeline.sh capture.mp4 work/ ../AnySplat 32   # workdir, AnySplat root, frames
```

There is no decimation stage: masking cuts the scene down to the subject
(~200k splats for a shoe at 64 frames). With `MASK=0` the output is millions of
splats; run `decimate_ply.py` on it by hand before deploying.

Subject masking is on by default: after frame extraction a window opens on
frame 0 (`pick_box.py`) -- drag a box around the subject and press Enter. SAM 2
(run through `conda run -n $SAM2_ENV`) then tracks it through every frame.

```bash
BOX="120 90 330 400" ./scripts/pipeline.sh capture.mp4   # no picker window
MASK=0 ./scripts/pipeline.sh capture.mp4                 # whole scene
MASKS=my_masks/ ./scripts/pipeline.sh capture.mp4        # precomputed masks
```

The envs default to `ANYSPLAT_ENV=anysplat` and `SAM2_ENV=sam2`. RTX 50xx
(Blackwell) cards need cu128 builds (torch 2.8, plus matching `torch_scatter`
and `xformers` for AnySplat), which `install.sh` does not install yet: on this
box both envs were upgraded to them after install. SAM 2 prints a
warning about skipping its post-processing CUDA extension; that is expected and
harmless.

Check `work/quest.ply` in [SuperSplat](https://playcanvas.com/supersplat/editor)
— **not foggy** (opacity correct), **not mirrored** (handedness correct) — then:

```bash
./scripts/deploy_quest.sh work/quest.ply    # install the APK, push, launch, show the log
```

The APK comes from the Unity project (see [docs/UNITY_SETUP.md](docs/UNITY_SETUP.md));
a new capture needs only `NO_INSTALL=1 ./scripts/deploy_quest.sh other.ply`.

## Capture with a phone, pick in the headset

`splat_server.py` runs the same stages as a service, so a capture never touches
the PC's keyboard:

```bash
./scripts/splat_server.py        # prints the phone upload URL, e.g. http://10.24.71.147:8765/
```

1. On the phone, open that URL and upload a video of the object. AnySplat
   starts at once.
2. In Quest Splat Viewer (it finds the server by UDP broadcast), frame 0
   appears on a panel. Right trigger: click the object or drag a box around
   it; **A** sends it, **B** clears.
3. About half a minute later the gaussians download and replace the scene.
   With nothing on the panel, **B** brings back the last capture to pick again
   (only SAM 2 onward re-runs).

The server keeps AnySplat and SAM 2 loaded in two resident worker processes
(`run_anysplat.py --serve`, `make_masks_sam2.py --serve`, protocol in
`worker_proto.py`), so model loading is paid once at startup rather than per
job: a warm 64-frame capture goes from ~60 s to ~44 s end to end on an RTX 5090.
Both models fit in its 32 GB together; on a smaller card, `--no-workers` falls
back to a fresh `conda run` per stage.

The app runs in passthrough, so the capture appears in your room. Hold the
**right grip** to pick it up and move it; let go above a real table and it
settles onto it. Tables come from the headset's **Space Setup** (Settings →
Physical Space; on a Quest 2, mark the table by hand), and the app asks for
"spatial data" permission on first launch. For this to look right, stage B3
(`level_ply.py`) stands each capture upright on the surface it was filmed on,
using the camera poses and that surface, with the origin at the object's base.

The phone, the Quest and the PC must reach each other on the network, and the
PC's firewall must allow TCP 8765 and UDP 8766. Networks with client isolation
(common on campus or guest Wi-Fi) block this; a phone hotspot or home router
works. If broadcast is blocked but traffic isn't, put the server's URL in
`server.txt` in the app's files folder:

```bash
echo http://10.24.71.147:8765 > server.txt
adb push server.txt /sdcard/Android/data/com.vrproj.questsplatviewer/files/
```

The server has no authentication: anyone on the network can upload and spend
GPU time, so run it only on a network you trust.

## Masking to a subject (stage B2, on by default)

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
   per-splat quaternions. `pipeline.sh` therefore keeps the file in that
   standard 3DGS frame (`--axes none`, which SuperSplat also shows upright), and
   the Unity app tells UnitySplats the source frame is `RDF`, so the package
   converts positions, rotations and SH on load. Baking the flip into the file
   as well (`fixup_ply.py`'s default, see `_apply_axes`) would get the
   package's own conversion applied on top and come out mirrored.
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

**Verified on the RTX 5090 box** (Unity 6000.6.8f1, UnitySplats pinned):
- the full pipeline, AnySplat and SAM 2 included, on a real capture
- the Unity project compiles, passes OpenXR project validation for Android
  with no errors, and in Play mode renders a pipeline `quest.ply` through
  `SplatSceneBrowser` upright and not mirrored

**Not verified yet:**
- the app on a headset (frame time, stereo under single-pass instanced)
