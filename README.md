# Video → feed-forward 3D Gaussians → Quest 2

A pipeline that takes a handheld video, runs a **feed-forward** model
(AnySplat — one forward pass, no COLMAP, no per-scene training from scratch),
and produces a 3D Gaussian Splat scene that renders on a **standalone Quest 2**
in Unity.

It runs as a **capture server**: film an object with a
phone, upload the video to the PC, pick the object in the headset, and about
half a minute later it is standing on a real table in passthrough. This has
been run end to end on a Quest 2 with the server on an RTX 5090.

See [docs/pipeline_diagram.html](docs/pipeline_diagram.html) for the stages at
a glance.

## Layout

```
scripts/
  ply_io.py             shared 3DGS PLY reader/writer + quaternion helpers
  pose_utils.py         shared pose loading, projection, convention detection
  extract_frames.py     A: video -> sharp, evenly-spread frames
  run_anysplat.py       B: inference CLI (+ dumps the camera poses)
  make_masks_sam2.py    A3: per-frame subject masks via SAM 2
  mask_splats.py        B2: keep only Gaussians inside the masks
  fixup_ply.py          C: opacity / axes / metric scale corrections
  decimate_ply.py       E: prune + subsample to a budget (manual tuning tool, not run by the server)
  level_ply.py          B3: stand the subject upright on its surface, origin at its base
  install.sh            clones AnySplat + SAM 2, fetches weights, creates both envs
  splat_server.py       the entry point: phone upload -> pick in the headset -> gaussians back (LAN)
  splat_server_upload.html  the phone's upload page
  worker_proto.py       stdio line protocol for the resident model workers
  deploy_quest.sh       install the APK on the headset over USB
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

## Install

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

The envs default to `ANYSPLAT_ENV=anysplat` and `SAM2_ENV=sam2`. RTX 50xx
(Blackwell) cards need cu128 builds (torch 2.8, plus matching `torch_scatter`
and `xformers` for AnySplat), which `install.sh` does not install yet: on this
box both envs were upgraded to them after install. SAM 2 prints a
warning about skipping its post-processing CUDA extension; that is expected and
harmless.

## Running: phone capture, pick in the headset

Install the viewer on the headset once, over USB (developer mode on, USB
debugging accepted in the headset). The APK is built from the Unity project,
see [docs/UNITY_SETUP.md](docs/UNITY_SETUP.md):

```bash
./scripts/deploy_quest.sh        # install the APK and launch it
```

Then start the server on the GPU box. It runs every stage itself, so a capture
never touches the PC's keyboard:

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

