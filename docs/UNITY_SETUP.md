# Unity project setup for Quest 2

The project now lives in `unity/QuestSplatViewer` (Unity 6000.6.8f1, URP),
and `Assets/Editor/QuestProjectSetup.cs` applies sections 2-4, the MSAA / HDR /
shadow rows of 5, and the scene in 8 from the command line, with an Editor
running that project. Foveation (6), render scale, and the splat sorter
settings (7) are not applied yet.

```bash
unity command eval "return QuestSplatViewer.EditorTools.QuestProjectSetup.Apply();"
unity command eval "return QuestSplatViewer.EditorTools.QuestProjectSetup.BuildApk();" --detach
```

The same two actions are in the Editor's **Quest Splat Viewer** menu. This file
remains the reasoning behind each setting; change the script to change one.

## Why these choices

Quest 2 is roughly half the GPU of a Quest 3 (Adreno 650 vs 740). Published
Quest 3 splat budgets are ~400–500k splats at 72 Hz, so **plan for 150–250k and
measure**. There are no published Adreno 650 splat benchmarks — the number has
to come from `PerfHud` on your own device.

Splats are **fill-rate and sort bound**, not vertex bound. Every setting below
is chosen to cut overdraw or sort cost.

## 1. Project creation

- Unity **6000.x LTS** (the splat packages require Unity 6000.0+)
- Template: **Universal 3D (URP)** — URP is the only pipeline with
  single-pass-instanced support in these packages
- Platform: **Android**

## 2. Renderer package

Install [`arloopa/UnitySplats`](https://github.com/arloopa/unitysplats) via
Package Manager → Add from git URL.

Dependencies it needs: `com.unity.mathematics` 1.3.2+, and `Unity.WebP` 0.3.22
only if you want `.sog` support.

> The package's install docs point at a feature branch rather than the default
> branch — check which branch you actually want before pinning.

Chosen over [`ninjamode/Unity-VR-Gaussian-Splatting`](https://github.com/ninjamode/Unity-VR-Gaussian-Splatting)
because it documents a **runtime** loading API, so a new capture is an
`adb push` rather than an APK rebuild. See the decision gate at the bottom.

## 3. Player settings (Project Settings → Player → Android)

| Setting | Value | Why |
|---|---|---|
| Scripting backend | IL2CPP | required for ARM64 |
| Target architecture | ARM64 only | Quest is ARM64; ARMv7 just bloats the APK |
| Graphics API | **Vulkan only** (remove GLES3) | the GPU radix sort path needs Vulkan; GLES 3.1 falls back to a CPU sort |
| Minimum API level | 29 | Quest requirement |
| Color space | Linear | |
| Multithreaded rendering | On | |

## 4. XR settings

- **XR Plug-in Management** → Android → enable **OpenXR**
- OpenXR → Feature groups → enable **Meta Quest Support**
- OpenXR → **Render Mode: Single Pass Instanced**
- Interaction profile: **Oculus Touch Controller Profile**

> **Verify stereo immediately after first deploy.** Splat shaders commonly break
> on instanced projection matrices — you'll see the scene correct in one eye and
> offset/absent in the other. Multi-pass "fixes" it by doubling render cost,
> which will blow the Quest 2 budget. If instanced is broken, that's a trigger to
> switch renderer, not to accept Multi-pass.

## 5. URP asset settings

| Setting | Value | Why |
|---|---|---|
| **MSAA** | **Disabled** | expensive, and does almost nothing for alpha-blended splats |
| Render Scale | 0.8–0.9 | the cheapest large fill-rate win |
| HDR | Off | not needed; costs bandwidth |
| Post-processing | Off | |
| Shadows | Off, 0 cascades | splats are unlit |
| Depth/Opaque texture | Off unless needed | saves a full-resolution copy |

## 6. Meta / foveation

- Enable **Fixed Foveated Rendering** at `High` (OpenXR → Meta Quest Support, or
  `OVRManager` foveation level). **This is the single biggest free win** here,
  because splat cost is dominated by peripheral overdraw.
- Set display refresh to **72 Hz** first. Get a stable 72 before trying 90.
- Avoid Application SpaceWarp initially — it interacts badly with the
  semi-transparent, high-frequency detail of splats.

## 7. Splat renderer settings

- Sort precision: **16-bit**
- **Center-eye sorting** (sort once, share between both eyes) — roughly halves
  sort cost for an in-practice-invisible error
- **Sort every 2–3 frames** rather than every frame
- Frustum culling: on

## 8. Scene setup

```
XR Origin
├── Camera Offset / Main Camera
│   └── WristHud (TextMesh)        <- PerfHud.target
├── SplatRoot (empty)              <- SplatSceneBrowser.splatRoot
└── AppRoot
    ├── SplatSceneBrowser          (splatRoot = SplatRoot)
    ├── SplatPlacementRig          (splatRoot = SplatRoot)
    └── PerfHud                    (target = WristHud, browser = SplatSceneBrowser)
```

## 9. Deploying a capture

`Application.persistentDataPath` needs **no storage permission**, which is why
`SplatSceneBrowser` checks it first:

```bash
adb push quest.ply /sdcard/Android/data/<your.package.name>/files/
```

Only use `/sdcard/Download` if you're willing to add `READ_EXTERNAL_STORAGE`
plus `requestLegacyExternalStorage` to the manifest — it buys nothing here.

## 10. Profiling loop

1. Deploy with a 150k file. Read `PerfHud`.
2. Watch **p95**, not mean — sorting spikes are what read as judder.
3. **Deliberately walk up close to the geometry.** Close-range collapse is the
   documented failure mode across every Quest splat project, caused by a few
   enormous splats covering the viewport. If it tanks, lower
   `decimate_ply.py --max-extent-frac` before lowering `--target`.
4. Sweep `--target` upward (150k → 200k → 250k) until p95 exceeds 13.9 ms.

## Decision gate

If after all of the above you cannot hold 72 Hz at a useful splat count, or
single-pass-instanced stereo is broken:

- Switch the rendering core to **ninjamode/Unity-VR-Gaussian-Splatting** (more
  adoption, explicitly VR-targeted, instanced + foveated rendering). Cost: asset
  conversion becomes editor-only, so each capture needs an APK rebuild.
- Stages A–E are unchanged either way — they emit plain `.ply`, which is exactly
  why the conversion scripts don't depend on the renderer.
- Last resort: splats render well in the **Quest browser via WebXR**
  (PlayCanvas / Three.js) for a fraction of the effort. Not a Unity app, but
  worth knowing before sinking days into the native path.
