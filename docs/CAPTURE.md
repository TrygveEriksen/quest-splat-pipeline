# Capture guide

## Why the rules are strict

AnySplat sees **448×448** frames and its examples run **2–64** of them. After
`extract_frames.py --max-frames 32`, a single bad frame is ~3% of everything the
model knows about your scene. Motion blur is unrecoverable downstream.

## Shooting

| Do | Don't |
|---|---|
| Static scene | People, pets, wind-blown plants |
| Slow, smooth orbit or pan | Fast whip pans (motion blur) |
| Even, diffuse light (overcast, shade) | Hard sun, blown windows, strong speculars |
| Keep the subject filling the frame | Wide shots where the subject is tiny |
| 20–40 s for ~32 frames | Very long clips (frames get too sparse per unit of motion) |
| Lock exposure and focus if your camera allows | Auto-exposure pumping between frames |

**Proving run:** one object, a slow 20–40 s orbit, filling the frame. Get this
working end to end before attempting a room.

**Scene scale:** feed-forward output has arbitrary scale, so measure one real
thing in the scene (a table edge, a door width) and note it. You'll need it for
`fixup_ply.py --scale`.

## If you only want one object in the scene

Shoot it the same way, then use stage B2 (`make_masks_sam2.py` +
`mask_splats.py`) to keep only the subject's Gaussians. Two things help the
masking work well:

- **Orbit the subject** so each part of it is seen by many views. The masking
  decision is a vote across the views that observe a Gaussian, so more views
  means a cleaner cut (the default needs at least 2).
- **Keep the subject away from the back wall** if you can. Background directly
  behind the subject is the one case the vote has to work hardest to reject.

Check a few generated masks by eye before running `mask_splats.py` — a SAM 2
track that drops mid-sequence shows up as a subject with chunks missing.

## Scaling up

- **One room** is near the limit of a single forward pass. Expect to reduce
  `--max-frames` if you hit CUDA OOM, which costs coverage.
- **Large / outdoor** is where feed-forward models degrade most *and* where the
  Quest 2 budget bites hardest. Prove a small scene first.

## Measured Quest 2 budgets

Fill this in from `PerfHud` — these are the numbers that matter, and nothing
published covers Adreno 650 splat rendering.

| Splats | Mean ms | p95 ms | 72 Hz? | Notes (close range especially) |
|---|---|---|---|---|
| 150,000 | | | | |
| 200,000 | | | | |
| 250,000 | | | | |

Budget at 72 Hz is **13.9 ms**. Test standing still *and* walking up close.
