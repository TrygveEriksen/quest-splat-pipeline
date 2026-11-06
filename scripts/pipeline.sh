#!/usr/bin/env bash
# Stages A -> C end to end, subject masking included. Run on the NVIDIA box from
# any shell: every step runs in its own conda env via `conda run`, so nothing
# needs activating first.
#
#   ./pipeline.sh capture.mp4                                  # pick the subject in a window
#   ./pipeline.sh capture.mp4 work/ ~/AnySplat 64              # all positional args
#   BOX="48 322 350 458" ./pipeline.sh capture.mp4             # no picker window
#   MASK=0 ./pipeline.sh capture.mp4                           # whole scene, no masking
#
# Masking (stages A2/A3 + B2) is on by default. The subject box (pixels, in
# frame 0) comes from $BOX, or, when that is unset, from pick_box.py, which
# needs a display. MASKS=dir uses precomputed masks and skips SAM 2.
#
# There is no decimation stage: masking already cuts the scene down to the
# subject (~200k splats for a shoe-sized object at 64 frames). decimate_ply.py
# is still there to run by hand if a capture comes out over budget.
#
# Stage D (refinement) is deliberately NOT automated: it needs the pose
# convention confirmed first, and a blind run can diverge instead of sharpen.
# The command to run is printed at the end.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -gt 4 ]]; then
    echo "usage: pipeline.sh <video> [workdir] [anysplat-root] [frames]" >&2
    echo "(the 5th 'target' argument went away with the decimation stage)" >&2
    exit 1
fi
VIDEO="${1:?usage: pipeline.sh <video> [workdir] [anysplat-root] [frames]}"
WORK="${2:-work}"
ANYSPLAT="${3:-$HERE/../../AnySplat}"
FRAMES="${4:-64}"
SCALE="${SCALE:-1.0}"
MASK="${MASK:-1}"           # 0 = reconstruct the whole scene, no subject masking
BOX="${BOX:-}"              # "x0 y0 x1 y1" around the subject in frame 0
MASKS="${MASKS:-}"          # optional: precomputed per-frame masks (skips SAM 2)
# On this box both envs have the cu128 builds RTX 50xx (Blackwell) needs.
ANYSPLAT_ENV="${ANYSPLAT_ENV:-anysplat}"
SAM2_ENV="${SAM2_ENV:-sam2}"
SAM2_CKPT="${SAM2_CKPT:-$HERE/../../sam2/checkpoints/sam2.1_hiera_large.pt}"
[[ "$MASK" == 0 ]] && MASKS=""

# `conda` is a shell function in an activated shell but not in a script, so
# find the executable itself.
CONDA="${CONDA_EXE:-$(command -v conda || true)}"
for c in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda"; do
    [[ -z "$CONDA" && -x "$c" ]] && CONDA="$c"
done
if [[ -z "$CONDA" ]]; then
    echo "conda not found: put it on PATH or set CONDA_EXE" >&2
    exit 1
fi
py()      { "$CONDA" run -n "$ANYSPLAT_ENV" --no-capture-output python "$@"; }
sam2_py() { "$CONDA" run -n "$SAM2_ENV" --no-capture-output python "$@"; }

# AnySplat (and the VGGT-1B backbone it pulls on construction) are cached here
# by install.sh; without this they would be re-downloaded into ~/.cache.
if [[ -z "${HF_HOME:-}" && -d "$HERE/../../hf_cache" ]]; then
    export HF_HOME="$(cd "$HERE/../../hf_cache" && pwd)"
fi
mkdir -p "$WORK"

echo "=== A. extract frames (target $FRAMES) ==="
py "$HERE/extract_frames.py" "$VIDEO" -o "$WORK/frames" --max-frames "$FRAMES"

if [[ "$MASK" != 0 && -z "$MASKS" ]]; then
    # Masks first, so the subject is picked before the long inference step.
    if [[ -z "$BOX" ]]; then
        if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
            echo "no display for the box picker: pass BOX=\"x0 y0 x1 y1\" (frame 0" \
                 "pixels) or MASK=0" >&2
            exit 1
        fi
        frames=("$WORK"/frames/*.png)
        echo "=== A2. pick the subject in ${frames[0]##*/} (drag a box, press Enter) ==="
        py "$HERE/pick_box.py" --frame "${frames[0]}" --save "$WORK/box.txt"
        BOX="$(<"$WORK/box.txt")"
    fi
    read -r -a box <<< "$BOX"
    if [[ ${#box[@]} -ne 4 ]]; then
        echo "BOX must be four numbers \"x0 y0 x1 y1\", got \"$BOX\"" >&2
        exit 1
    fi

    echo "=== A3. SAM 2 subject masks (box ${box[*]}) ==="
    # Masks are matched to frames by sorted order, so clear any left over
    # from an earlier run with more frames.
    mkdir -p "$WORK/masks"
    rm -f "$WORK"/masks/*.png
    sam2_py "$HERE/make_masks_sam2.py" --frames "$WORK/frames" \
        --out "$WORK/masks" --checkpoint "$SAM2_CKPT" --box "${box[@]}"
    MASKS="$WORK/masks"
fi

echo "=== B. AnySplat inference ==="
py "$HERE/run_anysplat.py" --frames "$WORK/frames" --out "$WORK/raw" \
    --anysplat-root "$ANYSPLAT"

STAGE_C_IN="$WORK/raw/gaussians.ply"
RECENTER=(--recenter)
if [[ -n "$MASKS" ]]; then
    # Must run BEFORE fixup_ply: the predicted poses are in the model's
    # coordinate frame, so they stop applying once the axes are flipped.
    echo "=== B2. mask to segmented subject ==="
    py "$HERE/mask_splats.py" --ply "$WORK/raw/gaussians.ply" \
        --poses "$WORK/raw/poses.npz" --masks "$MASKS" -o "$WORK/masked.ply"
    STAGE_C_IN="$WORK/masked.ply"

    # Level it on the surface it was filmed on, origin at its base, so the
    # headset can stand it on a real table. Falls back to centring.
    echo "=== B3. stand the subject on its surface ==="
    if py "$HERE/level_ply.py" --raw "$WORK/raw/gaussians.ply" --subject "$WORK/masked.ply" \
            --poses "$WORK/raw/poses.npz" -o "$WORK/leveled.ply"; then
        STAGE_C_IN="$WORK/leveled.ply"
        RECENTER=()
    else
        echo "(no surface to stand it on; centring it instead)"
    fi
fi

echo "=== C. fix up opacity / scale (scale=$SCALE) ==="
# --axes none keeps the standard 3DGS / OpenCV frame (x right, y down, z
# forward): SuperSplat shows it upright, and the Unity app tells UnitySplats
# it is RDF so the package does the Y-up conversion itself. Baking a flip in
# here instead would get the package's own conversion applied on top.
py "$HERE/fixup_ply.py" "$STAGE_C_IN" -o "$WORK/quest.ply" \
    --axes none --scale "$SCALE" "${RECENTER[@]}"

if [[ -n "$MASKS" ]]; then
    MASK_NOTE="on, masks in $MASKS${BOX:+ -- rerun without the picker with BOX=\"$BOX\"}"
else
    MASK_NOTE="off, whole scene -- millions of splats, far over a Quest 2 budget
  (~150-250k); mask it, or run decimate_ply.py on $WORK/quest.ply by hand"
fi

cat <<MSG

=== done ===
Quest-ready file: $WORK/quest.ply

Verify before deploying -- open it at https://playcanvas.com/supersplat/editor
and check it is NOT foggy (opacity) and NOT mirrored (handedness).

Deploy (headset on USB, developer mode on):
  $HERE/deploy_quest.sh $WORK/quest.ply

Subject masking: $MASK_NOTE

Optional stage D (refinement), only after checking the detected pose convention:
  $CONDA run -n $ANYSPLAT_ENV python $HERE/poses_to_colmap.py --poses $WORK/raw/poses.npz \\
      --ply $WORK/raw/gaussians.ply --frames $WORK/frames --out $WORK/colmap
  $CONDA run -n $ANYSPLAT_ENV python $ANYSPLAT/src/post_opt/simple_trainer.py default --data_dir $WORK/colmap
  # then re-run stage C on the refined output
MSG
