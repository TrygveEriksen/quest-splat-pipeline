#!/usr/bin/env bash
# Stages A -> E end to end. Run on the NVIDIA box inside AnySplat's conda env.
#
#   ./pipeline.sh capture.mp4 work/ ~/AnySplat
#
# Stage D (refinement) is deliberately NOT automated: it needs the pose
# convention confirmed first, and a blind run can diverge instead of sharpen.
# The command to run is printed at the end.
set -euo pipefail

VIDEO="${1:?usage: pipeline.sh <video> <workdir> <anysplat-root> [frames] [target]}"
WORK="${2:?missing workdir}"
ANYSPLAT="${3:?missing anysplat root}"
FRAMES="${4:-32}"
TARGET="${5:-150000}"
SCALE="${SCALE:-1.0}"
MASKS="${MASKS:-}"          # optional: directory of per-frame subject masks

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$WORK"

echo "=== A. extract frames (target $FRAMES) ==="
python "$HERE/extract_frames.py" "$VIDEO" -o "$WORK/frames" --max-frames "$FRAMES"

echo "=== B. AnySplat inference ==="
python "$HERE/run_anysplat.py" --frames "$WORK/frames" --out "$WORK/raw" \
    --anysplat-root "$ANYSPLAT"

STAGE_C_IN="$WORK/raw/gaussians.ply"
if [[ -n "$MASKS" ]]; then
    # Must run BEFORE fixup_ply: the predicted poses are in the model's
    # coordinate frame, so they stop applying once the axes are flipped.
    echo "=== B2. mask to segmented subject ==="
    python "$HERE/mask_splats.py" --ply "$WORK/raw/gaussians.ply" \
        --poses "$WORK/raw/poses.npz" --masks "$MASKS" -o "$WORK/masked.ply"
    STAGE_C_IN="$WORK/masked.ply"
fi

echo "=== C. fix up opacity / axes / scale (scale=$SCALE) ==="
python "$HERE/fixup_ply.py" "$STAGE_C_IN" -o "$WORK/unity_raw.ply" \
    --scale "$SCALE" --recenter

echo "=== E. decimate to $TARGET splats ==="
python "$HERE/decimate_ply.py" "$WORK/unity_raw.ply" -o "$WORK/quest.ply" \
    --target "$TARGET"

cat <<MSG

=== done ===
Quest-ready file: $WORK/quest.ply

Verify before deploying -- open it at https://playcanvas.com/supersplat/editor
and check it is NOT foggy (opacity) and NOT mirrored (handedness).

Deploy:
  adb push $WORK/quest.ply /sdcard/Android/data/<your.package>/files/

Optional stage B2 (mask to a subject) -- rerun with:
  MASKS=masks/ ./pipeline.sh ...
  (generate masks first: make_masks_sam2.py --frames $WORK/frames --out masks/ ...)

Optional stage D (refinement), only after checking the detected pose convention:
  python $HERE/poses_to_colmap.py --poses $WORK/raw/poses.npz \\
      --ply $WORK/raw/gaussians.ply --frames $WORK/frames --out $WORK/colmap
  python $ANYSPLAT/src/post_opt/simple_trainer.py default --data_dir $WORK/colmap
  # then re-run stages C and E on the refined output
MSG
