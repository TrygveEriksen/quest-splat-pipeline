#!/usr/bin/env bash
# One-shot install of everything the pipeline needs, laid out as:
#
#   <base>/AnySplat              clone of InternRobotics/AnySplat
#   <base>/sam2                  clone of facebookresearch/sam2 (+ checkpoint)
#   <base>/hf_cache              HF_HOME: AnySplat + VGGT-1B weights (~7.5 GB)
#   conda env "anysplat"         py3.10, torch 2.2.0+cu121, ffmpeg
#   conda env "sam2"             py3.11, torch 2.5.1+cu121
#
# SAM 2 gets its own env because it requires torch>=2.5.1, while AnySplat's
# xformers/gsplat/torch_scatter wheels are pinned to torch 2.2.
#
#   ./install.sh [base-dir]      (default: the directory above this repo)
#
# Safe to re-run: every step skips what is already there.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "${1:-$HERE/../..}" && pwd)"
CONDA="${CONDA:-$(command -v conda || echo "$HOME/anaconda3/bin/conda")}"
CONDA_ROOT="$("$CONDA" info --base)"

# Variables some sandboxes (e.g. snap) inject that silently redirect pip
# installs outside the env.
unset PIP_PREFIX PYTHONPATH PIP_BREAK_SYSTEM_PACKAGES
# conda's CUDA virtual-package probe needs POSIX semaphores; skip it.
export CONDA_OVERRIDE_CUDA="${CONDA_OVERRIDE_CUDA:-12.1}"
export HF_HOME="${HF_HOME:-$BASE/hf_cache}"

mkenv() {  # name pyver
    [[ -x "$CONDA_ROOT/envs/$1/bin/python" ]] || "$CONDA" create -y -q -n "$1" "python=$2"
}

echo "=== AnySplat ==="
[[ -d "$BASE/AnySplat" ]] || git clone https://github.com/InternRobotics/AnySplat "$BASE/AnySplat"
mkenv anysplat 3.10
A="$CONDA_ROOT/envs/anysplat/bin"
"$A/pip" install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121
# - pytorch3d is dropped: only the training chamfer loss uses it, and it has
#   to be compiled with nvcc.
# - torch_scatter comes as a prebuilt wheel from the PyG index (PyPI only has
#   the sdist, which also needs nvcc).
# - opencv-python is swapped for -headless: identical API, no libGL needed.
REQ="$(mktemp)"
grep -vE 'pytorch3d|^opencv-python' "$BASE/AnySplat/requirements.txt" > "$REQ"
echo "opencv-python-headless==4.11.0.86" >> "$REQ"
"$A/pip" install -q -r "$REQ" torch==2.2.0 -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
rm -f "$REQ"
"$CONDA" install -y -q -n anysplat -c conda-forge ffmpeg

echo "=== AnySplat weights -> $HF_HOME ==="
(cd "$BASE/AnySplat" && "$A/python" -c "
from src.model.model.anysplat import AnySplat
AnySplat.from_pretrained('lhjiang/anysplat')   # also pulls facebook/VGGT-1B
print('weights ok')")

echo "=== SAM 2 ==="
[[ -d "$BASE/sam2" ]] || git clone https://github.com/facebookresearch/sam2.git "$BASE/sam2"
mkenv sam2 3.11
S="$CONDA_ROOT/envs/sam2/bin"
"$S/pip" install -q torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
# SAM2_BUILD_CUDA=0 skips the optional hole-filling extension (needs nvcc);
# SAM 2 prints a warning about it at runtime, which is safe to ignore.
(cd "$BASE/sam2" && SAM2_BUILD_CUDA=0 "$S/pip" install -q --no-build-isolation -e .)
"$S/pip" install -q opencv-python-headless
CKPT="$BASE/sam2/checkpoints/sam2.1_hiera_large.pt"
[[ -s "$CKPT" ]] || "$S/python" -I -c "import sys, urllib.request; urllib.request.urlretrieve(
    'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', sys.argv[1])" "$CKPT"

cat <<MSG

=== installed ===
  AnySplat env : $A/python
  SAM 2 env    : $S/python
  SAM 2 ckpt   : $CKPT
  HF_HOME      : $HF_HOME  (pipeline.sh picks this up automatically)

Check the GPU is visible:
  $A/python -c "import torch; print(torch.cuda.is_available())"
MSG
