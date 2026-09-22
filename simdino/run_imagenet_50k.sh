#!/usr/bin/env bash
# Five complete epochs on 50,000 images, followed by frozen-feature evaluation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${SIMDINO_GPU:-2}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_bin="${SIMDINO_PYTHON:-/home/aghosh/anaconda3/envs/simdino/bin/python}"
data=/longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC
run=runs/imagenet-gpu-vits-50k
mkdir -p runs
trap 'code=$?; if (( code != 0 )); then printf "failed (exit %s)\n" "$code" > "$run.status"; fi' EXIT

active_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active_processes" ]]; then
  printf 'GPU %s already has compute processes; refusing to overlap them.\n' "$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

printf 'training\n' > "$run.status"
date -u '+Started: %Y-%m-%d %H:%M:%S UTC'
"$python_bin" -u run.py \
  --data "$data/train" --out "$run" \
  --classes 0 --per-class 50 --sampling epoch \
  --size 224 --patch 16 --width 384 --depth 12 --heads 6 --dim 128 \
  --batch 50 --steps 5000 --warmup 500 --workers 8 --bf16 "$@"

printf 'evaluating\n' > "$run.status"
"$python_bin" -u evaluate.py \
  --run "$run" --val "$data/val" \
  --train-per-class 50 --val-per-class 5
printf 'complete\n' > "$run.status"
date -u '+Finished: %Y-%m-%d %H:%M:%S UTC'
