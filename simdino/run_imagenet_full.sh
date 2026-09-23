#!/usr/bin/env bash
# ViT-S/16 on all of ImageNet across the GPUs of one node, then frozen-feature evaluation.
# Written for 8x V100: fp32, since V100 has no bf16 tensor cores. Extra arguments go to run.py.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${SIMDINO_GPUS:-0,1,2,3,4,5,6,7}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

python_bin="${SIMDINO_PYTHON:-python}"
data="${SIMDINO_DATA:-/longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC}"
run="${SIMDINO_RUN:-runs/imagenet-full-vits}"
workers="${SIMDINO_WORKERS:-$(( $(nproc) - 4 ))}"
epochs="${SIMDINO_EPOCHS:-100}"
per_gpu="${SIMDINO_PER_GPU:-64}"

gpus=$(( $(tr -cd , <<< "$CUDA_VISIBLE_DEVICES" | wc -c) + 1 ))
batch=$(( per_gpu * gpus ))
images=1281167
steps_per_epoch=$(( (images + batch - 1) / batch ))
steps=$(( epochs * steps_per_epoch ))
warmup=$(( 10 * steps_per_epoch ))
# DINO's linear scaling rule: 5e-4 per 256 images.
lr=$(awk -v b="$batch" 'BEGIN { printf "%.6g", 5e-4 * b / 256 }')

mkdir -p runs
trap 'code=$?; if (( code != 0 )); then printf "failed (exit %s)\n" "$code" > "$run.status"; fi' EXIT

active_processes=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active_processes" ]]; then
  printf 'GPUs %s already have compute processes; refusing to overlap them.\n' "$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

printf 'training\n' > "$run.status"
date -u '+Started: %Y-%m-%d %H:%M:%S UTC'
printf '%s GPUs, batch %s, lr %s, %s steps (%s per epoch), %s loader processes\n' \
  "$gpus" "$batch" "$lr" "$steps" "$steps_per_epoch" "$workers"
"$python_bin" -u run.py \
  --data "$data/train" --out "$run" \
  --classes 0 --per-class 0 --sampling epoch \
  --size 224 --patch 16 --width 384 --depth 12 --heads 6 --dim 128 \
  --batch "$batch" --steps "$steps" --warmup "$warmup" --lr "$lr" \
  --workers "$workers" --prefetch 6 --log-every 50 --save-every 1000 "$@"

printf 'evaluating\n' > "$run.status"
"$python_bin" -u evaluate.py \
  --run "$run" --val "$data/val" \
  --train-per-class 100 --val-per-class 50 --workers "$workers"
printf 'complete\n' > "$run.status"
date -u '+Finished: %Y-%m-%d %H:%M:%S UTC'
