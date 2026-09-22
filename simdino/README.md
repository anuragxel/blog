# SimDINO as mathematical pseudocode

The core is in `model.py` and `train.py`. It contains a ViT, normalized embedding alignment, coding-rate regularization, and the student/teacher updates. Data loading and experiment management live in separate files.

```python
trainer = Trainer(config)
state = trainer.init(jax.random.key(0))
state, step, placement = trainer.compile(state, mode="dp")
state, metrics = step(state, jax.device_put(views, placement))
```

`trainer.student` and `trainer.teacher` are separate `ViT` instances. Each model holds architecture configuration, and `apply(weights, images)` receives its weights explicitly. `State` names `student_weights`, `teacher_weights`, `optimizer_state`, and `step`.

`einx` names the axes for normalization, loss statistics, and CLS-token assembly. The forward pass composes image encoding, projection, and normalization. The loss composes cross-view alignment and coding rate. Array notation stays inside the functions that implement those operations. Parameter names and shapes are unchanged, so existing checkpoints still load.

Within `Trainer.step`, only the student weights receive gradients and optimizer updates. The teacher weights receive an EMA update toward the new student weights. The method returns a new state, which makes the same mathematical step usable with replicated or sharded storage.

## Files

| File | Responsibility |
|---|---|
| `model.py` | ViT initialization, forward pass, projection head, normalization |
| `train.py` | SimDINO objective, student/teacher updates, DP/FSDP placement |
| `config.py` | Model and training configuration |
| `data.py` | ImageFolder discovery, independent augmentations, evaluation preprocessing |
| `run.py` | CLI, batch loading, logging, experiment orchestration |
| `checkpoint.py` | Save and restore model, optimizer, and step state |
| `evaluate.py` | Frozen-backbone 1-NN evaluation against an initialization baseline |
| `test_train.py` | Numerical checks, including global-versus-sharded gradients |

## Setup

The `simdino` Conda environment has been created on this machine. To recreate the CPU environment elsewhere, run from this directory:

```bash
conda env create -f environment.yml
conda activate simdino
```

For the CPU checks and recorded CPU experiments, set these before importing JAX:

```bash
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

The existing environment also has optional CUDA dependencies from its initial setup. `export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1` suppresses their device-check warning during CPU runs; it does not enable GPU execution. A fresh install from `environment.yml` has no CUDA plugins.

## ImageNet smoke run

The local ImageNet root is `/longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC`, with `train/<class>/*.JPEG` and `val/<class>/*.JPEG`. No download or dataset modification is needed. Class labels are ignored during SSL and used only for evaluation.

From `simdino/`, this runs 30 updates on 160 real images, using two independently augmented global views per image:

```bash
python run.py \
  --data /longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC/train \
  --out runs/my-cpu-smoke \
  --classes 10 --per-class 16 \
  --size 32 --patch 8 --width 48 --depth 2 --heads 3 --dim 16 \
  --batch 8 --steps 30 --warmup 3 --workers 2

python evaluate.py \
  --run runs/my-cpu-smoke \
  --val /longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC/val
```

The subset is the first ten sorted class folders and the first sixteen images in each. Each update samples a batch without replacement from that subset; samples can reappear on subsequent updates. Augmentation randomness is indexed by seed and step. The runner uses threads for CPU image decoding and augmentation, and synchronizes each step for straightforward logging.

Outputs are `initial.pkl`, `last.pkl`, `config.json`, and `metrics.jsonl`. Checkpoints are written every 100 steps and at the end. Resume an interrupted run with the same arguments plus `--resume runs/my-cpu-smoke/last.pkl`; the optimizer, step, schedule, and per-step augmentation seeds continue. Load only trusted local pickle checkpoints. To compare initial and final features after resuming, retain the original run directory containing `initial.pkl`.

## ViT-S on 5,000 ImageNet images

With the CPU environment variables above, run from `simdino/`:

```bash
taskset -c 0-7 python run.py \
  --data /longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC/train \
  --out runs/imagenet-cpu-vits-5000 \
  --classes 50 --per-class 100 --sampling epoch \
  --size 64 --patch 16 --width 384 --depth 12 --heads 6 --dim 128 \
  --batch 25 --steps 200 --warmup 20 --workers 2

taskset -c 0-7 python evaluate.py \
  --run runs/imagenet-cpu-vits-5000 \
  --val /longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC/val \
  --train-per-class 100 --val-per-class 20
```

This uses ViT-S/16 depth and width (12 blocks, width 384, six heads), with
22,341,120 parameters including the projection head. Crops are reduced to
64×64 for the CPU experiment. The 200 updates visit all 5,000 training images
exactly once, with two augmented views each. One epoch at this resolution is
a training check, not a standard ImageNet benchmark. CPU affinity limits the
process to eight logical CPUs.

`--sampling epoch` shuffles the entire subset each epoch and pads an incomplete
last batch with images from the start of that epoch. Sampling is reproducible
on resume. The default `--sampling random` retains the original independent
batch sampling behavior.

## ViT-S on 50,000 images for five epochs

`bash run_imagenet_50k.sh` runs ViT-S/16 at 224×224 with bf16 activations,
batch 50, and 5,000 updates. It uses the first 50 sorted training images in
each of all 1,000 classes: exactly 50,000 images and 1,000 updates per epoch.
Parameters and coding-rate statistics remain fp32. Training starts from a
fresh initialization.

The launcher selects physical GPU 2 by default (`SIMDINO_GPU` overrides it),
checks that it has no compute processes immediately before starting, and
requires JAX's CUDA backend. This occupancy check is not a scheduler reservation.
`SIMDINO_PYTHON` can override the local environment's Python path.
It automatically evaluates initial and final frozen features using 50,000
training references and 5,000 validation images, with labels used only for
evaluation. Outputs are in `runs/imagenet-gpu-vits-50k/`; the adjacent `.status`
file records `training`, `evaluating`, `complete`, or failure.

To save the console output and limit CPU use, run from this directory:

```bash
mkdir -p runs
taskset -c 0-7 bash run_imagenet_50k.sh > runs/imagenet-gpu-vits-50k.log 2>&1
```

After an interruption, use the same launcher with
`--resume runs/imagenet-gpu-vits-50k/last.pkl`.

## What this implements

This is a two-global-view teaching variant of [SimDINO](https://proceedings.mlr.press/v267/wu25ar.html), not DINO's prototype cross-entropy objective. Student and teacher embeddings are unit-normalized; opposite views are aligned with `1 - cosine_similarity`. A coding-rate term rewards spread of student features, using the uncentered second moment as in the [released implementation](https://github.com/RobinWu218/SimDINO/blob/main/simdino/main_dino.py).

The rate is computed separately for each view and averaged. Its statistic is global across the data mesh: averaging local log-determinants would define a different loss. No teacher-logit centering, sharpening temperatures, or giant prototype layer is used. The ViT still uses ordinary softmax attention internally.

Simplifications include a small configurable ViT with bias-free linear maps, a two-layer projection head, two global crops without local crops, simplified color/blur augmentations, and fixed regularization coefficients. It is a training demonstration, not a reproduction of the paper's augmentation recipe, hyperparameters, multi-crop estimator, or ImageNet scores. The epsilon flag is the distortion scale; the formula uses its square.

`--mode fsdp` changes storage placement while retaining the same `Trainer.step`. Eligible matrix rows and matching optimizer state are sharded; small or indivisible leaves remain replicated. This is a single-host example. Exact communication scheduling and peak memory depend on JAX compilation and should be profiled before claiming FSDP efficiency at scale.

## CPU checks

```bash
python -m pip install pytest==8.4.2
XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m pytest test_train.py -q
```

The tests compare DP and FSDP updates with an unsharded reference, including Adam moments and teacher EMA. They also check loss geometry, the teacher's zero gradients, checkpoint continuation, deterministic two-view augmentation, and the blog's explicit `shard_map` loss and gradients. Two CPU devices simulate placements; they are not a multi-GPU throughput benchmark.

The recorded run is in `runs/imagenet-cpu-classes/`, with a short report in `EXPERIMENT.md`. Run outputs and checkpoints are ignored by Git, and this companion code directory is excluded from Jekyll output.
