# ViT-S on 5,000 ImageNet images — 2026-09-12

- Data: first 50 sorted ImageNet classes, 100 training images per class from `/longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC/train`. Labels are unused during SSL.
- Model: ViT-S/16 dimensions (width 384, 12 blocks, six attention heads), with bias-free linear maps and a two-layer projection head of output dimension 128; 22,341,120 total parameters.
- Resolution: 64×64 crops, giving 16 patches plus CLS. This reduced resolution is not the standard 224×224 ImageNet setting.
- Training: one complete shuffled epoch, 200 updates with batch 25, visiting all 5,000 distinct images exactly once and generating two global views each (10,000 augmented views). Seed 0, fp32, AdamW, peak learning rate 0.0003, 20 warmup steps, weight decay 0.04, teacher EMA starting at 0.996, epsilon 0.5, gamma 0.02.
- Hardware: one JAX CPU device, process restricted to eight logical CPUs. GPU execution was disabled throughout training and evaluation.
- Runtime: 201.58 seconds summed over logged training steps, including the first compiled update (11.06 seconds), excluding model initialization and checkpoint writes. Median subsequent update: 0.921 seconds.
- Outputs: `runs/imagenet-cpu-vits-5000/`, including initial/final checkpoints, per-step metrics, configuration, and `training_summary.json` with audited sampling coverage. Run artifacts are ignored by Git.

| Diagnostic | First step | Last step |
|---|---:|---:|
| Loss | -0.043760 | -0.213771 |
| Alignment term | 0.473583 | 0.375303 |
| Coding rate | 25.867176 | 29.453697 |
| Mean per-coordinate feature standard deviation | 0.065616 | 0.064944 |

All 200 steps had finite metrics and gradient norms. These endpoints use different batches and augmentations. A negative loss is possible because the coding-rate reward is subtracted from the alignment term.

Frozen-CLS cosine 1-NN evaluation used all 5,000 training images as references
and the first 20 validation images in each of the same 50 classes (1,000 total),
with deterministic resize-and-center-crop preprocessing:

| Model | 1-NN accuracy |
|---|---:|
| Random initialization | 5.80% |
| EMA teacher after 200 updates | 6.10% |
| Uniform random-label baseline | 2.00% |

The difference is only three additional correct predictions in one run; this
does not establish a reliable improvement in representation quality. The result
demonstrates that the ViT-S-sized training and evaluation paths run on real
ImageNet data. One epoch, reduced resolution, and a 50-class subset limit the
conclusions. Raw evaluation results are in `evaluation.json` alongside the checkpoints.

The README contains exact reproduction commands. The core model and trainer remain 224 lines; epoch sampling is a supporting data utility. Six CPU numerical tests passed, and separate sampling checks verified complete coverage, last-batch padding, deterministic resume, and reshuffling between epochs.

## Earlier tiny CPU smoke test — 2026-09-12

This checks the training loop and real data path. It is not a representation-quality benchmark.

- Data: `/longdata/anurag_storage/imagenet/ILSVRC/Data/CLS-LOC`.
- Subset: first 10 sorted classes, 16 training images per class; SSL does not use their labels.
- Model: 78,144 parameters; two ViT blocks, width 48, three heads, 8×8 patches on 32×32 crops, projection dimension 16.
- Training: 30 steps, batch 8, two independent global views; AdamW, peak learning rate 0.0003, three warmup steps, EMA starting at 0.996, epsilon 0.5, gamma 0.02, seed 0.
- Hardware: one JAX CPU device. No GPU computation was launched.
- Outputs: `runs/imagenet-cpu-classes/` (ignored by Git).

| Diagnostic | First step | Last step |
|---|---:|---:|
| Loss | 0.225203 | 0.164050 |
| Alignment term | 0.331692 | 0.253323 |
| Coding rate | 5.324434 | 4.463655 |
| Mean per-coordinate feature standard deviation | 0.141155 | 0.100106 |

All metrics and gradient norms remained finite. The lower combined loss alone does not demonstrate better representations; the feature-spread measures also decreased.

A frozen-CLS 1-nearest-neighbor check used the 160 training images as references and eight validation images per class (80 total). Deterministic resize-and-center-crop preprocessing and cosine similarity were used for both models:

| Model | 1-NN accuracy |
|---|---:|
| Random initialization | 16.25% |
| EMA teacher after 30 steps | 15.00% |
| Uniform random-label baseline | 10.00% |

The trained teacher did not improve this small evaluation. A larger dataset, more training, and an appropriate evaluation budget are needed before drawing conclusions about learned feature quality. An earlier development check used different evaluation preprocessing and is not the result reported here.

The README contains the exact training and evaluation commands. The separate two-CPU-device numerical tests compare unsharded, DP, and FSDP updates, including optimizer and teacher state, and verify the explicit collective example in the blog.
