"""Run the small SimDINO experiment on an ImageFolder training split."""
import argparse
import json
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path

import jax
import numpy as np

from checkpoint import load, save
from config import Config
from data import chunked, epoch_indices, image_files, load_view_batch, normalize_pixels, prefetched
from train import Trainer


def parse_args(
):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/simdino"))
    parser.add_argument("--mode", choices=["dp", "fsdp"], default="dp")
    parser.add_argument("--classes", type=int, default=0, help="First N classes; 0 means all")
    parser.add_argument("--per-class", type=int, default=0, help="First N files per class; 0 means all")
    parser.add_argument("--workers", type=int, default=8, help="Image-loading processes")
    parser.add_argument("--prefetch", type=int, default=4, help="Batches loaded ahead of training")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling", choices=["random", "epoch"], default="random")
    parser.add_argument("--resume", type=Path)
    for name, default in asdict(Config()).items():
        if isinstance(default, bool):
            parser.add_argument("--" + name, action="store_true", default=default)
        else:
            parser.add_argument("--" + name, type=type(default), default=default)
    args = parser.parse_args()
    config = Config(**{name: getattr(args, name) for name in asdict(Config())})
    positive = (config.size, config.patch, config.width, config.depth, config.heads, config.dim)
    if min(positive) < 1 or config.steps < 2 or config.batch < 2:
        parser.error("Model dimensions must be positive; steps and batch must be at least 2")
    if config.batch % jax.device_count() or config.size % config.patch or config.width % config.heads:
        parser.error("Batch must divide across devices, size into patches, and width into heads")
    if min(args.workers, args.prefetch, args.log_every, args.save_every) < 1:
        parser.error("Workers, prefetch, and logging and checkpoint intervals must be positive")
    if config.eps <= 0 or config.warmup < 0:
        parser.error("Workers and eps must be positive; warmup must be nonnegative")
    if min(args.classes, args.per_class, config.gamma, config.wd) < 0 or config.lr <= 0:
        parser.error("Subset sizes, gamma, and weight decay must be nonnegative; lr must be positive")
    if not 0 <= config.momentum <= 1:
        parser.error("Teacher momentum must be in [0, 1]")
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        parser.error("Output directory is nonempty; use a new path or --resume")
    return args, config


def batch_items(
    files,
    batch,
    seed,
    sampling,
    iteration,
):
    """(path, augmentation seed) pairs for one update, determined by the step alone."""
    # Step-indexed randomness makes augmentation reproducible on resume.
    rng = np.random.default_rng([seed, iteration])
    indices = rng.choice(len(files), batch, replace=False)
    if sampling == "epoch":
        indices = epoch_indices(len(files), batch, seed, iteration)
    seeds = rng.integers(2**32, size=batch)
    return [(files[i][0], int(seed)) for i, seed in zip(indices, seeds)]


def main(
):
    args, config = parse_args()
    files = image_files(args.data, args.classes, args.per_class)
    if len(files) < config.batch:
        raise ValueError("Dataset must contain at least one full batch")
    trainer = Trainer(config)
    if args.resume:
        previous_config, state = load(args.resume)
        if previous_config != config:
            raise ValueError("Resume with the same model and training configuration")
    else:
        state = trainer.init(jax.random.key(args.seed))
    start = int(state.step)

    metadata = dict(asdict(config), data=str(args.data.resolve()), images=len(files),
                    classes=args.classes, per_class=args.per_class, seed=args.seed, mode=args.mode,
                    sampling=args.sampling)
    metadata_path = args.out / "config.json"
    if args.resume and metadata_path.exists():
        previous = json.loads(metadata_path.read_text())
        if previous.get("sampling", "random") != args.sampling:
            raise ValueError("Resume must preserve the sampling method")
        if any(previous[key] != metadata[key] for key in ("data", "classes", "per_class", "seed")):
            raise ValueError("Resume must preserve the dataset subset and random seed")
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    save(args.out / ("resumed.pkl" if args.resume else "initial.pkl"), state, config)
    state, step, batch_sharding = trainer.compile(state, args.mode)
    num_parameters = sum(x.size for x in jax.tree.leaves(state.student_weights))
    print(f"{len(files)} images; {jax.devices()}; {num_parameters:,} parameters", flush=True)

    # Workers only decode and augment; spawning keeps them free of the CUDA process state.
    context = multiprocessing.get_context("spawn")
    chunk = max(1, config.batch // args.workers)
    plans = (chunked(batch_items(files, config.batch, args.seed, args.sampling, iteration), chunk)
             for iteration in range(start, config.steps))
    to_views = jax.jit(normalize_pixels, out_shardings=batch_sharding)
    with ProcessPoolExecutor(args.workers, mp_context=context) as pool:
        batches = prefetched(pool, partial(load_view_batch, size=config.size), plans, args.prefetch)
        with (args.out / "metrics.jsonl").open("a") as log:
            pending, tick = [], time.perf_counter()
            for iteration, chunks in zip(range(start, config.steps), batches):
                pixels = jax.device_put(np.concatenate(chunks, axis=1), batch_sharding)
                state, metrics = step(state, to_views(pixels))
                pending.append(metrics)
                done = iteration + 1
                saving = done % args.save_every == 0 or done == config.steps
                # Reading metrics waits for the device, so do it once per window, not per step.
                if iteration == start or done % args.log_every == 0 or saving:
                    seconds = (time.perf_counter() - tick) / len(pending)
                    for offset, metrics in enumerate(jax.device_get(pending)):
                        metrics = {key: float(value) for key, value in metrics.items()}
                        number = done - len(pending) + offset + 1
                        if not all(math.isfinite(value) for value in metrics.values()):
                            raise FloatingPointError(f"Non-finite metrics at step {number}: {metrics}")
                        metrics.update(step=number, seconds=seconds)
                        log.write(json.dumps(metrics) + "\n")
                    log.flush()
                    metrics["images_per_second"] = config.batch / seconds
                    print(json.dumps(metrics), flush=True)
                    pending, tick = [], time.perf_counter()
                if saving:
                    save(args.out / "last.pkl", state, config)
                    tick = time.perf_counter()


if __name__ == "__main__":
    main()
