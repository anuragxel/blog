"""Frozen-backbone 1-NN: compare the initial model with the trained EMA teacher."""
import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from checkpoint import load
from data import chunked, image_files, load_eval_batch, normalize_pixels, prefetched
from model import ViT


def features(
    model,
    weights,
    files,
    batch_size,
    pool,
    workers,
):
    """Frozen CLS features, with images loaded in worker processes and batches split across devices."""
    mesh = Mesh(np.array(jax.devices()), ("data",))
    batch_size = -(-batch_size // mesh.size) * mesh.size
    infer = jax.jit(
        lambda parameters, pixels: model.apply(parameters, normalize_pixels(pixels), backbone=True),
        in_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P("data"))),
        out_shardings=NamedSharding(mesh, P()),
    )
    weights = jax.device_put(weights, NamedSharding(mesh, P()))
    paths = [path for path, _ in files]
    batches = [paths[start:start + batch_size] for start in range(0, len(paths), batch_size)]
    chunk = max(1, batch_size // workers)
    load_batch = partial(load_eval_batch, size=model.config.size)
    output = []
    for batch, chunks in zip(batches, prefetched(pool, load_batch, (chunked(b, chunk) for b in batches), 4)):
        pixels = np.concatenate(chunks)
        # Pad the last batch to a fixed, device-divisible shape; padded rows are discarded.
        padded = np.resize(pixels, (batch_size, *pixels.shape[1:]))
        output.append(infer(weights, padded)[:len(batch)])
    return np.concatenate(jax.device_get(output))


def nearest_labels(
    queries,
    bank,
    bank_labels,
    chunk=256,
):
    """Cosine 1-NN on one device, in query chunks so the similarity matrix stays small."""
    bank = jnp.asarray(bank)
    nearest = jax.jit(lambda q, b: jnp.argmax(q @ b.T, axis=-1))
    indices = [nearest(queries[start:start + chunk], bank) for start in range(0, len(queries), chunk)]
    return bank_labels[np.concatenate(jax.device_get(indices))]


def evaluate(
    run,
    val_root,
    train_per_class=16,
    val_per_class=8,
    workers=8,
    batch_size=0,
):
    metadata = json.loads((run / "config.json").read_text())
    train_root, classes = Path(metadata["data"]), metadata["classes"]
    train_names = sorted(path.name for path in train_root.iterdir() if path.is_dir())
    val_names = sorted(path.name for path in val_root.iterdir() if path.is_dir())
    if classes:
        train_names, val_names = train_names[:classes], val_names[:classes]
    if train_names != val_names:
        raise ValueError("Train and validation class folders must match")
    training = image_files(train_root, classes, train_per_class)
    validation = image_files(val_root, classes, val_per_class)
    train_labels = np.array([label for _, label in training])
    val_labels = np.array([label for _, label in validation])
    results = {
        "classes": len(train_names), "train_images": len(training),
        "val_images": len(validation), "chance": 1 / len(train_names),
    }
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as pool:
        for name in ("initial", "last"):
            config, state = load(run / f"{name}.pkl")
            model = ViT(config)
            extract = partial(features, model, state.teacher_weights,
                              batch_size=batch_size or config.batch, pool=pool, workers=workers)
            bank, queries = extract(training), extract(validation)
            predictions = nearest_labels(queries, bank, train_labels)
            results[name + "_1nn"] = float(np.mean(predictions == val_labels))
            print(name, results[name + "_1nn"], flush=True)
    (run / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--val-per-class", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8, help="Image-loading processes")
    parser.add_argument("--batch", type=int, default=0, help="Images per forward pass; 0 uses training batch")
    args = parser.parse_args()
    evaluate(args.run, args.val, args.train_per_class, args.val_per_class, args.workers, args.batch)
