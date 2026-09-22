"""Frozen-backbone 1-NN: compare the initial model with the trained EMA teacher."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from checkpoint import load
from data import image_files, load_eval_image
from model import ViT


def features(model, weights, files, batch_size):
    infer = jax.jit(lambda parameters, images: model.apply(parameters, images, backbone=True))
    output = []
    for start in range(0, len(files), batch_size):
        batch = [load_eval_image(path, model.config.size) for path, _ in files[start:start + batch_size]]
        output.append(np.asarray(infer(weights, np.stack(batch))))
    return np.concatenate(output)


def evaluate(run, val_root, train_per_class=16, val_per_class=8):
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
    for name in ("initial", "last"):
        config, state = load(run / f"{name}.pkl")
        model = ViT(config)
        bank = features(model, state.teacher_weights, training, config.batch)
        queries = features(model, state.teacher_weights, validation, config.batch)
        # Evaluate in chunks so a larger reference set does not require one huge matrix.
        predictions = []
        for start in range(0, len(queries), config.batch):
            similarities = queries[start:start + config.batch] @ bank.T
            predictions.extend(train_labels[similarities.argmax(axis=-1)])
        results[name + "_1nn"] = float(np.mean(np.array(predictions) == val_labels))
    (run / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--val-per-class", type=int, default=8)
    args = parser.parse_args()
    evaluate(args.run, args.val, args.train_per_class, args.val_per_class)
