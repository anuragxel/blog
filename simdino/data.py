"""ImageFolder discovery and deterministic, per-sample CPU augmentations.

Loaders return uint8 pixels, which are 4x cheaper to pass between processes;
`normalize_pixels` converts them on whichever device holds the batch."""
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_files(
    root,
    classes=0,
    per_class=0,
):
    """Return (path, label) pairs; labels are used only by downstream evaluation."""
    folders = sorted(path for path in Path(root).iterdir() if path.is_dir())
    if classes:
        folders = folders[:classes]
    files = []
    for label, folder in enumerate(folders):
        images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
        if per_class:
            images = images[:per_class]
        files.extend((path, label) for path in images)
    if not files:
        raise ValueError(f"No class-folder images found in {root}")
    return files


def epoch_indices(
    count,
    batch,
    seed,
    step,
):
    """Shuffle each epoch, padding its last batch; any step can resume directly."""
    batches = (count + batch - 1) // batch
    epoch, offset = divmod(step, batches)
    order = np.random.default_rng([seed, epoch]).permutation(count)
    return np.resize(order, batches * batch)[offset * batch:(offset + 1) * batch]


def random_crop(
    image,
    size,
    rng,
):
    width, height = image.size
    for _ in range(10):
        area = width * height * rng.uniform(0.4, 1.)
        aspect = math.exp(rng.uniform(math.log(0.75), math.log(4 / 3)))
        crop_width = round(math.sqrt(area * aspect))
        crop_height = round(math.sqrt(area / aspect))
        if 0 < crop_width <= width and 0 < crop_height <= height:
            left = rng.integers(width - crop_width + 1)
            top = rng.integers(height - crop_height + 1)
            image = image.crop((left, top, left + crop_width, top + crop_height))
            break
    else:
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.BICUBIC)


def normalize_pixels(
    pixels,
):
    """uint8 pixels to standardized floats; accepts NumPy or JAX arrays."""
    return (pixels / 255 - IMAGE_MEAN) / IMAGE_STD


def open_rgb(
    path,
    min_side,
):
    """Let the JPEG decoder downscale by up to 8x while keeping both sides >= min_side."""
    with Image.open(path) as image:
        image.draft("RGB", (min_side, min_side))
        return image.convert("RGB")


def augment(
    image,
    size,
    rng,
):
    image = random_crop(image, size, rng)
    if rng.random() < 0.5:
        image = ImageOps.mirror(image)
    if rng.random() < 0.8:
        operations = [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]
        for operation in rng.permutation(operations):
            image = operation(image).enhance(rng.uniform(0.6, 1.4))
    if rng.random() < 0.2:
        image = ImageOps.grayscale(image).convert("RGB")
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.1, 2.)))
    if rng.random() < 0.1:
        image = ImageOps.solarize(image)
    return np.asarray(image, dtype=np.uint8)


def load_views(
    item,
    size,
):
    path, seed = item
    rng = np.random.default_rng(seed)
    # The smallest crop covers about 0.55 of a side, so keep 2x the output size.
    image = open_rgb(path, 2 * size)
    return np.stack([augment(image, size, rng) for _ in range(2)])


def load_view_batch(
    items,
    size,
):
    """One worker task: (path, seed) pairs to uint8 views shaped (2, b, h, w, c)."""
    return np.stack([load_views(item, size) for item in items], axis=1)


def load_eval_image(
    path,
    size,
):
    """Resize the shorter side, then center-crop without random augmentation."""
    image = open_rgb(path, round(size / 0.875))
    scale = round(size / 0.875) / min(image.size)
    resized = tuple(round(dimension * scale) for dimension in image.size)
    image = image.resize(resized, Image.Resampling.BICUBIC)
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return np.asarray(image.crop((left, top, left + size, top + size)), dtype=np.uint8)


def load_eval_batch(
    paths,
    size,
):
    return np.stack([load_eval_image(path, size) for path in paths])


def chunked(
    items,
    size,
):
    return [items[start:start + size] for start in range(0, len(items), size)]


def prefetched(
    pool,
    load,
    batches,
    depth,
):
    """Load each batch's chunks in worker processes, `depth` batches ahead; yield in order."""
    pending = deque()
    for chunks in batches:
        pending.append([pool.submit(load, chunk) for chunk in chunks])
        if len(pending) > depth:
            yield [future.result() for future in pending.popleft()]
    while pending:
        yield [future.result() for future in pending.popleft()]
