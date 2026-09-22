"""A pre-norm ViT with a CLS token and a normalized two-layer projection head."""
import math
from dataclasses import dataclass

import einx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float, PyTree

from config import Config


def normalize(z: Array) -> Array:
    norms = jnp.linalg.norm(z, axis=-1, keepdims=True)
    return z / jnp.maximum(norms, 1e-8)


def layer_norm(x: Array, parameters: PyTree) -> Array:
    # Accumulate statistics in fp32 even when the surrounding layers use bf16.
    values = x.astype(jnp.float32)
    mean = values.mean(axis=-1, keepdims=True)
    variance = values.var(axis=-1, keepdims=True)
    normalized = (values - mean) * jax.lax.rsqrt(variance + 1e-6)
    return (normalized * parameters["scale"] + parameters["bias"]).astype(x.dtype)


@dataclass(frozen=True)
class ViT:
    """Architecture only: weights are explicit inputs, never hidden mutable state."""
    config: Config

    def init(self, key: Array) -> PyTree:
        config = self.config
        keys = iter(jax.random.split(key, 4 + 4 * config.depth))
        width = config.width

        def weight(inputs, outputs):
            return jax.random.normal(next(keys), (inputs, outputs)) / math.sqrt(inputs)

        def norm():
            return {"scale": jnp.ones(width), "bias": jnp.zeros(width)}

        # All trainable leaves, including normalization, belong to this one pytree.
        embedding = weight(3 * config.patch**2, width)
        num_tokens = (config.size // config.patch)**2 + 1
        positions = 0.02 * jax.random.normal(next(keys), (1, num_tokens, width))
        blocks = []
        for _ in range(config.depth):
            blocks.append({
                "qkv": weight(width, 3 * width),
                "out": weight(width, width),
                "up": weight(width, 4 * width),
                "down": weight(4 * width, width),
                "n1": norm(),
                "n2": norm(),
            })
        return {
            "embed": embedding,
            "pos": positions,
            "cls": jnp.zeros((1, 1, width)),
            "norm": norm(),
            "blocks": blocks,
            "head1": weight(width, 4 * width),
            "head2": weight(4 * width, config.dim),
        }

    def apply(
        self,
        parameters: PyTree,
        images: Float[Array, "b h w c"],
        backbone: bool = False,
    ) -> Float[Array, "b d"]:
        config = self.config
        dtype = jnp.bfloat16 if config.bf16 else jnp.float32

        def linear(x, weight):
            return einx.dot("... [d], [d] e -> ... e", x, weight.astype(dtype))

        patches = rearrange(
            images, "b (h p) (w q) c -> b (h w) (p q c)",
            p=config.patch, q=config.patch,
        )
        x = linear(patches.astype(dtype), parameters["embed"])
        cls = jnp.broadcast_to(parameters["cls"].astype(dtype), (len(images), 1, config.width))
        x = jnp.concatenate((cls, x), axis=1) + parameters["pos"].astype(dtype)

        for block in parameters["blocks"]:
            qkv = linear(layer_norm(x, block["n1"]), block["qkv"])
            query, key, value = rearrange(
                qkv, "b t (q h d) -> q b t h d", q=3, h=config.heads,
            )
            attended = jax.nn.dot_product_attention(query, key, value)
            attended = rearrange(attended, "b t h d -> b t (h d)")
            x = x + linear(attended, block["out"])
            hidden = linear(layer_norm(x, block["n2"]), block["up"])
            x = x + linear(jax.nn.gelu(hidden), block["down"])

        features = layer_norm(x, parameters["norm"])[:, 0]
        if backbone:
            return normalize(features.astype(jnp.float32))
        hidden = jax.nn.gelu(linear(features, parameters["head1"]))
        embeddings = linear(hidden, parameters["head2"])
        return normalize(embeddings.astype(jnp.float32))
