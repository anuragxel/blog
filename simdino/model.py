"""A pre-norm ViT with a CLS token and a normalized two-layer projection head."""
import math
from dataclasses import dataclass

import einx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from config import Config


def normalize(
    z: Float[Array, "*b d"],
) -> Float[Array, "*b d"]:
    """Unit-length embeddings, with normalization computed in fp32."""
    z = z.astype(jnp.float32)
    squared_norms = einx.sum("... [d] -> ... 1", z * z)
    norms = jnp.sqrt(squared_norms)
    norms = jnp.maximum(norms, 1e-8)
    return z / norms


def layer_norm(
    x: Float[Array, "*b d"],
    parameters: PyTree,
) -> Float[Array, "*b d"]:
    # Accumulate statistics in fp32 even when the surrounding layers use bf16.
    values = x.astype(jnp.float32)
    mean = einx.mean("... [d] -> ... 1", values)
    variance = einx.var("... [d] -> ... 1", values)
    inverse_std = jax.lax.rsqrt(variance + 1e-6)
    normalized = (values - mean) * inverse_std
    output = normalized * parameters["scale"] + parameters["bias"]
    return output.astype(x.dtype)


def linear(
    x: Float[Array, "*b d"],
    weight: Float[Array, "d e"],
) -> Float[Array, "*b e"]:
    weight = weight.astype(x.dtype)
    return einx.dot("... [d], [d] e -> ... e", x, weight)


def split_qkv(
    qkv: Array,
    heads: int,
) -> tuple[Array, Array, Array]:
    query, key, value = einx.rearrange(
        "b t (qkv heads d) -> qkv b t heads d", qkv, qkv=3, heads=heads,
    )
    return query, key, value


def merge_heads(
    attended: Float[Array, "b t heads d"],
) -> Float[Array, "b t width"]:
    return einx.rearrange("b t heads d -> b t (heads d)", attended)


def attention(
    x: Float[Array, "b t d"],
    parameters: PyTree,
    heads: int,
) -> Float[Array, "b t d"]:
    qkv = linear(x, parameters["qkv"])
    query, key, value = split_qkv(qkv, heads)
    attended = jax.nn.dot_product_attention(query, key, value)
    attended = merge_heads(attended)
    return linear(attended, parameters["out"])


def mlp(
    x: Float[Array, "*b d"],
    up: Float[Array, "d hidden"],
    down: Float[Array, "hidden e"],
) -> Float[Array, "*b e"]:
    hidden = linear(x, up)
    hidden = jax.nn.gelu(hidden)
    return linear(hidden, down)


def projection_head(
    features: Float[Array, "b d"],
    parameters: PyTree,
) -> Array:
    return mlp(features, parameters["head1"], parameters["head2"])


def patchify(
    images: Float[Array, "b h w c"],
    patch_size: int,
) -> Float[Array, "b t patch"]:
    return einx.rearrange(
        "b (h ph) (w pw) c -> b (h w) (ph pw c)", images,
        ph=patch_size, pw=patch_size,
    )


def add_cls_and_positions(
    tokens: Float[Array, "b t d"],
    parameters: PyTree,
) -> Array:
    cls = parameters["cls"].astype(tokens.dtype)
    positions = parameters["pos"].astype(tokens.dtype)
    tokens = einx.rearrange("1 1 d, b t d -> b (1 + t) d", cls, tokens)
    return tokens + positions


def cls_token(
    tokens: Float[Array, "b t d"],
) -> Float[Array, "b d"]:
    cls, _ = einx.rearrange("b (1 + t) d -> b d, b t d", tokens)
    return cls


def transformer_block(
    x: Float[Array, "b t d"],
    weights: PyTree,
    heads: int,
) -> Array:
    normalized = layer_norm(x, weights["n1"])
    x = x + attention(normalized, weights, heads)
    normalized = layer_norm(x, weights["n2"])
    x = x + mlp(normalized, weights["up"], weights["down"])
    return x


@dataclass(frozen=True)
class ViT:
    """Architecture only: weights are explicit inputs, never hidden mutable state."""
    config: Config

    def init(
        self,
        key: Array,
    ) -> PyTree:
        config = self.config
        keys = jax.random.split(key, 4 + 4 * config.depth)
        keys = iter(keys)
        width = config.width

        def weight(
            inputs,
            outputs,
        ):
            key = next(keys)
            values = jax.random.normal(key, (inputs, outputs))
            return values / math.sqrt(inputs)

        def norm(
        ):
            return {"scale": jnp.ones(width), "bias": jnp.zeros(width)}

        # All trainable leaves, including normalization, belong to this one pytree.
        embedding = weight(3 * config.patch**2, width)
        num_tokens = (config.size // config.patch)**2 + 1
        position_key = next(keys)
        positions = 0.02 * jax.random.normal(position_key, (1, num_tokens, width))
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

    def embed_patches(
        self,
        parameters: PyTree,
        images: Float[Array, "b h w c"],
    ) -> Array:
        dtype = jnp.bfloat16 if self.config.bf16 else jnp.float32
        patches = patchify(images, self.config.patch)
        patches = patches.astype(dtype)
        return linear(patches, parameters["embed"])

    def encode(
        self,
        parameters: PyTree,
        images: Float[Array, "b h w c"],
    ) -> Float[Array, "b d"]:
        tokens = self.embed_patches(parameters, images)
        tokens = add_cls_and_positions(tokens, parameters)
        for weights in parameters["blocks"]:
            tokens = transformer_block(tokens, weights, self.config.heads)
        tokens = layer_norm(tokens, parameters["norm"])
        return cls_token(tokens)

    def apply(
        self,
        parameters: PyTree,
        images: Float[Array, "b h w c"],
        backbone: bool = False,
    ) -> Float[Array, "b d"]:
        features = self.encode(parameters, images)
        if backbone:
            return normalize(features)
        embeddings = projection_head(features, parameters)
        return normalize(embeddings)

    def encode_views(
        self,
        parameters: PyTree,
        views: Float[Array, "2 b h w c"],
    ) -> Float[Array, "2 b d"]:
        images = einx.rearrange("v b h w c -> (v b) h w c", views)
        embeddings = self.apply(parameters, images)
        return einx.rearrange("(v b) d -> v b d", embeddings, v=2)
