"""Student gradients, teacher EMA, and placement: the SimDINO training algorithm."""
from typing import NamedTuple

import einx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from einops import rearrange
from jaxtyping import Array, Float, PyTree
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from config import Config
from model import ViT


class State(NamedTuple):
    student_weights: PyTree
    teacher_weights: PyTree
    optimizer_state: PyTree
    step: Array


def coding_rate(z: Float[Array, "v b d"], eps: float) -> Float[Array, ""]:
    """Uncentered second moment, per view; b is the global batch under jit."""
    batch_size, dimension = z.shape[1:]
    moment = einx.dot("v [b] d, v [b] e -> v d e", z, z) / batch_size
    matrix = jnp.eye(dimension) + (dimension / eps**2) * moment
    cholesky = jnp.linalg.cholesky(matrix)
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    # logdet(A) / 2 = sum(log(diag(cholesky(A)))).
    return jnp.log(diagonal).sum(axis=-1).mean()


def ema(teacher_weights, student_weights, momentum):
    return jax.tree.map(
        lambda teacher, student: momentum * teacher + (1 - momentum) * student,
        teacher_weights, student_weights,
    )


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.student = ViT(config)
        self.teacher = ViT(config)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=config.lr,
            warmup_steps=min(config.warmup, config.steps - 1),
            decay_steps=config.steps, end_value=config.lr * 0.01,
        )
        # Decay matrix weights, but not CLS, positions, or normalization vectors.
        def decay_mask(parameters):
            return jax.tree.map(lambda x: x.ndim == 2, parameters)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(3.),
            optax.adamw(schedule, weight_decay=config.wd, mask=decay_mask),
        )

    def init(self, key: Array) -> State:
        weights = self.student.init(key)
        # JAX arrays are immutable; sharing initial values does not tie updates.
        return State(weights, weights, self.optimizer.init(weights), jnp.array(0, jnp.int32))

    def loss(self, student_weights, teacher_weights, views: Float[Array, "v b h w c"]):
        flat = rearrange(views, "v b h w c -> (v b) h w c")
        student_z = self.student.apply(student_weights, flat)
        teacher_z = jax.lax.stop_gradient(self.teacher.apply(teacher_weights, flat))
        student_z = rearrange(student_z, "(v b) d -> v b d", v=2)
        teacher_z = rearrange(teacher_z, "(v b) d -> v b d", v=2)

        # Each student view matches the OTHER teacher view of the same image.
        similarities = einx.dot("v b [d], v b [d] -> v b", student_z, teacher_z[::-1])
        alignment = 1 - similarities.mean()
        rate = coding_rate(student_z, self.config.eps)
        loss = alignment - self.config.gamma * rate
        metrics = {
            "loss": loss,
            "alignment": alignment,
            "rate": rate,
            "feature_std": student_z.std(axis=1).mean(),
        }
        return loss, metrics

    def step(self, state: State, views) -> tuple[State, dict]:
        # Differentiate only the first argument: the student's weight tree.
        (_, metrics), gradients = jax.value_and_grad(self.loss, argnums=0, has_aux=True)(
            state.student_weights, state.teacher_weights, views,
        )
        updates, optimizer_state = self.optimizer.update(
            gradients, state.optimizer_state, state.student_weights,
        )
        student_weights = optax.apply_updates(state.student_weights, updates)

        progress = state.step / self.config.steps
        momentum = 1 - (1 - self.config.momentum) * (1 + jnp.cos(jnp.pi * progress)) / 2
        teacher_weights = ema(state.teacher_weights, student_weights, momentum)
        new_state = State(student_weights, teacher_weights, optimizer_state, state.step + 1)
        return new_state, dict(metrics, grad_norm=optax.global_norm(gradients))

    def compile(self, state: State, mode="dp"):
        mesh = Mesh(np.array(jax.devices()), ("data",))
        replicated = NamedSharding(mesh, P())
        batch_sharding = NamedSharding(mesh, P(None, "data"))

        def placement(array):
            can_shard = array.ndim == 2 and array.shape[0] % mesh.size == 0
            if mode == "fsdp" and can_shard:
                return NamedSharding(mesh, P("data", None))
            return replicated

        state_shardings = jax.tree.map(placement, state)
        state = jax.tree.map(jax.device_put, state, state_shardings)
        compiled = jax.jit(
            self.step,
            in_shardings=(state_shardings, batch_sharding),
            out_shardings=(state_shardings, replicated),
        )
        return state, compiled, batch_sharding
