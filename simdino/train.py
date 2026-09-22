"""Student gradients, teacher EMA, and placement: the SimDINO training algorithm."""
from typing import NamedTuple

import einx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float, PyTree
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from config import Config
from model import ViT


class State(NamedTuple):
    student_weights: PyTree
    teacher_weights: PyTree
    optimizer_state: PyTree
    step: Array


def second_moment(
    z: Float[Array, "v b d"],
) -> Float[Array, "v d d"]:
    """One uncentered second-moment matrix per view, over the supplied batch."""
    batch_size = z.shape[1]
    moment = einx.dot("v [b] d, v [b] e -> v d e", z, z)
    return moment / batch_size


def coding_rate_from_moment(
    moment: Float[Array, "v d d"],
    eps: float,
) -> Float[Array, ""]:
    """Mean of 0.5 * logdet(I + dim / eps² * moment) across views."""
    dimension = moment.shape[-1]
    matrix = jnp.eye(dimension) + (dimension / eps**2) * moment
    cholesky = jnp.linalg.cholesky(matrix)
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    log_diagonal = jnp.log(diagonal)
    per_view = einx.sum("v [d] -> v", log_diagonal)
    return per_view.mean()


def coding_rate(
    z: Float[Array, "v b d"],
    eps: float,
) -> Float[Array, ""]:
    moment = second_moment(z)
    return coding_rate_from_moment(moment, eps)


def cross_view_alignment(
    student: Float[Array, "2 b d"],
    teacher: Float[Array, "2 b d"],
) -> Float[Array, ""]:
    """Mean cosine distance between opposite crops of each image."""
    opposite_views = teacher[::-1]
    similarities = einx.dot("v b [d], v b [d] -> v b", student, opposite_views)
    return 1 - similarities.mean()


def feature_spread(
    z: Float[Array, "v b d"],
) -> Float[Array, ""]:
    """Standard deviation across images, averaged over views and features."""
    feature_std = einx.std("v [b] d -> v d", z)
    return feature_std.mean()


def teacher_momentum(
    step: Array,
    initial: float,
    steps: int,
) -> Array:
    progress = step / steps
    cosine_decay = (1 + jnp.cos(jnp.pi * progress)) / 2
    return 1 - (1 - initial) * cosine_decay


def ema(
    teacher_weights: PyTree,
    student_weights: PyTree,
    momentum: Array | float,
) -> PyTree:
    return jax.tree.map(
        lambda teacher, student: momentum * teacher + (1 - momentum) * student,
        teacher_weights, student_weights,
    )


class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config
        self.student = ViT(config)
        self.teacher = ViT(config)
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0., peak_value=config.lr,
            warmup_steps=min(config.warmup, config.steps - 1),
            decay_steps=config.steps, end_value=config.lr * 0.01,
        )
        # Decay matrix weights, but not CLS, positions, or normalization vectors.
        def decay_mask(
            parameters,
        ):
            return jax.tree.map(lambda x: x.ndim == 2, parameters)

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(3.),
            optax.adamw(schedule, weight_decay=config.wd, mask=decay_mask),
        )

    def init(
        self,
        key: Array,
    ) -> State:
        weights = self.student.init(key)
        # JAX arrays are immutable; sharing initial values does not tie updates.
        optimizer_state = self.optimizer.init(weights)
        step = jnp.array(0, jnp.int32)
        return State(weights, weights, optimizer_state, step)

    def loss(
        self,
        student_weights,
        teacher_weights,
        views: Float[Array, "2 b h w c"],
    ):
        student_z = self.student.encode_views(student_weights, views)
        teacher_z = self.teacher.encode_views(teacher_weights, views)
        teacher_z = jax.lax.stop_gradient(teacher_z)

        alignment = cross_view_alignment(student_z, teacher_z)
        rate = coding_rate(student_z, self.config.eps)
        loss = alignment - self.config.gamma * rate
        spread = feature_spread(student_z)
        metrics = {
            "loss": loss,
            "alignment": alignment,
            "rate": rate,
            "feature_std": spread,
        }
        return loss, metrics

    def update_student(
        self,
        weights: PyTree,
        optimizer_state: PyTree,
        gradients: PyTree,
    ):
        updates, optimizer_state = self.optimizer.update(gradients, optimizer_state, weights)
        weights = optax.apply_updates(weights, updates)
        return weights, optimizer_state

    def update_teacher(
        self,
        teacher: PyTree,
        student: PyTree,
        step: Array,
    ) -> PyTree:
        momentum = teacher_momentum(step, self.config.momentum, self.config.steps)
        return ema(teacher, student, momentum)

    def step(
        self,
        state: State,
        views: Float[Array, "2 b h w c"],
    ) -> tuple[State, dict]:
        loss_and_grad = jax.value_and_grad(self.loss, argnums=0, has_aux=True)
        (_, metrics), gradients = loss_and_grad(
            state.student_weights, state.teacher_weights, views,
        )
        student, optimizer_state = self.update_student(
            state.student_weights, state.optimizer_state, gradients,
        )
        teacher = self.update_teacher(state.teacher_weights, student, state.step)
        new_state = State(student, teacher, optimizer_state, state.step + 1)
        metrics = dict(metrics)
        metrics["grad_norm"] = optax.global_norm(gradients)
        return new_state, metrics

    def compile(
        self,
        state: State,
        mode="dp",
    ):
        mesh = Mesh(np.array(jax.devices()), ("data",))
        replicated = NamedSharding(mesh, P())
        batch_sharding = NamedSharding(mesh, P(None, "data", None, None, None))

        def placement(
            array,
        ):
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
