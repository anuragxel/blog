---
layout: post
title: "Pretrain a vision model from scratch. Step 4: Write SimDINO in Jax"
description: Implementing SimDINO in Jax.
---

In the previous posts, we covered the [major families of self-supervised learning](https://anuragxel.github.io/blog/), then we understood that [distributed scaling is placement of arrays across devices](https://anuragxel.github.io/blog/) and how [different distributed strategies can be implemented](https://anuragxel.github.io/blog/) in JAX. Let's see how we can leverage these mental models to implement a self-supervised learning algorithm that can be scaled.

Let us implement a self-distillation based self-supervised learning algorithm, SimDINO. Quick recap of SimDINO {% cite wu2025simplifying %} from the [Visual SSL post](https://anuragxel.github.io/blog/): SimDINO is largely like DINO in-so-far as it aligns student and teacher embeddings over two augmented views of the same sample. Instead of relying on stability related tricks, it instead adds a coding-rate regularization to discourage collapse. The teacher is an exponential moving average of the student just like DINO.

For this example, use two global crops per image. Let $Z_s^{(v)}, Z_t^{(v)} \in \mathbb{R}^{B \times d}$ be their normalized embeddings, with views $v \in \{1,2\}$. Match opposite views and average the coding rate over the student views:

$$\mathcal{L}_{\mathrm{align}} = \frac{1}{2B}\sum_{i=1}^{B}\left[1-\langle z_{s,i}^{(1)},z_{t,i}^{(2)}\rangle + 1-\langle z_{s,i}^{(2)},z_{t,i}^{(1)}\rangle\right],$$

$$R(Z) = \frac{1}{2}\log\det\!\left(I + \frac{d}{B\epsilon^2}Z^TZ\right), \qquad \mathcal{L}=\mathcal{L}_{\mathrm{align}}-\frac{\gamma}{2}\left[R(Z_s^{(1)})+R(Z_s^{(2)})\right].$$

Unit normalization matters here, otherwise the model could increase the rate just by scaling up its embeddings. Here $\epsilon$ is the distortion scale and $\gamma$ sets the regularizer's strength.

## The model, in house style

Parameters are a pytree — here, a dictionary containing arrays and a list of per-block dictionaries. A `ViT` class holds the architecture configuration; its `apply` method takes weights explicitly.

```python
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
```

The embedding has dimension `config.dim`, not a dimension indexing thousands of prototypes. For downstream evaluation, `backbone=True` returns the normalized CLS representation before the projection head. `einx` names the contractions, and `einops` names the rearrangements, following the style from Part 3.

## The loss, and where stop-gradient lives

The trainer creates two model instances with the same architecture. Their weights live in a separate `State` pytree, so the student and teacher can follow different update rules. `Trainer` holds configuration and the optimizer transformation, not mutable training weights:

```python
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
```

The batch-coupled part of the objective is one small function:

```python
def coding_rate(z: Float[Array, "v b d"], eps: float) -> Float[Array, ""]:
    """Uncentered second moment, per view; b is the global batch under jit."""
    batch_size, dimension = z.shape[1:]
    moment = einx.dot("v [b] d, v [b] e -> v d e", z, z) / batch_size
    matrix = jnp.eye(dimension) + (dimension / eps**2) * moment
    cholesky = jnp.linalg.cholesky(matrix)
    diagonal = jnp.diagonal(cholesky, axis1=-2, axis2=-1)
    # logdet(A) / 2 = sum(log(diag(cholesky(A)))).
    return jnp.log(diagonal).sum(axis=-1).mean()
```

The matrix inside the determinant is positive definite because of the identity term. If $A=LL^T$ is its Cholesky factorization, $\tfrac12\log\det A=\sum_j\log L_{jj}$. The code uses that identity and computes the loss statistics in fp32, including when the ViT uses bf16.

```python
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
```

`views` has shape `(2, batch, height, width, channels)`. Both networks see both crops, but `teacher_z[::-1]` swaps the teacher views so each student output matches the other crop. The first term rewards agreement; the second rewards a more distributed set of embeddings.

Now the first payoff. Differentiation is with respect to the arguments we name:

```python
(_, metrics), gradients = jax.value_and_grad(trainer.loss, argnums=0, has_aux=True)(
    state.student_weights, state.teacher_weights, views,
)
```

That computes gradients for the student tree only. `stop_gradient` also makes the teacher boundary explicit inside the loss. The coding-rate calculation stays differentiable: stopping gradients through its batch statistic would remove the pressure it is supposed to apply to the student.

## The EMA teacher and the training step

The second payoff. A teacher update touches every array in the parameter tree. Because both trees are plain values with identical structure, we can write the update as follows:

```python
def ema(teacher_weights, student_weights, momentum):
    return jax.tree.map(
        lambda teacher, student: momentum * teacher + (1 - momentum) * student,
        teacher_weights, student_weights,
    )
```

That's the entire momentum update — the mechanism also used by MoCo {% cite he2020momentum %} and BYOL {% cite grill2020bootstrap %}. One `tree.map`. It returns a new teacher tree, without modifying the old one.

The full training step composes the loss, optimizer, and EMA update:

```python
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
```

The `State` fields are `student_weights`, `teacher_weights`, `optimizer_state`, and `step`. There is no running center. The batch second moment is an intermediate in the loss, recomputed each step, rather than persistent training state. `self.optimizer` is an Optax transformation with gradient clipping, AdamW, and a warmup/cosine learning-rate schedule; the teacher momentum approaches one over training.

The two updates are visible next to each other: the student follows the loss gradient through AdamW; the teacher moves toward the updated student by EMA. `Trainer.step` returns a new state without mutating the old one. That is the interface we preserve when changing placement.

## Sharding it: DP, then FSDP, by changing the placement

Following Parts 2 and 3, parallelizing this step is a placement decision. Replicate state and split the batch for DP; shard eligible parameter and optimizer arrays as well for FSDP:

```python
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
```

The view axis stays replicated and the batch axis is sharded: `P(None, "data")`. Both crops of an image therefore stay on the same device. Small arrays and matrices whose leading dimension is not divisible by the device count remain replicated; we do not blindly shard every leaf.

`Trainer.step` did not change. Under `jit`, the arrays describe the global computation. The contraction over `b` in `coding_rate` must therefore use the global batch, even though each device holds only part of it. The compiler inserts communication to preserve that meaning, along with the communication needed to combine parameter gradients.

This is the SimDINO version of the subtle distributed-statistics issue. Computing one coding rate per device and averaging those scalars is a different objective from computing the rate of the global second moment:

$$\frac{1}{N}\sum_{p=1}^{N}\log\det(I+\alpha C_p) \neq \log\det\!\left(I+\alpha\frac{1}{N}\sum_{p=1}^{N}C_p\right).$$

Here we explicitly choose a global-batch objective. Local or subsampled rate estimators are possible choices too; the point is to choose one deliberately, rather than changing the objective accidentally when adding devices.

With FSDP-style storage {% cite rajbhandari2020zero zhao2023pytorch %}, the compiler can gather sharded weights for use and reduce-scatter their gradients. The exact schedule and peak memory depend on the compiled program; a sharding specification alone does not guarantee the optimal layer-by-layer schedule. When corresponding student and teacher leaves have identical placements, their EMA update is local to each shard.

For this small model, FSDP is mainly illustrative. The useful property is that the loss and update rule survive the change from replicated to partitioned state.

## Seeing the collectives with shard_map

To close the loop with Part 2's vocabulary, here is the global coding-rate calculation with the collective written out. Inside `shard_map`, `z` contains only the device's local batch. For equally sized shards:

```python
def global_rate(z, eps):
    local_moment = einx.dot("v [b] d, v [b] e -> v d e", z, z) / z.shape[1]
    moment = jax.lax.pmean(local_moment, "data")
    matrix = jnp.eye(z.shape[-1]) + (z.shape[-1] / eps**2) * moment
    chol = jnp.linalg.cholesky(matrix)
    return jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)).sum(-1).mean()
```

That `pmean` averages the matrices **before** the nonlinear log-determinant. It belongs inside the differentiated loss. The following function returns the global loss. We differentiate the entire sharded function, so autodiff accounts for both the shared parameters and the statistic reduction:

```python
from functools import partial

@partial(jax.shard_map, mesh=mesh,
         in_specs=(P(), P(), P(None, "data")), out_specs=P())
def explicit_loss(student, teacher, views):
    flat = rearrange(views, "v b h w c -> (v b) h w c")
    zs = rearrange(trainer.student.apply(student, flat), "(v b) d -> v b d", v=2)
    zt = jax.lax.stop_gradient(rearrange(
        trainer.teacher.apply(teacher, flat), "(v b) d -> v b d", v=2))
    local_alignment = 1 - einx.dot("v b [d], v b [d] -> v b", zs, zt[::-1]).mean()
    return jax.lax.pmean(local_alignment, "data") - trainer.config.gamma * global_rate(zs, trainer.config.eps)

explicit_loss_and_grads = jax.jit(jax.value_and_grad(explicit_loss, argnums=0))
```

There are two distinct communication roles: combine feature statistics for the chosen objective, and combine parameter gradients for DP. The forward collectives are explicit here; differentiating the global function supplies the backward communication and gradient scaling. The returned gradients already correspond to the global loss, so we do not average them again. Replacing `global_rate` with a device-local rate would retain valid array shapes while changing the loss — precisely the kind of difference explicit collectives help us see.

## Why this was the right toy

A classifier demo often has one parameter tree, one update rule, and a loss separable over examples. SimDINO gives us two trees with different updates, an explicit differentiation boundary, and a loss that couples examples through a batch statistic. Jax lets us express those pieces directly, then change their placement without rewriting the objective.

The companion folder contains CPU checks comparing unsharded, DP, and FSDP updates, plus commands for an ImageNet class-folder dataset. A short run verifies the implementation and data path; representation quality needs a longer experiment and downstream evaluation. That is the distinction I want the small codebase to make easy: the core training algorithm is compact enough to read, while the experiment remains something we have to measure.

# References

{% bibliography --cited %}
