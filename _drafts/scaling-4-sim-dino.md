---
layout: post
title: "Pretrain a vision model from scratch. Step 4: Write SimDINO in JAX"
description: Implementing SimDINO in JAX.
---

In the previous posts, we covered the [major families of self-supervised learning]({% post_url 2026-09-18-scaling-1-visual-ssl %}), saw that [distributed scaling is just placement of arrays across devices]({% post_url 2026-09-18-scaling-2-parallelism %}), and [wrote those strategies in JAX]({% post_url 2026-09-19-scaling-3-why-jax-is-cool %}). Now let's put those mental models to work and implement a self-supervised learning algorithm that can actually be scaled.

The algorithm is SimDINO {% cite wu2025simplifying %}, the self-distillation method from the [Visual SSL post]({% post_url 2026-09-18-scaling-1-visual-ssl %}#simdino-deleting-the-training-stability-tricks). Quick recap: like DINO, it aligns student and teacher embeddings over two augmented views of the same image, and the teacher is an exponential moving average of the student. Unlike DINO, it drops the stability tricks and adds a coding-rate regularizer to discourage collapse.

For this example, we'll use two global crops per image. Let $Z_s^{(v)}, Z_t^{(v)} \in \mathbb{R}^{B \times d}$ be their normalized embeddings, with views $v \in \{1,2\}$. Match opposite views and average the coding rate over the student views:

$$\mathcal{L}_{\mathrm{align}} = \frac{1}{2B}\sum_{i=1}^{B}\left[1-\langle z_{s,i}^{(1)},z_{t,i}^{(2)}\rangle + 1-\langle z_{s,i}^{(2)},z_{t,i}^{(1)}\rangle\right],$$

$$R(Z) = \frac{1}{2}\log\det\!\left(I + \frac{d}{B\epsilon^2}Z^TZ\right), \qquad \mathcal{L}=\mathcal{L}_{\mathrm{align}}-\frac{\gamma}{2}\left[R(Z_s^{(1)})+R(Z_s^{(2)})\right].$$

Unit normalization matters here, since otherwise the model could crank up the rate just by scaling up its embeddings. Here $\epsilon$ is the distortion scale and $\gamma$ sets the regularizer's strength.

## The model, in house style

Parameters are a pytree, here a dictionary of arrays plus a list of per-block dictionaries. A `ViT` class holds only the architecture configuration, and its `apply` method takes the weights explicitly.

The array operations name the dimensions they act on:

```python
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
```

```python
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
```

Note that the embedding has dimension `config.dim`, instead of DINO's thousands of prototypes. For downstream evaluation, `backbone=True` returns the normalized CLS representation before the projection head. As promised in the last post, `einx` names the axes for every contraction, reduction, and rearrangement, and intermediate results get their own names so you can read each operation on its own.

## The loss, and where stop-gradient lives

The trainer creates two model instances with the same architecture. Their weights live in a separate `State` pytree, so the student and teacher can follow different update rules. `Trainer` itself only holds the configuration and the optimizer, never mutable training weights:

```python
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
```

The coding rate is computed from the second moment of the embeddings:

```python
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
```

The matrix inside the determinant is positive definite because of the identity term. If $A=LL^T$ is its Cholesky factorization, $\tfrac12\log\det A=\sum_j\log L_{jj}$. The code uses that identity and computes the loss statistics in fp32, including when the ViT uses bf16.

```python
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
```

`views` has shape `(2, batch, height, width, channels)`, and both networks see both crops. `cross_view_alignment` pairs each student output with the teacher's output on the other crop. The first term of the loss rewards agreement, and the second rewards embeddings that spread out.

Now for the first payoff: JAX differentiates with respect to exactly the arguments we name.

```python
loss_and_grad = jax.value_and_grad(trainer.loss, argnums=0, has_aux=True)
(_, metrics), gradients = loss_and_grad(
    state.student_weights, state.teacher_weights, views,
)
```

That computes gradients for the student tree only, and `stop_gradient` makes the teacher boundary explicit inside the loss too. The coding rate stays differentiable, though, because stopping gradients through its batch statistic would remove exactly the pressure it's supposed to put on the student.

## The EMA teacher and the training step

The second payoff is the teacher update. It touches every array in the parameter tree, but since both trees are plain values with identical structure, it's a single `tree.map`:

```python
def ema(
    teacher_weights: PyTree,
    student_weights: PyTree,
    momentum: Array | float,
) -> PyTree:
    return jax.tree.map(
        lambda teacher, student: momentum * teacher + (1 - momentum) * student,
        teacher_weights, student_weights,
    )
```

That's the entire momentum update, the same mechanism MoCo {% cite he2020momentum %} and BYOL {% cite grill2020bootstrap %} use. It returns a new teacher tree and leaves the old one alone.

The full training step composes the loss, optimizer, and EMA update:

```python
def teacher_momentum(
    step: Array,
    initial: float,
    steps: int,
) -> Array:
    progress = step / steps
    cosine_decay = (1 + jnp.cos(jnp.pi * progress)) / 2
    return 1 - (1 - initial) * cosine_decay


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
```

The `State` fields are `student_weights`, `teacher_weights`, `optimizer_state`, and `step`. Notice there's no running center like in DINO. The batch second moment is just an intermediate in the loss, recomputed every step. `self.optimizer` is an Optax transformation with gradient clipping, AdamW, and a warmup/cosine learning-rate schedule, and the teacher momentum ramps toward one over training.

The two updates sit right next to each other. The student follows the loss gradient through AdamW, and the teacher moves toward the updated student by EMA. `Trainer.step` returns a new state without mutating the old one, and that's the interface we'll keep fixed when we change placement.

## Sharding it: DP, then FSDP, by changing the placement

As in Parts 2 and 3, parallelizing this step is just a placement decision. For DP, replicate the state and split the batch. For FSDP, also shard the parameter and optimizer arrays that divide evenly:

```python
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
```

The view axis stays replicated and the batch axis is sharded: `P(None, "data", None, None, None)`. So both crops of an image stay on the same device. Small arrays, and matrices whose leading dimension doesn't divide by the device count, stay replicated, so we don't blindly shard every leaf.

Note that `Trainer.step` didn't change at all. Under `jit`, the arrays describe the global computation, so the contraction over `b` in `second_moment` uses the global batch even though each device holds only part of it. The compiler inserts the communication to make that true, along with the communication needed to combine parameter gradients.

This is SimDINO's version of the classic distributed-statistics trap (think synced BatchNorm). Computing one coding rate per device and averaging those scalars is a different objective from computing the rate of the global second moment:

$$\frac{1}{N}\sum_{p=1}^{N}\log\det(I+\alpha C_p) \neq \log\det\!\left(I+\alpha\frac{1}{N}\sum_{p=1}^{N}C_p\right).$$

Here we deliberately pick the global-batch objective. Local or subsampled rate estimators are valid choices too. The point is to pick one on purpose, and not change the objective by accident when you add devices.

With FSDP-style storage {% cite rajbhandari2020zero zhao2023pytorch %}, the compiler can gather sharded weights for use and reduce-scatter their gradients. The exact schedule and peak memory depend on what the compiler produces, and a sharding spec alone won't guarantee the ideal layer-by-layer schedule. When corresponding student and teacher leaves have identical placements, their EMA update is local to each shard.

For a model this small, FSDP is mostly for illustration. What I care about is that the loss and the update rule survive the switch from replicated to partitioned state untouched.

## Seeing the collectives with shard_map

To close the loop with Part 2's vocabulary, here is the global coding-rate calculation with the collective written out. Inside `shard_map`, `z` contains only the device's local batch. For equally sized shards:

```python
def global_rate(
    z,
    eps,
):
    local_moment = second_moment(z)
    moment = jax.lax.pmean(local_moment, "data")
    return coding_rate_from_moment(moment, eps)
```

That `pmean` averages the matrices **before** the nonlinear log-determinant, and it belongs inside the differentiated loss. The function below returns the global loss, and we differentiate the entire sharded function, so autodiff accounts for both the shared parameters and the statistic reduction:

```python
from functools import partial

@partial(jax.shard_map, mesh=mesh,
         in_specs=(P(), P(), P(None, "data")), out_specs=P())
def explicit_loss(
    student,
    teacher,
    views,
):
    student_z = trainer.student.encode_views(student, views)
    teacher_z = trainer.teacher.encode_views(teacher, views)
    teacher_z = jax.lax.stop_gradient(teacher_z)
    local_alignment = cross_view_alignment(student_z, teacher_z)
    alignment = jax.lax.pmean(local_alignment, "data")
    rate = global_rate(student_z, trainer.config.eps)
    return alignment - trainer.config.gamma * rate


explicit_loss_and_grads = jax.value_and_grad(explicit_loss, argnums=0)
explicit_loss_and_grads = jax.jit(explicit_loss_and_grads)
```

There are two distinct communication roles: combine feature statistics for the chosen objective, and combine parameter gradients for DP. The forward collectives are explicit here; differentiating the global function supplies the backward communication and gradient scaling. The returned gradients already correspond to the global loss, so we do not average them again. Replacing `global_rate` with a device-local rate would retain valid array shapes while quietly changing the loss, which is exactly the kind of bug explicit collectives help you see.

## Why this was the right toy

A classifier demo often has one parameter tree, one update rule, and a loss separable over examples. SimDINO gives us two trees with different updates, an explicit differentiation boundary, and a loss that couples examples through a batch statistic. JAX lets us express those pieces directly, then change their placement without rewriting the objective.

The companion folder contains CPU checks comparing unsharded, DP, and FSDP updates, plus commands for an ImageNet class-folder dataset. A short run checks the implementation and data path, but representation quality needs a longer run and a downstream evaluation. That's the split I want the small codebase to make easy: the training algorithm is compact enough to read in one sitting, and the experiment is still something you have to measure.

# References

{% bibliography --cited %}
