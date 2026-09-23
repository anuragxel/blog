---
layout: post
title: "Pretrain a vision model from scratch. Step 4: Write SimDINO in JAX"
description: Two parameter trees, two update rules, and one loss that couples the whole batch.
---

This is the last part of a four-part series on scaling up model (pre-)training. We covered the [major families of visual SSL]({% post_url 2026-09-18-scaling-1-visual-ssl %}), saw that [distributed training is mostly placement of arrays plus a handful of MPI collectives]({% post_url 2026-09-18-scaling-2-parallelism %}), and [wrote those placements in JAX]({% post_url 2026-09-19-scaling-3-why-jax-is-cool %}). Now let's put all of it to work on an actual self-supervised method, SimDINO {% cite wu2025simplifying %}.

I'm not going to walk through the ViT. It's a standard ViT, and you've seen a hundred of them (the full code is in the companion repo). What I want to show is the training loop, because that's where SimDINO gets interesting. Stripped down, a SimDINO step has:

1. **Two parameter trees**, a student and a teacher, with the same structure.
2. **Two update rules.** The student follows the gradient, and the teacher follows the student.
3. **One gradient boundary.** Gradients flow into the student and never into the teacher.
4. **One loss that couples the whole batch**, through the coding rate.

Each of these turns out to be a line or two of JAX. The fourth one is also where everything from parts 2 and 3 comes back to bite us, so it gets the most attention.

## SimDINO, again, but with intuition

Recall from part 1 that DINO {% cite caron2021emerging %} matches a student to an EMA teacher across crops, and that it avoids collapse through a fragile balancing act between centering and sharpening. SimDINO throws out that machinery (along with the prototypes and the softmax) and does two things instead.

The first is to pull the student's embedding of one crop toward the teacher's embedding of the *other* crop. With two global crops per image and unit-normalized embeddings $z$, this is just a cosine distance:

$$\mathcal{L}_{\mathrm{align}} = 1 - \frac{1}{2B}\sum_{i=1}^{B}\left[\langle z_{s,i}^{(1)},z_{t,i}^{(2)}\rangle + \langle z_{s,i}^{(2)},z_{t,i}^{(1)}\rangle\right]$$

On its own, this collapses immediately, since mapping every image to the same point gives perfect alignment. So the second thing is to add a direct penalty on collapse, the coding rate {% cite yu2020learning %} of the student's embeddings:

$$R(Z) = \frac{1}{2}\log\det\!\left(I + \frac{d}{\epsilon^2}\,C\right), \qquad C = \frac{1}{B} Z^{T} Z \in \mathbb{R}^{d \times d}, \qquad \mathcal{L} = \mathcal{L}_{\mathrm{align}} - \gamma \, \overline{R}$$

where $Z \in \mathbb{R}^{B \times d}$ stacks one view's embeddings, $\overline{R}$ averages the rate over the two student views, and $\gamma$ sets the regularizer's strength.

The way I think about $R$ is through the eigenvalues $\mu_1, \ldots, \mu_d$ of the second moment $C$:

$$R = \frac{1}{2}\sum_{j=1}^{d} \log\!\left(1 + \frac{d}{\epsilon^2}\mu_j\right), \qquad \sum_j \mu_j = \mathrm{tr}(C) = \frac{1}{B}\sum_i \lVert z_i \rVert^2 = 1.$$

Because the embeddings are unit-normalized, the total "energy" $\sum_j \mu_j$ is fixed at one. The only freedom left is how that energy is split across directions. $\log$ is concave, so the rate is maximized by spreading the energy evenly ($\mu_j = 1/d$) and minimized by piling it into one direction, which is exactly what collapse looks like. This is also why normalization isn't optional. Without it, the model could raise the rate just by making its embeddings longer.

So alignment pulls the views of each image together, and the rate pushes the whole cloud of images apart. Note that $C$ depends on *every* image in the batch. That one fact is going to matter a lot once we shard.

## The model is just a function

In JAX, the ViT is a pytree of weights plus a pure function. The only interface the training loop needs is this:

```python
def encode_views(
    params: PyTree, views: Float[Array, "2 b h w c"]
) -> Float[Array, "2 b d"]:
    """Unit-normalized embeddings for both crops of every image."""
```

Since the teacher and the student have the same architecture, the same function encapsulates both networks. We just input different `param` trees for student and teacher networks.

## Two trees, one state

The whole training state is a named tuple:

```python
class State(NamedTuple):
    student: PyTree
    teacher: PyTree
    opt_state: optax.OptState
    step: Int[Array, ""]


def init_state(key, model, optimizer) -> State:
    params = model.init(key)
    # Same values, two trees. JAX arrays are immutable, so nothing is tied.
    return State(params, params, optimizer.init(params), jnp.array(0))
```

## The loss, and the gradient boundary

The coding rate goes through the Cholesky factor. The matrix $I + \frac{d}{\epsilon^2} C$ is positive definite thanks to the identity, and if $A = LL^{T}$, then $\frac{1}{2}\log\det A = \sum_j \log L_{jj}$:

```python
def second_moment(z: Float[Array, "v b d"]) -> Float[Array, "v d d"]:
    return einx.dot("v [b] d, v [b] e -> v d e", z, z) / z.shape[1]


def rate_from_moment(moment: Float[Array, "v d d"], eps: float) -> Float[Array, ""]:
    d = moment.shape[-1]
    chol = jnp.linalg.cholesky(jnp.eye(d) + (d / eps**2) * moment)
    log_diag = jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1))
    return einx.sum("v [d]", log_diag).mean()  # mean over the two views


def cross_view_alignment(z_s: Float[Array, "2 b d"], z_t: Float[Array, "2 b d"]):
    # z_t[::-1] swaps the views, so crop 1 of the student meets crop 2 of the teacher.
    return 1 - einx.dot("v b [d], v b [d] -> v b", z_s, z_t[::-1]).mean()
```

I split `second_moment` out from `rate_from_moment` on purpose.

The loss itself reads almost like an equation:

```python
def loss_fn(student, teacher, views, model, config):
    z_s = model.encode_views(student, views)
    z_t = jax.lax.stop_gradient(model.encode_views(teacher, views))

    alignment = cross_view_alignment(z_s, z_t)
    rate = rate_from_moment(second_moment(z_s), config.eps)
    metrics = {
        "alignment": alignment,
        "rate": rate,
        # This goes to zero if every image maps to one point.
        "feature_std": einx.std("v [b] d -> v d", z_s).mean(),
    }
    return alignment - config.gamma * rate, metrics
```

In self-distillation, the teacher is a target. The loss is a function of three things, $\mathcal{L}(\theta_s, \theta_t, x)$, and we only ever take $\nabla_{\theta_s}\mathcal{L}$. While we take that derivative, the teacher's weights are held fixed. Instead, the teacher is updated via the EMA in the next section.

## The training step

Here is the whole step:

```python
def teacher_momentum(step, config):
    # Cosine ramp from config.momentum toward 1 over training.
    progress = step / config.steps
    return 1 - (1 - config.momentum) * (1 + jnp.cos(jnp.pi * progress)) / 2


def train_step(state: State, views, model, optimizer, config):
    grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)
    (loss, metrics), grads = grad_fn(state.student, state.teacher, views, model, config)

    # The student follows the gradient.
    updates, opt_state = optimizer.update(grads, state.opt_state, state.student)
    student = optax.apply_updates(state.student, updates)

    # The teacher follows the student.
    m = teacher_momentum(state.step, config)
    teacher = jax.tree.map(lambda t, s: m * t + (1 - m) * s, state.teacher, student)

    return State(student, teacher, opt_state, state.step + 1), metrics
```

The teacher update is the equation $\theta_t \leftarrow m\,\theta_t + (1-m)\,\theta_s$, applied to every weight. So the teacher is a running average of recent students, which makes it a slower and more stable target than the student itself (the same idea as in MoCo {% cite he2020momentum %} and BYOL {% cite grill2020bootstrap %}). As $m$ ramps toward one, the teacher changes less and less. Since the student and teacher are pytrees with the same structure, the equation is a simple `jax.tree.map`. Notice that `train_step` is also a pure function and it lets us perform sharding in the next section.

## Sharding it is a placement decision

Following part 3, we don't touch `train_step` to distribute it. We only say where the arrays live:

```python
mesh = jax.make_mesh((jax.device_count(),), ("data",))
replicated = NamedSharding(mesh, P())
# views is (view, batch, h, w, c): split the images and keep both crops of each together.
batch_sharding = NamedSharding(mesh, P(None, "data"))

def placement(x):
    if mode == "fsdp" and x.ndim == 2 and x.shape[0] % mesh.size == 0:
        return NamedSharding(mesh, P("data", None))
    return replicated

state_sharding = jax.tree.map(placement, state)
step = jax.jit(
    partial(train_step, model=model, optimizer=optimizer, config=config),
    in_shardings=(state_sharding, batch_sharding),
    out_shardings=(state_sharding, replicated),
)
```

With `mode = "dp"`, everything is replicated and only the batch is split. With `mode = "fsdp"`, the weight matrices (and their optimizer state) are split too, and the compiler gathers and reduce-scatters them as it goes {% cite rajbhandari2020zero zhao2023pytorch %}. Because the student and teacher leaves get identical placements, the EMA `tree.map` stays local to each shard and needs no communication at all.

## The batch-coupled loss, where sharding meets math

Data parallelism, as we set it up in part 2, relies on one property of the loss. If the loss is a mean over examples and the shards are equally sized, then the average of the per-device losses is the global loss, and averaging the per-device gradients gives the global gradient. Each device can compute its loss on its own slice without ever looking at anyone else's data.

The alignment term has this property, but the coding rate doesn't. The rate is a function of the second moment $C$, which is a mean over the whole batch, and the log-determinant is applied *after* that mean. With $N$ devices, the global moment is the average of the local ones, $C = \frac{1}{N}\sum_p C_p$, so there are two different things we could compute:

$$\underbrace{\frac{1}{N}\sum_{p=1}^{N}\log\det\!\left(I+\alpha C_p\right)}_{\text{average of local rates}} \;\le\; \underbrace{\log\det\!\left(I+\alpha\,\frac{1}{N}\sum_{p=1}^{N}C_p\right)}_{\text{rate of the global batch}}, \qquad \alpha = \frac{d}{\epsilon^2}$$

The inequality is Jensen's, since $\log\det$ is concave. The left side is what you get if every device computes the full loss on its own slice, the habit from plain data parallelism. That makes it a different objective, and one that depends on the number of devices. Take $d = 256$ and a global batch of 1024. On 64 devices, each device sees 16 images, so each $C_p$ has rank at most 16. Each local rate can then reward spreading across at most 16 of the 256 directions, however spread out the full batch is. Change the device count and you've changed the loss, without touching a line of it.

In JAX, we get the right side by default. Under `jit`, arrays describe the global computation, so when `second_moment` contracts over `b`, it contracts over the global batch, even though each device only holds a slice. The compiler inserts the communication to make that true, typically an all-reduce of the local moments before the log-determinant. That's a `(2, d, d)` matrix, which is tiny next to the gradients we're all-reducing anyway.

To see the collective explicitly, we can write the per-device code with `shard_map`, where `z_local` holds only the local batch. This is why we split `second_moment` from `rate_from_moment` earlier:

```python
def global_rate(z_local, eps):
    moment = jax.lax.pmean(second_moment(z_local), "data")
    return rate_from_moment(moment, eps)


def local_rate(z_local, eps):
    return jax.lax.pmean(rate_from_moment(second_moment(z_local), eps), "data")
```

Both have the same shapes, and the only difference is where the `pmean` goes. In `global_rate`, it averages the moments before the log-determinant, and in `local_rate`, it averages the rates after. When we differentiate through either one, autodiff inserts the matching collective in the backward pass.

This is an old problem. SimCLR {% cite chen2020simple %} aggregated batch-norm statistics across all devices for the same reason. Any loss term that isn't a mean over examples needs this care. A local rate can still be a reasonable choice, as long as we pick it deliberately.

## Does it actually train?

<!-- TODO(anurag): numbers. Candidates:
  - k-NN / linear-probe accuracy of the backbone after N epochs.
  - rate and feature_std over training, ideally next to a gamma = 0 run where feature_std goes to zero (collapse).
  - global_rate vs local_rate at a small per-device batch, if the gap shows up.
-->

## Two trees, two rules, one statistic

That's the whole mental model for self-distillation at scale. There are two parameter trees with different update rules, a gradient boundary that's a single argument, and a batch statistic that decides whether sharding changes your objective. Each of them is a line or two of JAX, and none of them needed the ViT to explain.

This also wraps up the series. We went from what visual SSL methods optimize, to where arrays live across devices, to how JAX lets us write placement separately from the math. My hope is that the next time you scale up a pretraining run, you skip the flags for a moment and ask which of your statistics just stopped meaning what you think they mean.

# References

{% bibliography --cited %}
