---
layout: post
title: "Pretrain a vision model from scratch. Step 3: Use JAX. It is very cool."
description: Meshes, shardings and named axes.
---

In the last couple of posts, we discussed the fundamentals of different [Visual SSL families]({% post_url 2026-09-18-scaling-1-visual-ssl %}) and then we discussed the fundamentals underpinning [distributed model scaling strategies]({% post_url 2026-09-18-scaling-2-parallelism %}). As we saw earlier, distributed model scaling strategies are nothing but placements of arrays on devices plus a handful of communication collectives, ranked by memory, compute and communication overheads.

This post is about why I think JAX fits that view best. It's not about speed. Its abstractions make scaling as simple as picking a mesh of devices and a sharding placement, and letting XLA cook. That makes distributed ML code in JAX a joy to read and write.

## The premise: programs as pure functions

JAX {% cite jax2018github %} lets us write numerical code as *pure functions* on arrays (and on *pytrees*, i.e., arbitrarily nested containers of arrays). In return, we get program transformations as higher-order functions:

- `jax.grad(f)`: a new function computing $\nabla f$,
- `jax.jit(f)`: $f$ traced and compiled, with compiled programs reused for matching input signatures,
- `jax.vmap(f)`: $f$ mapped over a new batch axis, without writing the batch axis.

Functional purity is what makes the above possible. `f` has no hidden state: parameters go in as arguments and anything that changes comes back as a return value. Think of `f` as a mathematical function. Because there are no side effects, tracing `f` once with abstract inputs captures the entire computation, and XLA can optimize the whole program rather than one op at a time.

While JAX has a steeper learning curve than PyTorch, the mental model of pure functions plus transformations helps untangle the mystery of distributed ML and makes scaling more straightforward. Let's now look at sharding neural networks with this mental model.

## Sharding as placement

Consider a two-layer MLP we'd like to train. `x` is a batch of input vectors with shape `(batch, width)` and `target` is the target array of the same shape. `params` is a pytree, here a dictionary with two weight matrices: `w_up` has shape `(width, hidden)` and `w_down` has shape `(hidden, width)`. The output has the same shape as `x`.

```python
import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


def mlp(params, x):
    h = jax.nn.gelu(x @ params["w_up"])
    return h @ params["w_down"]


def loss_fn(params, x, target):
    return ((mlp(params, x) - target) ** 2).mean()
```

We will change how these arrays are distributed while keeping the computation fixed.

### A 256-device example

Real GPU and TPU clusters are hierarchical: devices within a node (or a TPU slice) talk fast, nearby nodes a bit slower, and everything farther away slower still. As we saw last time, a decent default is TP inside the fastest group, DP across nearby groups, and PP across the slowest boundaries, so the chattiest communication stays local.

A **mesh** is a grid of devices with named axes. Ours has three, `replica`, `fsdp`, and `tensor`, and the size of each axis says how many devices take part in that form of parallelism.

For the code example, consider 32 hosts with 8 devices each. Assuming the device list is grouped by host, we can reshape it into the configuration we want:

```python
devices_by_host = np.array(jax.devices()).reshape(32, 8)
configurations = {
    "dp":      (256, 1, 1),
    "fsdp":    (1, 256, 1),
    "dp_tp":   (32, 1, 8),
    "fsdp_tp": (1, 32, 8),
    "dp_fsdp": (32, 8, 1),
}
mesh = Mesh(
    devices_by_host.reshape(configurations["dp_fsdp"]),
    ("replica", "fsdp", "tensor"),
    axis_types=(AxisType.Auto,) * 3,
)
```

Keeping each host's eight devices together lets us place TP or FSDP communication within that host:

<div class="collectives-table" markdown="1" role="region" aria-label="Parallelism configurations and device placement" tabindex="0">

| Configuration | Mesh shape: replica, FSDP, TP | Placement |
|---|---|---|
| DP | `(256, 1, 1)` | Replicate the model across all devices. |
| FSDP-style | `(1, 256, 1)` | Shard weights across all devices. |
| DP + TP | `(32, 1, 8)` | TP within hosts. DP across hosts. |
| FSDP-style + TP | `(1, 32, 8)` | TP within hosts. FSDP across hosts. |
| DP + FSDP-style | `(32, 8, 1)` | FSDP within hosts. DP across hosts. |

</div>

We can similarly group devices by neighbourhood or TPU slice to keep frequent communication on faster connections.

#### Defining the array placements

A `PartitionSpec` says, for each array dimension, which mesh axes split it (for a matrix, one entry for the rows and one for the columns). `None` leaves that dimension whole, and the array is replicated along any mesh axis the spec doesn't mention.

`NamedSharding` then pairs a `PartitionSpec` with the device `Mesh` to pin down where each piece of the array lives.

Take the `(32, 8, 1)` configuration we picked. Each host keeps a copy of the model split across its eight devices, and every device gets its own slice of the batch. So we split the batch over both `replica` and `fsdp`, and the weights over `fsdp` only.

TP follows the same rules. When `tensor` is larger than one, the hidden neurons get divided among devices, and each device holds the matching columns of `w_up` and rows of `w_down`. Here `tensor` has size one, so those dimensions stay whole.

```python
def placement(spec):
    return NamedSharding(mesh, spec)

batch_sharding = placement(P(("replica", "fsdp"), None))
param_shardings = {
    "w_up": placement(P("fsdp", "tensor")),
    "w_down": placement(P("tensor", "fsdp")),
}
```

Now we compile the loss and its gradient. We tell `jit` how the inputs are laid out and ask for gradients laid out like the weights, and JAX works out all the communication in between.

```python
loss_and_grad = jax.jit(
    jax.value_and_grad(loss_fn),
    in_shardings=(param_shardings, batch_sharding, batch_sharding),
    out_shardings=(placement(P()), param_shardings),
)
```

### Visualizing the placements

For the figures, we use two hosts with two devices each and an MLP with `batch = 8`, `width = 4`, and `hidden = 6`. The partition specs are unchanged.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-mlp.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-mlp.png' | relative_url }}" width="640" height="205" loading="lazy" alt="The global MLP: x, 8 by 4, multiplies w_up, 4 by 6. GeLU gives h, 8 by 6, which multiplies w_down, 6 by 4, to produce y, 8 by 4. Each grid cell represents one scalar.">
  </a>
  <figcaption>The MLP before sharding. Each square is one scalar.</figcaption>
</figure>

Below, U and D are `w_up` and `w_down`.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-dp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-dp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP: mesh (4, 1, 1). The 8 by 4 batch splits into four 2 by 4 pieces, one per device. Each device holds the complete 4 by 6 w_up and 6 by 4 w_down.">
  </a>
  <figcaption>DP, <code>(4, 1, 1)</code>: each device gets two input vectors and all the weights.</figcaption>
</figure>

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-fsdp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-fsdp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP plus FSDP: mesh (2, 2, 1). Four different 2 by 4 batch pieces remain. w_up splits by rows into U0 and U1, each 2 by 6. w_down splits by columns into D0 and D1, each 6 by 2. Each host stores both weight pieces across its two devices; matching weight pieces repeat across hosts.">
  </a>
  <figcaption>DP + FSDP, <code>(2, 2, 1)</code>: the batch is still split four ways. The weights are now divided between the two devices on each host.</figcaption>
</figure>

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-tp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-tp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP plus TP: mesh (2, 1, 2). The batch splits into two 4 by 4 pieces, with one piece repeated on both devices of each host. w_up splits by columns into two 4 by 3 pieces. w_down splits by rows into two 3 by 4 pieces. Matching weight pieces repeat across hosts.">
  </a>
  <figcaption>DP + TP, <code>(2, 1, 2)</code>: both devices on a host get the same four input vectors, but each handles three of the six hidden neurons.</figcaption>
</figure>

## shard_map, to manually write the collectives

With `jax.shard_map`, we write the per-device computation and its collectives ourselves. Here is the DP version for our 256-device mesh:

```python
from functools import partial

# Select the DP arrangement for the per-device program.
dp_mesh = Mesh(
    devices_by_host.reshape(configurations["dp"]),
    ("replica", "fsdp", "tensor"),
    axis_types=(AxisType.Auto,) * 3,
)

@partial(
    jax.shard_map,
    mesh=dp_mesh,
    check_vma=False,  # Handle gradient replication with the explicit pmean.
    in_specs=(P(), P("replica", None), P("replica", None)),
    out_specs=P(),
)
def dp_grads(params, x, target):
    grads = jax.grad(loss_fn)(params, x, target)
    return jax.lax.pmean(grads, axis_name="replica")
```

`loss_fn` sees the local batch. Because the batch shards are equally sized, averaging their gradients gives the gradient of the global mean loss. JAX provides the following low-level primitives for writing the communication collectives:

<div class="collectives-table" markdown="1" role="region" aria-label="MPI collective operations and JAX primitives" tabindex="0">

| MPI Primitive | JAX |
|---|---|
| all-reduce | `psum` / `pmean` |
| all-gather | `all_gather` |
| reduce-scatter | `psum_scatter` |
| all-to-all | `all_to_all` |

</div>

## Style choices

The rest of this post is about legibility, i.e., my style guide for writing JAX code that's easy to read. It all comes from one idea: **name the axes**.

### jaxtyping: shapes, names and types in signatures

Shape mistakes give you inscrutable reshape or broadcast errors or, worse, silent broadcasts that quietly produce wrong results. Then you end up tracing the whole forward pass by hand, usually across several files, to find the bug. `jaxtyping` puts the shape contract in the function signature, helping catch input and output shape mismatches at function boundaries when runtime checking is enabled:

```python
from jaxtyping import Array, Float, Int

def attention(
    q: Float[Array, "n d"],
    k: Float[Array, "m d"],
    v: Float[Array, "m dv"],
) -> Float[Array, "n dv"]:
    ...
```

With runtime checking enabled (via jaxtyping and beartype), symbolic dimensions (`"n d"`) unify across arguments within a call. If `q` and `k` disagree on `d`, that's an error *at this function's boundary*. The annotations become documentation that actually gets enforced. I think of them as the function's *interface contract*.

### einops and einx: shapes, names and types in operations

The same idea about *contracts* can be extended to function *bodies*. Older primitives like `reshape`/`transpose`/`unsqueeze` refer to axes by position, which says nothing about what each axis means, so a new reader has to re-derive the tensor layout (or you end up naming variables `"tensor_BHWC"`). It is thus better to **enforce named-axes contracts while manipulating tensors** too.

einops {% cite rogozhnikov2022einops %} instead allows tensor manipulation with spelled-out axes. For example, the ViT patchify operation can be written as follows:

```python
from einops import rearrange

patches = rearrange(imgs, "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
                    p1=16, p2=16)
```

einx {% cite fervers2024einx %} generalizes the notation to essentially every array op, providing helpers for dot products, reductions, and indexing, including a bracket syntax marking the axes. As an example, attention scores and a mean over tokens can be written as follows:

```python
import einx

scores = einx.dot("b q [d], b k [d] -> b q k", queries, keys)
pooled = einx.mean("b [s] d", tokens)
```

The bracket in `[d]` says "this axis is contracted." The bracket in `[s]` says "this axis is reduced."

We can also combine jaxtyping with einops or einx:

```python
import einx
from jaxtyping import Array, Float, Int

image_1d_tokens: Float[Array, "b n d"] = einx.id("b h w d -> b (h w) d", image_tokens)
```

Another useful style suggestion: **use consistent names for array dimensions, and make their mapping to device axes explicit**. A batch dimension called `b` in a jaxtyping signature should also be `b` in einops or einx. In our MLP, that dimension is distributed over the mesh axes `replica` and `fsdp`. Array dimensions describe the data while mesh axes describe how devices share it. Keeping that mapping visible makes both the model and its distributed execution easier to read.

## Name the axes, let XLA cook

In JAX, distributed training mostly comes down to saying where arrays live on a device mesh and letting the compiler handle the communication. Name the axes everywhere (in signatures, in einops/einx calls, and in the mesh), and you can read both what the code computes and where it runs straight off the page. That's why I think JAX is cool.

In the last post, we'll examine SimDINO, the Visual SSL method based on self-distillation [introduced earlier]({% post_url 2026-09-18-scaling-1-visual-ssl %}#simdino-deleting-the-training-stability-tricks), and observe the elegance of writing the EMA teacher, the stop-gradient, and the sharded training step in JAX.

# References

{% bibliography --cited %}
