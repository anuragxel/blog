---
layout: post
title: "Pretrain a vision model from scratch. Step 3: Use JAX. It is very cool."
description: Meshes, shardings and named axes.
---

In the last couple of posts, we discussed the fundamentals of different [Visual SSL families](https://anuragxel.github.io/blog/scaling-1-visual-ssl/) and then we discussed the fundamentals underpinning [distributed model scaling strategies](https://anuragxel.github.io/blog/scaling-2-parallelism/). As we saw earlier, distributed model scaling strategies are nothing but placements of arrays on devices plus a handful of communication collectives, ranked by memory, compute and communication overheads.

This post is about why I think JAX makes the most appropriate substrate for that view: not because it's faster, but because its abstractions make scaling as simple as considering the mesh of devices and the sharding placement, and letting XLA cook. It is thus beautiful to read and write distributed machine learning code in.

## The premise: programs as pure functions

JAX {% cite jax2018github %} lets us write numerical code as *pure functions* on arrays (and on *pytrees*, i.e., arbitrarily nested containers of arrays). In return, we get program transformations as higher-order functions:

- `jax.grad(f)` — a new function computing $\nabla f$,
- `jax.jit(f)` — $f$ traced and compiled, with compiled programs reused for matching input signatures,
- `jax.vmap(f)` — $f$ mapped over a new batch axis, without writing the batch axis.

Functional purity is what makes the above possible. `f` has no hidden state: parameters go in as arguments and anything that changes comes back as a return value. Think of `f` as a mathematical function. Because there are no side effects, tracing `f` once with abstract inputs captures the entire computation, and XLA can optimize the whole program rather than one op at a time.

While JAX has a steeper learning curve than PyTorch, the mental model of pure functions plus transformations helps untangle the mystery of distributed ML and makes scaling more straightforward. We shall now look at sharding neural networks using this framework.

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

GPU and TPU systems have communication locality, from devices within a node to nearby nodes or a TPU slice, and then across larger groups or slices. A useful starting point is TP within the fastest local group, DP among nearby groups, and PP across the slower boundaries. This keeps frequent communication close to the devices doing the work.

A **mesh** is a named grid of devices. For our DP, FSDP, and TP configurations, its axes are `replica`, `fsdp`, and `tensor`. Their sizes specify how many devices participate in each form of parallelism.

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

A `PartitionSpec` specifies which mesh axes split each array dimension. For a matrix, its entries describe the rows and columns. `None` leaves a dimension unsplit. The array is replicated along unused mesh axes.

Finally, `NamedSharding` combines a `PartitionSpec` with the device `Mesh` to determine where the array is stored.

Let's consider the selected `(32, 8, 1)` configuration. Each host keeps a copy of the model, divided among its eight devices. Each device also gets a different part of the batch. We express this by splitting the batch over `replica` and `fsdp`, and the weights over `fsdp`.

The same rules work for TP. When `tensor` is larger than one, we divide the hidden neurons among devices. Each device needs the corresponding columns of `w_up` and rows of `w_down`. Here `tensor` has size one, so those dimensions stay whole.

```python
def placement(spec):
    return NamedSharding(mesh, spec)

batch_sharding = placement(P(("replica", "fsdp"), None))
param_shardings = {
    "w_up": placement(P("fsdp", "tensor")),
    "w_down": placement(P("tensor", "fsdp")),
}
```

Now we compile the original loss and its gradient. We tell `jit` how the inputs are distributed and ask it to return gradients distributed like the weights. JAX works out the communication needed to perform the calculation.

```python
loss_and_grad = jax.jit(
    jax.value_and_grad(loss_fn),
    in_shardings=(param_shardings, batch_sharding, batch_sharding),
    out_shardings=(placement(P()), param_shardings),
)
```

Changing the mesh configuration thus changes the distribution of the same arrays, and the XLA compiler decides the execution schedule.

### A four-device walkthrough

We can see the same placement rules in the figures below using two hosts with two devices each. The code above uses 256 devices. Here, DP becomes `(4, 1, 1)`, DP + FSDP becomes `(2, 2, 1)`, and DP + TP becomes `(2, 1, 2)`. The partition specs stay the same.

To draw the arrays, we use `batch = 8`, `width = 4`, and `hidden = 6`. Each grid cell represents one scalar.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-mlp.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-mlp.png' | relative_url }}" width="640" height="205" loading="lazy" alt="The global MLP: x, 8 by 4, multiplies w_up, 4 by 6. GeLU gives h, 8 by 6, which multiplies w_down, 6 by 4, to produce y, 8 by 4. Each grid cell represents one scalar.">
  </a>
  <figcaption>We will use this MLP for the sharding example. The full array shapes stay the same across configurations.</figcaption>
</figure>

The input and output are both `(8, 4)`, with a hidden activation of shape `(8, 6)`. U and D denote the `(4, 6)` `w_up` and `(6, 4)` `w_down` matrices. Matching shard labels mean identical array contents. Each chip shows the arrays stored on one device.

With DP, `(4, 1, 1)`, each device stores a `(2, 4)` input, the full `(4, 6)` U, and the full `(6, 4)` D.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-dp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-dp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP: mesh (4, 1, 1). The 8 by 4 batch splits into four 2 by 4 pieces, one per device. Each device holds the complete 4 by 6 w_up and 6 by 4 w_down.">
  </a>
  <figcaption>With <code>(4, 1, 1)</code>, each device gets a quarter of the batch and a full copy of the weights.</figcaption>
</figure>

With DP + FSDP, `(2, 2, 1)`, the input is still split into four `(2, 4)` pieces. Each device stores a `(2, 6)` shard of U and a `(6, 2)` shard of D. Both hosts hold the same pair of weight shards.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-fsdp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-fsdp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP plus FSDP: mesh (2, 2, 1). Four different 2 by 4 batch pieces remain. w_up splits by rows into U0 and U1, each 2 by 6. w_down splits by columns into D0 and D1, each 6 by 2. Each host stores both weight pieces across its two devices; matching weight pieces repeat across hosts.">
  </a>
  <figcaption>With <code>(2, 2, 1)</code>, each device gets a different quarter of the batch, and the weights are split between the two devices on each host.</figcaption>
</figure>

With DP + TP, `(2, 1, 2)`, both devices on each host receive the same `(4, 4)` input. Each handles three hidden neurons, using a `(4, 3)` shard of U and the matching `(3, 4)` shard of D.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling3-tp-placement.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling3-tp-placement.png' | relative_url }}" width="640" height="660" loading="lazy" alt="DP plus TP: mesh (2, 1, 2). The batch splits into two 4 by 4 pieces, with one piece repeated on both devices of each host. w_up splits by columns into two 4 by 3 pieces. w_down splits by rows into two 3 by 4 pieces. Matching weight pieces repeat across hosts.">
  </a>
  <figcaption>With <code>(2, 1, 2)</code>, both devices on a host receive the same input. Each computes the activations for three of the six hidden neurons.</figcaption>
</figure>

The target follows `x`'s placement in every configuration. These pictures show input storage. Intermediate placements and the communication needed to compute the MLP are left to the compiler.

## shard_map, to manually write the collectives

For direct control, `jax.shard_map` lets us write the per-device program and call communication collectives ourselves. Returning to the 256-device code example, we select the `(256, 1, 1)` DP configuration. Each device holds the full parameter tree and one of 256 equal shards of `x` and `target`:

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

The rest of this post is about legibility and a style guide for writing clean code that is easy to read. These stylistic choices derive from the same philosophical point: **name the axes**.

### jaxtyping: shapes, names and types in signatures

Shape mistakes can cause inscrutable reshape or broadcast errors—or, worse, silent broadcasts that produce incorrect results. Often, the programmer then traces the entire forward pass manually to identify the issue. As the forward pass often spans multiple files, debugging is difficult. `jaxtyping` puts the shape contract in the function signature, helping catch input and output shape mismatches at function boundaries when runtime checking is enabled:

```python
from jaxtyping import Array, Float, Int

def attention(
    q: Float[Array, "n d"],
    k: Float[Array, "m d"],
    v: Float[Array, "m dv"],
) -> Float[Array, "n dv"]:
    ...
```

With runtime checking enabled (via jaxtyping and beartype), symbolic dimensions (`"n d"`) unify across arguments within a call. If `q` and `k` disagree on `d`, that's an error *at this function's boundary*. The annotations become enforced documentation. I think of jaxtyping annotations as defining the *interface contract*: the signature states the tensor type and shape semantics.

### einops and einx: shapes, names and types in operations

The same idea about *contracts* can be extended to function *bodies*. Older primitives like `reshape`/`transpose`/`unsqueeze` are limited: positional axis indices do not communicate the axes’ semantic roles, and a new reader must re-derive the tensor layout (or annotate the layout in the variable name as `"tensor_BHWC"`). It is thus better to **enforce named-axes contracts while manipulating tensors**.

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

### Concluding Remarks

We saw how JAX lets us express distributed training through array placement on a device mesh, with the compiler working out the communication. Naming the axes also makes the model easier to read, from shape annotations to tensor operations. I like this way of writing code because it makes both the computation and where it runs easier to follow.

In the last post, we'll examine SimDINO, the Visual SSL method based on self-distillation [introduced earlier]({% post_url 2026-09-18-scaling-1-visual-ssl %}#simdino-deleting-the-training-stability-tricks), and observe the elegance of writing the EMA teacher, the stop-gradient, and the sharded training step in JAX.

# References

{% bibliography --cited %}
