---
layout: post
title: "Pretrain a vision model from scratch. Step 2: Just Learn MPI"
description: DP, FSDP, TP, and pipeline parallelism as compositions of familiar communication primitives.
---

This is part 2 of a four-part series on scaling up model (pre-)training. We looked at the major families of Visual SSL algorithms, and while they are all fun to implement as toys, the real challenge is scaling up both the model (in terms of its parameters) and the amount of data the model is pre-trained on. Thus, we need to pre-train on a huge number of devices at once, which means borrowing techniques from parallel and distributed systems.

Some of these large-scale pretraining techniques have interesting names: data parallelism, fully sharded data parallelism, tensor parallelism, and pipeline parallelism. These strategies are usually presented as framework features: flags we flip or configs we edit. I think this framing obscures how simple they are. These abstractions were built to address old distributed-systems questions applied to deep learning: *where do the tensors and activations live, and how do you get through a forward and backward pass with the least communication overhead and the highest utilization?*

Almost all the communication in these four strategies boils down to five collective operations that [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) standardized back in 1994 {% cite mpiforum1994 %} (pipeline parallelism also needs point-to-point sends, which we'll get to). Once you know those five, the strategies are mostly arithmetic.

## Why shard at all: the memory footprint of training state

Let's start with why we shard in the first place. Consider a model trained with Adam. Each parameter costs approximately {% cite rajbhandari2020zero %}:

- 2 bytes for the bf16 working copy of the parameter
- 2 bytes for the bf16 gradient
- 4 bytes for the fp32 master copy of the parameter
- 4 + 4 bytes for the fp32 Adam first and second moments

That is **16 bytes per parameter** of *state*, before a single activation is computed. A 7B-parameter model therefore requires about 112 GB of persistent state. Activations are separate and scale roughly with $\text{batch} \times \text{sequence} \times \text{hidden} \times \text{depth}$, subject to checkpointing and implementation details. Smaller per-device batches and gradient checkpointing reduce activation memory. You can also offload state or keep the optimizer in lower precision, but the usual answer is to shard it across devices. Every strategy below is a choice of what to shard (data, state, or activations), plus a bill for the communication that choice incurs.

## Back to the 80s and 90s: the language of MPI

Before MPI, systems such as PVM, Intel's NX, Express, and PARMACS offered different interfaces for many of the same communication patterns. The MPI Forum standardized a common interface in 1994 {% cite mpiforum1994 %}. The problems were familiar: distribute some data, assemble pieces held on different machines, or combine results computed independently. These are also the operations we need to train a model across devices.

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling1-placements.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling1-placements.png' | relative_url }}" width="640" height="490" loading="lazy" alt="The same picture on four devices: replication puts the entire picture on each device, while sharding distributes its four quarters across the devices.">
  </a>
  <figcaption>With replication, every device stores the same array. With sharding, each device stores a different part of it.</figcaption>
</figure>

A **collective** is a communication operation that every process in a group takes part in. For this post, think of one process per device, and call a process's index within the group its **rank**. For each operation, we'll just track what each device holds before and after.

- **Broadcast: copy from one device to everyone.** One process, the **root**, copies its array to every other device. For example, one device can initialize the model's parameters and broadcast them so everyone starts with the same weights.
- **All-gather: assemble everyone's pieces on every device.** The devices exchange their arrays and concatenate them in rank order. Starting from distributed parameter shards, all devices end up holding the full layer.
- **All-reduce: combine everyone's contributions and give everyone the result.** Every device holds an array of the same shape, and a *reduction* (sum, max, etc.) combines corresponding entries. For gradients, every device ends up with the summed contribution to every parameter.
- **Reduce-scatter: combine contributions, then divide the result.** Like all-reduce, the full arrays are summed element by element, but then each device keeps only its own slice of the result (equally sized, for this post).
- **All-to-all: exchange different pieces with different devices.** Each device splits its array into one block per destination and sends each block where it belongs. This can rearrange activations from being split by tokens to being split by attention heads.

Here's a concrete example on two devices. Every operation starts from the arrays in the first row.

<div class="collectives-table" markdown="1" role="region" aria-label="Collective operations on two devices" tabindex="0">

| Operation / state | Device 0 | Device 1 |
| :--- | :---: | :---: |
| **Starting arrays** | `[1, 2]` | `[3, 4]` |
| Broadcast (root: device 0) | `[1, 2]` | `[1, 2]` |
| All-gather | `[1, 2, 3, 4]` | `[1, 2, 3, 4]` |
| All-reduce (sum) | `[4, 6]` | `[4, 6]` |
| Reduce-scatter (sum) | `[4]` | `[6]` |
| All-to-all (one element per block) | `[1, 3]` | `[2, 4]` |

</div>

The all-reduce and reduce-scatter rows reveal a useful identity:

$$\texttt{all-reduce} \;=\; \texttt{reduce-scatter} \;+\; \texttt{all-gather}$$

Reduce-scatter leaves `[4]` on device 0 and `[6]` on device 1. All-gathering those results gives `[4, 6]` to both devices, which is also the result of all-reduce. This decomposition will be useful when we get to FSDP.

### Understanding the communication cost model through ring all-reduce

The implementation of the collectives determines the communication overhead. Consider a ring implementation of all-reduce {% cite thakur2005optimization patarasuk2009bandwidth %} over $N$ devices with an input array of $V$ bytes per device. Arrange the devices in a ring and divide the input array into equal chunks, one per device, of $V/N$ bytes apiece. A reduce-scatter circulates and accumulates the chunks for $N-1$ steps, followed by an all-gather that circulates the completed chunks for another $N-1$. Counting bytes sent per device, the volume is

$$2 \cdot \frac{N-1}{N} V \;\approx\; 2V.$$

On a full-duplex link, each device sends and receives at the same time. The nice thing is that bandwidth cost barely grows with $N$. The catch is $2(N-1)$ sequential steps, so latency does grow.

For our 7B-parameter model, bf16 gradients occupy $V=14$ GB, so a large ring sends about 28 GB per device. At an effective 50 GB/s, that is roughly half a second. In practice, gradients are bucketed: a layer's gradients can start communicating as soon as backward produces them, while earlier layers are still computing. So the number that actually matters is how much communication is left over once you account for overlap.

One more piece of context for the communication cost model: interconnects are hierarchical. Devices connected by a local high-bandwidth fabric such as NVLink, or by TPU ICI within a slice, generally communicate much faster than devices reached through a cluster data-center network. Thus, a lot of placement decisions come down to putting each collective's traffic on the right tier of the network.

## Data parallelism: shard the data, all-reduce the gradients

The strategy is the most familiar: pure *data sharding* over fully replicated state. With $N$ devices and $P$ model parameters, replicate the model across devices and split the batch evenly among them. Let $\theta$ be the model weights and $\mathcal{L}_i$ the mean loss on device $i$. The gradient of the global mean loss $\mathcal{L}$ is the average of these local gradients:

$$\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i$$

The algorithm per step is: forward and backward locally, then an **all-reduce** to sum the gradients and a division by $N$ to average them (approximately $2 \cdot 2P(N-1)/N$ bytes sent per device in bf16), then an identical local optimizer step on every replica. The replicas stay in sync because they start from the same parameters and apply the same reduced gradient.

A few things to note. First, overlapping makes DP fast in practice: gradients are bucketed and reduced while backward is still running, and a well-tuned DP can hide much of its communication. Second, holding the per-device batch size fixed makes DP scale the *global batch size* with $N$. The fundamental limit of pure DP is memory: if the state does not fit on one device, pure DP is simply not available.

## FSDP: reschedule the all-reduce

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling1-fsdp.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling1-fsdp.png' | relative_url }}" width="640" height="515" loading="lazy" alt="Two device memories start with different colored parameter shards. All-gather temporarily gives both devices the full layer for computation. Freeing the temporary copies leaves the original shards.">
  </a>
  <figcaption>Each device gathers the layer’s weights for the forward pass, then keeps only the shard it owns.</figcaption>
</figure>

ZeRO's key observation {% cite rajbhandari2020zero %} is that no device needs to store and update the whole model. Dividing $P$ parameters across $N$ devices gives each device ownership of $P/N$ parameters and the corresponding gradients and optimizer state. Before a layer runs, its parameter shards are gathered into a temporary full copy on every device. After backward, gradient contributions are summed and partitioned so the owners can update their local parameters.

Those two exchanges are the collectives introduced above: an **all-gather** reconstructs the layer's parameters, and a **reduce-scatter** sums and repartitions its gradients. Fully sharded data parallelism (FSDP) performs them layer by layer {% cite zhao2023pytorch %}. Persistent model state falls from roughly $16P$ to $16P/N$. Under the bf16 accounting above, reconstructing each layer separately for forward and backward gives two parameter all-gathers and one gradient reduce-scatter, or about $6P(N-1)/N$ bytes sent per device per step. That's 1.5× DP's volume for the same model and device count. The exchanges can overlap with neighboring layers, although reconstructing a layer is still a synchronization point and temporarily raises peak memory.

Under this schedule, the temporary full weights are discarded after forward and gathered again for backward. Dividing the reduce-scattered gradient sums by $N$ gives each device the averaged gradient shard for its local optimizer update.

## Tensor parallelism: shard the matmul

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling1-tensor-parallel.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling1-tensor-parallel.png' | relative_url }}" width="640" height="565" loading="lazy" alt="Each square is an array element. A is 4 by 6, split into 4 by 3 column slices. B is 6 by 4, split into matching 3 by 4 row slices. On each device, the same 3 by 4 X times its A slice produces a 3 by 3 hidden array after GeLU. Multiplying by its B slice produces a 3 by 4 partial output. All-reduce sums these into the 3 by 4 Y, replicated on both devices.">
  </a>
  <figcaption>Each device produces a 3 × 4 partial output. All-reduce adds them elementwise, so the final output is also 3 × 4 and is available on both devices.</figcaption>
</figure>

DP and FSDP divide the batch, but each device still executes the full model for its local microbatch. Tensor parallelism (TP) instead partitions weight matrices so that devices share a layer's FLOPs and intermediate activations {% cite shoeybi2019megatron %}.

Consider an MLP $Y = \mathrm{GeLU}(XA)B$: $X$ is the input activation matrix, $A$ and $B$ are the first and second weight matrices, and $Y$ is the output. The input $X$ is replicated across devices. Splitting $A$ across columns produces separate activation shards, with GeLU applied locally. Splitting $B$ across rows lets those shards feed directly into the second matmul. An all-reduce sums the resulting partial outputs. No device has to materialize the full intermediate activation. Attention follows the same pattern by assigning heads to devices and combining them at the output projection.

These collectives operate on arrays of size $\text{microbatch} \times \text{sequence} \times \text{hidden}$, occur on every layer, and sit on the critical path. TP is therefore usually confined to the fastest interconnect, often within a node.

## Pipeline parallelism: the assembly line

<figure class="concept-figure">
  <a href="{{ '/assets/images/scaling/scaling1-pipeline.png' | relative_url }}">
    <img src="{{ '/assets/images/scaling/scaling1-pipeline.png' | relative_url }}" width="640" height="470" loading="lazy" alt="GPipe and 1F1B schedules for four microbatches on three devices, with device names on the left and layer ranges on the right. Numbers 1 through 4 identify microbatches. Hatched blue blocks are forward work, hatched green blocks are backward work, and gray cells are idle. GPipe groups forwards before backwards, while 1F1B alternates them after warmup.">
  </a>
  <figcaption>The numbers follow four microbatches through the pipeline. 1F1B alternates forward and backward passes after warmup, allowing it to free saved activations sooner than GPipe.</figcaption>
</figure>

The fourth strategy shards by *depth*. Split the model into $S$ stages, each owning a consecutive group of layers. Data flows through the stages like an assembly line. Between stages there are no collectives, only point-to-point handoffs of boundary activations ($\text{microbatch} \times \text{seq} \times \text{hidden}$ elements forward, with a similarly shaped activation gradient backward) between neighboring stages. Each transfer only crosses one boundary, so the volume is usually small, which makes pipeline parallelism a good fit for slower links.

The cost is utilization: stages sit idle while the pipeline fills and drains. GPipe {% cite huang2019gpipe %} addresses this by splitting the batch into $m$ microbatches that flow through the pipeline in a staggered fashion, so that with $S$ balanced stages and negligible communication overhead, the idle "bubble" occupies approximately $\frac{S-1}{m + S - 1}$ of the step.

## Composition for large-scale training

Large-scale training composes these along a *device mesh* {% cite narayanan2021efficient %}. The composition follows from a short accounting exercise:

1. **Count persistent state.** If the parameters, gradients, and optimizer state fit comfortably on one device, start with DP. Otherwise, consider FSDP.
2. **Count per-device work and activations.** DP and FSDP divide the batch but leave each device executing the full model. If one microbatch still requires too much compute or activation memory, shard within each layer using TP or across depth using pipeline parallelism.
3. **Inspect the topology.** Put latency-sensitive activation collectives, especially TP, on the fastest links. Parameter and gradient collectives from DP or FSDP are easier to overlap and can better tolerate slower links. Pipeline communication is point-to-point and often suits the slowest links.
4. **Check roofline estimates.** Divide the bytes moved by effective link bandwidth, add the latency of sequential collective steps, and compare the result with the compute available for overlap.
5. **Check the cost of parallelism.** Increase a parallelism dimension only when the memory or compute saved outweighs its communication and utilization costs.

A common layout uses TP within a node, DP or FSDP across nodes, and pipeline stages when the model or topology requires another dimension.

## It's all just MPI

So every one of these strategies is a choice of what to shard, plus a handful of MPI collectives to pay for it. Picking between them is bookkeeping: memory, compute, and communication, checked against the model, batch size, and hardware you actually have.

In the next post, I'll write this composition in JAX, where you just say how arrays are split across a device mesh and JAX/XLA figure out the communication for you. Most of the time you think about array axes and placement, and never write a collective by hand. I recommend the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/), which assumes a decent systems understanding but goes much further toward actually training an LLM at scale, with a lot more of the arithmetic (a.k.a. roofline estimates).

# References

{% bibliography --cited %}
