---
layout: post
title: "Scaling model (pre)-training. Part 2: Just Learn MPI"
description: DP, FSDP, TP, and pipeline parallelism as compositions of familiar communication primitives.
---

This is part 2 of a four-part series on scaling up model (pre)-training.

Data parallelism, fully-sharded data parallelism, tensor parallelism, and pipeline parallelism are usually presented as framework features: flags we flip or configs we edit. I think this framing obscures how simple they are. These abstractions were built to address old distributed-systems questions applied to deep learning: *where do the tensors and activations live, and how do you get through a forward and backward pass with the least communication overhead and the highest utilization?*

Most of the communication in these four strategies can be expressed with five collective operations standardized by [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) {% cite mpiforum1994 %}. Pipeline parallelism also uses point-to-point communication, which we treat separately. Once these communication patterns are familiar, the strategies become an exercise in arithmetic.

## Why shard at all: the memory footprint of training state

Let us start with why we shard in the first place. Consider a model trained with Adam. Following ZeRO's example, but using bf16 instead of fp16, each parameter costs approximately {% cite rajbhandari2020zero %}:

- 2 bytes — bf16 working copy of the parameter
- 2 bytes — bf16 gradient
- 4 bytes — fp32 master copy of the parameter
- 4 + 4 bytes — fp32 Adam first and second moments

That is **16 bytes per parameter** of *state*, before a single activation is computed. A 7B-parameter model therefore requires about 112 GB of persistent state. Activations are separate and scale roughly with $\text{batch} \times \text{sequence} \times \text{hidden} \times \text{depth}$, subject to checkpointing and implementation details. Smaller per-device batches and gradient checkpointing reduce activation memory; state is usually sharded across devices, although offloading and lower-precision optimizer states are also possible. The parallelism strategies differ in what they shard—data, state, or activations—and when they communicate.

## Back to the 80s and 90s: the language of MPI

Before MPI, systems such as PVM, Intel's NX, Express, and PARMACS offered different interfaces for many of the same communication patterns. The MPI Forum standardized a common interface in 1994 {% cite mpiforum1994 %}. The problems were familiar: distribute some data, assemble pieces held on different machines, or combine results computed independently. These are also the operations we need to train a model across devices.

A **collective** is a communication operation in which every process in a group participates. For this post, imagine one process per device. Its number within the group is called its **rank**. We will follow the data before and after an operation.

- **Broadcast: copy from one device to everyone.** Choose one process as the source, called the **root**. Its array is copied to every other device. For example, one device can initialize the model's parameters and broadcast them so everyone starts with the same weights.
- **All-gather: assemble everyone's pieces on every device.** The devices exchange their arrays and concatenate them in rank order. Starting from distributed parameter shards, all devices end up holding the full layer.
- **All-reduce: combine everyone's contributions and give everyone the result.** The input arrays have the same shape. A *reduction* combines corresponding entries using an operation such as sum or maximum; we use sums here. For gradients, the result contains the total contribution from all devices to every parameter.
- **Reduce-scatter: combine contributions, then divide the result.** As with all-reduce, the inputs are full arrays that are summed element by element. The result is then partitioned among the devices, with one slice going to its assigned owner. We use equally sized slices in this post.
- **All-to-all: exchange different pieces with different devices.** The input arrays are split into one block per destination. A destination receives its designated block from all senders. This can rearrange activations from being split by tokens to being split by attention heads.

For a concrete example, the table shows the starting arrays and the result of each operation. All operations start from the first row; they are not run in sequence. Broadcast uses device 0 as the root, both reductions use sums, and all-to-all sends one element per block.

<div class="collectives-table" markdown="1" role="region" aria-label="Collective operations on two devices" tabindex="0">

| Operation / state | Device 0 | Device 1 |
| :--- | :---: | :---: |
| **Starting arrays** | `[1, 2]` | `[3, 4]` |
| Broadcast | `[1, 2]` | `[1, 2]` |
| All-gather | `[1, 2, 3, 4]` | `[1, 2, 3, 4]` |
| All-reduce (sum) | `[4, 6]` | `[4, 6]` |
| Reduce-scatter (sum) | `[4]` | `[6]` |
| All-to-all | `[1, 3]` | `[2, 4]` |

</div>

The all-reduce and reduce-scatter rows reveal a useful identity:

$$\texttt{all-reduce} \;=\; \texttt{reduce-scatter} \;+\; \texttt{all-gather}$$

Reduce-scatter leaves `[4]` on device 0 and `[6]` on device 1. All-gathering those results gives `[4, 6]` to both devices, exactly the result of all-reduce. This decomposition will be useful when we get to FSDP.

### Understanding the communication cost model through ring all-reduce

The implementation of the collectives determines the communication overhead. Consider a ring all-reduce {% cite thakur2005optimization patarasuk2009bandwidth %} over $N$ devices with an input array of $V$ bytes per device. Arrange the devices in a ring and divide the input array into equal chunks, one per device, of $V/N$ bytes apiece. A reduce-scatter circulates and accumulates the chunks for $N-1$ steps; an all-gather circulates the completed chunks for another $N-1$. Counting bytes sent per device, the volume is

$$2 \cdot \frac{N-1}{N} V \;\approx\; 2V.$$

The send and receive volumes are equal. On a full-duplex link, sending and receiving can happen simultaneously. The attractive property is that bandwidth cost barely grows with $N$; the price is $2(N-1)$ sequential communication steps.

For our 7B-parameter model, bf16 gradients occupy $V=14$ GB, so a large ring sends about 28 GB per device. At an effective 50 GB/s, that is roughly half a second. In practice, gradients are bucketed: a layer's gradients can start communicating as soon as backward produces them, while earlier layers are still computing. The useful question is therefore how much communication remains after overlap, rather than the raw total alone.

One more piece of context for the communication cost model: interconnects are hierarchical. Devices connected by a local high-bandwidth fabric such as NVLink, or by TPU ICI within a slice, generally communicate much faster than devices reached through a cluster data-center network. Every placement decision is thus ultimately about matching a collective's traffic to the appropriate tier of networking.

## Data parallelism: shard the data, all-reduce the gradients

The strategy is the most familiar: pure *data sharding* over fully replicated state. With $N$ devices and $P$ model parameters, replicate the model across devices and split the batch evenly among them. Let $\theta$ be the model weights and $\mathcal{L}_i$ the mean loss on device $i$. The gradient of the global mean loss $\mathcal{L}$ is the average of these local gradients:

$$\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i$$

The algorithm per step is: forward and backward locally, then an **all-reduce** to sum the gradients and a division by $N$ to average them (approximately $2 \cdot 2P(N-1)/N$ bytes sent per device in bf16), then an identical local optimizer step on every replica. Logically, the replicas remain synchronized because they start from the same parameters and apply the same reduced gradient.

A few things to note. First, overlapping makes DP fast in practice: gradients are bucketed and reduced while backward is still running, and a well-tuned DP can hide much of its communication. Second, holding the per-device batch size fixed makes DP scale the *global batch size* with $N$. The fundamental limit of pure DP is memory: if the state does not fit on one device, pure DP is simply not available.

## FSDP: reschedule the all-reduce

ZeRO's key observation {% cite rajbhandari2020zero %} is that no device needs to store and update the whole model. Dividing $P$ parameters across $N$ devices gives each device ownership of $P/N$ parameters and the corresponding gradients and optimizer state. Before a layer runs, its parameter shards are gathered into a temporary full copy on every device. After backward, gradient contributions are summed and partitioned so the owners can update their local parameters.

Those two exchanges are the collectives introduced above: an **all-gather** reconstructs the layer's parameters, and a **reduce-scatter** sums and repartitions its gradients. Fully-sharded data parallelism (FSDP) performs them layer by layer {% cite zhao2023pytorch %}. Persistent model state falls from roughly $16P$ to $16P/N$. Under the bf16 accounting above, reconstructing each layer separately for forward and backward gives two parameter all-gathers and one gradient reduce-scatter, or about $6P(N-1)/N$ bytes sent per device per step—1.5× DP's volume for the same model and device count. The exchanges can overlap with neighboring layers, although reconstructing a layer is still a synchronization point and temporarily raises peak memory.

In this example, storage per device is a $1/N$ share of the parameters, gradients, and optimizer state. To execute a layer, the devices concatenate their bf16 parameter slices into a temporary full copy on every device. Under the schedule counted above, that copy is discarded after the forward pass and assembled again for backward. The reduce-scatter sums the devices' gradient contributions; dividing by $N$ produces the averaged gradient shards used for local parameter and optimizer updates.

## Tensor parallelism: shard the matmul

DP and FSDP divide the batch, but each device still executes the full model for its local microbatch. Tensor parallelism (TP) instead partitions weight matrices so that devices share a layer's FLOPs and intermediate activations {% cite shoeybi2019megatron %}.

Consider an MLP $Y = \mathrm{GeLU}(XA)B$: $X$ is the input activation matrix, $A$ and $B$ are the first and second weight matrices, and $Y$ is the output. The input $X$ is replicated across devices. Splitting $A$ across columns produces separate activation shards, with GeLU applied locally. Splitting $B$ across rows lets those shards feed directly into the second matmul. An all-reduce sums the resulting partial outputs. No device has to materialize the full intermediate activation. Attention follows the same pattern by assigning heads to devices and combining them at the output projection.

These collectives operate on arrays of size $\text{microbatch} \times \text{sequence} \times \text{hidden}$, occur on every layer, and sit on the critical path. TP is therefore usually confined to the fastest interconnect, often within a node.

## Pipeline parallelism: the assembly line

The fourth strategy shards by *depth*. Split the model into $S$ stages, each owning a consecutive group of layers. Data flows through the stages like an assembly line. Its communication profile is: no collectives are required between stages, only point-to-point handoffs of boundary activations ($\text{microbatch} \times \text{seq} \times \text{hidden}$ elements forward, with a similarly shaped activation gradient backward) between neighboring stages. This is often a low communication volume because each transfer crosses only one boundary, making pipeline parallelism attractive across slower links.

The cost is utilization: stages sit idle while the pipeline fills and drains. GPipe {% cite huang2019gpipe %} addresses this by splitting the batch into $m$ microbatches that flow through the pipeline in a staggered fashion, so that with $S$ balanced stages and negligible communication overhead, the idle "bubble" occupies approximately $\frac{S-1}{m + S - 1}$ of the step.

## Composition for large-scale training

Large-scale training composes these along a *device mesh* {% cite narayanan2021efficient %}. The composition follows from a short accounting exercise:

1. **Count persistent state.** If the parameters, gradients, and optimizer state fit comfortably on one device, start with DP. Otherwise, consider FSDP.
2. **Count per-device work and activations.** DP and FSDP divide the batch but leave each device executing the full model. If one microbatch still requires too much compute or activation memory, shard within each layer using TP or across depth using pipeline parallelism.
3. **Inspect the topology.** Put latency-sensitive activation collectives, especially TP, on the fastest links. Parameter and gradient collectives from DP or FSDP are easier to overlap and can better tolerate slower links. Pipeline communication is point-to-point and often suits the slowest links.
4. **Check roofline estimates.** Divide the bytes moved by effective link bandwidth, add the latency of sequential collective steps, and compare the result with the compute available for overlap.
5. **Check the cost of parallelism.** Increase a parallelism dimension only when the memory or compute saved outweighs its communication and utilization costs.

A common layout uses TP within a node, DP or FSDP across nodes, and pipeline stages when the model or topology requires another dimension.

In the next post I'll describe this composition in JAX. We specify how arrays are divided across the device mesh; JAX and XLA work out and insert the communication needed to execute that computation. In most of the code, we reason about array axes and device placement rather than writing collectives by hand. I recommend the [JAX Scaling Book](https://jax-ml.github.io/scaling-book/), which assumes a decent systems understanding but goes much further toward actually training an LLM at scale, with a lot more of the arithmetic (a.k.a. roofline estimates).

# References

{% bibliography --cited %}
