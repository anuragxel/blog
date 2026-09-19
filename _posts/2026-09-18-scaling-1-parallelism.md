---
layout: post
title: "Scaling model (pre)-training. Part 2: Parallelism is just MPI"
description: DP, FSDP, TP, and pipeline parallelism as compositions of familiar communication primitives.
---

This is part 2 of a four-part series on scaling up model (pre)-training.

Data parallelism, fully-sharded data parallelism, tensor parallelism, and pipeline parallelism are usually presented as framework features: flags we flip or configs we edit. I think this framing obscures how simple they are. These abstractions were built to address old distributed-systems questions applied to deep learning: *where do the tensors and activations live, and how do you get through a forward and backward pass with the least communication overhead and the highest utilization?* In fact, a systems person who has never done deep learning already knows all four strategies under older names:

- **Data parallelism** is data sharding over replicated state, run in bulk-synchronous Single Program, Multiple Data (SPMD) style {% cite valiant1990bridging %}. The intuition is: to count a country's population we do not ask every household to report to one office. Each district counts its own residents using identical instructions, and the totals are summed at the end. MapReduce {% cite dean2008mapreduce %} is the common example of this pattern.
- **Fully-sharded data parallelism** is partitioned state with an *owner-computes* rule. Think of bank branches: our home branch owns the ledger and is the only one that applies transactions. Any other branch that needs our balance requests a copy for the moment it needs one, rather than maintaining its own.
- **Tensor parallelism** is block-partitioned distributed matrix multiplication, which is as old as parallel computing itself: Cannon used it in 1969 to run the Kalman filter on a grid of processors {% cite cannon1969cellular %}, and SUMMA {% cite vandegeijn1997summa %} is another classic algorithm for the same problem.
- **Pipeline parallelism** is pipelining, the oldest trick in hardware: Ford's assembly line, the five-stage RISC pipeline, systolic arrays. Each station does one stage of the work and hands the piece downstream, and throughput comes from keeping every station busy on a *different* piece at the same time.

Most of the communication in these four strategies can be expressed with five collective operations standardized by [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) {% cite mpiforum1994 %}. Pipeline parallelism also uses point-to-point communication, which we treat separately. Once these communication patterns are familiar, the strategies become an exercise in arithmetic.

## Why shard at all: the memory footprint of training state

Let us start with why we shard in the first place. Consider a model with $\Psi$ parameters trained with Adam. Following ZeRO's example, but using bf16 instead of fp16, each parameter costs approximately {% cite rajbhandari2020zero %}:

- 2 bytes — bf16 working copy of the parameter
- 2 bytes — bf16 gradient
- 4 bytes — fp32 master copy of the parameter
- 4 + 4 bytes — fp32 Adam first and second moments

That is **16 bytes per parameter** of *state*, before a single activation is computed. A 7B-parameter model therefore requires about 112 GB of persistent state. Activations are separate and scale roughly with $\text{batch} \times \text{sequence} \times \text{hidden} \times \text{depth}$, subject to checkpointing and implementation details. Smaller per-device batches and gradient checkpointing reduce activation memory; state is usually sharded across devices, although offloading and lower-precision optimizer states are also possible. Each parallelism strategy chooses which of the data, state, and activations to shard and when to communicate.

## Back to the 80s and 90s: Remembering the MPI collectives

Before MPI, parallel computing was a wild west. Everyone had hand-rolled their own message-passing library (PVM, Intel's NX, Express, PARMACS), all encoding the communication patterns that kept recurring in scientific computing. The MPI Forum standardized the vocabulary in 1994 {% cite mpiforum1994 %}, and the core semantics of these collectives have remained stable because they form a natural basis for the space.

A collective is an operation that a group of $N$ processes executes together, each contributing and/or receiving data. Fix an array size $M$ and write $x^{(p)}$ for the data process $p$ holds, with $[\,\cdot \mid \cdot\,]$ denoting concatenation. Five collectives cover most of the layouts in this series:

- **broadcast** — one root holds $x \in \mathbb{R}^M$; afterward every process holds $x$. One use is distributing initialized parameters.
- **all-gather** — each process holds a shard $x^{(p)} \in \mathbb{R}^{M/N}$; afterward every process holds the concatenated full array.
- **reduce-scatter** — each process holds a full $x^{(p)} \in \mathbb{R}^M$; the arrays are summed and the result is sharded, leaving process $p$ with its $M/N$ slice.
- **all-reduce** — each process holds $x^{(p)} \in \mathbb{R}^M$; afterward every process holds the full sum $\sum_p x^{(p)}$.
- **all-to-all** — every process splits its input into $N$ blocks and sends block $q$ to process $q$. This changes the sharding axis: for example, token-sharded activations can be rearranged into head-sharded activations.

The single most useful identity in this whole subject is:

$$\texttt{all-reduce} \;=\; \texttt{reduce-scatter} \;+\; \texttt{all-gather}$$

Reduce-scatter produces summed shards $s_p$, and all-gathering those shards reassembles the full sum on all processes.

### Understanding the communication cost model through ring all-reduce

The ring algorithm {% cite thakur2005optimization patarasuk2009bandwidth %} makes the communication cost concrete and is widely used to implement all-reduce efficiently. Arrange the $N$ processes in a ring and cut each $M$-byte array into $N$ chunks. A reduce-scatter circulates and accumulates the chunks for $N-1$ steps; an all-gather circulates the completed chunks for another $N-1$. Counting bytes sent per device, the volume is

$$2 \cdot \frac{N-1}{N} M \;\approx\; 2M.$$

The volume of receiving chunks is the same and can occur simultaneously on a full-duplex link. The attractive property is that bandwidth cost barely grows with $N$; the price is $2(N-1)$ sequential communication steps.

For our 7B-parameter model, bf16 gradients occupy $M=14$ GB, so a large ring sends about 28 GB per device. At an effective 50 GB/s, that is roughly half a second. In practice, gradients are bucketed: layer $\ell$ can start communicating as soon as backward produces it, while earlier layers are still computing. The useful question is therefore how much communication remains after overlap, rather than the raw total alone.

One more piece of context for the communication cost model: interconnects are hierarchical. Devices connected by a local high-bandwidth fabric such as NVLink, or by TPU ICI within a slice, generally communicate much faster than devices reached through a cluster data-center network. Every placement decision is thus ultimately about matching a collective's traffic to the appropriate tier of networking.

## Data parallelism: shard the data, all-reduce the gradients

The strategy is the most familiar: pure *data sharding* over fully replicated state. Replicate all parameters on every device, give each device an equally sized slice of the batch, and observe that the gradient of a mean over the batch is the mean of per-device gradients:

$$\nabla_\theta \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \nabla_\theta \mathcal{L}_i$$

The algorithm per step is: forward and backward locally, then an **all-reduce** to sum the gradients and a division by $N$ to average them (approximately $2 \cdot 2\Psi(N-1)/N$ bytes sent per device in bf16), then an identical local optimizer step on every replica. Logically, the replicas remain synchronized because they start from the same parameters and apply the same reduced gradient.

A few things to note. First, overlapping makes DP fast in practice: gradients are bucketed and reduced while backward is still running, and a well-tuned DP can hide much of its communication. Second, holding the per-device batch size fixed makes DP scale the *global batch size* with $N$. The fundamental limit of pure DP is memory: if the state does not fit on one device, pure DP is simply not available.

## FSDP: reschedule the all-reduce

ZeRO's key observation {% cite rajbhandari2020zero %} is that no device needs to store and update the whole model. Instead, each device owns one shard of the parameters, gradients, and optimizer state. When a layer is about to run, the devices exchange their parameter shards so that each briefly reconstructs that layer. After backward, they add their gradient contributions while leaving the result sharded, and each device updates only the shard it owns.

Those two exchanges are the collectives introduced above: an **all-gather** reconstructs the layer's parameters, and a **reduce-scatter** sums and repartitions its gradients. Fully-sharded data parallelism (FSDP) performs them layer by layer {% cite zhao2023pytorch %}. Persistent model state falls from roughly $16\Psi$ to $16\Psi/N$. Under the bf16 accounting above, reconstructing each layer separately for forward and backward gives two parameter all-gathers and one gradient reduce-scatter, or about $6\Psi(N-1)/N$ bytes sent per device per step—1.5× DP's volume. The exchanges can overlap with neighboring layers, although reconstructing a layer is still a synchronization point and temporarily raises peak memory.

In this example, each device stores a $1/N$ share of the parameters, gradients, and optimizer state. To execute a layer, the devices concatenate their bf16 parameter slices into a temporary full copy on every device. Under the schedule counted above, that copy is discarded after the forward pass and assembled again for backward. The reduce-scatter sums the devices' gradient contributions; dividing by $N$ gives each device its slice of the averaged gradient. That device uses the slice to update its local parameters and optimizer state.

## Tensor parallelism: shard the matmul

DP and FSDP divide the batch, but each device still executes the full model for its local microbatch. Tensor parallelism (TP) instead partitions each weight matrix so that devices share a layer's FLOPs and intermediate activations {% cite shoeybi2019megatron %}. 

For the MLP $Y = \mathrm{GeLU}(XA)B$, $A$ is sharded across columns, so each device produces one shard of the intermediate activation and applies GeLU locally. Then $B$ is sharded across rows: each device consumes its corresponding activation shard and produces a partial output, which an all-reduce combines. No device has to materialize the full intermediate activation. Attention follows the same pattern by assigning heads to devices and combining them at the output projection.

These collectives operate on arrays of size $\text{microbatch} \times \text{sequence} \times \text{hidden}$, occur on every layer, and sit on the critical path. TP is therefore usually confined to the fastest interconnect, often within a node.

## Pipeline parallelism: the assembly line

The fourth strategy shards by *depth*. Device $p$ owns a contiguous block of layers, $[p\,L/N, (p{+}1)L/N)$ for an $L$-layer model, and data flows through the devices like an assembly line. Its communication profile is: no collectives are required between stages, only point-to-point handoffs of boundary activations ($\text{microbatch} \times \text{seq} \times \text{hidden}$ elements forward, with a similarly shaped activation gradient backward) between neighboring stages. This is often a low communication volume because each transfer crosses only one boundary, making pipeline parallelism attractive across slower links.

The cost is utilization: each stage waits while the pipeline fills and drains. GPipe {% cite huang2019gpipe %} addresses this by splitting the batch into $m$ microbatches that flow through the pipeline in a staggered fashion, so that with $N$ balanced stages and negligible communication overhead, the idle "bubble" occupies approximately $\frac{N-1}{m + N - 1}$ of the step.

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
