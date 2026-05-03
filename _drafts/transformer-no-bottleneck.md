---
layout: post
title: "Why is killing transformers hard? Part 1: no information bottleneck"
description: The capacity property that every alternative architecture has to match.
---

A friend of mine, very smart but not a machine learning person asked me recently, what's going on with the machine learning architecture world, is Mamba looking good compared to transformers?

I replied, "No, it has an information bottleneck embedded in its state-space operation, which is not good to have. Transformers don't have any such bottlenecks."

He responded, "I suppose, but it's hard to tell what the real reasons are for why it works. You guys sometimes add 7 layers, sometimes 8 and god knows why one works and the other doesn't."

I hated my own hand-waviness and decided to systematize what I know. This is the first of a four-part series on architectural properties that make transformers hard to displace. This post is about the simplest of them: the architecture preserves token dimensionality, so it never forces a low-dimensional summary that the rest of the model has to reconstruct.

## Self-attention and cross-attention

Let $X \in \mathbb{R}^{N \times d}$ be a set of $N$ tokens, each represented as a $d$-dimensional vector. **Self-attention** projects $X$ into queries, keys, and values via three learned matrices (or projectors) $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

The output is a weighted average of the values, where weights come from a row-wise softmax over scaled query-key dot products:

$$\mathrm{SelfAttn}(X) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V \in \mathbb{R}^{N \times d_v}$$

The scale $\sqrt{d_k}$ keeps the dot products from growing too large in magnitude as $d_k$ increases, which would otherwise push the softmax into saturated regions with vanishing gradients.

Note the use of *set* rather than *sequence*. The operation is permutation-equivariant on the token axis, so the structure of the input modality is contributed by the embedding step rather than by the attention itself.

**Cross-attention** is the same operation, but queries come from one set and keys/values from another. Given a target set $X \in \mathbb{R}^{N \times d}$ and a source set $Y \in \mathbb{R}^{M \times d}$,

$$Q = X W_Q, \quad K = Y W_K, \quad V = Y W_V$$

$$\mathrm{CrossAttn}(X, Y) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V \in \mathbb{R}^{N \times d_v}$$

Each of the $N$ target tokens attends over all $M$ source tokens. Self-attention is the special case $Y = X$ (so $M = N$).

## No bottleneck

The biggest question is: why is almost any concept or modality so easy to convert into a transformer?

Consider a standard ViT-B/16 {% cite dosovitskiy2021image %}. Each $16 \times 16$ RGB patch is $16 \times 16 \times 3 = 768$ floats. We linearly project each one to a $d = 768$-dimensional token, giving a set of $N$ tokens to feed into a stack of self-attention operations {% cite vaswani2017attention %}. Given $N$ input tokens, each self-attention layer outputs $N$ tokens of the same dimension.

Here is the key observation. At no point does the architecture reduce the dimensionality of any token. The output of self-attention sits in $\mathbb{R}^{N \times d}$, exactly like the input. There is no architectural bottleneck through which all $N$ tokens must be projected to lower dimensional space or to few number of vectors (i.e. tokens). With the right weights, $\mathrm{SelfAttn}(X)$ can losslessly carry every coordinate of $X$ forward.

Contrast this with a state-space model like Mamba {% cite gu2024mamba %}, whose forward pass is a recurrence

$$h_t = A h_{t-1} + B x_t, \quad y_t = C h_t$$

with hidden state $h_t \in \mathbb{R}^{d_h}$. Every token's contribution must squeeze through this fixed-dim $h_t$ before any later token sees it. Recent DiT work shows the same effect empirically: training loss has a nonzero lower bound when model width is smaller than token dimension, and drops sharply once the two are matched {% cite zheng2025rae %}.

## Non-parametric vs parametric estimation

There is a clean way to phrase the friend's question. The transformer-vs-SSM argument is the same non-parametric versus parametric estimation argument statisticians have been having for half a century {% cite wasserman2006nonparametric %}.

Self-attention is non-parametric in the per-layer aggregation, not in its parameter count. The projectors $W_Q, W_K, W_V$ are fixed-dimensional and do not grow with $N$. What grows is the reference set. Queries, keys, and values are all constructed from the input itself, so the operator's effective hypothesis class scales with $N$. Capacity scales with the data. State-space models like S4 {% cite gu2022efficiently %} and Mamba {% cite gu2024mamba %}, and linear-attention variants that recast attention as a recurrence with fixed state {% cite katharopoulos2020transformers %}, are parametric in exactly this sense. Their expressive capacity is whatever fits in $h_t \in \mathbb{R}^{d_h}$, regardless of how long the context gets. Selective SSMs make the recurrence input-dependent, but the state stays a fixed-capacity bottleneck.

The classical tradeoff shows up here. Parametric models have constant per-step cost and bounded capacity, and that bound is the functional-form bias I have been complaining about all along. The recurrent state $h_t$ is where the bound binds: anything the recurrence cannot encode in $h_t$ has to be reconstructed by later layers or skipped. Non-parametric operators grow capacity with the reference set, paying for it in compute and memory that scale with $N$.

The recurrence $h_t = f(h_{t-1}, x_t)$ is intrinsically ordered, so $h_t$ summarizes one particular causal traversal of the input. Self-attention is permutation-equivariant by default and acquires order through positional encodings. For modalities where order is not part of the data, like a set of images for novel view synthesis, a 3D point cloud, or the nodes of a graph, the recurrent route adds an unnecessary ordering inductive bias.

## Hard to kill

Any architecture that wants to compete with transformers on generality has to follow the notion of introducing ``no information bottlenecks''{% cite jelassi2024repeat %}. Also, the fact that the self-attention operator itself is non-parametric has deep implications in my view, which we shall cover in the next blog post.

# References

{% bibliography --cited %}
