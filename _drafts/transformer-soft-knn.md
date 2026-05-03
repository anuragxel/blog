---
layout: post
title: "Why is killing transformers hard? Part 2: self-attention as soft k-NN"
description: Self-attention through metric learning and kernel methods.
---

This is part 2 of a four-part series. Part 1 argued that the dim-preservation property of self-attention is one reason transformers are hard to displace. This post is about the other half of that picture: the self-attention operation is a non-parametric estimator mimicking a soft-KNN on a learned metric space.

## Recap

Self-attention on a token set $X \in \mathbb{R}^{N \times d}$ computes

$$\mathrm{SelfAttn}(X) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V$$

with $Q = X W_Q$, $K = X W_K$, $V = X W_V$. Output sits in the same $\mathbb{R}^{N \times d_v}$ as the input.

## Soft k-NN over a learned metric space

The projection weights $Q, K, V$ are themselves functions of the input, which has very deep connections to metric learning and classical learning literature. I like to think of self-attention as doing soft-KNN over a learned kernel/metric space.

To see why, look at the score we feed into softmax. For two tokens $x_i, x_j \in \mathbb{R}^d$, the unnormalized similarity expands as

$$\frac{q_i^{T} k_j}{\sqrt{d_k}} = \frac{x_i^{T} W_Q W_K^{T} x_j}{\sqrt{d_k}} = \frac{x_i^{T} M\, x_j}{\sqrt{d_k}}$$

where $M = W_Q W_K^{T} \in \mathbb{R}^{d \times d}$ is one learned matrix. The entire $QK^{T}$ is a single bilinear form on the input space. The factorization $M = W_Q W_K^{T}$ buys exactly one thing over learning $M$ directly: with $d_k < d$, it constrains $M$ to rank at most $d_k$.

This is the setup of classical *metric learning* {% cite weinberger2009distance %}. A bilinear form $\langle x_i, x_j \rangle_M = x_i^{T} M x_j$ defines a similarity (an inner product, when $M$ is symmetric positive-semidefinite). Metric learning and its many variants all amount to picking such an $M$ so that semantically similar points score high. Inner product in the raw input space is generally not meaningful but if you have a way to project your inputs to some metric space (which your self-attention operation did), you are golden. 

Thus, if we have a learned similarity, exponentiating and row-normalizing gives us,

$$\mathrm{softmax}\!\left( \frac{x_i^{T} M x_j}{\sqrt{d_k}} \right)_j$$

which is a Nadaraya-Watson estimator {% cite nadaraya1964estimating watson1964smooth %} with a learned kernel $k(x_i, x_j) = \exp(x_i^{T} M x_j / \sqrt{d_k})$ {% cite tsai2019transformer %}. Read the whole self-attention layer that way and the analogy is:

- Queries $q_i$ are test points.
- Keys $k_j$ are reference points.
- Values $v_j = W_V x_j$ are reference labels.
- $\mathrm{softmax}(q_i^{T} k_j / \sqrt{d_k})$ is a soft membership function. Instead of hard top-$k$ selection, you get a probability distribution over neighbors weighted by learned similarity.
- The output $\sum_j \alpha_{ij} v_j$ is the soft $k$-NN prediction, a weighted average of neighbor labels.

So the architecture is parameterizing soft $k$-NN regression, but with three distinct differences from classical $k$-NN:

1. **The metric, via the $Q, K$ projections.** Classical kernel methods commit to a kernel up front (RBF, polynomial, Matern), and the choice encodes a strong prior about what "similar" means. Self-attention learns the similarity from data.
2. **The label space, via the $V$ projection.** The "labels" attached to each reference point are themselves learned functions of the input. A token's $v_j$ is its "label" for the $k$-NN.
3. **The reference set, via the context window.** Classical $k$-NN draws references from a fixed training set. Self-attention draws them from whatever happens to be in the current input. Every input set $X$ gets its own reference pool, on the fly.

That is why the same operation works for image patches, text tokens, audio frames, point clouds {% cite zhao2021point %}, and graph nodes {% cite velickovic2018graph %}. Whatever the modality, the layer asks the same question: which of these other tokens are like me, in the metric I have learned to care about, and what should I copy from them?

## Implications from classical learning literature

Most learning methods are defined by the bias they bake in. Linear regression assumes the target is linear in features. Polynomial regression assumes it lives in a fixed polynomial family. Even a multilayer perceptron with fixed depth and width parameterizes a specific function class, universal approximation theorems notwithstanding {% cite hornik1991approximation %}. This is the classical bias-variance tradeoff at the level of function families, before any data is ever seen.

$k$-NN sits at the opposite end of the spectrum. It is the textbook *non-parametric* method {% cite devroye1996probabilistic %}, meaning the hypothesis class grows with the data and never commits to any closed-form structure. There is no assumption that the target is linear, smooth in any particular basis, or expressible by any finite parameterization. The only inductive bias is locality in the chosen metric. Bias and variance trade off through $k$ and the metric, not through a hand-picked function family.

This assumption-light rule has strong asymptotic guarantees, starting with the **Cover-Hart theorem** {% cite cover1967nearest %}. For a $c$-class problem, as the reference set size $n \to \infty$, the error rate of the $1$-nearest-neighbor classifier satisfies

$$R^* \le R_{1\text{-NN}} \le R^*\!\left(2 - \frac{c}{c-1}\, R^*\right) \le 2 R^*$$

where $R^*$ is the Bayes (irreducible) error. A parameter-free rule, $1$-NN, sits within a factor of two of the optimal classifier in the limit of infinite reference data.

The factor-of-two slack disappears once we let $k$ grow. Stone's theorem {% cite stone1977consistent %} shows that if $k \to \infty$ and $k/n \to 0$ as $n \to \infty$, the $k$-NN classifier is *universally consistent*. Its error converges to the Bayes rate $R^*$ for any data distribution, with no constant factor left over.

The catch is that both bounds are stated in the data-generating distribution's natural metric. With Euclidean distance on raw pixels, the "nearest neighbor" of a cat picture is often a different cat-shaped patch of color and not another cat. Classical $k$-NN binds against a meaningless notion of nearness, and degrades catastrophically in high dimensions, where pairwise distances concentrate and the very notion of a "nearest" point loses its meaning {% cite beyer1999nearest %}. The guarantees are real but contingent on having a metric worth using, and classical $k$-NN does not provide one.

This is the gap self-attention closes. Training $M = W_Q W_K^{T}$ end-to-end picks the inner product under which "nearest by $M$" correlates with "same downstream label". The classical statements do not transfer formally to attention (the reference set is in-context rather than iid, the metric is learned jointly with the labels, the aggregation is soft rather than top-$k$), but my conjecture is that self-attention is approximating: non-parametric k-NN whose worst case is bounded and whose limit case is Bayes-optimal.

The soft version also sidesteps the one remaining choice of classical $k$-NN, namely $k$ itself. Every query attends to all $N$ tokens, so $k = N$, and the softmax coefficients implicitly handle the effective neighborhood size. Tokens with low similarity get exponentially small weight and contribute almost nothing, while a few high-similarity tokens dominate the average.

## Multi-head attention as multiple parallel metrics

The standard motivation for multi-head attention is "different heads attend to different things", as in syntactic heads, positional heads, coreference heads. That is descriptively accurate but does not explain why the architecture needs multiple heads rather than one larger head with the same total parameter count.

Softmax is a soft argmax over the keys, with sharpness controlled by $\sqrt{d_k}$ and by how well-separated the keys are in the learned metric $M = W_Q W_K^{T}$. For a fixed query $x_i$, the per-query score $x_j \mapsto x_i^{T} M x_j$ is linear in $x_j$, so it has one direction of maximum increase in key-space. A single head's softmax tends to concentrate mass along that direction, which makes one head a reasonable way to express one mode of relevance.

Once a task requires picking out two unrelated things at once (attend to the subject *and* the object, look back $\delta_1$ tokens *and* $\delta_2$ tokens, match shape *and* texture), a single linear score function has to pick a direction that compromises between them, and the softmax assigns mass accordingly. Multi-head attention learns $h$ different metrics $M_h = W_{Q,h} W_{K,h}^{T}$ in parallel, so each query gets $h$ different linear score functions and can put mass in $h$ different directions of key-space at once. The concatenation is the union of $h$ soft $k$-NN lookups, each in its own metric.

The number of heads is best read as the number of distinct modes of relevance the task needs to express within a single layer. Subspace metric learning in the classical sense {% cite weinberger2009distance %} did this by hand, picking a few different metrics for a few different aspects of the data. Multi-head attention does it end-to-end, with the $h$ metrics learned jointly with the rest of the model.

## The unreasonable effectiveness of linear and k-NN probes after pretraining

Pretraining is metric learning. Every forward pass through a stacked self-attention model is one big soft $k$-NN over a learned metric, so any training loss that updates $W_Q$ and $W_K$ is also optimizing the inner product that decides which tokens score high together. The objective does not have to be a contrastive metric-learning loss explicitly. Masked language modeling, masked image modeling, and contrastive image-text losses all train the same metric, just with different supervision signals.

Consider DINO {% cite caron2021emerging %} and DINOv2 {% cite oquab2024dinov2 %}, or any image/video foundation model. Why are their features so general? You did layer-wise soft $k$-NN over an astonishingly huge reference set, and the network learned the metric space in which that layer-wise soft $k$-NN works. Both report $k$-NN classification accuracy on frozen features as a flagship evaluation alongside linear probing, on the basis that a well-trained representation should already place same-class examples near each other in the learned space. CLIP {% cite radford2021learning %} works the same way at inference time, scoring an image embedding against a set of text embeddings with a single dot product, which is the cross-attention score with hard top-1 selection.

## Harder to kill

For a competing architecture to work across modalities the way transformers do, it has to support learned similarity, learned labels, and a context-shaped reference set. Architectures that fix any of the three (RNNs and SSMs that summarize the reference set into hidden state, MLPs without context-dependent retrieval) have to pay for it in modality-specific tuning or task-specific architecture.

# References

{% bibliography --cited %}
