---
layout: post
title: "Why won't the transformer die? Part 3: three design choices"
description: Why three projectors, why inner product, why softmax.
---

This is part 3 of a four-part series. Parts 1 and 2 set up the picture of self-attention as a non-parametric, dim-preserving, K-NN-mimicking lookup table. This post takes apart the three specific design choices inside that lookup table and asks why each is the way it is.

## Why three projectors? QKV as content-addressable lookup

Why three projections? Why not two (tied $Q = K$) or four? In my view, each of the three projections has a distinct role.

### The connection

One framing is that we need read, write, and address modes to implement a content-addressable memory, and three projections give such an API. To make it concrete, picture an ordinary associative array, where you write `memory[key] = value` and read it back by handing the same key in. A classical lookup needs an exact key match. A *content-addressable* lookup relaxes that to "return the value whose key is most similar to the query" under some chosen similarity. Soft attention is the differentiable version of the same retrieval, returning a weighted average of values where the weight on the $i$-th entry is $\exp(\langle q, k_i \rangle / \sqrt{d_k})$ after row-normalization across the keys {% cite vaswani2017attention %}.

In *self*-attention, input tokens play all three roles via learned projectors. For a query $q$, a head computes approximately

$$\mathrm{attn}(q) \approx \texttt{memory}[k_{j^*}], \qquad j^* = \arg\max_j \, \langle q, k_j \rangle$$

with the softmax replacing the hard $\arg\max$ by a weighted average. In `memory[key] = value` words, $W_K$ files each token under a key $k_j$, $W_V$ sets the value $v_j$ stored under that key, i.e. the content that gets returned on a match, and $W_Q$ forms the query that a token hands to the memory. The Turing-tape analogy is that the key is the addressing of the tape cells, the value is what the cells contain, and the query is the head's read pattern. The transformer is then a differentiable read on a soft, content-addressable tape.

### From classical theory of computation

Differentiable content-addressable memory has been a recurring research goal for decades.

- **Hopfield's associative memory** {% cite hopfield1982neural %} stores patterns as fixed points of a network and retrieves them by partial-pattern similarity to a query. A pattern goes in, the closest stored pattern comes out. The query and the stored items live in the same space, but the read-out is a separately recoverable item. That is already a Q/K/V picture in spirit.
- **Neural Turing Machines** {% cite graves2014neural %} and the **Differentiable Neural Computer** {% cite graves2016hybrid %} bolted a Turing-machine-style external memory onto a recurrent controller, with separate read heads, write heads, and content-based addressing as the lookup mechanism.
- **End-to-End Memory Networks** {% cite sukhbaatar2015end %} and **Key-Value Memory Networks** {% cite miller2016key %} cleaned this up by separating out the key vector used for addressing from the value vector that gets returned. By the time the transformer arrived, the K/V split was already standard vocabulary in the memory-network line of work.
- **Fast weight programmers** {% cite schmidhuber1992learning schlag2021linear %} are an older lineage where one network produces the weights of another, and the linear-attention reduction shows that this is exactly what an attention layer does in disguise.

### Why not two projectors

Tying $Q = K$ forces queries and keys to share a representation, limiting the ability to distinguish what a token searches for from how it is addressed. There is no asymmetry with two projections, because with shared $W_Q = W_K$ the pre-softmax score matrix $X W W^{T} X^{T}$ is symmetric, so token A scores B exactly as B scores A. Without masking or other positional mechanisms, any remaining asymmetry comes from each row's normalization. This learned asymmetry can be useful for causal language modeling, dependency-style relations, and other directional computations. Tied-QK variants exist and work in some settings, but they give up this asymmetry, so three independent projections is the natural choice for an asymmetric, differentiable, content-addressable lookup.

## Why inner product?

We have settled on three projections, but why is the score itself an inner product? Why not Euclidean distance, cosine, an MLP applied to concatenated $(q, k)$, or any of the other plausible similarities?

### Universal similarity measure

Recall that attention's effective similarity is $\exp(x_i^{T} M x_j / \sqrt{d_k})$ with $M = W_Q W_K^{T}$ {% cite tsai2019transformer %}. The natural worry is that "exp of an inner product" might be a restrictive family. Kernel methods {% cite scholkopf2002learning %} suggest why inner products can be expressive when applied to suitable learned features. Every positive-definite kernel decomposes as $K(x, y) = \langle \phi(x), \phi(y) \rangle$ in some (possibly infinite-dimensional) feature space. Writing the similarity as an inner product of *learned* features $q = W_Q^{T} x$ and $k = W_K^{T} x$ is just: instead of fixing the kernel and discovering its features implicitly, we set up the inner product and learn the appropriate kernel. This motivates the connection, although linear Q/K maps alone do not represent every kernel.

With $d_k$ large enough, $W_Q W_K^{T}$ can express any rank-$d_k$ bilinear similarity on the input space. Some kernels (like RBF on the raw inputs) correspond to infinite-dimensional feature spaces, so no finite-rank bilinear score reproduces them exactly, but the relevant question is whether the rank-$d_k$ family is rich enough at the scales we work at, and in practice it is. The inner-product score also lines up directly with the metric-learning view from the earlier post, since picking the kernel and picking the metric are the same problem expressed in two different ways.

## Why softmax?

Softmax has the form $\mathrm{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)$. It does two things at once: it exponentiates, and it normalizes the result to the probability simplex. The simplex is a geometric space where every point represents a valid probability distribution over a set of mutually exclusive outcomes. The two steps deserve separate analysis.

### The exp part: maximum entropy

The exponential is downstream of one of the most-used variational arguments in statistical inference. Among all distributions on a discrete set with a fixed expected value of the logits, the maximum-entropy distribution has the form $p_i \propto \exp(\lambda z_i)$. This is Jaynes' principle {% cite jaynes1957information %} in the discrete case: the exponential is the least-committal probability assignment given linear constraints on the logits, and it is uniquely determined by that property.

Moreover, the exponential is the unique continuous solution (up to a choice of base) of $f(a + b) = f(a) f(b)$, which is what gives softmax shift-invariance in the logits, since multiplying every $\exp(z_i)$ by the same constant cancels under normalization. That is why we can subtract $\max_i z_i$ before exponentiating for numerical stability, and why attention is invariant to constant shifts in $QK^{T}$.

### Probability simplex

Mapping $\mathbb{R}^N$ to the $(N-1)$-dimensional probability simplex by requiring $\alpha_i \geq 0$ and $\sum_i \alpha_i = 1$ is a choice. We can exponentiate without normalizing, normalize without exponentiation (e.g. linear attention {% cite katharopoulos2020transformers %}), or have completely different constraints (sigmoid attention {% cite ramapuram2024sigmoid %}, where each weight is independent in $[0, 1]$).

What does the probability simplex commit us to? Three things that get conflated:

1. **Boundedness.** $\alpha_i \in [0, 1]$, which is useful for numerically stable downstream computation.
2. **Comparability across contexts.** $\sum_i \alpha_i = 1$, so a weight of $0.9$ means the same thing whether the context has 3 keys or 1000 keys. Without it, comparing attention patterns across sequence lengths is not well-defined.
3. **Convex combination.** The output $\sum_i \alpha_i v_i$ lives in the convex hull of the values $\{v_i\}$. The contextualized output is *somewhere among* the value vectors, never outside.

The third property is a choice and one could have chosen a different inductive bias (for example, sigmoid attention).

## Just won't die

An alternative architecture that also has [dim-preservation properties](https://anuragxel.github.io/blog/transformer-no-bottleneck/) and the [soft-k-NN behavior](https://anuragxel.github.io/blog/transformer-soft-knn/) to emulate a soft lookup has to make three more decisions. We looked through those decisions and came away with a few interesting realizations. Tying up Q and K gives up on directional asymmetry. Non-inner-product scores give up on some really nice kernel properties. Non-softmax aggregation changes whether the per-head output is a convex combination of the values. The combination that the transformer makes is general enough that it makes the architecture stable across a variety of modalities and tasks.

## References

{% bibliography --cited %}
