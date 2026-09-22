---
layout: post
title: "Why won't the transformer die? Part 3: three design choices"
description: Why three projection weight matrices, why inner product, why softmax.
---

This is part 3 of a four-part series. Parts [1]({% post_url 2026-05-03-transformer-no-bottleneck %}) and [2]({% post_url 2026-05-03-transformer-soft-knn %}) set up a way to think about self-attention: it keeps the token dimensions intact and looks up information among the input tokens using learned similarities. Calling Q, K, and V "queries, keys, and values" gives me names for the matrices but doesn't provide intuition for why the operation looks the way it does.

I also want to know what freedom I lose if I tie two of their weight matrices together, for example. These are the sorts of questions I want to be able to answer before changing an attention layer. This post takes apart each specific choice, in terms of the number of projection weight matrices, the inner product, and softmax.

## Why three projection weight matrices? QKV as content-addressable lookup

Why three projections? Why not two (tied $Q = K$) or four? In my view, each of the three projections has a distinct role.

### The connection

One framing is that we need read, write, and address modes to implement a content-addressable memory, and three projections give such an API. To make it concrete, picture an ordinary associative array, where you write `memory[key] = value` and read it back by handing the same key in. A classical lookup needs an exact key match. A *content-addressable* lookup relaxes that to "return the value whose key is most similar to the query" under some chosen similarity. Soft attention is the differentiable version of the same retrieval, returning a weighted average of values where the weight on the $i$-th entry is $\exp(\langle q, k_i \rangle / \sqrt{d_k})$ after row-normalization across the keys {% cite vaswani2017attention %}.

In *self*-attention, input tokens play all three roles via learned projection weights. For a query $q$, a head computes approximately

$$\mathrm{attn}(q) \approx \texttt{memory}[k_{j^*}], \qquad j^* = \arg\max_j \, \langle q, k_j \rangle$$

with the softmax replacing the hard $\arg\max$ by a weighted average. In `memory[key] = value` words, $W_K$ files each token under a key $k_j$, $W_V$ sets the value $v_j$ stored under that key, i.e. the content that gets returned on a match, and $W_Q$ forms the query that a token hands to the memory. The Turing-tape analogy is that the key is the addressing of the tape cells, the value is what the cells contain, and the query is the head's read pattern. The transformer is then a differentiable read on a soft, content-addressable tape.

<figure class="concept-figure">
  <a href="{{ '/assets/images/transformer/array-vs-attention.png' | relative_url }}">
    <img src="{{ '/assets/images/transformer/array-vs-attention.png' | relative_url }}" width="640" height="650" loading="lazy" alt="A memory lookup requests address a 2 and returns its contents v 2. In attention, the query plays the requested-address role, keys play stored-address roles, and values are the stored contents. Attention produces a weighted sum of the values.">
  </a>
  <figcaption>The requested address is the query. Memory addresses are the keys. Memory contents are the values. An exact lookup returns one stored value. Attention returns a weighted sum.</figcaption>
</figure>

### From classical theory of computation

Differentiable content-addressable memory has been a recurring research goal for decades.

- **Hopfield's associative memory** {% cite hopfield1982neural %} stores patterns as fixed points of a network and retrieves them by partial-pattern similarity to a query. A pattern goes in, and the closest stored pattern comes out. The query and the stored items live in the same space, but the read-out is a separately recoverable item. That is already a Q/K/V picture in spirit.
- **Neural Turing Machines** {% cite graves2014neural %} and the **Differentiable Neural Computer** {% cite graves2016hybrid %} bolted a Turing-machine-style external memory onto a recurrent controller, with separate read heads, write heads, and content-based addressing as the lookup mechanism.
- **End-to-End Memory Networks** {% cite sukhbaatar2015end %} and **Key-Value Memory Networks** {% cite miller2016key %} cleaned this up by separating out the key vector used for addressing from the value vector that gets returned. By the time the transformer arrived, the K/V split was already standard vocabulary in the memory-network line of work.
- **Fast weight programmers** {% cite schmidhuber1992learning schlag2021linear %} are an older lineage where one network produces the weights of another, and the linear-attention reduction shows that this is exactly what an attention layer does in disguise.

### Why not two projection weight matrices?

Tying $Q = K$ forces queries and keys to share a representation, limiting the ability to distinguish what a token searches for from how it is addressed. There is no asymmetry with two projections, because with shared $W_Q = W_K$ the pre-softmax score matrix $X W W^{T} X^{T}$ is symmetric, so token A scores B exactly as B scores A. Without masking or other positional mechanisms, any remaining asymmetry comes from each row's normalization. This learned asymmetry can be useful for causal language modeling, dependency-style relations, and other directional computations. Tied-QK variants exist and work in some settings, but they give up this asymmetry, so three independent projections are the natural choice for an asymmetric, differentiable, content-addressable lookup.

<figure class="concept-figure">
  <a href="{{ '/assets/images/transformer/tied-qk-scores.png' | relative_url }}">
    <img src="{{ '/assets/images/transformer/tied-qk-scores.png' | relative_url }}" width="640" height="340" loading="lazy" alt="With tied query and key weights, the A-to-B and B-to-A entries of the score matrix must match. Separate weights allow those two entries to differ.">
  </a>
  <figcaption>Tying Q/K weights forces symmetric dot-product scores. This is before softmax, masking, or additional positional terms. Row normalization can still produce asymmetric attention weights.</figcaption>
</figure>

## Why the inner product?

We have settled on three projections, but why is the score itself an inner product? Why not Euclidean distance, cosine, an MLP applied to concatenated $(q, k)$, or any of the other plausible similarities?

### Expressiveness of learned similarity

Recall that attention's effective similarity is $\exp(x_i^{T} M x_j / \sqrt{d_k})$ with $M = W_Q W_K^{T}$ {% cite tsai2019transformer %}. The natural worry is that "exp of an inner product" might be a restrictive family. Kernel methods {% cite scholkopf2002learning %} suggest why inner products can be expressive when applied to suitable learned features. Every positive-definite kernel decomposes as $K(x, y) = \langle \phi(x), \phi(y) \rangle$ in some (possibly infinite-dimensional) feature space. Attention's learned features $q = W_Q^{T} x$ and $k = W_K^{T} x$ suggest a related idea: learn the representations used in the inner product. The analogy has limits, though: separate Q/K maps need not define a positive-definite kernel, and linear maps alone do not represent every kernel.

With unconstrained, separate $W_Q$ and $W_K$, their product can in principle represent any bilinear form of rank at most $d_k$ on the input space. This is an algebraic capacity statement, not a guarantee that training will find a useful similarity. Increasing $d_k$ relaxes the rank constraint, but does not make linear Q/K maps universal: a bilinear score on raw inputs cannot generally reproduce a kernel such as RBF. Whether this family is expressive enough depends on the task and the input representations, which may already contain nonlinear features learned by preceding layers. The connection to metric and kernel learning is useful as an intuition, rather than an equivalence, since $W_Q W_K^{T}$ need not be symmetric or positive-semidefinite.

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

<figure class="concept-figure">
  <a href="{{ '/assets/images/transformer/attention-convex-hull.png' | relative_url }}">
    <img src="{{ '/assets/images/transformer/attention-convex-hull.png' | relative_url }}" width="640" height="325" loading="lazy" alt="Three value vectors form a triangle. Nonnegative attention weights that sum to one place their weighted sum inside the triangle.">
  </a>
  <figcaption>The softmax-weighted sum lies in the values’ convex hull, before attention dropout, output projection, or residual addition.</figcaption>
</figure>

The third property is a choice and one could have chosen a different inductive bias (for example, sigmoid attention).

## Just won't die

An alternative architecture that also has [dim-preservation properties](https://anuragxel.github.io/blog/transformer-no-bottleneck/) and the [soft-k-NN behavior](https://anuragxel.github.io/blog/transformer-soft-knn/) to emulate a soft lookup has to make three more decisions. The kernel connection motivates inner-product scores without making them necessary. Non-softmax aggregation can change whether the per-head output is a convex combination of the values. This combination may help explain the transformer's versatility across modalities and tasks. If I tie $W_Q = W_K$, there is a tradeoff: fewer parameters, but the learned score can no longer distinguish A looking for B from B looking for A.

There is still one loose end in this picture. If attention behaves like a non-parametric lookup, where do all the learned parameters fit in? In the [last post]({% post_url 2026-09-12-transformer-projectors-vs-projections %}), we'll separate the weights from the activations and look at what that distinction lets us do.

## References

{% bibliography --cited %}
