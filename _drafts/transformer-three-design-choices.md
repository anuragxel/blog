---
layout: post
title: "Why won't the transformer die? Part 3: three design choices"
description: Why three projectors, why inner product, why softmax.
---

This is part 3 of a four-part series. Parts 1 and 2 set up the picture of self-attention as a non-parametric, dim-preserving lookup. This post takes apart the three specific design choices inside that lookup and asks why each is the way it is.

## Why three projectors? QKV as content-addressable lookup

Why three projections? Why not two (tied $Q = K$) or four? Most people do not ask, because the architecture just works, but each of the three projections does distinct work that the others cannot cover.

### The connection

One framing is that you need read, write, and address modes to implement a content-addressable memory, and three projections gives you that API. To make it concrete, picture an ordinary associative array, where you write `memory[key] = value` and read it back by handing the same key in. A classical lookup needs an exact key match. A *content-addressable* lookup relaxes that to "return the value whose key is most similar to the query" under some chosen similarity. Soft attention is the differentiable version of the same retrieval, returning a weighted average of values where the weight on the $i$-th entry is $\exp(\langle q, k_i \rangle / \sqrt{d_k})$ after row-normalization across the keys {% cite bahdanau2015neural %}.

In *self*-attention, every token plays all three roles at once over the same underlying vector. Through $W_Q$, the token forms its query, namely what it is asking the rest of the input about. Through $W_K$, it advertises itself as a catalog entry, a key under which others can look it up when their queries line up. Through $W_V$, it exposes its payload, the content that gets returned when it is matched. One vector, three independently learned projections, three separate roles in the same lookup. The memory bank itself is not stored in weights. It is built on the fly out of the current input and torn down at the end of the layer.

Read self-attention this way and the layer is doing one step of "every token issues a query against an associative memory built out of everyone else, and overwrites itself". The Turing-tape analogy is that K is the addressing of the tape cells, V is what the cells contain, and Q is the head's read pattern. The transformer is shaped like a differentiable read step on a soft, content-addressable tape.

### From classical theory of computation

Differentiable content-addressable memory has been a recurring research goal for decades. The QKV split is the formulation that ended up in the transformer.

- **Hopfield's associative memory** {% cite hopfield1982neural %} stores patterns as fixed points of a network and retrieves them by partial-pattern similarity to a query. A pattern goes in, the closest stored pattern comes out. The query and the stored items live in the same space, but the read-out is a separately recoverable item. That is already a Q/K/V picture in spirit.
- **Neural Turing Machines** {% cite graves2014neural %} and the **Differentiable Neural Computer** {% cite graves2016hybrid %} bolted a Turing-machine-style external memory onto a recurrent controller, with separate read heads, write heads, and content-based addressing as the lookup mechanism. The "find by content, return separately stored value" split shows up there exactly as Q/K/V does in attention.
- **End-to-End Memory Networks** {% cite sukhbaatar2015end %} and **Key-Value Memory Networks** {% cite miller2016key %} cleaned this up by separating out the key vector used for addressing from the value vector that gets returned. By the time the transformer arrived, the K/V split was already standard vocabulary in the memory-network line of work.
- **Fast weight programmers** {% cite schlag2021linear %} are an even older lineage where one network produces the weights of another, and the linear-attention reduction shows that this is exactly what an attention layer does in disguise. The QKV split is also the read/write/address API of a fast-weight controller.

**Modern Hopfield networks** {% cite ramsauer2021hopfield %} show that the softmax attention update is, up to scaling, the retrieval rule of a Hopfield network. The associated capacity bound is exponential in $d$ conditional on the stored keys being sufficiently well-separated, so the effective in-practice capacity depends on how spread out the keys actually are. **Pérez and collaborators** {% cite perez2019turing %} show that transformers with arbitrary precision and unbounded positional encodings are Turing complete. The arbitrary-precision assumption is unphysical, so this is a theoretical statement about the family rather than a claim about any particular trained model, but it does say that the QKV-plus-softmax primitive is at least as expressive as the standard memory-and-control idealizations the field has been chasing.

### Why not two projectors

Two projections (tying $Q = K$) collapses address-space into content-space. You can only retrieve things by their similarity to themselves, which loses the ability to query for something different from what you are. It also kills asymmetry, because with shared $W_Q = W_K$ the pre-softmax score matrix $X W W^{T} X^{T}$ is symmetric, so token A attending to B differently than B attending to A becomes impossible. Causal language modeling, dependency-style relations, and any computation with directionality depend on that asymmetry. Tied-QK variants exist and work in some settings, but they give up this asymmetry, so three independent projections is the natural choice for an asymmetric, differentiable, content-addressable lookup.

## Why inner product?

We have settled on three projections, but why is the score itself an inner product? Why not Euclidean distance, cosine, an MLP applied to concatenated $(q, k)$, or any of the other plausible similarities?

### Universal similarity measure

Recall that attention's effective similarity is $\exp(x_i^{T} M x_j / \sqrt{d_k})$ with $M = W_Q W_K^{T}$. The natural worry is that "exp of an inner product" might be a restrictive family. Mercer's theorem and the broader kernel-methods literature {% cite scholkopf2002learning %} say it is not. Every positive-definite kernel decomposes as $K(x, y) = \langle \phi(x), \phi(y) \rangle$ in some (possibly infinite-dimensional) feature space. Writing the similarity as an inner product of *learned* features $q = W_Q x$ and $k = W_K x$ is just running this construction in reverse: instead of fixing the kernel and discovering its features implicitly, set up the inner product and let the learned kernel emerge. Performers run this connection the other way and approximate softmax attention through random feature maps {% cite rahimi2007random choromanski2021rethinking %}.

With $d_k$ large enough, $W_Q W_K^{T}$ can express any rank-$d_k$ bilinear similarity on the input space. Some kernels (like RBF in raw input space) have infinite-rank bilinear form, but the relevant question is whether the rank-$d_k$ family is rich enough at the scales we work at, and in practice it is. The inner-product score also lines up directly with the metric-learning view from the earlier post, since picking the kernel and picking the metric are the same problem expressed in two different ways.

## Why softmax?

Softmax has the form $\mathrm{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)$. It does two things at once: it exponentiates, and it normalizes the result to the probability simplex. The two steps deserve separate analysis.

### The exp part: maximum entropy

The exponential drops out of one of the most-used variational arguments in statistical inference. Among all distributions on a discrete set with a fixed expected value of the logits, the maximum-entropy distribution has the form $p_i \propto \exp(\lambda z_i)$. This is Jaynes' principle {% cite jaynes1957information %}, the same derivation that gives the Boltzmann distribution in statistical physics. Exp is the least-committal probability assignment given linear constraints on the logits, and it is uniquely determined by that property.

The functional-equation view says the same thing from a different angle. Exp is the unique continuous solution of $f(a + b) = f(a) f(b)$, which is what gives softmax shift-invariance in the logits, since multiplying every $\exp(z_i)$ by the same constant cancels under normalization. That is why you can subtract $\max_i z_i$ before exponentiating and get a numerically stable implementation, and why attention is invariant to constant shifts in $QK^{T}$. You cannot get this property from any other choice of $f$.

So if you want a least-committal probability assignment under a linear constraint on the logits, exp is what you get.

### The simplex part

The other half, mapping $\mathbb{R}^N$ to the $(N-1)$-simplex by requiring $\alpha_i \geq 0$ and $\sum_i \alpha_i = 1$, is a choice. You can have the exp without the normalization (energy-based attention, where the weights stay as unnormalized similarities), normalization without the exp (linear-attention variants built on positive feature maps {% cite katharopoulos2020transformers %}), or different constraints entirely (sigmoid attention {% cite ramapuram2024sigmoid %}, where each weight is independent in $[0, 1]$). The empirical record is that these alternatives behave measurably differently from softmax.

What does the simplex commit you to? Three things that get conflated:

1. **Boundedness.** $\alpha_i \in [0, 1]$, which is necessary for numerically stable downstream computation.
2. **Comparability across contexts.** $\sum_i \alpha_i = 1$, so a weight of $0.9$ means the same thing whether the context has 3 keys or 1000 keys. Without it, comparing attention patterns across sequence lengths is not even well-defined.
3. **Convex combination.** The output $\sum_i \alpha_i v_i$ lives in the convex hull of the values $\{v_i\}$. The contextualized output is *somewhere among* the value vectors, never outside.

The third property is a choice and one could have chosen a different inductive bias.

## Just won't die

An alternative architecture that wants to keep the dim-preservation of part 1 and the soft-k-NN behavior of part 2 has to make these three choices in some form. Tied-QK gives up directional asymmetry. Non-inner-product scores give up the kernel-methods backbone and the random-features approximations that ride on it. Non-softmax aggregation changes whether the per-head output is a convex combination of the values. Each choice has sensible alternatives. The combination is what makes the architecture stable across modalities and tasks.

# References

{% bibliography --cited %}
