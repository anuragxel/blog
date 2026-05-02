---
layout: post
title: Why is killing Transformers Hard?
description: Exploring One General Inductive Bias of Self-Attention.
---

A friend of mine, very smart but not a machine learning person asked me recently, what's going on with the machine learning architecture world, is Mamba looking good compared to transformers?

I replied, "No, it has an information bottleneck embedded in its state-space operation, which is not good to have. Transformers don't have any such bottlenecks."

He responded, "I suppose, but it's hard to tell what the real reasons are for why it works. You guys sometimes add 7 layers, sometimes 8 and god knows why one works and the other doesn't."

I replied, "Yes, but transformers are uniquely beautiful and special. I don't have the best way to communicate it, but let me try, by drawing on existing literature and some of my own connections and thoughts."

I hated my own hand-waviness and decided to systematize what I know.

## Motivation

I'm trying to group and systematize all the intuitions I've picked up about the transformer architecture and the self-attention and cross-attention mechanisms in one place. There are various ways to look at this architecture, and they in turn follow different lineages of machine learning history and research. I'll use some math to explain the intuitions, and compare with classical algorithms whose properties are well studied. Sadly, I'm not a theorist, so I find it difficult to actually derive proofs of what's possible but I do think the exploration itself has value.

## Definition of Self-Attention and Cross-Attention

Let $X \in \mathbb{R}^{N \times d}$ be a set of $N$ tokens, each represented as a $d$-dimensional vector. **Self-attention** projects $X$ into queries, keys, and values via three learned matrices (or projectors) $W_Q, W_K \in \mathbb{R}^{d \times d_k}$ and $W_V \in \mathbb{R}^{d \times d_v}$:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

The output is then a weighted average of the values, where weights come from a row-wise softmax over scaled query–key dot products:

$$\mathrm{SelfAttn}(X) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V \in \mathbb{R}^{N \times d_v}$$

The scale $\sqrt{d_k}$ keeps the dot products from growing too large in magnitude as $d_k$ increases, which would otherwise push the softmax into saturated regions with vanishing gradients.

Note the use of *set* rather than *sequence* (as the language folks would say). The operation is permutation-equivariant and thus makes no assumption about the structure of the input modality.

**Cross-attention** is the same operation, but queries come from one set and keys/values from another. Given a target set $X \in \mathbb{R}^{N \times d}$ and a source set $Y \in \mathbb{R}^{M \times d}$,

$$Q = X W_Q, \quad K = Y W_K, \quad V = Y W_V$$

$$\mathrm{CrossAttn}(X, Y) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V \in \mathbb{R}^{N \times d_v}$$

Each of the $N$ target tokens attends over all $M$ source tokens. Self-attention is the special case $Y = X$ (so $M = N$).

# What's cool about this specific operation and its sister?

## No information bottlenecks

Okay, we wrote it down. A good start. There are many questions here, but the biggest one is: why is almost any concept or modality so easy to convert into a transformer?

Consider a standard ViT-B/16 {% cite dosovitskiy2021image %}. Each $16 \times 16$ RGB patch is $16 \times 16 \times 3 = 768$ floats. We linearly project each one to a $d = 768$-dimensional token, giving a set of $N$ tokens to feed into a stack of self-attention operations {% cite vaswani2017attention %}. Consider just the first one. Given $N$ input tokens, it outputs $N$ tokens of the same dimension. For now, imagine the learned parameters are optimally set. We won't worry about how one would actually train such a method.

Stack the $N$ queries as rows of a matrix, and the $N$ keys the same way. We compute an inner product between every query–key pair, then take a row-wise softmax. That gives us $N$ coefficients per row, which we use to weight the $N$ value vectors computed from $X$.

Here is the key observation. At no point does the architecture reduce the dimensionality of any token. The output of self-attention sits in $\mathbb{R}^{N \times d}$, exactly like the input. There is no architectural bottleneck through which all $N$ tokens must be projected to lower dimensional space or to few number of vectors (i.e. tokens). Contrast this with a state-space model like Mamba {% cite gu2024mamba %}, where every token must pass through a fixed-size hidden state $h_t \in \mathbb{R}^{d_h}$. With the right weights, $\mathrm{SelfAttn}(X)$ can losslessly carry every coordinate of $X$ forward, since one-hot attention with $W_V = I$ would just permute the input. This can't be said about Mamba.

## Data-Dependent Projection as "Soft KNN over Learned Metric Space"

The projection weights $Q, K, V$ are themselves functions of the input, which actually has very deep connections to metric learning and classical learning literature. It is the second piece of what makes self-attention so general. For this piece I like to think of self-attention/cross-attention as doing soft-KNN over a learned kernel/metric space.

To see why, look at the score we feed into softmax. For two tokens $x_i, x_j \in \mathbb{R}^d$, the unnormalized similarity expands as

$$\frac{q_i^{T} k_j}{\sqrt{d_k}} = \frac{x_i^{T} W_Q W_K^{T} x_j}{\sqrt{d_k}} = \frac{x_i^{T} M\, x_j}{\sqrt{d_k}}$$

where $M = W_Q W_K^{T} \in \mathbb{R}^{d \times d}$ is one learned matrix. The entire $QK^{T}$ machinery collapses to a single bilinear form on the input space. The factorization $M = W_Q W_K^{T}$ buys us exactly one thing over learning $M$ directly. With $d_k < d$, it constrains $M$ to rank at most $d_k$.

This is the setup of classical *metric learning* {% cite weinberger2009distance %}. A bilinear form $\langle x_i, x_j \rangle_M = x_i^{T} M x_j$ defines a similarity (an inner product, when $M$ is symmetric positive-semidefinite). Mahalanobis metric learning and its many variants all amount to picking such an $M$ so that semantically similar points score high. Inner product in the raw input space is meaningless. Inner product in the $M$-warped space is whatever similarity the training objective shaped it to be.

Once we have a learned similarity, exponentiating and row-normalizing gives us

$$\mathrm{softmax}\!\left( \frac{x_i^{T} M x_j}{\sqrt{d_k}} \right)_j$$

which is exactly a Nadaraya-Watson estimator {% cite nadaraya1964estimating watson1964smooth %} with a learned kernel $k(x_i, x_j) = \exp(x_i^{T} M x_j / \sqrt{d_k})$ {% cite tsai2019transformer %}. Read the whole self-attention layer that way and the analogy is:

- Queries $q_i$ are test points.
- Keys $k_j$ are reference points.
- Values $v_j = W_V x_j$ are reference labels.
- $\mathrm{softmax}(q_i^{T} k_j / \sqrt{d_k})$ is a soft membership function. Instead of hard top-$k$ selection, you get a probability distribution over neighbors weighted by learned similarity.
- The output $\sum_j \alpha_{ij} v_j$ is the soft $k$-NN prediction, a weighted average of neighbor labels.

So the architecture is parameterizing soft $k$-NN regression, but with three distinct knobs that classical $k$-NN does not have:

1. **The metric, via the $Q, K$ projections.** Classical kernel methods commit to a kernel up front (RBF, polynomial, Matern), and the choice encodes a strong prior about what "similar" means. Self-attention learns the similarity from data.
2. **The label space, via the $V$ projection.** The "labels" attached to each reference point are themselves learned functions of the input. A token's $v_j$ is its "label" for the $k$-NN.
3. **The reference set, via the context window.** Classical $k$-NN draws references from a fixed training set. Self-attention draws them from whatever happens to be in the current input. Every input set $X$ gets its own reference pool, on the fly.

Three knobs, all learned, all data-dependent. That is why the same operation works for image patches, text tokens, audio frames, point clouds {% cite zhao2021point %}, and graph nodes {% cite velickovic2018graph %}. Whatever the modality, the layer asks the same question: which of these other tokens are like me, in the metric I have learned to care about, and what should I copy from them?

### Some implications from classical learning literature

Most learning methods are defined by the bias they bake in. Linear regression assumes the target is linear in features. Polynomial regression assumes it lives in a fixed polynomial family. Even a multilayer perceptron with fixed depth and width parameterizes a specific function class, universal approximation theorems notwithstanding {% cite hornik1991approximation %}. Each of these wins when its prior matches reality and loses when it does not. That is the classical bias-variance tradeoff at the level of function families, before any data is ever seen.

$k$-NN sits at the opposite end of the spectrum. It is the textbook *non-parametric* method {% cite devroye1996probabilistic %}, meaning the hypothesis class grows with the data and never commits to any closed-form structure. There is no assumption that the target is linear, smooth in any particular basis, or expressible by any finite parameterization. The only inductive bias is locality in the chosen metric. Bias and variance trade off through $k$ and the metric, not through a hand-picked function family. Almost every other method you reach for encodes some particular bias about what the function should look like, and $k$-NN simply does not.

This assumption-light rule has strong asymptotic guarantees, starting with the **Cover-Hart theorem** {% cite cover1967nearest %}. For a $c$-class problem, as the reference set size $n \to \infty$, the error rate of the $1$-nearest-neighbor classifier satisfies

$$R^* \le R_{1\text{-NN}} \le R^*\!\left(2 - \frac{c}{c-1}\, R^*\right) \le 2 R^*$$

where $R^*$ is the Bayes (irreducible) error. A parameter-free rule, $1$-NN, sits within a factor of two of the optimal classifier in the limit of infinite reference data. Hard to beat for something that does no learning at all.

The factor-of-two slack is the worst case for $k = 1$, and it disappears once we let $k$ grow. Stone's theorem {% cite stone1977consistent %} shows that if $k \to \infty$ and $k/n \to 0$ as $n \to \infty$, the $k$-NN classifier is *universally consistent*. Its error converges to the Bayes rate $R^*$ for any data distribution, with no constant factor left over. So the price for committing to no functional form is, in the limit, nothing at all.

The catch is that both bounds are stated in the data-generating distribution's natural metric. With Euclidean distance on raw pixels, the "nearest neighbor" of a cat picture is often a different cat-shaped patch of color and not another cat. Classical $k$-NN binds against a meaningless notion of nearness, and degrades catastrophically in high dimensions, where pairwise distances concentrate and the very notion of a "nearest" point loses its meaning {% cite beyer1999nearest %}. The guarantees are real but contingent on having a metric worth using, and classical $k$-NN does not provide one.

This is the gap self-attention closes. Training $M = W_Q W_K^{T}$ end-to-end picks the inner product under which "nearest by $M$" correlates with "same downstream label", which is the precondition Cover-Hart and Stone need. The classical statements do not transfer formally to attention (the reference set is in-context rather than iid, the metric is learned jointly with the labels, the aggregation is soft rather than top-$k$), but they tell you what kind of learner attention is approximating: a non-parametric one whose worst case is bounded and whose limit case is Bayes-optimal.

The soft version also sidesteps the one remaining choice classical $k$-NN forces on you, namely $k$ itself. Every query attends to all $N$ tokens, so $k = N$ structurally, and the softmax coefficients implicitly handle the effective neighborhood size. Tokens with low similarity get exponentially small weight and contribute almost nothing, while a few high-similarity tokens dominate the average. The "how many neighbors to trust" knob becomes a continuous, per-query, learned function of the inputs rather than a global hyperparameter set by hand.

Put it all together and self-attention parameterizes only the metric and the labels, leaving the aggregation rule itself non-parametric, while approximating the universal-consistency story of $k$-NN and routing around its two classical failure modes (a bad metric and a hand-picked $k$). The bias lives in the metric and in the convex-combination aggregation, not in any closed-form function family. Stacking layers compounds the operation but does not change what kind of operation it is.

### Back to Mamba: non-parametric vs parametric estimation

There is a clean way to phrase the friend's question from the top of the post. The transformer-vs-SSM argument is the same non-parametric versus parametric estimation argument statisticians have been having for half a century {% cite wasserman2006nonparametric %}.

Self-attention is non-parametric in the function it represents. The reference set is whatever sits in the context, the metric and labels are learned, and no fixed-capacity bottleneck stands between the input and the output. Capacity scales with the data. State-space models like S4 {% cite gu2022efficiently %} and Mamba {% cite gu2024mamba %}, and linear-attention variants that recast attention as a recurrence with fixed state {% cite katharopoulos2020transformers %}, are parametric in exactly the place where attention is not. Every token in the past has to be compressed into a fixed-size hidden state $h_t \in \mathbb{R}^{d_h}$ before the next token sees it. Selective SSMs make the recurrence input-dependent, but the state stays a fixed-capacity bottleneck.

The classical tradeoff falls out cleanly. Parametric models have constant per-step cost and bounded capacity, and that bound is the functional-form bias I have been complaining about all along. Here it shows up as a literal information bottleneck, where anything that does not fit through $h_t$ is gone for good. Non-parametric models have no such ceiling and grow capacity with the reference set, paying for it in compute and memory that scale with $N$. There is no free lunch, only a choice about where the cost lives.

The recurrence $h_t = f(h_{t-1}, x_t)$ is intrinsically ordered, so $h_t$ summarizes one particular causal traversal of the input. There is no version of Mamba or linear attention you can run on a set and get a permutation-equivariant output. Self-attention is permutation-equivariant by default, and you opt into ordering through positional encodings only when you actually want it. For modalities where order is not part of the data, like a set of images for novel view synthesis, a 3D point cloud, or the nodes of a graph, the recurrent route bakes in an additional inductive bias.

## Why 3 Projectors? QKV as the turing tape or "content-addressable array lookups"

Why three projections? Why not two (tied $Q = K$) or four? Most people do not ask, because the architecture just works, but each of the three projections does distinct work that the others cannot cover.

### The connection

The cleanest framing is that you need read, write, and address modes to implement a content-addressable memory, and three projections is the minimum-viable signature for implementing memory in the turing-sense. To make that concrete, picture an ordinary associative array, where you write `memory[key] = value` and read it back by handing the same key in. A classical lookup needs an exact key match. A *content-addressable* lookup relaxes that to "return the value whose key is most similar to the query" under some chosen similarity. Soft attention is the differentiable version of the same retrieval, returning a weighted average of values where the weight on the $i$-th entry is $\exp(\langle q, k_i \rangle / \sqrt{d_k})$ after row-normalization across the keys.

In *self*-attention, every token plays all three roles at once over the same underlying vector. Through $W_Q$, the token forms its query, namely what it is asking the rest of the input about. Through $W_K$, it advertises itself as a catalog entry, a key under which others can look it up when their queries line up. Through $W_V$, it exposes its payload, the content that gets returned when it is matched. One vector, three independently learned projections, three separate roles in the same lookup. The memory bank itself is not stored in weights. It is built on the fly out of the current input and torn down at the end of the layer.

Read self-attention this way and the layer is doing one step of "every token issues a query against an associative memory built out of everyone else, and overwrites itself with what it finds". The Turing-tape analogy is that K is the addressing of the tape cells, V is what the cells contain, and Q is the head's read pattern. The transformer is shaped like a differentiable read step on a soft, content-addressable tape.

### From classical theory of computation

Differentiable content-addressable memory has been a recurring research goal for decades. The QKV split is the formulation that ended up in the transformer.

- **Hopfield's associative memory** {% cite hopfield1982neural %} stores patterns as fixed points of a network and retrieves them by partial-pattern similarity to a query. A pattern goes in, the closest stored pattern comes out. The query and the stored items live in the same space, but the read-out is a separately recoverable item. That is already a Q/K/V picture in spirit.
- **Neural Turing Machines** {% cite graves2014neural %} and the **Differentiable Neural Computer** {% cite graves2016hybrid %} bolted a Turing-machine-style external memory onto a recurrent controller, with separate read heads, write heads, and content-based addressing as the lookup mechanism. The "find by content, return separately stored value" split shows up there exactly as Q/K/V does in attention.
- **End-to-End Memory Networks** {% cite sukhbaatar2015end %} and **Key-Value Memory Networks** {% cite miller2016key %} cleaned this up by separating out the key vector used for addressing from the value vector that gets returned. By the time the transformer arrived, the K/V split was already standard vocabulary in the memory-network line of work.

**Modern Hopfield networks** {% cite ramsauer2021hopfield %} prove that the softmax attention update is, up to scaling, exactly the retrieval rule of a Hopfield network. The associated capacity bound is exponential in $d$, but conditional on the stored keys being sufficiently well-separated, so the effective in-practice capacity depends on how spread out the keys actually are. Every attention layer is therefore an associative memory with at most that capacity, and QKV is its read interface. Building on that primitive, **Pérez and collaborators** {% cite perez2019turing %} show that the transformer with arbitrary precision and unbounded positional encodings is Turing complete: given enough depth and context, three projections plus softmax can simulate any algorithm a Turing machine can.

### Why not two, why not four

Two projections (tying $Q = K$) collapses address-space into content-space. You can only retrieve things by their similarity to themselves, which loses the ability to query for something different from what you are. It also kills asymmetry, because with shared $W_Q = W_K$ the pre-softmax score matrix $X W W^{T} X^{T}$ is symmetric, so token A attending to B differently than B attending to A becomes impossible. Causal language modeling, dependency-style relations, and any computation with directionality depend on that asymmetry. Two projections is a strict expressivity loss.

Going to four does not buy anything either. You might think you need separate read-value and write-value channels, but the residual stream is already the write channel, threading the output of every attention block back into the next layer's input. A token's "write" is whatever it adds to its own residual, and that infrastructure exists outside the attention operation itself. So three is the minimum for an asymmetric, differentiable, content-addressable lookup, and the architecture supplies the fourth implicitly through the residual.

## Why Inner Product?

We have settled on three projections, but why is the score itself an inner product? Why not Euclidean distance, cosine, an MLP applied to concatenated $(q, k)$, or any of the other plausible similarities?

### Kernel-trick strikes back: Universal similarity measure

Recall from the soft $k$-NN section that attention's effective similarity is $\exp(x_i^{T} M x_j / \sqrt{d_k})$ with $M = W_Q W_K^{T}$. The natural worry is that "exp of an inner product" might be a restrictive family. Mercer's theorem and the broader kernel-methods literature {% cite scholkopf2002learning %} say it is not. Every positive-definite kernel decomposes as $K(x, y) = \langle \phi(x), \phi(y) \rangle$ in some (possibly infinite-dimensional) feature space. Writing the similarity as an inner product of *learned* features $q = W_Q x$ and $k = W_K x$ is just running this construction in reverse: instead of fixing the kernel and discovering its features implicitly, set up the inner product and let the learned kernel emerge.

With $d_k$ large enough, $W_Q W_K^{T}$ can express any rank-$d_k$ bilinear similarity on the input space, which already covers every kernel worth caring about at finite data scales. The inner-product score also lines up directly with the metric-learning view from earlier, since picking the kernel and picking the metric are the same problem expressed two different ways.

### The RBF answer

Under LayerNorm, or any other step that controls $\ell_2$ norms, the queries and keys come out with roughly constant magnitude. In that regime the exponentiated dot product *is* a Gaussian / RBF kernel. Expanding,

$$\|q - k\|^2 = \|q\|^2 + \|k\|^2 - 2 \langle q, k \rangle,$$

so if $\|q\|$ and $\|k\|$ are pinned, then $\exp(\langle q, k \rangle / \sqrt{d_k})$ and $\exp(-\|q - k\|^2 / (2\sigma^2))$ differ only by a constant that drops out under row-wise softmax. Normalized dot-product attention is implicitly RBF-kernel attention, and RBF kernels are universal approximators on compact sets.

This is also why random Fourier features {% cite rahimi2007random %} and the Performer line {% cite choromanski2021rethinking %} are useful reference points. Random features approximate any shift-invariant kernel by an inner product of finite-dimensional features, and Performer runs that idea in the other direction to approximate softmax attention as a linear map in feature space. Both only work because the underlying score is an inner product to begin with.

### Tying back

The earlier soft $k$-NN section identified attention as kernel regression with a learned kernel. This section says the kernel is parameterized as exp of an inner product not because that family is special, but because (a) Mercer guarantees inner-product-of-learned-features covers every kernel we could want and (b) under standard normalization it coincides with the universal RBF family.

## Why Softmax?

Softmax has the form $\mathrm{softmax}(z)_i = \exp(z_i) / \sum_j \exp(z_j)$. It does two things at once: it exponentiates, and it normalizes the result to the probability simplex. The two steps deserve separate analysis, because one is principled and the other is a real bias.

### The exp part: maximum entropy

The exponential drops out of one of the most-used variational arguments in statistical inference. Among all distributions on a discrete set with a fixed expected value of the logits, the maximum-entropy distribution has the form $p_i \propto \exp(\lambda z_i)$. This is Jaynes' principle {% cite jaynes1957information %}, the same derivation that gives the Boltzmann distribution in statistical physics. Exp is the least-committal probability assignment given linear constraints on the logits, and it is uniquely determined by that property.

The functional-equation view says the same thing from a different angle. Exp is the unique continuous solution of $f(a + b) = f(a) f(b)$, which is what gives softmax shift-invariance in the logits, since multiplying every $\exp(z_i)$ by the same constant cancels under normalization. That is why you can subtract $\max_i z_i$ before exponentiating and get a numerically stable implementation, and why attention is invariant to constant shifts in $QK^{T}$. You cannot get this property from any other choice of $f$.

So the exp half of softmax is forced by the maximum-entropy argument.

### The simplex part: a real inductive bias

The other half, mapping $\mathbb{R}^N$ to the $(N-1)$-simplex by requiring $\alpha_i \geq 0$ and $\sum_i \alpha_i = 1$, is a choice. You can have the exp without the normalization (energy-based attention, where the weights stay as unnormalized similarities), normalization without the exp (linear-attention variants built on positive feature maps {% cite katharopoulos2020transformers %}), or different constraints entirely (sigmoid attention {% cite ramapuram2024sigmoid %}, where each weight is independent in $[0, 1]$). The empirical record is that these alternatives behave measurably differently from softmax, so the simplex normalization counts as a separate commitment that the choice of exp does not force on you.

What does the simplex commit you to? Three things that get conflated:

1. **Boundedness.** $\alpha_i \in [0, 1]$, which is necessary for numerically stable downstream computation.
2. **Comparability across contexts.** $\sum_i \alpha_i = 1$, so a weight of $0.9$ means the same thing whether the context has 3 keys or 1000 keys. Without it, comparing attention patterns across sequence lengths is not even well-defined.
3. **Convex combination.** The output $\sum_i \alpha_i v_i$ lives in the convex hull of the values $\{v_i\}$. The contextualized output is *somewhere among* the value vectors, never outside.

The third property is the substantive one. It says formally that within a single attention head the output is interpolative among the values, never extrapolative. A token whose value ought to express "not X" cannot, in one head, produce an output outside the convex hull of the $v_i$'s it has access to. Attention layers handle negation and out-of-context phenomena by routing them through the $V$ projection (which can flip signs) and the residual stream (which adds the head's output back into the token). That is the architecture working *around* the convex-combination constraint, not *with* it.

This is a substantive inductive bias, and it is exactly the formal version of the distributional theory of meaning {% cite harris1954distributional %}: a token's contextualized representation is built only from the things sitting next to it. The simplex normalization is the architectural commitment to that theory. It is also the only inductive bias classical $k$-NN carries, namely predict from nearby labels in the chosen metric. The convex-hull part comes from the aggregation choice rather than the bias itself, since any weighted averaging over neighbors ($k$-NN regression, Nadaraya-Watson, softmax attention) lands the output inside the convex hull of those neighbor labels by construction. Self-attention inherits both the locality bias and the weighted-averaging aggregation, with the metric and the labels learned end-to-end.

## Other interesting ideas from our framing

### The unreasonable effectiveness of linear probe or k-NN probes after pretraining.

Pretraining is metric learning. Every forward pass through a stacked self-attention model is one big soft $k$-NN over a learned metric, so any training loss that updates $W_Q$ and $W_K$ is also optimizing the inner product that decides which tokens score high together. The objective does not have to be a contrastive metric-learning loss explicitly. Masked language modeling, masked image modeling, and contrastive image-text losses all train the same metric, just with different supervision signals.

Consider DINO {% cite caron2021emerging %} and DINOv2 {% cite oquab2024dinov2 %}, or any image/video foundation model. Why are their features so general? You did soft $k$-NN over an astonishingly huge reference set, and the network learned the metric in which that $k$-NN works. Both report $k$-NN classification accuracy on frozen features as a flagship evaluation alongside linear probing, on the basis that a well-trained representation should already place same-class examples near each other in the learned space. CLIP {% cite radford2021learning %} works the same way at inference time, scoring an image embedding against a set of text embeddings with a single dot product, which is the cross-attention score with hard top-1 selection.

This is also why frozen features plus a linear probe generalizes so well across downstream tasks. A linear classifier on top of inner-product features is a one-shot $k$-NN with class prototypes as the keys. The pretraining stage already did the metric-learning work. The probe only has to pick the directions in that metric space that correspond to the downstream labels.

### Multi-heads from a new metric-learning viewpoint

The standard motivation for multi-head attention is "different heads attend to different things", as in syntactic heads, positional heads, coreference heads. That is descriptively accurate but does not explain why the architecture needs multiple heads rather than one larger head with the same total parameter count.

Softmax is a soft argmax over the keys. With temperature controlled by $\sqrt{d_k}$, a single head concentrates its weight on whichever key is closest to the query in the learned metric $M = W_Q W_K^{T}$, leaving the rest of the simplex mass to the surrounding keys. The output of one head is therefore close to a single weighted nearest-neighbor lookup in one metric space, and it can express only *unimodal* relevance: one peak, falling off in all directions away from the argmax.

The moment a task requires picking out two unrelated things at once (attend to the subject *and* the object, look back $\delta_1$ tokens *and* $\delta_2$ tokens, match shape *and* texture), one metric is not enough. You cannot pick a single bilinear form $M$ so that two separated regions both come out high relative to everything else, because the soft-argmax of one metric concentrates on its one mode. Multi-head attention learns $h$ different metrics $M_h = W_{Q,h} W_{K,h}^{T}$ in parallel and concatenates the per-head outputs. Each head is its own soft-argmax in its own learned subspace, and the concatenation is the union of $h$ separate $k$-NN lookups, each in a metric tuned to a different relevance pattern.

The number of heads is best read as the number of distinct modes of relevance the task needs to express within a single layer. Subspace metric learning in the classical sense {% cite weinberger2009distance %} did this by hand, picking a few different metrics for a few different aspects of the data. Multi-head attention does it end-to-end, with the $h$ metrics learned jointly with the rest of the model.


# References

{% bibliography --cited %}
