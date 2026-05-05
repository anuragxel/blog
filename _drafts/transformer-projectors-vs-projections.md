---
layout: post
title: "Why won't the transformer die? Part 4: projectors vs projections"
description: Where the trillion parameters live and what's their role.
---

This is part 4 of a four-part series. The first three posts argued for self-attention as a non-parametric, dim-preserving lookup with a learned metric (parts 1 and 2) implemented through three specific design choices (part 3). The k-NN reading covers per-layer aggregation. A trained transformer has hundreds of billions of parameters and thus we need to think about their role. The right resolution in my view is the distinction between *projections* and *projectors*, which is also what makes the architecture compositional.

## Projectors vs projections

Recall the per-layer setup:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\mathrm{SelfAttn}(X) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V$$

The matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are *projectors*: fixed parameters of the model, learned during training and frozen at inference. They contribute roughly $3 \cdot d \cdot d_k$ parameters per head per layer, and once you account for $h$ heads, $L$ layers, output projections $W_O$, and the MLP blocks between attention layers, this is where the parameter count concentrates.

The matrices $Q, K, V$ are *projections*: data-dependent activations you get by applying the projectors to whatever input $X$ shows up. They are recomputed every forward pass and discarded after. The effective similarity between two tokens,

$$\langle q_i, k_j \rangle = x_i^{T} (W_Q W_K^{T}) x_j = x_i^{T} M x_j,$$

is a fixed bilinear form $M = W_Q W_K^{T}$ applied to fresh inputs. The metric $M$ is parametric and the same for every input the model ever sees. The reference set, queries, and values are non-parametric in the sense that they are constructed entirely from the current input.

That separation is what lets attention compose. The same learned metric gets applied to every input, including to the *previous* layer's output, which is itself the result of the same metric structure applied to *that* layer's input, and so on for $L$ layers. The attention parts of a stacked transformer can be read as $L$ soft $k$-NN lookups, each with its own learned metric and head structure, each operating on the reference set produced by the layer below. The MLP blocks between attention layers add parametric nonlinearities that the $k$-NN view says nothing about, and most of the parameter budget goes there rather than into the projectors.

So a trillion-parameter transformer decomposes into:

1. **Per-head, per-layer projectors** $W_{Q,h}^{\ell}, W_{K,h}^{\ell}, W_{V,h}^{\ell}$, encoding $hL$ learned metrics and the corresponding value transforms.
2. **Output projections** $W_O^{\ell}$ that mix per-head outputs back to model dimension at each layer.
3. **MLP blocks** between attention layers, where most of the parameter budget concentrates. These supply the parametric nonlinearities that compose with the soft $k$-NN aggregation.
4. **Embedding and unembedding matrices.**

The $k$-NN view explains the per-layer aggregation operation. The model's stored knowledge sits in the projectors and (mostly) in the MLPs, which the $k$-NN view does not address. Probing the MLP blocks directly {% cite geva2021transformer %} shows that the two feed-forward matrices behave like a static key-value memory of their own, with keys that pattern-match against residual-stream activations and values that read out distributions over the vocabulary. The "non-parametric" label from earlier in the series is about the per-layer aggregation rule, not the model as a whole.

A handful of implications fall out cleanly from this picture.

## RAG is attention with additional external references

Retrieval-augmented generation {% cite lewis2020retrieval %} extends the context window to include relevant documents from a corpus. Under the $k$-NN view, this is just handing the non-parametric estimator more reference points. $k$-NN language models {% cite khandelwal2020generalization %} take this one step further and interpolate the model's softmax with a literal nearest-neighbor lookup over a datastore of training-time hidden states, which is the framing of "soft k-NN over a learned metric" pushed to its logical end. The "RAG vs fine-tuning" debate is roughly "non-parametric estimation with a bigger reference set against parametric fitting on it", and the trade-offs look like the classical ones: bigger reference sets favor non-parametric methods, and small well-targeted training sets favor parametric ones.

## The KV cache is the cached reference set

At autoregressive inference, the $K$ and $V$ projections of past tokens never change once computed, because the projectors $W_K, W_V$ are fixed and the past tokens are fixed. They are cached once and reused for every subsequent generation step. Each new query position recomputes only its own $q, k, v$ and runs one fresh attention lookup against the cached reference set. The memory cost of the cache scales linearly with context length, because it is literally storing the reference set the per-layer $k$-NN runs over.

## Generalized tool calling fits the projector/projection picture

Since $W_Q, W_K, W_V$ are data-invariant, what they encode at training time is the *kind* of similarity that matters, not any particular tuples of (query, key, value). When a model is trained on tool use, the projectors plausibly pick up something like "match a user request to an interface description, then route to the matching argument slots". The specific tool name, parameter schema, and function ABI live in the data-dependent projections at inference time.

That separation is consistent with models calling tools they have not seen at training time, as long as the new tool descriptions live in the context {% cite schick2023toolformer %}. Generalization to unseen tool schemas is real but bumpy and format-sensitive in practice, so the picture explains why transfer happens at all rather than predicting that it always works.

## Test-time compute scaling via layer reuse

Two earlier framings combine here. The projector/projection split says the per-layer projectors $W_Q, W_K, W_V$ are data-invariant, so the same attention operation can be applied to whatever activations it sees, including activations that came out of the same operation a moment earlier. The no-information-bottleneck property says the output of a self-attention layer sits in the same $\mathbb{R}^{N \times d}$ as the input, so feeding the layer's output back into itself is a type-correct thing to do. Together, the architecture supports running the same layer multiple times against an evolving reference set, deepening the computation at test time without any new parameters.

This is the architectural premise behind Universal Transformer {% cite dehghani2019universal %} and PonderNet {% cite banino2021pondernet %}, which tie attention and MLP weights across depth and learn a halting policy that decides when to stop iterating and emit a prediction. Looped Transformers {% cite giannou2023looped %} use a tied stack as a programmable computational substrate. Recent recurrent-depth setups scale the same trick to language-model sizes and use it as a test-time compute knob {% cite geiping2025scaling %}. What the data-invariant projectors buy you here is that the lookup operation has the same input-output type at every step, so the loop is well-defined without any sample-specific bookkeeping. RNNs and SSMs can loop too, but the loop semantics are different: their state is what evolves, whereas in a looped transformer it is the reference set that evolves while the lookup operation stays fixed.

There is one caveat worth flagging. The MLP block applied between attention layers is also data-invariant in its weights, but it implements a fixed nonlinear transformation on each token independently, with no reference-set lookup involved. The attention part loops cleanly because the operation it implements is a lookup against a reference set that itself evolves as the layer iterates. The MLP part loops less cleanly because it applies the same token-wise function to whatever sits in the residual stream, which does not refine in the same way. Iterated-depth setups in the literature typically need a per-iteration timestep embedding to disambiguate which loop step the tied layer is running {% cite dehghani2019universal %}. I conjecture that the MLP block also needs more capacity to compensate for being applied repeatedly, since it is doing more per-parameter work each iteration, though I do not know of a clean empirical result that confirms this. Either way, this is part of the architectural cost of sharing weights across depth.

## Refuses to die

The per-layer operation does not care where the reference set comes from, as long as the projectors and the projections are consistently typed. An alternative architecture that wants to support all four of these affordances has to give up either the data-invariance of the projectors or the fixed-typed lookup against a reference set.

Most of what makes trained transformers useful at deployment time (long-context inference, retrieval, tool use, scratchpad reasoning, test-time compute) rides on properties this lens makes visible, and a replacement architecture has to either replicate them or cope without them.

# References

{% bibliography --cited %}
