---
layout: post
title: "Why won't the transformer die? Part 4: projectors vs projections"
description: What about the trillion parameters?
---
The first three posts argued for self-attention as a somewhat-non-parametric, dim-preserving lookup table with a learned metric  implemented through three specific design choices. However, a trained transformer has hundreds of billions of parameters and thus we need to think about their role. The right resolution in my view is the distinction between *projections* and *projectors*, which is also what makes the architecture compositional.

## Projectors vs projections

Recall the per-layer setup:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

$$\mathrm{SelfAttn}(X) = \mathrm{softmax}\!\left( \frac{Q K^{T}}{\sqrt{d_k}} \right) V$$

The matrices $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are *projectors*: fixed parameters of the model, learned during training and frozen at inference. They contribute roughly $3 \cdot d \cdot d_k$ parameters per head per layer, and once you account for $h$ heads, $L$ layers, output projections $W_O$, and the MLP blocks between attention layers, this is where the parameter count concentrates.

The matrices $Q, K, V$ are *projections*: data-dependent activations you get by applying the projectors to whatever input $X$ shows up. They are computed from the input, with past keys and values retained during cached decoding. The effective similarity between two tokens,

$$\langle q_i, k_j \rangle = x_i^{T} (W_Q W_K^{T}) x_j = x_i^{T} M x_j,$$

is a fixed form $M = W_Q W_K^{T}$. The metric $M$ is parametric and the same for every input the model ever sees. The reference set, queries, and values are non-parametric in the sense that they are constructed entirely from the current input.

The separation enables composition. The attention parts of a stacked transformer can be read as $L$ soft $k$-NN lookups, each with its own learned metric and head structure, each operating on the reference set produced by the layer below. The MLP blocks between attention layers add parametric nonlinearities that the $k$-NN view ignores, and most of the parameter budget is concentrated there rather than the projectors.

So a trillion-parameter transformer decomposes into:

1. **Per-head, per-layer projectors** $W_{Q,h}^{\ell}, W_{K,h}^{\ell}, W_{V,h}^{\ell}$, encoding $hL$ learned metrics and the corresponding value transforms.
2. **Output projections** $W_O^{\ell}$ that mix per-head outputs back to model dimension at each layer.
3. **MLP blocks** between attention layers, the parametric nonlinearities that compose with the soft $k$-NN aggregation. These blocks typically account for most of a transformer's parameters.
4. **Embedding and unembedding matrices.**

The $k$-NN view explains the per-layer aggregation operation. MLPs also store learned associations, which the $k$-NN view does not address. Probing the MLP blocks directly {% cite geva2021transformer %} shows that the two feed-forward matrices behave like a static key-value memory of their own.

A handful of implications fall out cleanly from this picture.

## RAG is attention with additional external references

Retrieval-augmented generation {% cite lewis2020retrieval %} augments the context to include relevant documents from a corpus. Under the $k$-NN view, this is just handing the non-parametric estimator more reference points. $k$-NN language models {% cite khandelwal2020generalization %} take this one step further and interpolate the model's softmax with a literal nearest-neighbor lookup over a datastore of training-time hidden states, which is the framing of "soft k-NN over a learned metric" pushed to its logical end. The "RAG vs fine-tuning" debate is roughly "non-parametric estimation with a bigger reference set against parametric fitting on it". The trade-offs resemble classical ones, although data relevance and the task matter alongside reference-set size.

## The KV cache is the cached reference set

At autoregressive inference, the $K$ and $V$ projections of past tokens never change once computed, because the projectors $W_K, W_V$ are fixed, the past tokens are fixed, and causal masking prevents past positions from attending to newly appended tokens. They are cached once and reused for every subsequent generation step. Each new query position recomputes only its own $q, k, v$ and runs one fresh attention lookup against the cached reference set. The memory cost of the cache scales linearly with context length, because it is literally storing the reference set the per-layer $k$-NN runs over.

## Generalized tool calling fits the projector/projection picture

Since $W_Q, W_K, W_V$ are data-invariant, what they encode at training time is the *notion* of similarity, not any particular tuples of (query, key, value). Thus, when a model is trained on tool use, the projectors plausibly pick up something like "match a user request to an interface description, then route to the matching slots". The specific tool name, parameter schema, and function ABI live in the data-dependent projections (available reference set) at inference time. That separation is consistent with models calling tools they have not seen at training time, as long as the new tool descriptions live in the context.

## Test-time compute scaling via layer reuse

Two earlier framings combine here. The projector/projection split says the per-layer projectors $W_Q, W_K, W_V$ are data-invariant, so the same attention operation can be applied to whatever activations it sees, including activations that came out of the same operation a moment earlier. The no-information-bottleneck property says the output of a self-attention layer sits in the same $\mathbb{R}^{N \times d}$ as the input, so feeding the layer's output back into itself is a type-correct thing to do (i.e. obeys the I/O contract semantics of the operation). In this type-based view, the layer accepts and returns tokens of `Type[Concept]`, so its outputs can be fed back in for further refinement. These are conceptual types; useful refinement still depends on training. Together, the architecture supports running the same layer multiple times against an evolving reference set, deepening the computation at test time without any new parameters. 

This is the architectural premise behind Universal Transformer {% cite dehghani2019universal %}, which ties attention and MLP weights across depth and includes an adaptive halting mechanism. PonderNet {% cite banino2021pondernet %} explores the related question of learning when to halt recurrent computation. Looped Transformers {% cite giannou2023looped %} use a tied stack as a programmable computational substrate. Recent recurrent-depth setups scale the same trick to language-model sizes and use it as a test-time compute knob {% cite geiping2025scaling %}. What the data-invariant projectors buy you here is that the lookup operation has the same input-output type at every step, so the loop is well-defined without any sample-specific bookkeeping. Here, recurrence updates the token representations, and therefore the reference set, across depth, rather than summarizing successive tokens into a recurrent state.

There is one caveat here. The MLP block applied between attention layers is also data-invariant, but it implements a fixed nonlinear transformation on each token independently. Iterated-depth setups in the literature sometimes use a per-iteration timestep embedding to disambiguate which loop step the tied layer is running {% cite dehghani2019universal %}. I conjecture that the MLP block also needs more capacity to compensate for being applied repeatedly, since it is doing more per-parameter work each iteration, though I do not know of a clean empirical result that confirms this. This may be one architectural cost of sharing weights across depth.

## Chaining multimodal models

We can then think about how pretrained models can be chained across modalities: an encoder supplies tokens, a learned adapter maps them to the language model's reference set manifold, and the language model processes the resulting tokens. Consider a vision encoder connected through a linear adapter, as in LLaVA {% cite liu2023visual %}. In our row-vector convention, an image $I$ becomes a sequence of visual features $Z = E_{\mathrm{vision}}(I) \in \mathbb{R}^{N_v \times d_v}$, and the adapter maps these to:

$$H_v = Z W_A, \qquad W_A \in \mathbb{R}^{d_v \times d}.$$

These visual tokens enter the input sequence alongside text embeddings. From there, the language model applies its usual attention and MLP blocks.

In the lookup picture developed here, the adapter supplies additional reference points. The adapter $W_A$ is a *projector*, fixed at inference; its outputs $H_v$ are image-dependent *projections*. The interface lets a pretrained model supply activations that another model can use as its reference set. 

Matching dimensions is one aspect; training must make the encoder's features useful to the receiving model. For example, original LLaVA first trains the adapter with both pretrained models frozen, then tunes the adapter and language model on visual instructions while keeping the vision encoder frozen {% cite liu2023visual %}. In the same type-based view, the vision transformer produces `Type[Visual]` tokens, and the adapter maps them to `Type[LinguoVisual]`. With alignment training, the LLM learns to accept both language and adapted visual tokens, broadening its input type from `Type[Language]` to `Type[Union[Language, LinguoVisual]]`. My interpretation is that the common token interface makes composition architecturally straightforward, while learned alignment makes it useful.

## Refuses to die

The per-layer operation does not care where the reference set comes from, as long as the projectors and the projections are consistently employed in the "typing sense". Most of what makes trained transformers useful at deployment time (long-context inference, retrieval, tool use, scratchpad reasoning, test-time compute) rides on properties this lens makes visible, and a replacement architecture would need to support similar capabilities, potentially through different mechanisms.

# References

{% bibliography --cited %}
