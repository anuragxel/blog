---
layout: post
title: "Scaling model (pre)-training. Part 1: An opinionated intro. to visual self-supervised learning"
description: Explaining SimCLR, MAE/SimMIM, and DINO/SimDINO SSL families.
---

I'm in the final year of my PhD, and I want to write down some of the things I learned along the way that did not belong in a paper. This series is about the mental models I use to understand pre-training and distributing it across devices.

This is part 1 of a four-part series on scaling up model (pre-)training. Supervised pretraining on ImageNet was the default recipe for a while: train a classifier on a million labeled images, chop off the head, and fine-tune the backbone on your task. Self-supervised learning (SSL) instead creates a proxy task so that we can train a backbone on a much larger number of images without the million labels by manufacturing the supervision signal from the images themselves.

I will walk through three influential SSL families in vision: contrastive learning (SimCLR), masked image modeling (MAE and SimMIM), and self-distillation (DINO and its recent simplification, SimDINO). I may look into video SSL later, but I feel this is a good illustrative task for understanding how to scale vision and robotics models in terms of data and compute.

## Prehistory: pretext tasks

The first wave of visual SSL manufactured labels through hand-designed puzzles. For example, a model could predict the relative position of two patches cropped from the same image {% cite doersch2015unsupervised %}, solve a jigsaw of shuffled patches {% cite noroozi2016unsupervised %}, colorize a grayscale image {% cite zhang2016colorful %}, predict which of four rotations was applied {% cite gidaris2018unsupervised %}, or inpaint a masked region {% cite pathak2016context %}.

The failure mode of this era was *proxy mismatch*. The network is free to solve the pretext task with shortcuts, and the features that solve the puzzle need not be the features you wanted. Some pretext-era ideas did survive. For example, *instance discrimination* treats every image in the dataset as its own class {% cite dosovitskiy2014discriminative wu2018unsupervised %}. Combined with data augmentation, it is a statement of invariance of the representation: two augmented views of image $i$ must map to the same point, while views of image $j \ne i$ must map elsewhere.

## Contrastive learning: SimCLR

SimCLR {% cite chen2020simple %} is instance discrimination stripped to its minimal form. Take a batch of $B$ images, augment each one twice (random crop, color jitter, blur), and push all $2B$ views through an encoder $f$ and a small MLP projection head $g$ to get embeddings $z = g(f(x))$. For a positive pair $(i, j)$ consisting of two views of the same image, the loss is

$$\ell_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2B} \mathbb{1}_{[k \neq i]} \exp(\mathrm{sim}(z_i, z_k)/\tau)}$$

where $\mathrm{sim}(u, v) = u^{T} v / \lVert u \rVert \lVert v \rVert$ is cosine similarity and $\tau$ is a temperature. This is a $(2B-1)$-way softmax classification problem: given view $i$, identify its partner $j$ among the other views in the batch. The other $2B - 2$ views act as *negatives*.

The loss has the functional form of InfoNCE {% cite oord2018representation %}. Under its usual sampling assumptions, minimizing InfoNCE maximizes a lower bound on mutual information between the two views. With $2B-1$ candidates, that bound cannot exceed $\log(2B-1)$. I treat this as motivation rather than a complete explanation for SimCLR's batch-size results. Larger batches, including batches of thousands of images, provide more in-batch negatives, which SimCLR found empirically useful.

Three design decisions in SimCLR are worth dwelling on:

**Augmentations are the supervision.** The loss says, "Learn a representation that is invariant to the augmentation pipeline." Random crop implies global identity is not sensitive to individual regions or pixels; color jitter implies global identity is not encoded in color statistics. This is the pretext-task guess in a new costume, but stated as an invariance on the latent representation space and not in pixel space, which turns out to be far more robust.

**The projection head is a buffer.** The loss is applied to $z = g(h)$, but the representation we keep is $h = f(x)$, before the head. The contrastive objective demands invariance to augmentation, but the augmentation *could be useful* downstream and the parameters of the head account for it. Nearly every method since has kept some version of this trick.

**Negatives stop collapse.** The trivial solution—mapping every image to the same point—is directly penalized. InfoNCE optimizes *alignment* (positives close) plus *uniformity* (embeddings spread uniformly on the hypersphere). Contrastive methods generally have this property: collapse is solved explicitly at the price of needing lots of negatives. Barlow Twins {% cite zbontar2021barlow %} avoids collapse without explicit negatives by pushing the cross-correlation matrix between the two views toward the identity.

## Masked image modeling: MAE and SimMIM

The second family imports the BERT recipe {% cite devlin2019bert %}: hide part of the input and predict the hidden part, i.e., use some kind of inpainting mechanism for supervision. The lineage in vision is older: denoising autoencoders {% cite vincent2008extracting %} and the inpainting Context Encoder {% cite pathak2016context %} are good examples. The Vision Transformer {% cite dosovitskiy2021image %} helped make masked image modeling effective at scale.

MAE {% cite he2022masked %} and SimMIM {% cite xie2022simmim %} then showed, more or less simultaneously, that regressing pixels works with substantial masking: roughly 60–75% of image patches or regions, depending on the method. The MAE objective is almost embarrassingly simple. Split the image into patches, mask a random 75%, and minimize

$$\mathcal{L} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \lVert \hat{x}_i - x_i \rVert^2$$

over the masked set $\mathcal{M}$ only, where $x_i$ is the (per-patch normalized) pixel content of patch $i$. Some points to note:

**The masking ratio matters.** BERT masks 15% of text tokens; MAE masks 75% of patches. Images are spatially redundant: a masked patch can usually be interpolated from its neighbors. Light masking creates a task solvable by low-level texture statistics and reintroduces the danger of proxy mismatch. Aggressive masking makes local interpolation less useful and encourages the encoder to model broader structure. The masking ratio plays the role that negatives played in contrastive methods: it's the knob that makes the task hard enough to require semantics.

**Asymmetry is the systems win.** MAE's encoder sees *only* the 25% of patches that are visible. A lightweight decoder takes the encoded visible patches plus learned mask tokens (with positional embeddings) and reconstructs the image. Skipping masked tokens in the encoder reduces training FLOPs and produced a 2.8× wall-clock speedup in the paper's default ViT-L comparison; other tested configurations reached 3.5–4.1×. SimMIM instead does the opposite: the full masked sequence goes through the encoder, and the "decoder" is a single linear layer predicting pixels with an $\ell_1$ loss. It is simpler and works with hierarchical backbones like Swin and even convolutional backbones, but it does not get MAE's encoder-side savings from dropping masked tokens. Interpreting both MAE and SimMIM together provides us with a nice picture: *high masking ratio + direct pixel regression* is the core recipe.

**Sidestepping collapse.** As the target is the data itself, the trivial constant solution has enormous loss. However, MAE features can be less linearly separable than those from contrastive methods, even while performing well after end-to-end fine-tuning. Reconstruction demands that the representation retain everything, including color statistics and low-level pixel information that may not be necessary.

## Self-distillation: DINO

The third family is strange because, on paper, it feels like it shouldn't work. DINO {% cite caron2021emerging %} casts the self-supervised learning problem as knowledge distillation {% cite hinton2015distilling %} with no pretrained teacher and two asymmetric branches, unlike the symmetric branches in contrastive methods. The teacher is the student's own exponential moving average (EMA); hence, it is *self*-distillation (an idea with roots in Mean Teacher from semi-supervised learning {% cite tarvainen2017mean %}). Both networks output a distribution over $K$ prototypes ($K$ is large in practice), and the student matches the teacher's distribution with a cross-entropy loss:

$$P_s(x) = \mathrm{softmax}\!\left(\frac{g_{\theta_s}(x)}{\tau_s}\right), \qquad P_t(x) = \mathrm{softmax}\!\left(\frac{g_{\theta_t}(x) - c}{\tau_t}\right)$$

$$\mathcal{L} = \sum_{x \in \{x^g_1, x^g_2\}} \; \sum_{\substack{x' \in V \\ x' \neq x}} - P_t(x)^{T} \log P_s(x')$$

The organizing idea is *information asymmetry*: the student often has to match a teacher target while seeing less of the image, and the teacher changes more slowly through EMA. DINO creates this asymmetry with crops: the teacher sees only the two global crops $x^g_1, x^g_2$, while the student sees both global and local crops in $V$. The loss excludes matching a view to itself. The teacher is updated as $\theta_t \leftarrow \lambda \theta_t + (1-\lambda)\theta_s$, with $\lambda$ ramping toward 1, and gradients flow only through the student. iBOT {% cite zhou2021ibot %} creates information asymmetry through masking: the student predicts the teacher's patch tokens at masked positions, alongside the DINO [CLS] loss.

Collapse prevention is where DINO gets weird. With no negative samples and no reconstruction loss, there are two trivial solutions: the output can collapse to one specific bin (a delta function) or to the uniform distribution. DINO avoids these collapse options with two hand-tuned mechanisms in tension:

- **Centering**: Subtract a running mean $c \leftarrow m c + (1-m) \frac{1}{B}\sum_i g_{\theta_t}(x_i)$ from teacher logits. This prevents any single bin from dominating but pushes the learning dynamics toward the uniform solution.
- **Sharpening**: Use a teacher temperature $\tau_t < \tau_s$ (with its own warmup schedule). This prevents the uniform solution but pushes toward a delta solution.

The balancing act between these two heuristics is tricky and genuinely fragile. There's also the headache of tuning the EMA schedule, the temperature schedules, weight-decay schedules, and the last-layer freezing tricks. While the method works spectacularly (the emergent attention-map segmentation in DINO and the DINOv2 {% cite oquab2024dinov2 %} features are absolutely beautiful, especially with registers {% cite darcet2024vision %}), it is a stack of empirically discovered training stability tricks. Why these mechanisms avoid collapse is still not fully settled. The DINO paper shows empirically that centering and sharpening together prevent the two observed forms of output collapse. Their interaction shapes the training dynamics.

### SimDINO: Deleting the training stability tricks

SimDINO {% cite wu2025simplifying %} asks whether many of DINO's collapse-prevention mechanisms can be replaced with a *direct* penalty on collapsed representations. The authors repurpose a mechanism known as the coding rate {% cite yu2020learning %}. For a batch of $n$ student embeddings $Z \in \mathbb{R}^{d \times n}$, the coding rate is

$$R(Z) = \frac{1}{2} \log \det\!\left(I + \frac{d}{n\epsilon^2} Z Z^{T}\right)$$

This measures (up to distortion $\epsilon$) how many bits you'd need to code the batch. Intuitively, it rewards embeddings that spread across different directions. If all normalized embeddings are identical, $ZZ^{T}$ has rank one and $R(Z)$ is lower than it would be for embeddings spread across several directions. Maximizing the rate therefore discourages collapse. SimDINO drops the softmax and cross-entropy entirely, replaces their alignment term with squared Euclidean distance between normalized student and teacher embeddings, and trains with

$$\mathcal{L}_{\mathrm{SimDINO}} = \mathbb{E}\left[\, \tfrac12\lVert z_s-z_t\rVert_2^2\right] \; - \; \gamma \, R(Z_s)$$

The $\log\det$ term is, in a precise sense, the "spread out" pressure that negatives were providing, computed from $ZZ^T/n$ instead of pairwise comparisons.

# References

{% bibliography --cited %}
