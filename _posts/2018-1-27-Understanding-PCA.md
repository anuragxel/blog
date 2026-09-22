---
layout: post
title: Understanding PCA as an Optimization Problem
---

## Why?
I’m usually miffed at the whole covariance-maximization-to-the-algorithm hand-waviness that goes on when talking about PCA. Here, we’ll see it purely as an optimization problem and derive the simple algorithm.

## PCA: Problem Statement
Let’s assume that we have a set of zero-centred samples $X = \{x_{1}, x_{2}, .. x_{n}\}^{T}$ all of dimensionality $d$.  We wish to transform these vectors to a lower-dimensional space $X^{\prime} = \{ x^{\prime}\_{1}, x^{\prime}\_{2} ,... x^{\prime}\_{n} \}^{T}$ of size, say, $k$, such that $k \lt\lt d$. We wish to perform this dimensionality reduction using a linear transformation $P$ such that

$$ x^{\prime}_{k \times 1} = P_{k \times d} x_{d \times 1} $$

The subscript denotes the dimensions of the matrices (we’ll drop the subscripts below for brevity).

## Ill-formed problem?
Given that we can choose any arbitrary $P$, we will get a corresponding dimensionality reduction. How do we know which $P$ to choose from the span of all real matrices of size $k \times d$?

The trick is to formulate an optimization problem, so that from the set of all $P$’s, we choose the “best” $P$. As with any optimization problem, we need to define an objective function which we would minimize or maximize subject to certain constraints.

## Defining how to get the “best” $P$

There are two ways of defining the optimization problem of PCA. We’ll look at both ways and then prove their equivalence.

### Reconstruction Loss Minimization

Now, how do we get $x$ back from its projection? $P$ is $k \times d$ with $k < d$, so it has no true inverse. But $P$ is essentially a set of directions we project onto, so we can take those directions to be orthonormal, i.e., $P P^{T} = I_{k}$. Then $P^{T}$ acts as a right pseudoinverse and maps a low-dimensional code back into $\mathbb{R}^{d}$.

For some vector $x$, the projection is $x^{\prime} = Px$. Thus, the reconstructed $x$, let’s call it $\hat{x}$, would be

$$ \hat{x} = P^{T} x^{\prime}$$

$$ \hat{x} = P^{T} P x$$

It’s now apparent that we wish to minimize the reconstruction loss incurred because of projection and backprojection due to $P$, that is,

$$ \mathbf{min} \quad || x − \hat{x} || $$

However, for simplicity, **let’s assume we wish to find just one direction, let’s call it $p$**. We’ll get rid of this assumption later. Also, we will use the whole dataset instead of one sample. Thus, we can formulate the following optimization problem:

$$ \mathbf{min} \quad || X - (Xp)p^{T} || $$

$$ s.t. \quad\quad p^{T}p = 1 $$

Convince yourself that this is equivalent to the reconstruction loss mentioned above. Also, the constraint just says $p$ should be a unit vector, so that $(Xp)p^{T}$ is an orthogonal projection (scale $p$ and the reconstruction changes).

### Covariance Maximization

The other way of looking at PCA is to find the best set of directions such that the variability of the data is maximized in the lower-dimensional space (this is what we usually see).

Why is this a good idea? Because **the direction with the maximum variance preserves the most information about the data after projection**. (Convince yourself by imagining a distribution of students with a variable (say their grade) that never changes. That direction tells you nothing about any student, and dropping it loses nothing.) Note that “information” here means reconstruction, and not how well you can tell classes apart. PCA is unsupervised and doesn’t care about class boundaries (that’s what LDA is for).

Say $X^{\prime} = Xp$ where $p$ is that direction of maximum variance. Thus, we can write the optimization as follows:

$$ \mathbf{max} \quad ||X^{\prime}||^{2} $$

$$ = \mathbf{max} \quad ||Xp||^{2} \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad tr((Xp)^{T}(Xp)) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad p^{T}X^{T}Xp \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Here, $S = X^{T}X$ is called the scatter matrix (or the unnormalized covariance matrix). It’s important to remember the data is zero-centred, and the equation is pretty similar otherwise.

<figure class="concept-figure">
  <a href="{{ '/assets/images/pca/projection-reconstruction.png' | relative_url }}">
    <img src="{{ '/assets/images/pca/projection-reconstruction.png' | relative_url }}" width="640" height="380" loading="lazy" alt="Centered points project orthogonally onto the principal direction p. Dashed perpendicular segments connect each point to its reconstruction. Total squared norm splits into projected squared norm and squared reconstruction error.">
  </a>
  <figcaption>Projecting centered data onto this line preserves the most variance and gives the smallest squared reconstruction error.</figcaption>
</figure>

### Proof of Equivalence

Now, a little bit of linear algebra to show that they are equivalent. Let’s start with the first formulation and show that it’s equivalent to the second formulation.

$$ \mathbf{min} \quad || X - (Xp)p^{T} ||  \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad || X - (Xp)p^{T} ||^{2}  \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad tr((X - (Xp)p^{T})(X - (Xp)p^{T})^{T}) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad tr((X - (Xp)p^{T})(X^{T} - pp^{T}X^{T})) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad tr(XX^{T} - 2Xpp^{T}X^{T} + Xp(p^{T}p)p^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad tr(XX^{T} - Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad tr(XX^{T}) - tr(Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad -tr(Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{min} \quad -tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad p^{T}X^{T}Xp \quad s.t. \quad p^{T}p = 1 $$

$$ = \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Honestly, the lowest-reconstruction-loss view convinces me much more than the covariance maximization one.

## Deriving the Algorithm

The beauty of this optimization is that it’s an eigenvector-eigenvalue problem. Let’s see how, starting with the second formulation:

$$ \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Now, we use Lagrange multipliers to turn this constrained problem into an unconstrained one.

$$ \mathbf{max} \quad p^{T}Sp - \lambda(p^{T}p - 1) $$

We can consider the objective as $L(p,\lambda) = p^{T}Sp - \lambda(p^{T}p - 1)$ and we wish to maximize $L$.

Taking the partial derivative with respect to $\lambda$ and setting it to zero,

$$ \frac{\partial L}{\partial \lambda} = 0 $$

$$ \Longrightarrow \quad p^{T}p - 1 = 0 $$

Taking the partial derivative with respect to $p$ and setting it to zero,

$$ \frac{\partial L}{\partial p} = 0 $$

$$ \Longrightarrow \quad 2Sp - 2\lambda p = 0  \quad \Longrightarrow \quad Sp = \lambda p $$

Plugging both back into $L(p, \lambda)$,

$$ L(p, \lambda) = p^{T}(Sp) - \lambda(p^{T}p - 1) $$

$$ = p^{T}\lambda p - \lambda(1 - 1) = \lambda p^{T}p = \lambda $$

Thus, the optimization reduces to

$$ \mathbf{max} \quad \lambda $$

$$ s.t. \quad Sp = \lambda p $$

which is essentially finding the eigenvector $p$ of the scatter matrix $S$ corresponding to the maximum eigenvalue $\lambda$. Now, we can discard our assumption of finding only one direction. To find the $k$ directions, all we need is to find the eigenvectors corresponding to each of the $k$ eigenvalues sorted in descending order.

## PCA: The Algorithm

Thus, the algorithm can be implemented simply as follows:

```
def pca(X, k):
    cov = np.cov(X, rowvar=False)
    w, v = np.linalg.eigh(cov)
    return v[:, ::-1][:, :k]
```

This returns the principal directions as columns, so the result is $P^{T}$ in our earlier notation. To project the samples and reconstruct them in the original space:

```python
mean = X.mean(axis=0)
X_centered = X - mean
directions = pca(X_centered, k)
X_projected = X_centered @ directions
X_reconstructed = X_projected @ directions.T + mean
```
