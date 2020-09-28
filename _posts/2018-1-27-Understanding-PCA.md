---
layout: post
title: Understanding PCA as an Optimization Problem
---

## Why?
I’m usually miffed at the whole co-variance maximization to the algorithm hand-waviness that goes on when talking about PCA. Here, we’ll see it purely as an optimization problem and derive the simple algorithm.

## PCA: Problem Statement
Let’s assume the we have a set of zero centred samples $X = \{x_{1}, x_{2}, .. x_{n}\}^{T}$ all of dimensionality $d$.  We wish to transform these vectors to a lower dimensional space $X^{\prime} = \{ x^{\prime}\_{1}, x^{\prime}\_{2} ,... x^{\prime}\_{n} \}^{T}$ of size say $k$, such that $k \lt\lt d$. We wish to perform this dimensionality reduction using a linear transformation $P$ such that

$$ x^{\prime}_{k \times 1} = P_{k \times d} x_{d \times 1} $$

The subscript denotes the dimensions of the matrices (we’ll drop the subscript ahead for brevity).

## Ill-formed problem?
Given that we chose any arbitrary $P$, we will get a corresponding dimensionality reduction. How do we know which $P$ to chose from the span of all real matrices of size $k \times d$?

The trick is to formulate an optimization problem, so that from the set of all $P$’s, we choose the “best” $P$. As any optimization problem, we need to define an objective function which we would minimize or maximize subject to certain constraints.

## Defining how to get the “best” $P$

There are two ways of defining the optimization problem of PCA, we’ll look at both the ways and then prove their equivalence.

### Reconstruction Loss Minimization

Consider the inverse transformation $P^{−1}$. Given that we know that $P$ is essentially a set of directions over which we project our vectors, we can take that $P$ is a orthogonal matrix. Thus, we can say that $P^{−1}=P^{T}$.

For some vector $x$, the projection is $x^{\prime} = P \cdot x$. Thus, the reconstructed $x$, let’s call it x^{\cap}, would be
$$ x^{\cap} = P^{-1} x^{\prime}$$
$$ x^{\cap} = P^{T} x^{\prime}$$
$$ x^{\cap} = P^{T} P x^{\prime}$$

It’s now apparent that we wish to minimize the reconstruction loss that encurred because of projection and backprojection due to $P$. (say, $||x − x^{\cap}||$).

However, for simplicity, **let’s assume we wish to find just one direction, let’s call it $p$**. We’ll get rid of this assumption later. Also, we will use the whole dataset instead of one sample. Thus, we can formulate an optimization problem as the following,

$$ \mathbf{min} \quad || X - (Xp)p^{T} || $$
$$ s.t. \quad\quad p^{T}p = 1 $$

Convince yourself that this is equivalent to the reconstruction loss mentioned above. Also, the constraint just implies the direction $p$ to be found should be a unit vector. This is because any scaled version of $p$ will also correspond to a solution.

### Co-Variance Maximization

The other way of looking at PCA is to find the best set of directions such that variability of the data is maximized in the lower dimensional space (This is what we usually see).

Why is this a good idea? Because, **maximum discriminability is obtained in the direction with the maximum amount of variance.** (Convince yourself by imagining a distribution of students with a variable (say their grade) that never changes)

Say $X^{\prime} = Xp$ where $p$ is that direction of maximum variance. Thus, we can write the optimization as,

$$ \mathbf{max} \quad ||X^{\prime}||^{2} $$
$$ \mathbf{max} \quad ||Xp||^{2} \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad tr((Xp)^{T}(Xp)) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad p^{T}X^{T}Xp \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Here, $$S = X^{T}X$$ is called the scatter matrix (or the unnormalized covariance matrix). It’s important to remember the data is zero centred and the equation is pretty similar otherwise.

### Proof of Equivalence

Now, a little bit of linear algebra to show that they are equivalent. Let’s start with the first formulation and show that it’s equivalent to the second formulation.

$$ \mathbf{min} \quad || X - (Xp)p^{T} ||  \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad || X - (Xp)p^{T} ||^{2}  \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad tr((X - (Xp)p^{T})(X - (Xp)p^{T})^{T}) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad tr((X - (Xp)p^{T})(X^{T} - pp^{T}X^{T})) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad tr(XX^{T} - 2Xpp^{T}X^{T} + Xp(p^{T}p)p^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad tr(XX^{T} - Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad tr(XX^{T}) - tr(Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad -tr(Xpp^{T}X^{T}) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{min} \quad -tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad tr(p^{T}X^{T}Xp) \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad p^{T}X^{T}Xp \quad s.t. \quad p^{T}p = 1 $$
$$ \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Honestly, I’m much more convinced that the direction we are trying to find has the lowest reconstruction loss rather than looking at it as co-variance maximization problem.

## Deriving the Algorithm

The beauty of this optimization is that it’s an eigenvector-eigenvalue problem. Let’s see how, let’s start with the second formulation,

$$ \mathbf{max} \quad p^{T}Sp \quad s.t. \quad p^{T}p = 1 $$

Now, using Lagrangian multipliers to convert this constrainted optimization to an unconstrainted optimization problem.

$$ \mathbf{max} \quad p^{T}Sp - \lambda(p^{T}p - 1) \quad s.t. \quad \lambda \gt 0 $$

We can consider the objective as $L(p,\lambda) = p^{T}Sp - \lambda(p^{T}p - 1)$ and we wish to maximize $L$.

Taking partial with respect to $\lambda$ and setting as zero,

$$ \frac{\partial L}{\partial \lambda} = 0 $$
$$ \Longrightarrow \quad p^{T}p - 1 = 0 $$

Taking partial with respect to $p$ and setting as zero,

$$ \frac{\partial L}{\partial p} = 0 $$
$$ \Longrightarrow \quad 2Sp - 2\lambda p = 0  \quad \Longrightarrow \quad Sp = \lambda p $$

Using both the equation and plugging in $L(p, \lambda)$,

$$ L(p, \lambda) = p^{T}(Sp) - \lambda(p^{T}p - 1) $$
$$ = p^{T}\lambda\p - \lambda(1 - 1) = \lambda p^{T}p = \lambda $$

Thus, the optimization reduces to,

$$ \mathbf{max} \quad \lambda $$
$$ s.t. \quad Sp = \lambda p $$

which is essentially finding the eigenvector $p$ of scatter matrix $S$ corresponding to the maximum eigenvalue $\lambda$. Now, we can discard our assumption of finding only one direction. To find the $k$ directions, all we need is to find the eigenvectors corresponding to the each of the $k$ eigenvalues sorted in descending order.

## PCA: The Algorithm

Thus the algorithm can implemented simply as,

``
def pca(X, k):
    cov = np.cov(X)
    w, v = np.linalg.eig(cov)
    w, v = sort_vecs(w, v, reverse=True)
    return v[:k]
``
