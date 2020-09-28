---
layout: post
title: Understanding PCA as an Optimization Problem
---

## Why?
I’m usually miffed at the whole co-variance maximization to the algorithm hand-waviness that goes on when talking about PCA. Here, we’ll see it purely as an optimization problem and derive the simple algorithm.

## PCA: Problem Statement
Let’s assume the we have a set of zero centred samples $X = \{x_{1}, x_{2}, .. x_{n}\}^{T}$ all of dimensionality $d$.  We wish to transform these vectors to a lower dimensional space ( X^{\prime} = \{ x^{\prime}_{1}, x^{\prime}_{2} ,... x^{\prime}_{n} \}^{T}) of size say $k$, such that $k \lt\lt d$. We wish to perform this dimensionality reduction using a linear transformation $P$ such that

$$ x^{\prime}_{k \times 1} = P_{k \times d} \cdot x_{d \times 1} $$
