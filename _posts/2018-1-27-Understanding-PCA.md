---
layout: post
title: Understanding PCA as an Optimization Problem
---

## Why?
I’m usually miffed at the whole co-variance maximization to the algorithm hand-waviness that goes on when talking about PCA. Here, we’ll see it purely as an optimization problem and derive the simple algorithm.

## PCA: Problem Statement
Let’s assume the we have a set of zero centred samples $X = \{x_{1}, x_{2}, .. x_{n}\}^{T}$ all of dimensionality $d$.
