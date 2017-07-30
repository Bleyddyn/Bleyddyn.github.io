---
title: "Eligibility Traces"
collection: publications
permalink: /publications/2017-07-17-eligibility-traces
excerpt: "Notes on using Eligibility Traces with neural networks"
date: 2017-07-17
paperurl: /publications/2017-07-17-eligibility-traces
use_math: true
---

## From ME539, Oregon State

The last slide from [these lecture notes](http://classes.engr.oregonstate.edu/mime/fall2008/me539/Lectures/ME539-w6-RL2_notes.pdf):

Gradient descent Sarsa((λ)):

$\Delta \vec \theta_t = \alpha \delta \vec e_t$

where

$\delta_t = r_{t+1} + \gamma Q_t(s_{t+1},a_{t+1}) − Q_t(s_t,a_t)$

$\vec e = \gamma \lambda \vec e_{t-1} +\nabla_{\vec \theta} Q_t(s_t,a_t)$

I understand and have implemented most of that, in one place or another. Just the last part of that last line I need to figure out.

## TF implementation

[TD-Gammon](https://github.com/fomorians/td-gammon/blob/master/model.py)

As usual with Tensorflow implementations, it's very hard to follow.

Do you only need to update traces with the derivitive of the output with respect to the weights of only the last (output) layer, or for every layer?

## Biologically plausible learning in RNNs

From [Biologically plausible learning in recurrent neural networks reproduces neural dynamics observed during cognitive tasks](http://www.biorxiv.org/content/biorxiv/early/2017/02/22/057729.full.pdf).


$e_{i,j}(t) = e_{i,j}(t-1) + S(r_j(t-1)(x_i(t) - \bar x_i))$

"where $r_j$ represents the output of neuron j, and thus the current input at this synapse. $x_i$ represents
the current excitation (or potential) of neuron i and $\bar x_i$ represents a short-term 
running average of $x_i$, and thus $x(t) - \bar x$ tracks the fast fluctuations of neuron output."

S must be a monotonic, supralinear function. In this paper they used the cubic function $S(x) = x^3$.

This one seems to relatively easy to implement, although it does seem like it would need a decay parameter on the previous values.
