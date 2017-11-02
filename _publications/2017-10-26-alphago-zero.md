---
title: "Mastering the game of Go without human knowledge"
collection: publications
permalink: /publications/2017-10-26-alphago-zero
excerpt: "AlphaGo Zero, all RL self-play."
date: 2017-10-26
paperurl: https://www.nature.com/nature/journal/v550/n7676/full/nature24270.html
use_math: true
---


They use a shared representation for the policy and value outputs with different output heads.

Policy head: Conv-2 stride 1, BN, Relu, Fully connected with 362 outputs (192 + 1)

Value head: Conv-1 stride 1, BN, Relu, FC 256, Relu, FC with one output, tanh ([-1, 1])

Loss function:

l=(z−v)2 −πT logp + c||θ||^2

MSE for the value output, Categorical cross entry for the policy and L2 regularizer with c=10-4

> By using a combined policy and value network architecture, and by using a low weight on the value component, it was possible to avoid overfitting to the values (a problem described in previous work12).

I don't see the 'low weight on the value component' in the description of the loss function.
