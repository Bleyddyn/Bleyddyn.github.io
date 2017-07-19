---
title: "Eligibility Traces"
collection: publications
permalink: /publications/2017-07-01-off-policy-actor-critic
excerpt: "Notes on using Eligibility Traces with neural networks"
date: 2017-07-17
paperurl: /publications/2017-07-01-off-policy-actor-critic
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
