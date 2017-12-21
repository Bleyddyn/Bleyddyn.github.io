---
title: "Deep Learning in Robotics: A Review of Recent Research"
collection: publications
permalink: /publications/2017-08-15-deep-learning-in-robotics
excerpt: "A long review of the use of DL in robotics"
date: 2017-08-15
paperurl: https://arxiv.org/abs/1707.07217
usemath: false
---

## Four broad categories of Neural Networks

A) Feed-Forward network (Function approximator)

B) Auto-Encoder (Encoder/Decoder)

C) Recurrent Network

D) Q-Learning network

## Seven challenges they see in robotics that could be addressed by DL

1. Learning complex, high-dimensional, and novel dynamics.
2. Learning control policies in dynamic environments.
3. Advanced manipulation.
4. Advanced object recognition.
5. Interpreting and anticipating human actions
6. Sensor fusion & dimensionality reduction.
7. High-level task planning.

## From the section “Practical recommendations for working with Structure D”

> Another important technique is to train in simulation before attempting to train with an actual robot. This reduces wear on physical equipment, as well reduces training time. Even if only a crude simulation is available, a model that has been pre-trained on a similar challenge will converge much more quickly to fit the real challenge than one that was trained from scratch.

> They also suggest using more intelligent exploration strategies than epsilon-greedy.

