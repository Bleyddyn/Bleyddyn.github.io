---
title: "Sample-Efficient Deep RL with Generative Adversarial Tree Search"
collection: publications
permalink: /publications/2018-06-27-rl-with-generative-adversarial-tree-search
excerpt: "Learned dynamics model with a GAN for image generation and MCTS for planning."
date: 2018-06-27
paperurl: https://arxiv.org/abs/1806.05780
usemath: true
---

# Overview

They use a GAN for the dynamics model, based on PIX2PIX with Wasserstein metric as the loss and spectral normalization to make training more stable. Input to the GAN is four consecutive frames plus gaussian noise plus a sequence of actions.

The Wasserstein distance can be used to approximate optimism in the Q-function for a better method of exploration than e-greedy. 

They say:

> In order to improve the quality of the generated frames, it is common to also add a class of multiple losses and capture different frequency aspects of the frames. Therefore, we also add 10 * L1 + 90 * L2 loss to the GAN loss in order to improve the training process.

Refs:

* [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)
* [Action-Conditional Video Prediction using Deep Networks in Atari Games](https://arxiv.org/abs/1507.08750)

# Learned Environment Models

* [World Models](/publications/2018-04-09-world-models)
* [Learning Robot Policies by Dreaming](/publications/2018-06-25-learning-robot-policies-by-dreaming)
* Sample-Efficient Deep RL with Generative Adversarial Tree Search

---
