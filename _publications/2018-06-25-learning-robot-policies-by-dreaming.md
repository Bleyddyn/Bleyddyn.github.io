---
title: "Learning Real-World Robot Policies by Dreaming"
collection: publications
permalink: /publications/2018-06-25-learning-robot-policies-by-dreaming
excerpt: "Unsupervised learning of image encoding, dynamics and reward models."
date: 2018-06-25
paperurl: https://arxiv.org/abs/1805.07813
usemath: true
---

# Overview

Very similar to the World Models paper. Train a variational autoencoder to encode input images into a representation, train a CNN to predict next state from current state and action (also encoded), and train a model to predict reward given a state encoding.

One major difference between World Models and this paper is that the dynamics model is stateless, and doesn't include any uncertainly measure.

Another difference is that in this paper the encoder and dynamics models are trained in an end-to-end fashion.

If I ever get World Models working, I will want to try changing it so that it trains end-to-end, like this one.

# Learned Environment Models

* [World Models](/publications/2018-04-09-world-models)
* Learning Real-World Robot Policies by Dreaming
* [Sample-Efficient Deep RL with Generative Adversarial Tree Search](/publications/2018-06-27-rl-with-generative-adversarial-tree-search)

---
