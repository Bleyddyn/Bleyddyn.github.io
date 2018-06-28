---
title: "World Models"
collection: publications
permalink: /publications/2018-04-09-world-models
excerpt: "Unsupervised learning of image encoding and dynamics model."
date: 2018-04-09
paperurl: https://arxiv.org/abs/1803.10122
usemath: true
---

# See also 

* An [interactive blog post](https://worldmodels.github.io/)
* A [Keras/Tensorflow implementation](https://github.com/AppliedDataSciencePartners/WorldModels)
* A [Tensorflow implementation](https://github.com/hardmaru/WorldModelsExperiments)
  * This one seems to have a more stable VAE implementation.


# Overview

* Vision Model (V)
  * A variational autoencoder that learns a compressed representation (latent space), z.
* Memory RNN (M)
  * An RNN with a Mixture Density Network on top that predicts the distribution of the next z, given current z and previous action.
* Controller (C)
  * A simple network that maps from current z plus M's internal hidden vector to an action.


Still trying to get the implementation to run all the way through...

# Learned Environment Models

* World Models
* [Learning Robot Policies by Dreaming](/publications/2018-06-25-learning-robot-policies-by-dreaming)
* [Sample-Efficient Deep RL with Generative Adversarial Tree Search](/publications/2018-06-27-rl-with-generative-adversarial-tree-search)

---
