---
layout: archive
title: "Bayesian Nets"
permalink: /bayesian-nets/
author_profile: true
---

{% include base_path %}


* [Deep Prior](https://arxiv.org/abs/1712.05016)
* [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424), e.g. Bayes by Backprop
  * Follow-on paper: [Bayesian Hypernetworks](https://arxiv.org/abs/1710.04759)
    * Train a neural net to output the distribution of the desired neural net.
    * Once trained, input random noise to sample the parameters of the desired net.
    * A lot of the algorithm has to do with being able to scale up to lots of parameters.
* [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)
* [Dropout as a bayesian approximation: Representing model uncertainty in deep learning](https://arxiv.org/abs/1506.02142)
* [An Approximate Bayesian Long Short-Term Memory Algorithm for Outlier Detection](https://arxiv.org/abs/1712.08773)
* [Probabilistic supervised learning](https://arxiv.org/abs/1801.00753)
* [Using Deep Neural Network Approximate Bayesian Network](https://arxiv.org/abs/1801.00282)
* [Bayesian Neural Networks](https://arxiv.org/abs/1801.07710)
* [Efficient Exploration through Bayesian Deep Q-Networks](https://arxiv.org/abs/1802.04412)
  * The last layer is a Bayesian Linear Regression Model
  * Performs better than Double DQN and is easier to implement than others
* [Bayesian Uncertainty Estimation for Batch Normalized Deep Networks](https://arxiv.org/abs/1802.06455)
  * Similar to the 'Dropout as a bayesian approximation...' paper, but using Batch Norm layers to calculate uncertainty
* [Variational Inference for Policy Gradient](https://arxiv.org/abs/1802.07833)
* [Modern Computational Methods for Bayesian Inference — A Reading List](https://eigenfoo.xyz/bayesian-inference-reading/)

---
