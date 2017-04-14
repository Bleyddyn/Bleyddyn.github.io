---
title: "Deep Reinforcement Learning with Double Q-learning"
collection: publications
permalink: /publications/2015-12-08-double-q-learning
excerpt: "Improved Q-value estimation by reducing overestimates of Deep Q-networks."
date: 2015-12-08
paperurl: https://arxiv.org/abs/1509.06461v3
use_math: true
---

## Notes

Because Q-learning uses a max operation, it can overstimate state/action values, which can lead to training problems.

Since Deep Q-Learning already uses two networks, a target network and an online (or behavior) network, overstimation can be reduced by using the target network to estimate Q-values. So the update target for DQN becomes:

$$Y_t^{DoubleQ} = R_{t+1} + \gamma Q(S_{t+1},\underset{a}{argmax} Q(S_{t+1},a;\theta_t);\theta_t^\prime)$$

Where $\theta_t$ are the weights for the behaviour Q-values, and $\theta_t^\prime$ are the weights for the target Q-values.
