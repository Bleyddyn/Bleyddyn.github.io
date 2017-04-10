---
title: "Deep Reinforcement Learning with Double Q-learning"
collection: publications
permalink: /publications/2015-12-08-double-q-learning
venue: "Association for the Advancement of Artificial Intelligence"
excerpt: "Improved Q-value estimation by reducing overestimates of Deep Q-networks."
date: 2015-12-08
paperurl: https://arxiv.org/abs/1509.06461v3
#citation: 'Your Name, You. (2009). "Paper Title Number 1." <i>Journal 1</i>. 1(1).'
---

## Notes

Because Q-learning uses a max operation, it can overstimate state/action values, which can lead to training problems.

Since Deep Q-Learning already uses two networks, a target network and an online (or behavior) network, overstimation can be reduced by using the target network to estimate Q-values. So the update target for DQN becomes:

Y = R(t+1)  + gamma * Q-target( s(t+1), argmax(Q-online( S(t+1),a )) )
