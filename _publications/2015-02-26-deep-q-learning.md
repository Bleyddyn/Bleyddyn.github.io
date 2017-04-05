---
title: "Human-level control through deep reinforcement learning"
collection: publications
permalink: /publications/2015-02-26-deep-q-learning.md
venue: "Nature"
excerpt: "One of the first deep reinforcement learning papers."
date: 2015-02-26
paperurl: http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
#citation: 'Your Name, You. (2009). "Paper Title Number 1." <i>Journal 1</i>. 1(1).'
---

## Notes

Reinforcement learning methods that use non-linear function approximators (e.g. neural networks) for the action value function are not theoretically stable.

This paper gets around that problem with two main changes to training:

1. Experience replay

Each time step is recorded with: state, action, reward, and next state.
A random mini-batch of experiences is drawn and used for a Q-learning update.

2. Separate target and behavior value networks

Updates are performed on a copy of the network. Only after a number of such updates is the improved network copied back into the one used to choose actions.
