---
title: "Pre-training Neural Networks with Human Demonstrations for Deep Reinforcement Learning"
collection: publications
permalink: /publications/2017-09-14-pre-training-for-rl
excerpt: "Pre-train using supervised learning on human provided demonstations."
date: 2017-09-14
paperurl: https://arxiv.org/abs/1709.04083
usemath: false
---

The idea is to generate experience data from a human demonstration (playing a game or in my case manually driving MaLPi around) and use it to pre-train the network, or most of it, using supervised learning. This step is essentially how [Donkey Car](http://www.donkeycar.com) works. Once that step is done, use the pre-trained weights to initialize the network and continue training using Reinforcement Learning.

They tried initializing with the full network and also with everything except the final output layer, getting similar results.

They also tried using the human data to pre-load the Replay memory in DQN, which would normally be initialized via random play. By itself this wasn't very useful, however when combined with pre-training this had similar results to pre-training alone, on two games, but was beneficial on the third. In order for the combination to work, the initial exploration rate (epsilon) had to be lowered considerably which could be useful in some cases, including MaLPi.
