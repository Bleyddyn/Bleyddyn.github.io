---
title: "Pretraining Deep Actor-Critic Reinforcement Learning Algorithms With Expert Demonstrations"
collection: publications
permalink: /publications/2018-02-01-pretraining-ac-with-expert
excerpt: "Pretraining Deep Actor-Critic Reinforcement Learning Algorithms With Expert Demonstrations."
date: 2018-02-01
paperurl: https://arxiv.org/abs/1801.10459
usemath: true
---

One of the assumptions in this paper is that expert policies don't have reward data associated with them, so they can't be used to directly pre-train the Critic portion of an Actor/Critic network. I don't think that assumption will hold true if/when I switch MaLPi from imitation learning to RL.

Calculate the Advantage of the expert policy. If the Advantage is greater than zero, then multiply that by the gradient of the weights of the critic and add that to the gradient calculated by the base RL algorithm. It looks like they use a hyperparamter to control how much of the expert policy is used, and elsewhere they mention they only do this 'pre-training' for a limited time, so either it gets set to zero at zome point, or is slowly reduced over training time. The only time I see it mentioned, though, it's just set to one.

Despite their claims, it doesn't look to me like there's a huge benefit to pre-training in many cases.

# References

* Ziyu Wang, Victor Bapst, Nicolas Heess, Volodymyr Mnih, Remi Munos, Koray Kavukcuoglu, and Nando de Freitas. Sample efficient actor-critic with expe- rience replay. [arXiv preprint arXiv:1611.01224](https://arxiv.org/abs/1611.01224), 2016.

---
