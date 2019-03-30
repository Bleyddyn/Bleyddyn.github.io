---
layout: archive
title: "Multi-Task Learning"
permalink: /multi-task-learning
author_profile: true
---

{% include base_path %}

# To Read

* [Beyond Shared Hierarchies: Deep Multitask Learning through Soft Layer Ordering](https://arxiv.org/abs/1711.00108)
* [Pseudo-task Augmentation: From Deep Multitask Learning to Intratask Sharing---and Back](https://arxiv.org/abs/1803.04062)
* [Zero-Shot Task Generalization with Multi-Task Deep Reinforcement Learning](https://arxiv.org/abs/1706.05064)
* [Multi-task Learning for Continuous Control](https://arxiv.org/abs/1802.01034)
* [Multi-Goal Reinforcement Learning: Challenging Robotics Environments and Request for Research](https://arxiv.org/abs/1802.09464)
* [DiGrad: Multi-Task Reinforcement Learning with Shared Actions](https://arxiv.org/abs/1802.10463)
* [Distral: Robust Multitask Reinforcement Learning](https://arxiv.org/abs/1707.04175)
* [End-to-End Video Captioning with Multitask Reinforcement Learning](https://arxiv.org/abs/1803.07950)
* [RL2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/pdf/1611.02779)

# Read

* [Vision Based Multi-task Manipulation for Inexpensive Robots Using End-to-End Learning from Demonstration](https://arxiv.org/abs/1707.02920)
  * Includes a diagram for how to add a GAN to the network as an auxiliary task
* [Multi-Task Learning Objectives for Natural Language Processing](http://ruder.io/multi-task-learning-nlp/index.html)
  * Specifically about NLP, but some ideas might be useful for MaLPi.
  * Auxilliary tasks should complement the main task.
  * Adversarial loss
    * (Ganin, Y., & Lempitsky, V. (2015). Unsupervised Domain Adaptation by Backpropagation. In Proceedings of the 32nd International Conference on Machine Learning. (Vol. 37).
    * [Domain-Adversarial Training of Neural Networks](http://www.jmlr.org/papers/volume17/15-239/source/15-239.pdf)
  * Predicting the next frame in video, [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551).
* [Hierarchical and Interpretable Skill Acquisition in Multi-Task Reinforcement Learning](https://einstein.ai/static/images/pages/research/hierarchical-reinforcement-learning/iclr2018_HRL.pdf)
* [Multi-Task Sequence To Sequence Learning](https://arxiv.org/abs/1511.06114)
  * All of the examples are for text related tasks.
  * Sequence auto-encoders were one of the auxiliary tasks they used which showed benefit.
* [MultiNet: Multi-Modal Multi-Task Learning for Autonomous Driving](https://arxiv.org/abs/1709.05581)
  * They allow the driver to override the NN during autonomous operation, rather than having the expert edit collected data after the fact.
  * They collect two stero pairs of images 33ms apart, then use those four images to make ten control decisions (power plus steering) over the next 330ms. However, only the final control decision is used, the others are considered 'side tasks' or auxiliary tasks.
  * They insert a binary modality input after the first convolutional layer, before the second. Modality would be something like driving at home versus at the track.
* [An Overview of Multi-Task Learning in Deep Neural Networks](http://ruder.io/multi-task/)

---
