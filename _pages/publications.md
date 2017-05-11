---
layout: archive
title: "Papers"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

## Papers read, with notes

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

## Papers to read

### Relevant to MaLPi

* [Safe and efficient off-policy reinforcement learning](https://arxiv.org/abs/1606.02647)
* [A Robust Adaptive Stochastic Gradient Method for Deep Learning](https://arxiv.org/abs/1703.00788)
* [Sample Efficient Actor-Critic With Experience Replay](https://arxiv.org/abs/1611.01224)
  * [ACER implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/acer.py)
* [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)
  * [PCL implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [On Generalized Bellman Equations and Temporal-Difference Learning](https://arxiv.org/abs/1704.04463)
* [The Reactor: A Sample-Efficient Actor-Critic Architecture](https://arxiv.org/abs/1704.04651)
* [Learning to Act By Predicting the Future](https://arxiv.org/abs/1611.01779)
* [Neural Episodic Control](https://arxiv.org/abs/1703.01988)
* [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)
* [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)
  * Replacing epsilon greedy exploration with a generalized count-based exploration strategy.

### Not as relevant to MaLPi, but interesting

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Beating Atari with Natural Language Guided Reinforcement Learning](http://web.stanford.edu/class/cs224n/reports/2762090.pdf)
* [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
* [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
  * [A Tensorflow implementation](https://github.com/DeNeutoy/bayesian-rnn)
