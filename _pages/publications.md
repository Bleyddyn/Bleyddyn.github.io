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

### Actor/Critic papers

* [Off-Policy Actor-Critic](https://arxiv.org/abs/1205.4839)
  * Sutton, et. al. 2012. Includes elegibility traces.
* [Sample Efficient Actor-Critic With Experience Replay](https://arxiv.org/abs/1611.01224)
  * [ACER implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/acer.py)
* [The Reactor: A Sample-Efficient Actor-Critic Architecture](https://arxiv.org/abs/1704.04651)
  * Also compares time stacked inputs versus LSTMs in section 3.3.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
  * The A3C paper.
* [A Survey of Actor-Critic Reinforcement Learning: Standard and Natural Policy Gradients](https://pdfs.semanticscholar.org/145a/42e83ec142a125da3ad845ee95027ef702e5.pdf)
  * 2010, maybe?
* [ON ACTOR-CRITIC ALGORITHMS](http://www.mit.edu/~jnt/Papers/J094-03-kon-actors.pdf)
  * 2003

### Relevant to MaLPi

* [Learning Atari: An Exploration of the A3C Reinforcement Learning Method](https://instructure-uploads.s3.amazonaws.com/account_10720000000090242/attachments/70223081/learning-atari-exploration.pdf?response-content-disposition=attachment%3B%20filename%3D%22AtariReport.pdf%22%3B%20filename%2A%3DUTF-8%27%27AtariReport.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJFNFXH2V2O7RPCAA%2F20170617%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20170617T002647Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=6d0f68826df0ae0c3c35de54aa8387701592a4ea64e7d06dd74efc7972ff20d9)
* [Safe and efficient off-policy reinforcement learning](https://arxiv.org/abs/1606.02647)
* [A Robust Adaptive Stochastic Gradient Method for Deep Learning](https://arxiv.org/abs/1703.00788)
* [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)
  * [PCL implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [On Generalized Bellman Equations and Temporal-Difference Learning](https://arxiv.org/abs/1704.04463)
* [Learning to Act By Predicting the Future](https://arxiv.org/abs/1611.01779)
* [Neural Episodic Control](https://arxiv.org/abs/1703.01988)
* [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)
  * [Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/6bi6np/d_glearning_taming_the_noise_in_reinforcement/)
* [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310)
  * Replacing epsilon greedy exploration with a generalized count-based exploration strategy.
* [Learning to act by predicting the future](https://openreview.net/forum?id=rJLS7qKel&noteId=rJLS7qKel)
  * Predicting the future as a supervised learning task
* [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/)
* [One-Shot Imitation Learning](https://arxiv.org/abs/1703.07326)
* [Efficient Parallel Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1705.04862v2)
* [Recurrent Additive Networks](http://www.kentonl.com/pub/llz.2017.pdf)
  * A simpler type of RNN. Not sure if/where it's been published. Only tested on language tasks?
* [Feature Control as Intrinsic Motivation for Hierarchical Reinforcement Learning](https://arxiv.org/pdf/1705.06769.pdf)
  * Followup to the auxiliary tasks paper.
* [Non-Markovian Control with Gated End-to-End Memory Policy Networks](https://arxiv.org/abs/1705.10993)
* [Experience Replay Using Transition Sequences](https://arxiv.org/abs/1705.10834)
* [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  * A replacement for RELU activation. Looks fairly simple to implement and try. A quote from the abstract, "...thus, vanishing and exploding gradients are impossible."
* [Grounded Language Learning in a Simulated 3D World](https://arxiv.org/abs/1706.06551)
  * Rewards for completing tasks given in written instructions.
* [MEC: Memory-efficient Convolution for Deep Neural Network](https://arxiv.org/abs/1706.06873)

### Not as relevant to MaLPi, but interesting

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Beating Atari with Natural Language Guided Reinforcement Learning](http://web.stanford.edu/class/cs224n/reports/2762090.pdf)
* [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
* [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
  * [A Tensorflow implementation](https://github.com/DeNeutoy/bayesian-rnn)
* [ML for analyzing unix log files](https://www.google.com/#safe=off&q=unix+log+files+machine+learning+detect+outliers)
ss [Concrete Dropout](https://arxiv.org/abs/1705.07832)
* [Bayesian Reinforcement Learning: A Survey](https://arxiv.org/abs/1609.04436)
* [Beyond Monte Carlo Tree Search: Playing Go with Deep Alternative Neural Network and Long-Term Evaluation](https://arxiv.org/abs/1706.04052)
* [Meta learning Framework for Automated Driving](https://arxiv.org/abs/1706.04038)
