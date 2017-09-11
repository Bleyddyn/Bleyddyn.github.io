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

## Papers read, without notes

* [Revisiting natural gradient for deep networks](https://arxiv.org/abs/1301.3584v7)
  * Lots of math I didn't understand, but it does seem like natural gradients would be a better choice for RL.
* [Notes on Gradient Descent](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/notes/gradientDescent.pdf)
* [Gradual Learning of Deep Recurrent Neural Networks](https://arxiv.org/pdf/1708.08863.pdf)
  * This suggests it should be possible to start MaLPi with a one-layer LSTM, then add more layers and continue training. See Figure 1.
* [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)
* [An Information-Theoretic Optimality Principle for Deep Reinforcement Learning](https://arxiv.org/abs/1708.01867)
  * Their algorithm outperforms DQN and DDQN in terms of game-play and sample complexity.

## Papers to read

### Actor/Critic papers

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
* [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144)
  * This algorithm is in OpenAI's baseline repo.
  * It's a natural gradient actor critic method ([Natural Gradients](http://kvfrans.com/a-intuitive-explanation-of-natural-gradient-descent/)).

### Reviews

* [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)

### Relevant to MaLPi

* Learning Atari: An Exploration of the A3C Reinforcement Learning Methoda
  * This paper is from Berkeley class, but I don't have a direct link for it. Google search should work.
* [Safe and efficient off-policy reinforcement learning](https://arxiv.org/abs/1606.02647)
* [A Robust Adaptive Stochastic Gradient Method for Deep Learning](https://arxiv.org/abs/1703.00788)
* [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)
  * [PCL implementation](https://github.com/pfnet/chainerrl/blob/master/chainerrl/agents/pcl.py)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [On Generalized Bellman Equations and Temporal-Difference Learning](https://arxiv.org/abs/1704.04463)
* [Learning to Act By Predicting the Future](https://arxiv.org/abs/1611.01779)
* [Neural Episodic Control](https://arxiv.org/abs/1703.01988)
* [Mean Actor Critic](https://arxiv.org/abs/1709.00503)
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
* [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389)
  * CNN -> LSTM architecture
* [Neural SLAM](http://arxiv.org/abs/1706.09520v1)
* [Expected Policy Gradients](https://arxiv.org/abs/1706.05374)
* [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
  * Replace e-greedy or entropy methods of exploration with noisy parameters
* [Distral: Robust Multitask Reinforcement Learning](https://arxiv.org/abs/1707.04175)
* [Learning from Demonstrations for Real World Reinforcement Learning](https://arxiv.org/abs/1704.03732)
* [Trial without Error: Towards Safe Reinforcement Learning via Human Intervention](https://arxiv.org/abs/1707.05173)
  * [Blog](https://owainevans.github.io/blog/hirl_blog.html)
  * Very nice idea of having a layer between the agent and the environment for preventing disastrous behavior.
  * Initially handled by a human but later by a learned system.
* [Bayesian Neural Networks with Random Inputs for Model Based Reinforcement Learning](https://medium.com/towards-data-science/bayesian-neural-networks-with-random-inputs-for-model-based-reinforcement-learning-36606a9399b4)
  * I read through this once, but don't understand most of it.
* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
  * From OpenAI.org: "outperforms other online policy gradient methods"
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
* [Better Exploration with Parameter Noise](https://blog.openai.com/better-exploration-with-parameter-noise/)
  * Looks like I would need [Layer Normalization](https://arxiv.org/abs/1607.06450) first.
* [Guiding Reinforcement Learning Exploration Using Natural Language](https://arxiv.org/abs/1707.08616)
* [Leveraging Demonstrations for Deep Reinforcement Learning on Robotics Problems with Sparse Rewards](https://arxiv.org/abs/1707.08817)
* [DARLA: Improving Zero-Shot Transfer in Reinforcement Learning](https://arxiv.org/abs/1707.08475)
* [Decoupled Learning of Environment Characteristics for Safe Exploration](https://arxiv.org/abs/1708.02838)
* [Knowledge Sharing for Reinforcement Learning: Writing a BOOK](https://arxiv.org/abs/1709.01308)
* [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](https://arxiv.org/abs/1512.05287)
  * Variational or Bayesian Dropout, for use with RNN's.
* [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)

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
* [Representation Learning for Grounded Spatial Reasoning](https://arxiv.org/pdf/1707.03938.pdf)
  * Instruction text -> LSTM -> vectors 1 and 2
  * V1 is used as a kernel in a convolution over the state space object embeddings (hand built?)
  * V2 is used to make a global map representation of the input
  * both outputs are concatenated and input to a CNN to predict the final map value
* [Early Stage Malware Prediction Using Recurrent Neural Networks](https://arxiv.org/abs/1708.03513)
* Hwang J, Jung M, Madapana N, et al. Achieving "synergy" in cognitive behavior of humanoids via deep learning of dynamic visuo-motor-attentional coordination. Humanoid Robots (Humanoids), 2015 IEEE-RAS 15th International Conference on; Seoul. 2015. p. 817-824.
  * Combined human gesture recogniztion, attention, object detection and grasping.
  * [Arxiv page](https://arxiv.org/abs/1507.02347)
* Deep Mixture Density Network (MDN)
  *  "MDNs combine the benefits of DNNs and GMMs (Gaussian mixture model)  by using the DNN to model the complex relationship between input and output data, but providing probability distributions as output"
  * C. Bishop. Mixture density networks, Tech. Rep. NCRG/94/004, Neural Computing Research Group. Aston University, 1994.
  * H. Zen, A. Senior. Deep mixture density networks for acoustic modeling in statistical parametric speech synthesis, ICASSP, 2014.
* [Anytime Neural Networks via Joint Optimization of Auxiliary Losses](https://arxiv.org/abs/1708.06832)
* [Language Grounding for Robotics accepted papers](https://robonlp2017.github.io/accepted.html)
* Yuke Zhu, Roozbeh Mottaghi, Eric Kolve, Joseph J Lim, Abhinav Gupta, Li Fei- Fei, and Ali Farhadi. Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning. In ICRA, 2017.
