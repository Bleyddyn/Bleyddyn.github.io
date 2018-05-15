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

# Sections

* [Papers read, with notes](#papers-read-with-notes)
* [Current](#current)
* [Papers read, with minimal notes](#papers-read-with-minimal-notes)
* [Papers to Read, by Category](#papers-to-read-by-category)
* [Reviews](#reviews)
* [Relevant to MaLPi](#relevant-to-malpi)
* [Not as relevant to MaLPi, but interesting](#not-as-relevant-to-malpi-but-interesting)
* [Classes/Education](#classeseducation)
* [Simulators](#simulators)

## Papers read, with notes

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

## Current

* [World Models](https://arxiv.org/abs/1803.10122)
  * [Source (not official)](https://github.com/AppliedDataSciencePartners/WorldModels)
* [Active Robotic Mapping through Deep Reinforcement Learning](https://arxiv.org/abs/1712.10069)
* [Boosting the Actor with Dual Critic](https://arxiv.org/abs/1712.10282)
* [Vision Based Multi-task Manipulation for Inexpensive Robots Using End-to-End Learning from Demonstration](https://arxiv.org/abs/1707.02920)
  * Includes a diagram for how to add a GAN to the network as an auxiliary task
* [Robust Reinforcement Learning](https://papers.nips.cc/paper/1841-robust-reinforcement-learning.pdf)
* [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702)
* [Unicorn: Continual Learning with a Universal, Off-policy Agent](https://arxiv.org/abs/1802.08294)
* [Unsupervised learning in LSTM recurrent neural networks](ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf)
  * Possible method for clustering accelerometer data for a reward function
* [Recurrent Predictive State Policy Networks](https://arxiv.org/abs/1803.01489)

## Papers read, with minimal notes

* [Revisiting natural gradient for deep networks](https://arxiv.org/abs/1301.3584v7)
  * Lots of math I didn't understand, but it does seem like natural gradients would be a better choice for RL.
* [Notes on Gradient Descent](https://ipvs.informatik.uni-stuttgart.de/mlr/marc/notes/gradientDescent.pdf)
* [Gradual Learning of Deep Recurrent Neural Networks](https://arxiv.org/pdf/1708.08863.pdf)
  * This suggests it should be possible to start MaLPi with a one-layer LSTM, then add more layers and continue training. See Figure 1.
* [A Brief Survey of Deep Reinforcement Learning](https://arxiv.org/abs/1708.05866)
* [An Information-Theoretic Optimality Principle for Deep Reinforcement Learning](https://arxiv.org/abs/1708.01867)
  * Their algorithm outperforms DQN and DDQN in terms of game-play and sample complexity.
* [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf)
  * This paper included a comparison of A3C with A3C+LSTM, and it looks to me like the LSTM version performs better on many games.
* [Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents](https://arxiv.org/abs/1709.06009)
  * Score using the average of the last n games, both during training and afterward.
  * Add stochasticity by having 'sticky actions.'
* [Online Learning of a Memory for Learning Rates](https://arxiv.org/pdf/1709.06709.pdf)
  * Learns a memory that's used to set the learning rate.
  * Something to look for as an option in Keras or Tensorflow.
  * Was compared with an optimizer learning system called L2LBGDBGD.
* [Probabilistic machine learning and artificial intelligence](https://www.cse.iitk.ac.in/users/piyush/courses/pml_winter16/nature14541.pdf)
  * Review article. Basically an add for why all ML should be probabilistic.
* [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560)
  * Everything seems to have a large effect on performance: network architecture, hyperparameters, reward scaling, random seeds, number of trials, algorithm and implementation details.
  * Evaluation metrics might include: confidence bounds, power analysis, and significance of the metric.
* [Learning Diverse Skills via Maximum Entropy Deep Reinforcement Learning](http://bair.berkeley.edu/blog/2017/10/06/soft-q-learning/)
  * Soft Q-Learning, also called Maximal Entropy Models. Benefits:
    * Better Exploration
    * Fine-Tuning Maximum Entropy Policies
    * Compositionality
    * Robustness
    * Paper: [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165)
    * Code: https://github.com/haarnoja/softqlearning
* [Interpretable ML](https://medium.com/south-park-commons/how-to-make-your-data-and-models-interpretable-by-learning-from-cognitive-science-a6a29867790)
  * Generate prototypes and critics of a cluster for visualizing data in a human-centric way: [MMD-critic](https://github.com/BeenKim/MMD-critic)
* [Learning Long Duration Sequential Task Structure From Demonstrations with Application in Surgical Robotics](http://bair.berkeley.edu/blog/2017/10/17/lfd-surgical-robots/)
  * The last section describes a policy that chooses sub-policies. I think.
* [Mixup](https://twitter.com/hardmaru/status/925145276016345088)
  * Data augmentation by linear interpretation between two x's and two y's.
  * They suggest randomly selected samples, but I would probably need to keep them close in time.
  * Yes, I'm referencing a tweet. Haven't found the original paper.
* [Does batch size matter?](https://blog.janestreet.com/does-batch-size-matter/)
  * Interesting but I'm still not sure how it helps me pick a learning rate, especially if I'm using RMSProp or Adam.
  * lr = BatchSize / NumberOfSamples?
  * What about learning rate decay?
* [The impossibility of intelligence explosion](https://medium.com/@francois.chollet/the-impossibility-of-intelligence-explosion-5be4a9eda6ec)
* [NIPS 2017 Notes](https://cs.brown.edu/%7Edabel/blog/posts/misc/nips_2017.pdf)
* [Machine Teaching: A New Paradigm for Building Machine Learning Systems](https://arxiv.org/abs/1707.06742)
* [Recent Advances in Recurrent Neural Networks](https://arxiv.org/abs/1801.01078)
  * High level overview of many different types of RNNs.
  * [Tunable Efficient Unitary Neural Networks (EUNN) and their application to RNNs](https://arxiv.org/abs/1612.05231)
  * Gated Orthogonal Recurrent Unit
  * Two regularization methods I've never heard of:
    * Activation Stabilization. D. Krueger and R. Memisevic, “Regularizing rnns by stabilizing activations,” arXiv preprint arXiv:1511.08400, 2015.
    * Hidden Activation Preservation. D. Krueger, et al., “Zoneout: Regularizing rnns by randomly preserving hidden activations,” arXiv preprint arXiv:1606.01305, 2016.
* [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400)(MAML)
  * Meta-train a model so that it is good at quickly learning (fine-tuning) a new task
  * [Code is available](https://github.com/cbfinn/maml_rl)
* [Recasting Gradient-Based Meta-Learning as Hierarchical Bayes](https://arxiv.org/abs/1801.08930)
  * Follow-up to MAML
  * Reformulates MAML as Hierarchical Bayesian inference, and uses that to improve MAML.
* [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561)
  * Really should take notes on this and try to implement it.
* [Learning Longer-Term Dependencies In RNNs With Auxiliary Losses](https://openreview.net/pdf?id=Hy9xDwyPM)
  * They suggest having an RNN try to predict sequences from 'anchors' chosen at random as an unsupervized auxiliary task.
  * Not much in the way of implementation details.
  * [Unsupervised Pretraining for Sequence to Sequence Learning](https://arxiv.org/abs/1611.02683)
  * and Andrew M. Dai and Quoc V. Le. [Semi-supervised sequence learning](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning)
* [Deep Reinforcement Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)
* [Semi-supervised Sequence Learning](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning) (NIPS 2015)
  * Among other methods, they used a 'sequence autoencoder' to pre-train an LSTM.
  * The autoencoder was trained to memorize input sequences and generate them as output.
* [L4: Practical loss-based stepsize adaptation for deep learning](https://arxiv.org/abs/1802.05074)
   * [Github with code](https://github.com/martius-lab/l4-optimizer/)
   * Should be easy to try out with a Keras wrapper: keras.optimizers.TFOptimizer(optimizer)
* [Towards Principled Design of Deep Convolutional Networks: Introducing SimpNet](https://arxiv.org/abs/1802.06205)
  * Overall a pretty bad paper, far too wordy.
  * My only takeaway is that I should try testing max-pooling versus my current strided convolutions.
* [An Outsider’s Tour of Reinforcement Learning, part 4](http://www.argmin.net/2018/02/08/lqr/)
  * And very similar: [Reinforcement Learning never worked, and 'deep' only helped a bit.](https://himanshusahni.github.io/2018/02/23/reinforcement-learning-never-worked.html)
  * Having MaLPi learn a model and use it for planning seems like a very good idea.
* [Optimization for Deep Learning Highlights in 2017](http://ruder.io/deep-learning-optimization-2017/index.html)
  * Decoupled weight decay. [Not yet in Keras](https://github.com/keras-team/keras/pull/9189)
  * Try AMSGrad instead of Adam
  * Learning rate annealing with Adam. Should be able to try this out
    * β1=0.9
    * a non-default β2=0.98
    * ϵ=10−9
    * η = dmodel^−0.5⋅min(step_num^−0.5,step_num⋅warmup_steps^−1.5)
    * where dmodel is the number of parameters of the model and warmup_steps=4000
  * SGD with warm restarts
* [The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://arxiv.org/abs/1802.10031)
  * They describe a Horizon Aware Value Function that takes into account discounting
  * They also link to implementations of several Policy Gradient methods
* [Deep Reinforcement Learning for Vision-Based Robotic Grasping: A Simulated Comparative Evaluation of Off-Policy Methods](https://arxiv.org/pdf/1802.10264.pdf)
  * Deep-Q Learning was more stable than DDPG (an actor-critic method) over hyperparameter ranges and random seed.
  * Monte Carlo and a novel Corrected MC both performed fairly well under high data conditions.
* [Setting up a Reinforcement Learning Task with a Real-World Robot](https://arxiv.org/abs/1803.07067)
  * Time delays can have a significant impact on ability to learn
  * Actions closer to the hardware make learning easier. Velocity control versus positional control
  * "Too small action cycle times make learning harder. Too long action cycle times also impede performance as they reduce precision and cause slow data collection."
* [DAgger algorithm](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)
  * This is the imitation learning algorithm used in CS 294
  * [Blog Post 1](https://bleyddyn.github.io/2018-02-27-dagger-1) and [Blog Post 2](https://bleyddyn.github.io/2018-03-12-dagger-2)
* [Reinforcement Learning with a Corrupted Reward Channel](https://arxiv.org/abs/1705.08417)
  * One method they suggest for overcoming this is "Quantilization". "Rather than choosing the state with the highest observed reward, these agents instead randomly choose a state from a top quantile of high-reward states."
* [Differentiable plasticity: training plastic neural networks with backpropagation](https://arxiv.org/abs/1804.02464)
  * Use a Hebbian trace: "the Hebbian trace is simply a running average of the product of pre- and post-synaptic activity"
  * Effective weight for every i/j neuron pair is baseline weight (normal NN) plus a plasticity coefficient times the hebbian trace. Plasticity coefficient is learned along with normal weights.
* [The unreasonable effectiveness of the forget gate](https://arxiv.org/abs/1804.04849)
  * A version of the LSTM layer that only has a forget gate. Also needs a new initilizer for the biases of the forget gate called a 'chrono initializer'.

## Papers to Read, by Category

* [Bayesian Neural Nets](/bayesian-nets)
* [Actor/Critic papers](/actor-critic)
* [Multi-Task papers](/multi-task-learning)

### Reviews

* [Deep Reinforcement Learning: An Overview](https://arxiv.org/abs/1701.07274)
* [Continual Lifelong Learning with Neural Networks: A Review](https://arxiv.org/abs/1802.07569)
* [A Survey of Deep Learning Techniques for Mobile Robot Applications](https://arxiv.org/abs/1803.07608)

### Relevant to MaLPi

* [Curiosity-driven Exploration by Self-supervised Prediction](https://pathak22.github.io/noreward-rl/)
  * [Arxiv link](https://arxiv.org/abs/1705.05363)
* Learning Atari: An Exploration of the A3C Reinforcement Learning Methods.
  * This paper is from Berkeley class, but I don't have a direct link for it. Google search should work.
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
  * [A blog post about it](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
  * [And Keras code](https://github.com/flyyufelix/C51-DDQN-Keras)
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
* [The Uncertainty Bellman Equation and Exploration](https://arxiv.org/abs/1709.05380)
* [Lifelong Learning with Dynamically Expandable Networks](https://arxiv.org/abs/1708.01547)
* [Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.10089)
  * Mentions something called Hindsight Experience Replay.
* [Self-supervised Deep Reinforcement Learning with Generalized Computation Graphs for Robot Navigation](https://arxiv.org/abs/1709.10489)
* [Embodied Question Answering](http://embodiedqa.org)
  * [Paper](http://embodiedqa.org/paper.pdf)
* [Time Limits in Reinforcement Learning](https://arxiv.org/abs/1712.00378)
  * For value networks that will be used in a non-episodic way, don't end bootstraping at training episode boundaries.
* [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888)
  * [Website](https://sermanet.github.io/imitate/)
  * Upvote
* [Reverse Curriculum Generation for Reinforcement Learning Agents](http://bair.berkeley.edu/blog/2017/12/20/reverse-curriculum/)
  * This could be very useful when I try to train MaLPi to find its charging station.
  * Paper: [Reverse Curriculum Generation for Reinforcement Learning](http://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf)
  * [Code](https://sites.google.com/view/reversecurriculum)
* [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)
* [Ray RLLib](https://t.co/7pfYViDchA)
* [Attention based neural networks](https://towardsdatascience.com/memory-attention-sequences-37456d271992)
* [Expected Policy Gradients for Reinforcement Learning](https://arxiv.org/abs/1801.03326)
* [Model-Based Action Exploration](http://arxiv.org/abs/1801.03954v1)
* [Curiosity-driven reinforcement learning with homeostatic regulation](https://arxiv.org/abs/1801.07440)
* [Regret Minimization for Partially Observable Deep Reinforcement Learning](https://openreview.net/forum?id=BJoBhUUUG)
* [One-shot Imitation from Humans via Domain-Adaptive Meta-Learning](https://arxiv.org/abs/1802.01557)
* [Alleviating catastrophic forgetting using context-dependent gating and synaptic stabilization](https://arxiv.org/abs/1802.01569)
* [Temporal Difference Models: Model-Free Deep RL for Model-Based Control](https://arxiv.org/abs/1802.09081), BAIR
* [Reinforcement and Imitation Learning for Diverse Visuomotor Skills](https://arxiv.org/abs/1802.09564)
* [Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835)
* [Composable Deep Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/1803.06773), BAIR
* [Recall Traces: Backtracking Models for Efficient Reinforcement Learning](https://arxiv.org/abs/1804.00379), BAIR
* [Universal Planning Networks](https://arxiv.org/abs/1804.00645)
* [Reinforcement Learning With Unsupervised Auxiliary Tasks](https://arxiv.org/pdf/1611.05397.pdf)
* [Latent Space Policies for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1804.02808)
* [Averaging Weights Leads to Wider Optima and Better Generalization](http://arxiv.org/abs/1803.05407)
  * Running average of weights during training to create an effect similar to ensembling
  * [Blog post about implementing it](https://medium.com/@hortonhearsafoo/adding-a-cutting-edge-deep-learning-training-technique-to-the-fast-ai-library-2cd1dba90a49)
  * [PyTorch implementation](https://github.com/fastai/fastai/pull/276/files)
* [Temporal Difference Models: Model-Free Deep RL for Model-Based Control](https://arxiv.org/abs/1802.09081), BAIR
* [Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review](https://arxiv.org/abs/1805.00909) BAIR

### Not as relevant to MaLPi, but interesting

* [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
* [Beating Atari with Natural Language Guided Reinforcement Learning](http://web.stanford.edu/class/cs224n/reports/2762090.pdf)
* [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)
* [Hybrid computing using a neural network with dynamic external memory](https://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)
* [Bayesian Recurrent Neural Networks](https://arxiv.org/abs/1704.02798)
  * [A Tensorflow implementation](https://github.com/DeNeutoy/bayesian-rnn)
  * [Another TF implementation](https://github.com/mirceamironenco/BayesianRecurrentNN)
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
* [Augmenting End-to-End Dialog Systems with Commonsense Knowledge](https://arxiv.org/abs/1709.05453)
* [Predictive representations can link model-based reinforcement learning to model-free mechanisms](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005768)
* [Neural Task Programming: Learning to Generalize Across Hierarchical Tasks](https://arxiv.org/abs/1710.01813)
* [Understanding Generalization and Stochastic Gradient Descent](https://arxiv.org/abs/1710.06451)
  * Includes how to choose the best batch size for test set accuracy.
* [A generative vision model that trains with high data efficiency and breaks text-based CAPTCHAs](http://science.sciencemag.org/content/early/2017/10/25/science.aag2612.full)
* [Generalized Grounding Graphs: A Probabilistic Framework for Understanding Grounded Commands](https://arxiv.org/abs/1712.01097)
* [Peephole: Predicting Network Performance Before Training](https://arxiv.org/abs/1712.03351)
* [Peano-HASEL actuators: Muscle-mimetic, electrohydraulic transducers that linearly contract on activation](http://robotics.sciencemag.org/content/3/14/eaar3276)
* [Hydraulically amplified self-healing electrostatic actuators with muscle-like performance](http://science.sciencemag.org/content/359/6371/61)
* [Unsupervised Low-Dimensional Vector Representations for Words, Phrases and Text that are Transparent, Scalable, and produce Similarity Metrics that are Complementary to Neural Embeddings](https://arxiv.org/abs/1801.01884)
* [Emergent complexity via multi-agent competition](https://arxiv.org/abs/1710.03748)
* [PRNN: Recurrent Neural Network with Persistent Memory](https://arxiv.org/abs/1801.08094)
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
  * Might be useful to classify task descriptions for a multi-task system.
* [Directly Estimating the Variance of the λ-Return Using Temporal-Difference Methods](https://arxiv.org/abs/1801.08287)
* [Scalable Meta-Learning for Bayesian Optimization](https://arxiv.org/abs/1802.02219)
* [Learning to Play with Intrinsically-Motivated Self-Aware Agents](https://arxiv.org/abs/1802.07442)
* [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)
* [Accelerated Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1803.02811). Adam Stooke and Pieter Abbeel
* [Learning and Querying Fast Generative Models for Reinforcement Learning](https://arxiv.org/abs/1802.03006)
* [Learning by Playing - Solving Sparse Reward Tasks from Scratch](https://arxiv.org/abs/1802.10567)
* [Selective Experience Replay for Lifelong Learning](https://arxiv.org/abs/1802.10269)
* [Semi-Parametric Topological Memory For Navigation](https://openreview.net/pdf?id=SygwwGbRW)
* [Shifting Mean Activation Towards Zero with Bipolar Activation Functions](https://arxiv.org/abs/1709.04054)
  * Alternative to Batch Norm for normalization
* [Strategic attentive writer for learning macro-actions](https://arxiv.org/abs/1606.04695)
* [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
  * Sounds like this has some of the benefits of model ensembles, but at training time instead of run/inference time.
* [The Limits and Potentials of Deep Learning for Robotics](https://arxiv.org/abs/1804.06557)

## Autoencoders

* [Density Estimation: A Neurotically In-Depth Look At Variational Autoencoders](http://ruishu.io/2018/03/14/vae/)
* [Variational autoencoders](https://www.jeremyjordan.me/variational-autoencoders/)

## Classes/Education

* [CS 294: Deep Reinforcement Learning](http://rll.berkeley.edu/deeprlcourse/)
* [CMU 10703, Spring 2017 Deep Reinforcement Learning and Control](https://katefvision.github.io)
* [Stanford CS234: Reinforcement Learning](http://web.stanford.edu/class/cs234/index.html)
* [David Silver's UCL Course on RL](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
* [Deep Learning (DLSS) and Reinforcement Learning (RLSS) Summer School, Montreal 2017 (videos)](http://videolectures.net/deeplearning2017_montreal/)
* [Deep RL Bootcamp (Aug 2017, Berkeley](https://sites.google.com/view/deep-rl-bootcamp/lectures)
* [Theories of Deep Learning (STATS 385), Stanford 2017](https://stats385.github.io)
* [Bayesian Deeplearning](http://bayesiandeeplearning.org)
* [CS20: TensorFlow for Deep Learning Research](https://cs20.stanford.edu)
* [A list of fifteen more classes](https://sky2learn.com/deep-learning-reinforcement-learning-online-courses-and-tutorials-theory-and-applications.html) (Some overlap)
* [Hierarchical RL Workshop](https://sites.google.com/view/hrlnips2017)
  * Includes lectures by David Silver
* [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/) (15 hours, lessons, videos, exercises)

## Simulators

* [HoME: a Household Multimodal Environment](https://arxiv.org/abs/1711.11017)
  * [Github](https://github.com/HoME-Platform/home-platform), [Website](https://home-platform.github.io)
* [MINOS: Multimodal Indoor Simulator](https://minosworld.github.io)
* [CHALET: Cornell House Agent Learning Environment](https://arxiv.org/abs/1801.07357)

---
