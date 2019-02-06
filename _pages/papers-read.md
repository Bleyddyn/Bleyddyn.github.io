---
layout: archive
title: "Papers Read"
permalink: /papers-read
author_profile: true
usemath: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

# Sections

* [Papers read, with notes](#papers-read-with-notes)
* [Papers read, with minimal notes](#papers-read-with-minimal-notes)
* [Papers to Read](/papers-to-read)

## Papers read, with notes

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

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
    * [Code](https://github.com/haarnoja/softqlearning)
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
  * [Implementation from DeepMind](https://github.com/deepmind/scalable_agent)
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
* [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
* [Episodic Memory Deep Q-Networks](https://arxiv.org/abs/1805.07603)
  * A variant of experience replay (H) that keeps only the highest rewarding epsiodes for each state (action? state/action?) and adds a second loss term of lambda * (Q(s,a) - H(s,a))^2.
  * Supposed to be more sample efficient.
* [Why you need to improve your training data, and how to do it](https://petewarden.com/2018/05/28/why-you-need-to-improve-your-training-data-and-how-to-do-it/)
  * Lot's of stuff in there but calculating a confusion matrix seems like a good second step (I already look at samples of the data).
  * What's the equivalent of a confusion matrix for continuous outputs? A plot with output range on one axis and standard deviation on the other?
* [Been There, Done That: Meta-Learning with Episodic Recall](https://arxiv.org/abs/1805.09692)
  * They use a key/value memory that feeds past data into the LSTM not concatenated to the input, but through a new 'reinstatement' gate that works alongside the forget and input gates.
* [Hierarchical Reinforcement Learning with Hindsight](https://arxiv.org/abs/1805.08180)
* [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114)
  * Two types of uncertainty
    * "Aleatoric uncertainty, arises from inherent stochasticities of a system, e.g. observation noise and process noise."
    * "Epistemic uncertainty – corresponds to subjective uncertainty about the dynamics function, due to a lack of sufficient data to uniquely determine the underlying system exactly."
  * Their algorithm is: probabilistic ensembles with trajectory sampling (PETS)
* [Memory Augmented Self-Play](https://arxiv.org/abs/1805.11016)
  * Followup to: Sukhbaatar, et. al. Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play. ArXiv e-prints, March 2017.
  * Episodic feature vectors from past steps are passed through an LSTM and the hidden state (?) is used a input to the current time step.
* [Integrating Episodic Memory into a Reinforcement Learning Agent using Reservoir Sampling](https://arxiv.org/abs/1806.00540)
  * Main idea is a way to choose which states to save to memory when training in an online way, with no experience replay.
* [Unsupervised learning in LSTM recurrent neural networks](ftp://ftp.idsia.ch/pub/juergen/icann2001unsup.pdf)
  * Possible method for clustering accelerometer data for a reward function
* [Vision Based Multi-task Manipulation for Inexpensive Robots Using End-to-End Learning from Demonstration](https://arxiv.org/abs/1707.02920)
  * Includes a diagram for how to add a GAN to the network as an auxiliary task
* [Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam](https://arxiv.org/abs/1806.04854)
  * Variational replacements for Adam, AdaGrad and RMSProp that include uncertainty estimates in the gradient updates.
  * Includes an RL example where the uncertainty is used for exploration.
  * Might be worth trying if I can find an implementation.
* [An Intriguing Failing of Convolutional Neural Networks and the CoordConv  Solution](https://arxiv.org/abs/1807.03247)
  * [Blog Post](https://eng.uber.com/coordconv/)
  * Definitely want to try this both in my MaLPi code and in the VAE portion of World Models.
* [Learning to Drive in a Day](https://arxiv.org/abs/1807.00412)
  * Add a VAE to the convolutional layers as an auxiliary task.
* Recurrent Predictive State Policy Networks
  * Use a predictive state representation instead of an RNN and a simple feed-forward ‘reactive’ policy.
  * Needs to be initialized to work well.
  * Sounds like a version of the MERLIN algorithm, possibly with more theoretically grounded predictions.
* [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
  * [A blog post about it](https://flyyufelix.github.io/2017/10/24/distributional-bellman.html)
  * [And Keras code](https://github.com/flyyufelix/C51-DDQN-Keras)
  * Instead of learning a single value for expected reward, learn a distribution.
* [Robot Learning in Homes: Improving Generalization and Reducing Dataset Bias](https://arxiv.org/abs/1807.07049)
  * 6 DOF, $3k robot arm. Collect 28k grasps in 6 different homes (cluttered, noisy training data). Noise is modeled with one network (includes arm ID as one input) and grasping angle with another. Model generalizes better when trained this way. Noise modelling gives another 10% improvement.
* [Episodic Curiosity through Reachability](https://arxiv.org/abs/1810.02274)
  * Episodic Curiosity Module: Embedding network + Comparator network, Memory buffer with M embeddings (replace randomly once full), Reward Bonus Estimator.
    *   O -(embed)-> e
    *   Compare e with all M
    *   Similarity score C(M,e) = F(c1,...,cm), F = max is prone to outliers, so use F = 90th percentile (?)
    *   Curiosity bonus b = B(M, e) = α(β − C(M, e)), β = 0.5 works well. α depends on reward scale.
    *   If bonus is > a novelty threshold, add e to memory. bnovelty = 0 works well.
  * Reachability network training: This procedure takes as input a sequence of observations o1 , . . . , oN and forms pairs from those observations. The pairs (oi , oj) where \|i − j \| ≤ k are taken as positive (reachable) examples while the pairs with \|i − j\| > γk become negative examples. The hyperparameter γ is necessary to create a gap between positive and negative examples. In the end, the network is trained with logistic regression loss to output the probability of the positive (reachable) class.
  * Training the agent is PPO with the environment reward plus the bonus reward.
* [Empiricism and the limits of gradient descent](http://togelius.blogspot.com/2018/05/empiricism-and-limits-of-gradient.html)
  * Logical Empiricism: sense impressions are all there is. They cause/induce knowledge as our minds generalize over sense impressions.
  * Critical Rationalism: Karl Popper. We formulate hypotheses and test them against sense impressions
  * “learning by gradient descent is an implementation of empiricist induction, whereas evolutionary computation is much closer to the hypothetico-deductive process of Popper's critical rationalism.”
* [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
* [Learning Shared Dynamics with Meta-World Models](https://arxiv.org/abs/1811.01741)
* [Experience Replay for Continual Learning](https://arxiv.org/abs/1811.11682)
* [IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis](https://arxiv.org/abs/1807.06358)
  * VAE+GAN in a simple network.
* [The Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning](https://storage.googleapis.com/obstacle-tower-build/Obstacle_Tower_Paper_Final.pdf)
* [Revisiting Self-Supervised Visual Representation Learning](https://arxiv.org/abs/1901.09005)
  * "self-supervised learning techniques are likely to benefit from using CNNs with increased number of channels across wide range of scenarios."
  * Skip connections help maintain generalizability in later layers (not so useful for MaLPi/DonkeyCar).
  * Multiple different types of self-supervision are tried: Predict rotation (0°, 90°, 180°, 270°), each image is its own example and must be identified after augmentation (translation, scaling, rotation, etc), Jigsaw (permute image patches and predict permutation). Patch based methods need to do some kind of averaging over patches when used for the down-stream task.

---
